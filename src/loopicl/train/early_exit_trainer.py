"""EarlyExitTrainer — Trainer subclass for Ouro-style early exit training.

Reads from config.training:
    early_exit_beta : float         = 0.05    entropy coefficient (Stage I)
    stage1_steps    : int | None    = 18_000  auto-switch to Stage II after N steps
    gate_only_stage : bool          = False   manual Stage II override (used when
                                              stage1_steps is None)

Automated two-stage schedule (stage1_steps is set):
    Steps 0 … stage1_steps-1  →  Stage I  (joint LM + gate, entropy-regularised loss)
    Steps stage1_steps … end  →  Stage II (LM frozen, gate BCE loss)
    The transition freezes LM params exactly once, in-place.

Stage I loss:
    ℒ = Σₜ p_ϕ(t) · ℒ^(t)  −  β · H(p_ϕ(·))

Stage II loss:
    I_i^(t)  = max(0, ℒ_i^(t-1) − ℒ_i^(t))
    w_i^(t)  = σ(k · (mean_i(I_i^(t)) − γ))
    ℒ_gate   = BCE(1 − λₜ,  w^(t))

EarlyExitTrainerV2 extends EarlyExitTrainer to support both classification
(CE loss) and quantile regression (pinball loss). Task type is read from
``config.prior.prior_tasktype``.

Reference: Ouro (arxiv 2510.25741)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from loopicl.evaluation.builtin import get_benchmark
from loopicl.train.trainer import Trainer, _validate_micro_batch, _align_micro_batch
from loopicl.validation.early_exit_val import EarlyExitStatsValidator

__all__ = ["EarlyExitTrainer", "EarlyExitTrainerV2"]

# Stage II gate supervision hyperparameters (from paper, Table 1)
_GATE_K:     float = 50.0
_GATE_GAMMA: float = 0.005


class EarlyExitTrainer(Trainer):
    """Trainer that implements Ouro Stage I / Stage II early exit losses.

    Supports fully automated stage switching via config.training.stage1_steps.
    """

    # ------------------------------------------------------------------
    # Build model — handle manual Stage II freezing at startup
    # ------------------------------------------------------------------

    def build_model(self):
        """Build model; freeze LM now only if starting directly in Stage II."""
        super().build_model()
        self._stage2_active = False  # tracks whether LM is already frozen

        stage1_steps = getattr(self.config.training, "stage1_steps", None)
        gate_only    = getattr(self.config.training, "gate_only_stage", False)

        # Manual mode: gate_only_stage=True with no automated schedule
        if gate_only and stage1_steps is None:
            self._freeze_lm()

    def _freeze_lm(self):
        """Freeze all params except exit_gate. Called once at Stage II entry."""
        if self._stage2_active:
            return
        frozen, trainable = 0, 0
        for name, param in self.raw_model.named_parameters():
            if name.startswith("exit_gate"):
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        self._stage2_active = True
        if self.master_process:
            print(
                f"[Step {self.curr_step}] Stage II: froze {frozen:,} LM params, "
                f"training {trainable:,} gate params only."
            )

    # ------------------------------------------------------------------
    # Validation — add EarlyExitStatsValidator on real benchmark datasets
    # ------------------------------------------------------------------

    def configure_validation(self):
        """Standard benchmark validators + EarlyExitStatsValidator."""
        super().configure_validation()

        val_cfg = getattr(self.config, "validation", None)
        if val_cfg is None or not val_cfg.enabled:
            return

        ee_cfg = getattr(val_cfg, "early_exit_stats", None)
        if ee_cfg is None:
            return

        benchmarks = getattr(val_cfg, "benchmarks", [])
        if not benchmarks:
            return

        bcfg   = benchmarks[0]
        bmark  = get_benchmark(bcfg.benchmark)
        exit_q = getattr(ee_cfg, "exit_q",       0.5)
        max_ds = getattr(ee_cfg, "max_datasets", 20)

        self.validators.append(
            EarlyExitStatsValidator(
                benchmark=bmark,
                benchmark_dir=bcfg.benchmark_dir,
                exit_q=exit_q,
                max_datasets=max_ds,
            )
        )

    # ------------------------------------------------------------------
    # Micro-batch forward + loss
    # ------------------------------------------------------------------

    def _run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """Forward + backward with Stage I or Stage II early exit loss."""
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = _validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y   = _align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.training.device)
        micro_y = micro_y.to(self.config.training.device)
        micro_d = micro_d.to(self.config.training.device)

        if not torch.isfinite(micro_X).all():
            print(
                f"Warning: NaN/Inf in input X at micro-batch {micro_batch_idx}. Skipping."
            )
            return {"ce": 0.0, "accuracy": 0.0}

        y_train = micro_y[:, :train_size]
        y_test  = micro_y[:, train_size:]
        true    = y_test.long().flatten()          # (M*B,)

        beta         = getattr(self.config.training, "early_exit_beta", 0.05)
        stage1_steps = getattr(self.config.training, "stage1_steps", None)

        # Automated stage transition: freeze LM once we cross stage1_steps
        if stage1_steps is not None and self.curr_step >= stage1_steps:
            self._freeze_lm()

        gate_only = self._stage2_active

        if self.ddp:
            self.model.require_backward_grad_sync = False

        with self.amp_ctx:
            all_logits, all_lambdas = self.model(
                micro_X, y_train, train_size,
                return_exit_info=True,
            )
            # all_logits  : List[T] of Tensor (M, B, T_cls)
            # all_lambdas : Tensor (T, B)

            T = len(all_logits)
            B = all_lambdas.shape[1]

            # Exit distribution p_ϕ(t)
            one_minus = 1.0 - all_lambdas                              # (T, B)
            survival  = torch.cumprod(one_minus, dim=0)                # (T, B)
            S_prev    = torch.cat(
                [torch.ones(1, B, device=survival.device, dtype=survival.dtype),
                 survival[:-1]],
                dim=0,
            )                                                           # (T, B)
            p_exit = all_lambdas * S_prev                              # (T, B)
            p_exit = p_exit / (p_exit.sum(dim=0, keepdim=True) + 1e-8)

            # Per-step task losses
            M = true.shape[0] // B
            step_losses_per_sample = torch.stack([
                F.cross_entropy(logits_t.flatten(end_dim=-2), true, reduction="none")
                .view(M, B).mean(0)
                for logits_t in all_logits
            ])  # (T, B)
            step_losses = step_losses_per_sample.mean(dim=1)           # (T,)

            steps                = torch.arange(1, T + 1, device=all_lambdas.device, dtype=all_lambdas.dtype)
            avg_exit_per_dataset = (p_exit * steps.unsqueeze(1)).sum(dim=0)  # (B,)

            with torch.no_grad():
                optimal_loop = step_losses_per_sample.argmin(dim=0).float() + 1  # (B,)

            if gate_only:
                loss = self._stage2_loss(all_logits, all_lambdas, true, T, B)
            else:
                loss = self._stage1_loss(p_exit, step_losses_per_sample, beta)

            gap_loss_weight = getattr(self.config.training, "exit_gap_loss_weight", 0.0)
            if gap_loss_weight > 0.0:
                loss = loss + gap_loss_weight * ((avg_exit_per_dataset - optimal_loop) ** 2).mean()

            final_loss = step_losses[-1]

        if not torch.isfinite(loss):
            print(
                f"Warning: NaN/Inf loss at micro-batch {micro_batch_idx} "
                f"(step {self.curr_step}). Skipping."
            )
            return {"ce": 0.0, "accuracy": 0.0}

        self.scaler.scale(loss / num_micro_batches).backward()

        with torch.no_grad():
            pred           = all_logits[-1].flatten(end_dim=-2)
            accuracy       = (pred.argmax(dim=1) == true).float().mean()
            entropy        = -(p_exit * p_exit.clamp(min=1e-8).log()).sum(0).mean()
            avg_exit_layer = avg_exit_per_dataset.mean()
            exit_loop_gap  = (avg_exit_per_dataset.detach() - optimal_loop).mean()

            return {
                "ce":             final_loss.item()      / num_micro_batches,
                "accuracy":       accuracy.item()        / num_micro_batches,
                "exit_entropy":   entropy.item()         / num_micro_batches,
                "avg_exit_layer": avg_exit_layer.item()  / num_micro_batches,
                "exit_loop_gap":  exit_loop_gap.item()   / num_micro_batches,
            }

    # ------------------------------------------------------------------
    # Stage I — entropy-regularised expected loss
    # ------------------------------------------------------------------

    @staticmethod
    def _stage1_loss(
        p_exit:      torch.Tensor,  # (T, B)
        step_losses: torch.Tensor,  # (T, B)  per-sample per-step CE loss
        beta:        float,
    ) -> torch.Tensor:
        """ℒ = E_b[Σₜ p_ϕ(t|b) · ℒ^(t,b)]  −  β · H(p_ϕ(·))"""
        expected_loss = (p_exit * step_losses).sum(dim=0).mean()
        entropy = -(p_exit * p_exit.clamp(min=1e-8).log()).sum(dim=0).mean()
        return expected_loss - beta * entropy

    # ------------------------------------------------------------------
    # Stage II — gate BCE loss (frozen LM)
    # ------------------------------------------------------------------

    @staticmethod
    def _stage2_loss(
        all_logits:  "list[torch.Tensor]",  # List[T] of (M, B, T_cls)
        all_lambdas: torch.Tensor,           # (T, B)
        true:        torch.Tensor,           # (M*B,)
        T:           int,
        B:           int,
    ) -> torch.Tensor:
        """Gate BCE loss using best-future loss improvement as supervision."""
        M = true.shape[0] // B
        per_tok_per_b = torch.stack([
            F.cross_entropy(logits_t.flatten(end_dim=-2), true, reduction="none")
            .view(M, B).mean(0)
            for logits_t in all_logits
        ])  # (T, B)

        best_from_inclusive = per_tok_per_b.flip(0).cummin(0).values.flip(0)  # (T, B)
        best_future = best_from_inclusive[1:]  # (T-1, B)

        improvement = (per_tok_per_b[:-1] - best_future).clamp(min=0.0)  # (T-1, B)
        w = torch.sigmoid(_GATE_K * (improvement - _GATE_GAMMA))          # (T-1, B)

        best_loss_ever = per_tok_per_b.min(dim=0).values                   # (B,)
        regret = (per_tok_per_b[:-1] - best_loss_ever.unsqueeze(0)).clamp(min=0.0)  # (T-1, B)

        _eps = 1e-6
        continue_prob = 1.0 - all_lambdas[:-1]  # (T-1, B)
        gate_loss = F.binary_cross_entropy_with_logits(
            torch.logit(continue_prob.float().clamp(_eps, 1 - _eps)),
            w.detach().float(),
            weight=(1.0 + regret).detach().float(),
        )

        last_gate_loss = F.binary_cross_entropy_with_logits(
            torch.logit((1.0 - all_lambdas[-1]).float().clamp(_eps, 1 - _eps)),
            torch.zeros(B, device=all_lambdas.device, dtype=torch.float32),
        )

        return (gate_loss * (T - 1) + last_gate_loss) / T

    # ------------------------------------------------------------------
    # Validation — gate statistics
    # ------------------------------------------------------------------

    def run_validation(self) -> dict[str, float]:
        """Run benchmark validators, then append gate statistics."""
        results = super().run_validation()
        if self.master_process:
            results.update(self._gate_validation_stats())
        return results

    def _gate_validation_stats(self, n_batches: int = 8) -> dict[str, float]:
        """Sample prior batches and report per-loop exit probabilities and accuracy."""
        raw = self.raw_model.module if hasattr(self.raw_model, "module") else self.raw_model
        raw.eval()
        device  = self.config.training.device
        T_max   = raw.num_icl_loops
        mb_size = self.config.training.micro_batch_size

        all_avg_exit: list[float] = []
        all_entropy:  list[float] = []
        all_p_exit:   list[list[float]] = []
        all_acc_t:    list[list[float]] = []

        with torch.no_grad():
            for _ in range(n_batches):
                try:
                    batch = next(iter(self.dataloader))
                except StopIteration:
                    break

                batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
                sort_idx = torch.argsort(batch[-1])
                batch = [t[sort_idx] for t in batch]
                if len(batch) in (7, 8):
                    batch = [batch[0], batch[2], batch[-3], batch[-2], batch[-1]]

                micro_batch = tuple(t[:mb_size] for t in batch)
                micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
                seq_len, train_size = _validate_micro_batch(micro_seq_len, micro_train_size)
                micro_X, micro_y   = _align_micro_batch(micro_X, micro_y, micro_d, seq_len)
                micro_X = micro_X.to(device)
                micro_y = micro_y.to(device)

                if not torch.isfinite(micro_X).all():
                    continue

                y_train = micro_y[:, :train_size]
                y_test  = micro_y[:, train_size:]
                true    = y_test.long().flatten()
                B       = micro_X.shape[0]

                all_logits, all_lambdas = raw(
                    micro_X, y_train, train_size=train_size,
                    return_exit_info=True,
                    num_loops=T_max,
                )
                T = len(all_logits)

                one_minus = 1.0 - all_lambdas
                survival  = torch.cumprod(one_minus, dim=0)
                S_prev    = torch.cat([
                    torch.ones(1, B, device=survival.device, dtype=survival.dtype),
                    survival[:-1],
                ], dim=0)
                p_exit = all_lambdas * S_prev
                p_exit = p_exit / (p_exit.sum(0, keepdim=True) + 1e-8)

                steps    = torch.arange(1, T + 1, device=all_lambdas.device, dtype=all_lambdas.dtype)
                avg_exit = (p_exit * steps.unsqueeze(1)).sum(0).mean()
                entropy  = -(p_exit * p_exit.clamp(min=1e-8).log()).sum(0).mean()

                all_avg_exit.append(avg_exit.item())
                all_entropy.append(entropy.item())
                all_p_exit.append([p_exit[t].mean().item() for t in range(T)])

                M = true.shape[0] // B
                all_acc_t.append([
                    (logits_t.flatten(end_dim=-2).argmax(-1) == true).float().mean().item()
                    for logits_t in all_logits
                ])

        raw.train()

        n = len(all_avg_exit)
        if n == 0:
            return {}

        stats: dict[str, float] = {
            "val_ee/avg_exit_layer": sum(all_avg_exit) / n,
            "val_ee/exit_entropy":   sum(all_entropy)  / n,
        }
        T = len(all_p_exit[0]) if all_p_exit else 0
        for t in range(T):
            stats[f"val_ee/p_exit_loop_{t + 1}"] = (
                sum(b[t] for b in all_p_exit if t < len(b)) / n
            )
            stats[f"val_ee/acc_loop_{t + 1}"] = (
                sum(b[t] for b in all_acc_t if t < len(b)) / n
            )
        return stats


# ---------------------------------------------------------------------------
# EarlyExitTrainerV2 — task-agnostic extension (classification + regression)
# ---------------------------------------------------------------------------


def _pinball_per_sample(
    logits_t: torch.Tensor,   # (M, B, Q) quantile predictions
    y_test:   torch.Tensor,   # (B, M)    continuous targets
) -> torch.Tensor:            # (B,)
    """Mean pinball loss per batch element, averaged over M test points and Q quantiles."""
    q = logits_t.sort(dim=-1).values       # enforce monotonicity (M, B, Q)
    Q = q.shape[-1]
    alphas = torch.linspace(0.0, 1.0, Q + 2, device=q.device, dtype=q.dtype)[1:-1]
    y = y_test.float().T.unsqueeze(-1)     # (M, B, 1)
    errors = y - q                         # (M, B, Q)
    pb = torch.maximum(alphas * errors, (alphas - 1) * errors)  # (M, B, Q)
    return pb.mean(dim=(0, 2))             # (B,)


class EarlyExitTrainerV2(EarlyExitTrainer):
    """EarlyExitTrainer extended to support quantile-regression tasks.

    Only ``_run_micro_batch``, ``_stage2_loss_v2``, and
    ``_gate_validation_stats`` are overridden.  Everything else
    (stage scheduling, freezing, validation plumbing) is inherited.
    """

    # ------------------------------------------------------------------
    # Micro-batch forward + loss
    # ------------------------------------------------------------------

    def _run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """Forward + backward; dispatches task loss by ``prior_tasktype``."""
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = _validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y   = _align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.training.device)
        micro_y = micro_y.to(self.config.training.device)
        micro_d = micro_d.to(self.config.training.device)

        if not torch.isfinite(micro_X).all():
            print(
                f"Warning: NaN/Inf in input X at micro-batch {micro_batch_idx}. Skipping."
            )
            return {"ce": 0.0, "accuracy": 0.0}

        task_type = getattr(self.config.prior, "prior_tasktype", "classification")
        is_reg    = task_type == "regression"

        y_train = micro_y[:, :train_size]
        y_test  = micro_y[:, train_size:]    # (B, M)

        beta         = getattr(self.config.training, "early_exit_beta",  0.05)
        stage1_steps = getattr(self.config.training, "stage1_steps",     None)

        if stage1_steps is not None and self.curr_step >= stage1_steps:
            self._freeze_lm()

        gate_only = self._stage2_active

        if self.ddp:
            self.model.require_backward_grad_sync = False

        with self.amp_ctx:
            all_logits, all_lambdas = self.model(
                micro_X, y_train, train_size,
                return_exit_info=True,
            )
            # all_logits  : List[T] of (M, B, C) or (M, B, Q)
            # all_lambdas : (T, B)

            T = len(all_logits)
            B = all_lambdas.shape[1]
            M = y_test.shape[1]

            # Per-loop per-sample losses — shape (T, B)
            if is_reg:
                step_losses_per_sample = torch.stack([
                    _pinball_per_sample(logits_t, y_test)
                    for logits_t in all_logits
                ])  # (T, B)
            else:
                true = y_test.long().flatten()   # (M*B,)
                step_losses_per_sample = torch.stack([
                    F.cross_entropy(
                        logits_t.flatten(end_dim=-2), true, reduction="none"
                    ).view(M, B).mean(0)
                    for logits_t in all_logits
                ])  # (T, B)

            step_losses = step_losses_per_sample.mean(dim=1)   # (T,)

            # Exit distribution p_ϕ(t)
            one_minus = 1.0 - all_lambdas                              # (T, B)
            survival  = torch.cumprod(one_minus, dim=0)                # (T, B)
            S_prev    = torch.cat([
                torch.ones(1, B, device=survival.device, dtype=survival.dtype),
                survival[:-1],
            ], dim=0)                                                   # (T, B)
            p_exit = all_lambdas * S_prev
            p_exit = p_exit / (p_exit.sum(dim=0, keepdim=True) + 1e-8)

            steps                = torch.arange(1, T + 1, device=all_lambdas.device,
                                                dtype=all_lambdas.dtype)
            avg_exit_per_dataset = (p_exit * steps.unsqueeze(1)).sum(dim=0)  # (B,)

            with torch.no_grad():
                optimal_loop = step_losses_per_sample.argmin(dim=0).float() + 1  # (B,)

            if gate_only:
                loss = self._stage2_loss_v2(step_losses_per_sample, all_lambdas, T, B)
            else:
                loss = self._stage1_loss(p_exit, step_losses_per_sample, beta)

            gap_loss_weight = getattr(self.config.training, "exit_gap_loss_weight", 0.0)
            if gap_loss_weight > 0.0:
                gap = avg_exit_per_dataset - optimal_loop
                late_mult = getattr(self.config.training, "exit_gap_late_multiplier", 2.0)
                weighted_gap = torch.where(gap > 0, late_mult * gap, gap)
                loss = loss + gap_loss_weight * (weighted_gap ** 2).mean()

            final_loss = step_losses[-1]

        if not torch.isfinite(loss):
            print(
                f"Warning: NaN/Inf loss at micro-batch {micro_batch_idx} "
                f"(step {self.curr_step}). Skipping."
            )
            return {"loss": 0.0}

        self.scaler.scale(loss / num_micro_batches).backward()

        with torch.no_grad():
            entropy        = -(p_exit * p_exit.clamp(min=1e-8).log()).sum(0).mean()
            avg_exit_layer = avg_exit_per_dataset.mean()
            exit_loop_gap  = (avg_exit_per_dataset.detach() - optimal_loop).mean()

            metrics: dict = {
                "loss":           final_loss.item()     / num_micro_batches,
                "exit_entropy":   entropy.item()         / num_micro_batches,
                "avg_exit_layer": avg_exit_layer.item()  / num_micro_batches,
                "exit_loop_gap":  exit_loop_gap.item()   / num_micro_batches,
            }

            if not is_reg:
                true     = y_test.long().flatten()
                pred     = all_logits[-1].flatten(end_dim=-2)
                accuracy = (pred.argmax(dim=1) == true).float().mean()
                metrics["accuracy"] = accuracy.item() / num_micro_batches
            else:
                metrics["pinball"] = final_loss.item() / num_micro_batches

        return metrics

    # ------------------------------------------------------------------
    # Stage II — task-agnostic BCE gate loss
    # ------------------------------------------------------------------

    @staticmethod
    def _stage2_loss_v2(
        step_losses: torch.Tensor,   # (T, B) per-sample per-step loss (any task)
        all_lambdas: torch.Tensor,   # (T, B)
        T: int,
        B: int,
    ) -> torch.Tensor:
        """Task-agnostic gate BCE loss."""
        best_from_inclusive = step_losses.flip(0).cummin(0).values.flip(0)  # (T, B)
        best_future         = best_from_inclusive[1:]                         # (T-1, B)

        improvement = step_losses[:-1] - best_future                          # (T-1, B)
        w = torch.sigmoid(_GATE_K * (improvement - _GATE_GAMMA))             # (T-1, B)

        best_loss_ever = step_losses.min(dim=0).values                        # (B,)
        regret = (step_losses[:-1] - best_loss_ever.unsqueeze(0)).clamp(min=0.0)  # (T-1, B)

        _eps         = 1e-6
        continue_prob = 1.0 - all_lambdas[:-1]                               # (T-1, B)
        gate_loss = F.binary_cross_entropy_with_logits(
            torch.logit(continue_prob.float().clamp(_eps, 1 - _eps)),
            w.detach().float(),
            weight=(1.0 + regret).detach().float(),
        )

        last_gate_loss = F.binary_cross_entropy_with_logits(
            torch.logit((1.0 - all_lambdas[-1]).float().clamp(_eps, 1 - _eps)),
            torch.zeros(B, device=all_lambdas.device, dtype=torch.float32),
        )

        return (gate_loss * (T - 1) + last_gate_loss) / T

    # ------------------------------------------------------------------
    # Validation gate statistics (task-aware)
    # ------------------------------------------------------------------

    def _gate_validation_stats(self, n_batches: int = 8) -> dict[str, float]:
        """Gate stats per loop; reports task-appropriate quality metric."""
        raw = self.raw_model.module if hasattr(self.raw_model, "module") else self.raw_model
        raw.eval()
        device    = self.config.training.device
        T_max     = raw.num_icl_loops
        mb_size   = self.config.training.micro_batch_size
        task_type = getattr(self.config.prior, "prior_tasktype", "classification")
        is_reg    = task_type == "regression"

        all_avg_exit:  list[float]       = []
        all_entropy:   list[float]       = []
        all_p_exit:    list[list[float]] = []
        all_metric_t:  list[list[float]] = []

        with torch.no_grad():
            for _ in range(n_batches):
                try:
                    batch = next(iter(self.dataloader))
                except StopIteration:
                    break

                batch    = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
                sort_idx = torch.argsort(batch[-1])
                batch    = [t[sort_idx] for t in batch]
                if len(batch) in (7, 8):
                    batch = [batch[0], batch[2], batch[-3], batch[-2], batch[-1]]

                micro_batch = tuple(t[:mb_size] for t in batch)
                micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
                seq_len, train_size = _validate_micro_batch(micro_seq_len, micro_train_size)
                micro_X, micro_y   = _align_micro_batch(micro_X, micro_y, micro_d, seq_len)
                micro_X = micro_X.to(device)
                micro_y = micro_y.to(device)
                if not torch.isfinite(micro_X).all():
                    continue

                y_train = micro_y[:, :train_size]
                y_test  = micro_y[:, train_size:]
                B       = micro_X.shape[0]

                all_logits_b, all_lambdas_b = raw(
                    micro_X, y_train, train_size=train_size,
                    return_exit_info=True, num_loops=T_max,
                )
                T = len(all_logits_b)

                one_minus = 1.0 - all_lambdas_b
                survival  = torch.cumprod(one_minus, dim=0)
                S_prev    = torch.cat([
                    torch.ones(1, B, device=survival.device, dtype=survival.dtype),
                    survival[:-1],
                ], dim=0)
                p_exit = all_lambdas_b * S_prev
                p_exit = p_exit / (p_exit.sum(0, keepdim=True) + 1e-8)

                steps    = torch.arange(1, T + 1, device=all_lambdas_b.device,
                                        dtype=all_lambdas_b.dtype)
                avg_exit = (p_exit * steps.unsqueeze(1)).sum(0).mean()
                entropy  = -(p_exit * p_exit.clamp(min=1e-8).log()).sum(0).mean()

                all_avg_exit.append(avg_exit.item())
                all_entropy.append(entropy.item())
                all_p_exit.append([p_exit[t].mean().item() for t in range(T)])

                if is_reg:
                    all_metric_t.append([
                        _pinball_per_sample(logits_t, y_test).mean().item()
                        for logits_t in all_logits_b
                    ])
                else:
                    true = y_test.long().flatten()
                    all_metric_t.append([
                        (logits_t.flatten(end_dim=-2).argmax(-1) == true).float().mean().item()
                        for logits_t in all_logits_b
                    ])

        raw.train()
        n = len(all_avg_exit)
        if n == 0:
            return {}

        metric_name = "pinball_loop" if is_reg else "acc_loop"
        stats: dict[str, float] = {
            "val_ee/avg_exit_layer": sum(all_avg_exit) / n,
            "val_ee/exit_entropy":   sum(all_entropy)  / n,
        }
        T = len(all_p_exit[0]) if all_p_exit else 0
        for t in range(T):
            stats[f"val_ee/p_exit_loop_{t + 1}"] = (
                sum(b[t] for b in all_p_exit   if t < len(b)) / n
            )
            stats[f"val_ee/{metric_name}_{t + 1}"] = (
                sum(b[t] for b in all_metric_t if t < len(b)) / n
            )
        return stats
