"""EarlyExitStatsValidator — gate diagnostics for LoopICLEarlyExit on real benchmarks.

Runs the model with ``return_exit_info=True`` on a sample of benchmark datasets
and logs the exit-gate distribution:

    ee_stats/{tag}/avg_exit_layer   — expected loop index at exit (1-indexed)
    ee_stats/{tag}/exit_entropy     — entropy of the exit distribution
    ee_stats/{tag}/p_exit_loop_{t}  — mean exit probability at loop t
    ee_stats/{tag}/acc_loop_{t}     — classification accuracy if exited at loop t
    ee_stats/{tag}/acc_full         — accuracy at the final loop (all loops run)
    ee_stats/{tag}/acc_ee           — accuracy when using exit_q threshold
    ee_stats/{tag}/acc_delta        — acc_ee - acc_full (positive = ee wins)
    ee_stats/{tag}/avg_baseline_ms  — wall-clock time, full loops (tau=1.0)
    ee_stats/{tag}/avg_ee_ms        — wall-clock time, early exit at exit_q
    ee_stats/{tag}/actual_speedup   — avg_baseline_ms / avg_ee_ms

``acc_ee`` is computed by selecting, for each dataset, the logits from the first
loop t where CDF_t = 1 − S_t exceeds exit_q (matching inference behaviour).
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import torch
import torch.nn as nn

from loopicl.evaluation.benchmark import Benchmark
from loopicl.evaluation.setup import iter_benchmark_datasets
from loopicl.validation.base import BaseValidator


class EarlyExitStatsValidator(BaseValidator):
    """Collect exit-gate statistics from LoopICLEarlyExit on real benchmark datasets.

    Parameters
    ----------
    benchmark : Benchmark
        Benchmark definition (same one used by BenchmarkValidator).
    benchmark_dir : str
        Root directory holding the benchmark datasets.
    exit_q : float
        Quantile threshold used for early-exit inference (matches model's exit_q).
        Default: 0.5.
    max_datasets : int
        Maximum number of individual datasets to evaluate.  Gate stats are fast
        but this prevents validation time from growing unboundedly.  Default: 20.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        benchmark_dir: str,
        exit_q: float = 0.5,
        max_datasets: int = 20,
    ):
        self.benchmark = benchmark
        self.benchmark_dir = benchmark_dir
        self.exit_q = exit_q
        self.max_datasets = max_datasets

    @property
    def name(self) -> str:
        return "ee_stats"

    def validate(
        self,
        model: nn.Module,
        device: str,
        step: int | None = None,
        dtype: str = "float32",
    ) -> dict[str, float]:
        model.eval()

        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        amp_dtype = _dtype_map.get(dtype)
        use_cuda = "cuda" in str(device)

        # Accumulators
        all_avg_exit:    list[float]        = []
        all_entropy:     list[float]        = []
        all_p_exit:      list[list[float]]  = []
        all_acc_t:       list[list[float]]  = []
        all_acc_ee:      list[float]        = []
        all_baseline_ms: list[float]        = []
        all_ee_ms:       list[float]        = []

        n_collected = 0

        for _, ds_task, datasets in iter_benchmark_datasets(
            self.benchmark, self.benchmark_dir, "cls", prior_n_val=True
        ):
            if ds_task != "cls":
                continue  # gate stats are only meaningful for classification
            for X_train, X_test, y_train, y_test in datasets:
                if n_collected >= self.max_datasets:
                    break

                X_full = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
                X_t    = torch.as_tensor(X_full, device=device).unsqueeze(0)   # (1, N, C)
                y_t    = torch.as_tensor(y_train.astype(np.float32), device=device).unsqueeze(0)
                y_test_t = torch.as_tensor(y_test.astype(np.int64), device=device)
                train_size = len(y_train)

                try:
                    with torch.no_grad():
                        ctx = (
                            torch.autocast(device_type="cuda", dtype=amp_dtype)
                            if amp_dtype is not None and use_cuda
                            else torch.autocast(device_type="cpu", enabled=False)
                        )
                        with ctx:
                            # --- Baseline: full loops, timed ---
                            if use_cuda:
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            all_logits_full, all_lambdas = model(
                                X_t, y_t,
                                train_size=train_size,
                                return_exit_info=True,
                            )
                            if use_cuda:
                                torch.cuda.synchronize()
                            baseline_ms = (time.perf_counter() - t0) * 1000

                            # --- Early-exit inference, timed ---
                            if use_cuda:
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            logits_ee = model(
                                X_t, y_t,
                                train_size=train_size,
                                exit_q=self.exit_q,
                            )
                            if use_cuda:
                                torch.cuda.synchronize()
                            ee_ms = (time.perf_counter() - t0) * 1000

                    # all_logits_full : List[T] of (M=1, B=1, T_cls)
                    # all_lambdas     : (T, B=1)
                    T = len(all_logits_full)
                    B = all_lambdas.shape[1]  # == 1

                    # Exit distribution p_ϕ(t) from the full-loop pass
                    one_minus = 1.0 - all_lambdas                          # (T, 1)
                    survival  = torch.cumprod(one_minus, dim=0)            # (T, 1)
                    S_prev    = torch.cat([
                        torch.ones(1, B, device=survival.device, dtype=survival.dtype),
                        survival[:-1],
                    ], dim=0)
                    p_exit = all_lambdas * S_prev
                    p_exit = p_exit / (p_exit.sum(0, keepdim=True) + 1e-8)  # (T, 1)

                    steps    = torch.arange(1, T + 1, device=all_lambdas.device, dtype=all_lambdas.dtype)
                    avg_exit = (p_exit * steps.unsqueeze(1)).sum(0).mean()
                    entropy  = -(p_exit * p_exit.clamp(min=1e-8).log()).sum(0).mean()

                    # Per-loop accuracy
                    n_test = len(y_test)
                    acc_per_loop: List[float] = []
                    for logits_t in all_logits_full:
                        # logits_t: (M=1, B=1, T_cls) → (n_test, T_cls) after squeeze
                        pred = logits_t.squeeze(0).squeeze(0)[:n_test]    # (n_test, T_cls)
                        acc = (pred.argmax(-1).cpu() == y_test_t.cpu()).float().mean().item()
                        acc_per_loop.append(acc)

                    # acc_full: accuracy at the final loop
                    acc_full = acc_per_loop[-1]

                    # acc_ee: accuracy from the early-exit forward pass
                    # logits_ee: (M=1, B=1, T_cls) → squeeze to (n_test, T_cls)
                    pred_ee = logits_ee.squeeze(0).squeeze(0)[:n_test]
                    acc_ee  = (pred_ee.argmax(-1).cpu() == y_test_t.cpu()).float().mean().item()

                    all_avg_exit.append(avg_exit.item())
                    all_entropy.append(entropy.item())
                    all_p_exit.append([p_exit[t, 0].item() for t in range(T)])
                    all_acc_t.append(acc_per_loop)
                    all_acc_ee.append(acc_ee)
                    all_baseline_ms.append(baseline_ms)
                    all_ee_ms.append(ee_ms)
                    n_collected += 1

                except Exception as exc:
                    print(f"[EarlyExitStatsValidator] Skipping dataset: {exc}")
                    continue

            if n_collected >= self.max_datasets:
                break

        if not all_avg_exit:
            return {}

        n   = len(all_avg_exit)
        tag = f"q{int(self.exit_q * 100):02d}"

        mean_acc_full    = float(np.mean([acc_per_loop[-1] for acc_per_loop in all_acc_t]))
        mean_acc_ee      = float(np.mean(all_acc_ee))
        mean_baseline_ms = float(np.mean(all_baseline_ms))
        mean_ee_ms       = float(np.mean(all_ee_ms))

        stats: dict[str, float] = {
            f"ee_stats/{tag}/avg_exit_layer":  float(np.mean(all_avg_exit)),
            f"ee_stats/{tag}/exit_entropy":    float(np.mean(all_entropy)),
            f"ee_stats/{tag}/acc_full":        mean_acc_full,
            f"ee_stats/{tag}/acc_ee":          mean_acc_ee,
            f"ee_stats/{tag}/acc_delta":       mean_acc_ee - mean_acc_full,
            f"ee_stats/{tag}/avg_baseline_ms": mean_baseline_ms,
            f"ee_stats/{tag}/avg_ee_ms":       mean_ee_ms,
            f"ee_stats/{tag}/actual_speedup":  mean_baseline_ms / max(mean_ee_ms, 1e-3),
        }

        # Per-loop averages (p_exit and acc)
        T = len(all_p_exit[0]) if all_p_exit else 0
        for t in range(T):
            stats[f"ee_stats/{tag}/p_exit_loop_{t + 1}"] = (
                float(np.mean([b[t] for b in all_p_exit if t < len(b)]))
            )
            stats[f"ee_stats/{tag}/acc_loop_{t + 1}"] = (
                float(np.mean([b[t] for b in all_acc_t if t < len(b)]))
            )

        return stats
