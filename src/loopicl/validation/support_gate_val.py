"""SupportGateStatsValidator — gate diagnostics for LoopICLSupportGate.

Runs a direct model forward pass (no sklearn pipeline) on a sample of
benchmark datasets and logs gate distribution metrics:

    gate_stats/{tag}/avg_gate          — mean gate value across support tokens
    gate_stats/{tag}/avg_keep_ratio    — fraction of tokens kept at gate_tau
                                         (equals avg_gate in soft mode)
    gate_stats/{tag}/avg_speedup       — theoretical decoder speedup (1 / keep_ratio)
    gate_stats/{tag}/avg_baseline_ms   — wall-clock time with no pruning (tau=1.0)
    gate_stats/{tag}/avg_inference_ms  — wall-clock time with gate_tau applied
    gate_stats/{tag}/actual_speedup    — avg_baseline_ms / avg_inference_ms

avg_baseline_ms is measured first on each dataset (tau=1.0, full support set),
then avg_inference_ms is measured with the configured gate_tau.  The ratio gives
the true wall-clock speedup from token pruning on this hardware.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from loopicl.evaluation.benchmark import Benchmark
from loopicl.evaluation.setup import iter_benchmark_datasets
from loopicl.validation.base import BaseValidator


class SupportGateStatsValidator(BaseValidator):
    """Collect gate statistics from LoopICLSupportGate at inference time.

    Parameters
    ----------
    benchmark : Benchmark
        Benchmark definition (same one used by BenchmarkValidator).
    benchmark_dir : str
        Root directory holding the benchmark datasets.
    gate_tau : float
        Hard-pruning threshold.  1.0 = soft mode (keep_ratio = avg_gate).
        < 1.0 = binary keep/prune at this threshold.
    max_datasets : int
        Maximum number of individual datasets to sample across the benchmark.
        Gate stats are fast but limiting this avoids extending validation
        time significantly.  Default: 20.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        benchmark_dir: str,
        gate_tau: float = 1.0,
        max_datasets: int = 20,
    ):
        self.benchmark = benchmark
        self.benchmark_dir = benchmark_dir
        self.gate_tau = gate_tau
        self.max_datasets = max_datasets

    @property
    def name(self) -> str:
        return "gate_stats"

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

        all_avg_gates: list[float] = []
        all_keep_ratios: list[float] = []
        all_baseline_ms: list[float] = []
        all_inference_ms: list[float] = []
        use_cuda = "cuda" in str(device)
        n_collected = 0

        for _, _, datasets in iter_benchmark_datasets(
            self.benchmark, self.benchmark_dir, "all", prior_n_val=True
        ):
            for X_train, X_test, y_train, _ in datasets:
                if n_collected >= self.max_datasets:
                    break

                X_full = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
                X_t = torch.as_tensor(X_full, device=device).unsqueeze(0)   # (1, N_total, C)
                y_t = torch.as_tensor(
                    y_train.astype(np.float32), device=device
                ).unsqueeze(0)                                               # (1, N_train)
                train_size = len(y_train)

                try:
                    with torch.no_grad():
                        ctx = (
                            torch.autocast(device_type="cuda", dtype=amp_dtype)
                            if amp_dtype is not None and use_cuda
                            else torch.autocast(device_type="cpu", enabled=False)
                        )
                        with ctx:
                            def _timed_forward(tau: float):
                                if use_cuda:
                                    torch.cuda.synchronize()
                                t0 = time.perf_counter()
                                _, g = model(
                                    X_t, y_t,
                                    train_size=train_size,
                                    return_gate_info=True,
                                    gate_tau=tau,
                                )
                                if use_cuda:
                                    torch.cuda.synchronize()
                                return g, (time.perf_counter() - t0) * 1000

                            # Baseline: full support set (no token pruning)
                            _, baseline_ms = _timed_forward(tau=1.0)
                            # Configured mode: token pruning at gate_tau
                            gates, elapsed_ms = _timed_forward(tau=self.gate_tau)

                    # gates : (1, N_train) ∈ (0, 1)
                    avg_gate = gates.mean().item()
                    keep_ratio = (
                        (gates >= self.gate_tau).float().mean().item()
                        if self.gate_tau < 1.0
                        else avg_gate
                    )
                    all_avg_gates.append(avg_gate)
                    all_keep_ratios.append(keep_ratio)
                    all_baseline_ms.append(baseline_ms)
                    all_inference_ms.append(elapsed_ms)
                    n_collected += 1

                except Exception as exc:
                    print(f"[SupportGateStatsValidator] Skipping dataset: {exc}")
                    continue

            if n_collected >= self.max_datasets:
                break

        if not all_avg_gates:
            return {}

        mean_keep     = float(np.mean(all_keep_ratios))
        mean_baseline = float(np.mean(all_baseline_ms))
        mean_infer    = float(np.mean(all_inference_ms))
        tag = f"tau{self.gate_tau:.2f}"
        return {
            f"gate_stats/{tag}/avg_gate":         float(np.mean(all_avg_gates)),
            f"gate_stats/{tag}/avg_keep_ratio":   mean_keep,
            f"gate_stats/{tag}/avg_speedup":      1.0 / max(mean_keep, 1e-3),
            f"gate_stats/{tag}/avg_baseline_ms":  mean_baseline,
            f"gate_stats/{tag}/avg_inference_ms": mean_infer,
            f"gate_stats/{tag}/actual_speedup":   mean_baseline / max(mean_infer, 1e-3),
        }
