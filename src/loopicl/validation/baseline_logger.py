"""Baseline wandb logging utilities.

Provides helpers to build a flat metrics dict from pre-computed baseline
results on disk and to publish those metrics as static per-baseline wandb runs
(one run per baseline model, acting as a flat reference line alongside the
candidate training curve).
"""

from __future__ import annotations

import numpy as np

from loopicl.evaluation.benchmark import Benchmark
from loopicl.evaluation.setup import load_all_results
from loopicl.evaluation.metrics import _PRIMARY_METRIC_PRIORITY, _pick_primary, compute_aggregated_stats, compute_improvability


def build_baseline_log_dict(
    benchmarks: list[Benchmark],
    benchmark_dir: str,
) -> dict[str, float]:
    """Build a flat metrics dict from pre-computed baseline result files.

    Reads ``{benchmark_dir}/{benchmark.name}/{dataset}/{model}.json`` for
    every benchmark and returns averaged scores in the same key format used
    by the candidate during training::

        results/{bench_name}/performance/{model}
        results/{bench_name}/logloss/{model}
        results/{bench_name}/r2/{model}
        results/{bench_name}/rmse/{model}
        results/{bench_name}/nrmse/{model}
        results/{bench_name}/win_rate_vs_random_forest/{model}
        more_results/{bench_name}/pr_auc/{model}
        more_results/{bench_name}/mcc/{model}

    The returned dict is consumed by :func:`create_baseline_wandb_runs` and
    can also be logged directly to the active wandb run as flat reference lines.

    Parameters
    ----------
    benchmarks:
        List of :class:`~ticla.benchmarks.benchmark.Benchmark` objects.
    benchmark_dir:
        Root directory that holds the per-benchmark result files.
    """
    log_dict: dict[str, float] = {}
    _more_stat_keys = {"mean_skill_score"}

    for benchmark in benchmarks:
        all_results = load_all_results(benchmark_dir, benchmark.name)
        if not all_results:
            continue
        prefix = f"results/{benchmark.name}"
        more_prefix = f"more_results/{benchmark.name}"

        perf_acc: dict[str, list[float]] = {}
        logloss_acc: dict[str, list[float]] = {}
        pr_auc_acc: dict[str, list[float]] = {}
        mcc_acc: dict[str, list[float]] = {}
        r2_acc: dict[str, list[float]] = {}
        rmse_acc: dict[str, list[float]] = {}
        nrmse_acc: dict[str, list[float]] = {}

        for model_scores in all_results.values():
            for bname, scores in model_scores.items():
                primary_key = next((k for k in _PRIMARY_METRIC_PRIORITY if k in scores), None)
                if primary_key:
                    perf_acc.setdefault(bname, []).append(scores[primary_key])
                if "logloss" in scores:
                    logloss_acc.setdefault(bname, []).append(scores["logloss"])
                if "pr_auc" in scores:
                    pr_auc_acc.setdefault(bname, []).append(scores["pr_auc"])
                if "mcc" in scores:
                    mcc_acc.setdefault(bname, []).append(scores["mcc"])
                if "r2" in scores:
                    r2_acc.setdefault(bname, []).append(scores["r2"])
                if "rmse" in scores and np.isfinite(scores["rmse"]):
                    rmse_acc.setdefault(bname, []).append(scores["rmse"])
                if "nrmse" in scores and np.isfinite(scores["nrmse"]):
                    nrmse_acc.setdefault(bname, []).append(scores["nrmse"])

        for bname, vals in perf_acc.items():
            log_dict[f"{more_prefix}/performance/{bname}"] = float(np.mean(vals))
        for bname, vals in logloss_acc.items():
            log_dict[f"{prefix}/logloss/{bname}"] = float(np.mean(vals))
        for bname, vals in pr_auc_acc.items():
            log_dict[f"{more_prefix}/pr_auc/{bname}"] = float(np.mean(vals))
        for bname, vals in mcc_acc.items():
            log_dict[f"{more_prefix}/mcc/{bname}"] = float(np.mean(vals))
        for bname, vals in r2_acc.items():
            log_dict[f"{prefix}/r2/{bname}"] = float(np.mean(vals))
        for bname, vals in rmse_acc.items():
            log_dict[f"{prefix}/rmse/{bname}"] = float(np.mean(vals))
        for bname, vals in nrmse_acc.items():
            log_dict[f"{prefix}/nrmse/{bname}"] = float(np.mean(vals))

        # Aggregate stats (improvability, skill score, win rates) per baseline model
        all_model_names = sorted({m for ds_scores in all_results.values() for m in ds_scores})
        for bname in all_model_names:
            b_rows: list[tuple[str, dict[str, float], dict[str, float]]] = []
            for ds_name, model_scores in all_results.items():
                if bname not in model_scores:
                    continue
                cand_scores = dict(model_scores[bname])
                baselines_row: dict[str, float] = {}
                for other_m, other_s in model_scores.items():
                    if other_m == bname:
                        continue
                    pk = _pick_primary(other_s)
                    if pk:
                        baselines_row[other_m] = other_s[pk]
                b_rows.append((ds_name, cand_scores, baselines_row))
            if not b_rows:
                continue
            agg = compute_aggregated_stats(b_rows, extra_baseline_keys=["tabpfn_v2_5"])
            for stat_name, stat_val in agg.items():
                p = more_prefix if stat_name in _more_stat_keys else prefix
                log_dict[f"{p}/{stat_name}/{bname}"] = stat_val
            imp_entries: list[tuple[str, dict[str, float]]] = []
            for ds_name, cand_sc, bl_row in b_rows:
                if not cand_sc:
                    continue
                pk = _pick_primary(cand_sc)
                if pk is None:
                    continue
                row: dict[str, float] = {bname: cand_sc[pk]}
                row.update(bl_row)
                imp_entries.append((pk, row))
            imp = compute_improvability(imp_entries, bname)
            if imp is not None:
                log_dict[f"{prefix}/improvability/{bname}"] = imp

    return log_dict


def create_baseline_wandb_runs(
    baseline_log_dict: dict[str, float],
    project: str,
    max_steps: int,
    entity: str | None = None,
    wandb_dir: str | None = None,
    wandb_mode: str = "online",
    group: str = "baselines",
) -> None:
    """Create one wandb run per baseline model as a static reference.

    Groups all keys in *baseline_log_dict* by the last path segment (the
    baseline model name), then creates one wandb run per group.  Each run is
    tagged ``"baseline"`` and finished immediately after logging, so it appears
    as a flat reference line alongside the candidate training curve.

    Runs that already exist on the server (matched by name and the
    ``"baseline"`` tag) are skipped to avoid duplicates.

    Parameters
    ----------
    baseline_log_dict:
        Flat dict produced by :func:`build_baseline_log_dict`.
    project:
        wandb project name.
    max_steps:
        Total training steps; metrics are logged at 10 evenly-spaced steps so
        the reference lines span the full x-axis.
    entity:
        wandb entity (team/user).  Defaults to the API default entity.
    wandb_dir:
        Local directory for wandb run files.
    wandb_mode:
        wandb mode (``"online"``, ``"offline"``, ``"disabled"``).
    group:
        wandb group name for all baseline runs.
    """
    if not baseline_log_dict:
        return

    try:
        import wandb
    except ImportError:
        print("[baseline_logger] wandb not installed, skipping baseline run creation.")
        return

    api = wandb.Api()
    existing_names: set[str] = set()
    try:
        resolved_entity = entity or api.default_entity
        api_runs = api.runs(
            f"{resolved_entity}/{project}",
            filters={"tags": {"$in": ["baseline"]}},
        )
        existing_names = {r.name for r in api_runs}
    except Exception as e:
        print(f"[baseline_logger] Could not query existing baseline runs: {e}")

    # Group metrics by baseline name (last path segment).
    grouped: dict[str, dict[str, float]] = {}
    for key, val in baseline_log_dict.items():
        baseline_name = key.split("/")[-1]
        grouped.setdefault(baseline_name, {})[key] = val

    num_points = 10
    log_steps = sorted(set(
        round(max_steps * i / (num_points - 1)) for i in range(num_points)
    ))

    for baseline_name, metrics in grouped.items():
        run_name = f"baseline/{baseline_name}"
        if run_name in existing_names:
            continue

        run = wandb.init(
            dir=wandb_dir,
            project=project,
            name=run_name,
            group=group,
            tags=["baseline"],
            mode=wandb_mode,
        )
        for step in log_steps:
            run.log(metrics, step=step)
        run.finish()
