"""Benchmark validator: evaluate candidate model and compare with pre-computed baselines."""

from __future__ import annotations

import numpy as np
from torch import nn

from loopicl.evaluation.benchmark import Benchmark
from loopicl.evaluation.metrics import (
    mean_metric_on_datasets, compute_summary_metrics, compute_aggregated_stats,
    _pick_primary, _PRIMARY_METRIC_PRIORITY,
)
from loopicl.evaluation.setup import load_all_results, iter_benchmark_datasets
from loopicl.validation.base import BaseValidator
from loopicl.interface.wrap_interface import wrap_interface
from loopicl.interface.wrap_model import TabICLWrapper, IrisMELTTwoPhaseWrapper, LoopICLRegWrapper


def _build_candidate_clf(
    model: nn.Module,
    device: str,
    task_type: str,
    inference_batch_size: int,
    dtype: str,
    num_loops: int | None,
    n_ensemble: int | None,
    mrl_dim: int | None = None,
    use_two_phase_inference: bool = False,
    loop_residual_scaling_fn: str | None = None,
    exit_q: float | None = None,
):
    """Build the sklearn-compatible pipeline for the candidate model."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    WrapperCls = IrisMELTTwoPhaseWrapper if use_two_phase_inference else TabICLWrapper
    wrapper_kwargs = dict(dtype=dtype, num_loops=num_loops, mrl_dim=mrl_dim)
    if not use_two_phase_inference and loop_residual_scaling_fn is not None:
        wrapper_kwargs["loop_residual_scaling_fn"] = loop_residual_scaling_fn
    if not use_two_phase_inference and exit_q is not None:
        wrapper_kwargs["exit_q"] = exit_q
    base_clf = wrap_interface(
        "tabicl",
        task_type=task_type,
        model=WrapperCls(model, device, **wrapper_kwargs),
        device=device,
        batch_size=inference_batch_size,
    )
    if n_ensemble is not None:
        base_clf.n_estimators = n_ensemble
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", base_clf)])


def _score_candidate(
    model: nn.Module,
    device: str,
    datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    task_type: str,
    inference_batch_size: int,
    dtype: str,
    num_loops: int | None,
    n_ensemble: int | None,
    mrl_dim: int | None = None,
    use_two_phase_inference: bool = False,
    loop_residual_scaling_fn: str | None = None,
    exit_q: float | None = None,
) -> dict[str, float]:
    """Fit the candidate pipeline on each dataset and return mean scores.

    Delegates to :func:`~ticla.benchmarks.metrics.mean_metric_on_datasets` so
    that the same scoring code is used for both the candidate and baselines.
    """
    clf = _build_candidate_clf(model, device, task_type, inference_batch_size, dtype, num_loops, n_ensemble, mrl_dim, use_two_phase_inference, loop_residual_scaling_fn, exit_q)

    # Skip classification datasets whose max class label exceeds the model's
    # class embedding table size (model-specific constraint, not a metric issue).
    _col_enc = getattr(model, "col_y_encoder", None)
    _model_max_classes: int | None = (
        getattr(_col_enc, "embedding", None) and _col_enc.embedding.num_embeddings
    ) or None

    if task_type == "cls" and _model_max_classes is not None:
        filtered = []
        for ds in datasets:
            if int(np.max(ds[2])) >= _model_max_classes:
                print(
                    f"[BenchmarkValidator] WARNING: max label {int(np.max(ds[2]))} "
                    f">= model max_classes={_model_max_classes}. Skipping dataset."
                )
            else:
                filtered.append(ds)
    else:
        filtered = datasets

    return mean_metric_on_datasets(clf, filtered, task_type, skip_on_error=True)


class BenchmarkValidator(BaseValidator):
    """Validate the candidate model on a fixed benchmark and compare with pre-computed baselines.

    The baseline results are computed once (via ``scripts/evaluate_baselines.py``)
    and saved to ``{benchmark_dir}/{benchmark.name}/baselines.json``.  During
    each validation step only the candidate model is evaluated; the baselines are
    simply loaded from disk and the deltas are logged to wandb.

    Metrics logged
    --------------
    ``benchmark/{name}/{dataset}/{metric}``
        Candidate mean metric score on the dataset group.
        Metric is one of: ``roc_auc`` (binary cls), ``neg_log_loss`` (multiclass cls),
        ``neg_rmse`` (regression).
    ``benchmark/{name}/delta_vs_{baseline}/{dataset}/{metric}``
        Candidate score minus baseline score (positive = candidate wins).

    Parameters
    ----------
    benchmark:
        Benchmark definition produced by :func:`ticla.benchmarks.builtin.get_benchmark`.
    benchmark_dir:
        Root directory that holds both the saved prior datasets and
        ``baselines.json`` (e.g. ``"results/benchmarks/"``).
    inference_batch_size:
        Max test samples per forward pass for the candidate model.
    candidate_name:
        Identifier used to tag results for this candidate.
    inference_num_blocks:
        Number of ICL blocks to use during inference. If None, uses the
        model's default (``icl_num_blocks``).
    """

    def __init__(
        self,
        benchmark: Benchmark,
        benchmark_dir: str,
        inference_batch_size: int = 512,
        candidate_name: str = "candidate",
        inference_num_blocks: int | None = None,
        task_type: str = "all",
        n_ensemble: int | None = None,
        improvability_exclude_keys: list[str] | None = None,
        mrl_dim: int | None = None,
        use_two_phase_inference: bool = False,
        max_train_size: int | None = None,
        max_test_size: int | None = None,
        val_every_multiplier: int = 1,
        loop_residual_scaling_fn: str | None = None,
        exit_q: float | None = None,
    ):
        self.benchmark = benchmark
        self.benchmark_dir = benchmark_dir
        self.inference_batch_size = inference_batch_size
        self.candidate_name = candidate_name
        self.inference_num_blocks = inference_num_blocks
        self.task_type = task_type  # "all", "cls", or "reg"
        self.n_ensemble = n_ensemble
        self.improvability_exclude_keys = improvability_exclude_keys or []
        self.mrl_dim = mrl_dim
        self.use_two_phase_inference = use_two_phase_inference
        self.max_train_size = max_train_size
        self.max_test_size = max_test_size
        self.val_every_multiplier = val_every_multiplier
        self.loop_residual_scaling_fn = loop_residual_scaling_fn
        self.exit_q = exit_q
        self._baseline_results: dict[str, dict[str, float]] | None = None
        self._baseline_r2_results: dict[str, dict[str, float]] = {}

    @property
    def name(self) -> str:
        return f"benchmark/{self.benchmark.name}/{self.candidate_name}"

    def _load_baselines(self) -> dict[str, dict[str, float]]:
        all_results = load_all_results(self.benchmark_dir, self.benchmark.name)
        if not all_results:
            print(
                f"[BenchmarkValidator] No baseline results found in "
                f"{self.benchmark_dir}/{self.benchmark.name}/. "
                "Run  python scripts/evaluate_baselines.py  to generate them."
            )
            return {}
        results: dict[str, dict[str, float]] = {}
        r2_results: dict[str, dict[str, float]] = {}
        for ds_name, model_scores in all_results.items():
            row: dict[str, float] = {}
            for m, s in model_scores.items():
                # Multiclass (has balanced_accuracy or _n_classes) → use logloss as primary
                if ("balanced_accuracy" in s or "_n_classes" in s) and "logloss" in s:
                    row[m] = s["logloss"]
                else:
                    primary_key = next((k for k in _PRIMARY_METRIC_PRIORITY if k in s), None)
                    if primary_key:
                        row[m] = s[primary_key]
            if row:
                results[ds_name] = row
            r2 = {m: s["r2"] for m, s in model_scores.items() if "r2" in s}
            if r2:
                r2_results[ds_name] = r2
        self._baseline_r2_results = r2_results
        return results

    def _load_baseline_metric(self, metric: str) -> dict[str, dict[str, float]]:
        """Return {ds_name: {model_name: score}} for a specific stored metric."""
        all_results = load_all_results(self.benchmark_dir, self.benchmark.name)
        out: dict[str, dict[str, float]] = {}
        for ds_name, model_scores in all_results.items():
            row = {m: s[metric] for m, s in model_scores.items() if metric in s}
            if row:
                out[ds_name] = row
        return out

    @staticmethod
    def _fmt(value: float) -> str:
        if 0.0 <= value <= 1.0:
            width = 30
            filled = round(value * width)
            return f"[{'█' * filled}{'░' * (width - filled)}] {value:.4f}"
        return f"{value:.4f}"

    @staticmethod
    def _avg_rows(
        rows: list[tuple[str, dict[str, float], dict[str, float]]]
    ) -> dict[str, tuple[float, dict[str, float]]]:
        """Return {metric_name: (mean_candidate, {bname: mean_bscore})} averaged over all rows.

        Baselines (primary metric only) are associated with the primary key of each row.
        """
        cand_acc: dict[str, list[float]] = {}
        base_acc: dict[str, dict[str, list[float]]] = {}
        for _, cand_scores, baselines in rows:
            if not cand_scores:
                continue
            primary = _pick_primary(cand_scores) or next(
                (k for k in cand_scores if not k.startswith("_")), next(iter(cand_scores))
            )
            for k, v in cand_scores.items():
                if not k.startswith("_"):
                    cand_acc.setdefault(k, []).append(v)
            for bname, bscore in baselines.items():
                base_acc.setdefault(primary, {}).setdefault(bname, []).append(bscore)
        return {
            m: (
                float(np.mean(cand_acc[m])),
                {b: float(np.mean(s)) for b, s in base_acc.get(m, {}).items()},
            )
            for m in cand_acc
        }

    def _print_summary(
        self, rows: list[tuple[str, dict[str, float], dict[str, float]]]
    ) -> None:
        all_names = [self.candidate_name] + [bname for _, _, bl in rows for bname in bl]
        col_w = max(len(n) for n in all_names) if all_names else 9

        title = f" Benchmark: {self.benchmark.name} "
        print(f"\n{'─' * 60}")
        print(title.center(60))
        print(f"{'─' * 60}")
        for ds_name, cand_scores, baselines in rows:
            if not cand_scores:
                print(f"  {ds_name}  [skipped — no valid predictions]")
                continue
            primary = _pick_primary(cand_scores) or next(
                (k for k in cand_scores if not k.startswith("_")), next(iter(cand_scores))
            )
            print(f"  {ds_name}  [{primary}]")
            print(f"    {self.candidate_name:<{col_w}}  {self._fmt(cand_scores[primary])}")
            for bname, bscore in baselines.items():
                print(f"    {bname:<{col_w}}  {self._fmt(bscore)}")

        print(f"{'─' * 60}")
        for metric_name, (mean_cand, mean_bases) in self._avg_rows(rows).items():
            print(f"  Average  [{metric_name}]")
            print(f"    {self.candidate_name:<{col_w}}  {self._fmt(mean_cand)}")
            for bname, bscore in mean_bases.items():
                print(f"    {bname:<{col_w}}  {self._fmt(bscore)}")
        agg = compute_aggregated_stats(rows)
        if agg:
            print(f"{'─' * 60}")
            for stat_name, stat_val in agg.items():
                print(f"  {stat_name}: {stat_val:.4f}")
        print(f"{'─' * 60}\n")


    def _summary_metrics(
        self,
        summary_rows: list[tuple[str, dict[str, float], dict[str, float]]],
    ) -> dict[str, float]:
        """Return summary metrics for the trainer to log to wandb.

        Delegates to :func:`~ticla.benchmarks.metrics.compute_summary_metrics`,
        the same function used for baselines in ``log_baselines_to_wandb``.
        """
        label = self.candidate_name
        if self.mrl_dim is not None:
            label = f"{label}_mrl{self.mrl_dim}"
        return compute_summary_metrics(
            summary_rows,
            candidate_name=self.candidate_name,
            benchmark_name=self.benchmark.name,
            label=label,
            extra_baseline_keys=["tabpfn_v2_5"],
            improvability_exclude_keys=self.improvability_exclude_keys,
        )

    @staticmethod
    def _subsample(
        datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        max_train_size: int | None,
        max_test_size: int | None,
        seed: int = 0,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Randomly subsample train/test splits to the given size limits."""
        if max_train_size is None and max_test_size is None:
            return datasets
        rng = np.random.default_rng(seed)
        result = []
        for X_train, X_test, y_train, y_test in datasets:
            if max_train_size is not None and len(X_train) > max_train_size:
                idx = rng.choice(len(X_train), max_train_size, replace=False)
                X_train, y_train = X_train[idx], y_train[idx]
            if max_test_size is not None and len(X_test) > max_test_size:
                idx = rng.choice(len(X_test), max_test_size, replace=False)
                X_test, y_test = X_test[idx], y_test[idx]
            result.append((X_train, X_test, y_train, y_test))
        return result

    def validate(self, model: nn.Module, device: str, step: int | None = None, dtype: str = "float32") -> dict[str, float]:
        if self._baseline_results is None:
            self._baseline_results = self._load_baselines()

        model.eval()
        summary_rows: list[tuple[str, dict[str, float], dict[str, float]]] = []

        for ds_key, ds_task, datasets in iter_benchmark_datasets(
            self.benchmark, self.benchmark_dir, self.task_type, prior_n_val=True
        ):
            datasets = self._subsample(datasets, self.max_train_size, self.max_test_size)
            cand_scores = _score_candidate(
                model, device, datasets, ds_task,
                self.inference_batch_size, dtype, self.inference_num_blocks, self.n_ensemble,
                mrl_dim=self.mrl_dim,
                use_two_phase_inference=self.use_two_phase_inference,
                loop_residual_scaling_fn=self.loop_residual_scaling_fn,
                exit_q=self.exit_q,
            )
            baselines = self._baseline_results.get(ds_key, {})
            summary_rows.append((ds_key, cand_scores, baselines))

        self._print_summary(summary_rows)
        return self._summary_metrics(summary_rows)
