"""Setup utilities: generate prior datasets and evaluate baselines for a benchmark.

These functions are called by the two CLI scripts and can also be triggered
automatically by the trainer when ``auto_setup=True`` is set in the benchmark
validator config.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from loopicl.evaluation.benchmark import Benchmark
from loopicl.evaluation.prior import _prior_dataset_dir, load_prior_datasets, save_prior_dataset
from loopicl.evaluation.openml import get_openml_suite_task_ids, get_openml_task_type, get_task_ids_from_dataset_ids, load_openml_dataset, load_openml_dataset_fold
from loopicl.evaluation.metrics import mean_metric_on_datasets, _PRIMARY_METRIC_PRIORITY, _pick_primary, compute_aggregated_stats, compute_improvability, compute_summary_metrics


# ---------------------------------------------------------------------------
# Per-dataset / per-model file helpers
# ---------------------------------------------------------------------------

def _model_result_path(
    benchmark_dir: str, benchmark_name: str, dataset_name: str, model_name: str
) -> Path:
    return Path(benchmark_dir) / benchmark_name / dataset_name / f"{model_name}.json"


def _save_model_result(
    benchmark_dir: str,
    benchmark_name: str,
    dataset_name: str,
    model_name: str,
    scores: dict,
) -> None:
    path = _model_result_path(benchmark_dir, benchmark_name, dataset_name, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(), **scores}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_model_result(
    benchmark_dir: str, benchmark_name: str, dataset_name: str, model_name: str
) -> dict | None:
    path = _model_result_path(benchmark_dir, benchmark_name, dataset_name, model_name)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_results(
    benchmark_dir: str, benchmark_name: str
) -> dict[str, dict[str, dict]]:
    """Load all per-dataset/per-model results from disk.

    Returns nested dict: {dataset_name: {model_name: {metric: score}}}.
    """
    root = Path(benchmark_dir) / benchmark_name
    out: dict[str, dict[str, dict]] = {}
    if not root.exists():
        return out
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for model_file in sorted(ds_dir.glob("*.json")):
            model_name = model_file.stem
            with open(model_file) as f:
                data = json.load(f)
            scores = {k: v for k, v in data.items() if k != "updated_at"}
            out.setdefault(ds_dir.name, {})[model_name] = scores
    return out


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def load_baseline_clf(model_name: str, task_type: str = "cls", n_ensemble: int | None = None):
    """Load a baseline model by name using its native sklearn-compatible interface.

    Parameters
    ----------
    model_name:
        Name of the baseline model.
    task_type:
        ``"cls"`` for classification or ``"reg"`` for regression.
    n_ensemble:
        If not None, passed as ``n_ensemble`` to TabICL models.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    def _with_imputer(estimator):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])

    key = model_name.lower()

    if key in ("tabicl_v1", "tabicl_1"):
        if task_type == "reg":
            raise ValueError("TabICL v1 only supports classification.")
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(checkpoint_version="tabicl-classifier-v1-20250208.ckpt")
        if n_ensemble is not None:
            clf.n_estimators = n_ensemble
        return _with_imputer(clf)

    if key in ("tabicl_v1_1", "tabicl_1_1"):
        if task_type == "reg":
            raise ValueError("TabICL v1.1 only supports classification.")
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(checkpoint_version="tabicl-classifier-v1.1-20250506.ckpt")
        if n_ensemble is not None:
            clf.n_estimators = n_ensemble
        return _with_imputer(clf)

    if key in ("tabicl", "tabicl_v2", "tabicl_2"):
        if task_type == "reg":
            from tabicl import TabICLRegressor
            clf = TabICLRegressor(checkpoint_version="tabicl-regressor-v2-20260212.ckpt")
            if n_ensemble is not None:
                clf.n_estimators = n_ensemble
            return _with_imputer(clf)
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(checkpoint_version="tabicl-classifier-v2-20260212.ckpt")
        if n_ensemble is not None:
            clf.n_estimators = n_ensemble
        return _with_imputer(clf)

    if key == "tabicl_v2-def":
        # TabICL v2 with its built-in default ensemble (n_estimators not overridden).
        if task_type == "reg":
            from tabicl import TabICLRegressor
            clf = TabICLRegressor(checkpoint_version="tabicl-regressor-v2-20260212.ckpt")
            return _with_imputer(clf)
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(checkpoint_version="tabicl-classifier-v2-20260212.ckpt")
        return _with_imputer(clf)

    if key == "tabpfn_v2":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if task_type == "reg" else TabPFNClassifier
        clf = cls.create_default_for_version(ModelVersion.V2)
        clf.ignore_pretraining_limits = True
        if n_ensemble is not None:
            clf.n_estimators = n_ensemble
        return clf

    if key in ("tabpfn_v2_5", "tabpfn_2_5"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if task_type == "reg" else TabPFNClassifier
        clf = cls.create_default_for_version(ModelVersion.V2_5)
        clf.ignore_pretraining_limits = True
        if n_ensemble is not None:
            clf.n_estimators = n_ensemble
        return clf

    if key in ("tabpfn_v2_6", "tabpfn_2_6"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if task_type == "reg" else TabPFNClassifier
        clf = cls.create_default_for_version(ModelVersion.V2_6)
        clf.ignore_pretraining_limits = True
        if n_ensemble is not None:
            clf.n_estimators = n_ensemble
        return clf

    if key in ("tabpfn_v3", "tabpfn_3"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if task_type == "reg" else TabPFNClassifier
        clf = cls.create_default_for_version(ModelVersion.V3)
        clf.ignore_pretraining_limits = True
        if n_ensemble is not None:
            clf.n_estimators = n_ensemble
        return clf

    if key == "tabpfn_v3-def":
        # TabPFN v3 with its built-in default ensemble (n_estimators not overridden).
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if task_type == "reg" else TabPFNClassifier
        clf = cls.create_default_for_version(ModelVersion.V3)
        clf.ignore_pretraining_limits = True
        return clf

    if key == "limix":
        from limix import LimiXClassifier
        return LimiXClassifier()

    if key in ("tabfm", "tabfm-def"):
        import torch
        from tabfm import TabFMClassifier, TabFMRegressor
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if task_type == "reg":
            model = tabfm_v1_0_0.load(model_type="regression", device=device)
            clf = TabFMRegressor(model=model)
            if key == "tabfm-def":
                pass  # use model default ensemble
            elif n_ensemble is not None:
                clf.n_estimators = n_ensemble
            else:
                clf.n_estimators = 1
            return _with_imputer(clf)
        model = tabfm_v1_0_0.load(device=device)
        clf = TabFMClassifier(model=model)
        if key == "tabfm-def":
            pass  # use model default ensemble
        elif n_ensemble is not None:
            clf.n_estimators = n_ensemble
        else:
            clf.n_estimators = 1
        return _with_imputer(clf)

    if key in ("random_forest", "rf"):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task_type == "reg":
            return RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    raise ValueError(
        f"Unknown baseline: {model_name!r}. "
        f"Valid options: tabicl_v1, tabicl_v1_1, tabicl_v2, tabicl_v2-def, "
        f"tabpfn_v2, tabpfn_v2_5, tabpfn_v2_6, tabpfn_v3, tabpfn_v3-def, limix, "
        f"tabfm (1 estimator), tabfm-def (default ensemble), random_forest."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_openml_task_ids(benchmark: Benchmark) -> list[int]:
    """Return hardcoded task IDs plus any IDs fetched from suite definitions."""
    ids = list(benchmark.openml_task_ids)
    for suite_id in benchmark.openml_suite_ids:
        try:
            ids.extend(get_openml_suite_task_ids(suite_id))
        except Exception as exc:
            print(f"[setup] WARNING: could not fetch OpenML suite {suite_id}: {exc}. Skipping.")
    return ids


def iter_benchmark_datasets(
    benchmark: Benchmark,
    benchmark_dir: str,
    task_type: str = "all",
    prior_n_val: bool = False,
):
    """Yield ``(ds_key, ds_task, datasets)`` for every dataset source in *benchmark*.

    This is the single place that knows how to iterate all benchmark dataset
    sources (prior, openml, talent, tabularmath, multitab, pmlb, tabnetics,
    beyond_arena, tabular_reasoning).  Both :func:`evaluate_baselines` and
    :class:`~ticla.validation.benchmark_val.BenchmarkValidator` use this to
    guarantee they evaluate on exactly the same set of datasets.

    Parameters
    ----------
    benchmark:
        Benchmark definition.
    benchmark_dir:
        Root directory for benchmark data (needed for prior datasets).
    task_type:
        ``"cls"``, ``"reg"``, or ``"all"`` — filters which dataset types to yield.
    prior_n_val:
        If ``True``, load only ``prior_cfg.n_val_datasets`` per prior config
        (used during validation to keep evaluation fast).  If ``False``, load
        all available datasets (used when evaluating baselines).
    """
    # Prior datasets
    for prior_cfg in benchmark.prior_datasets:
        ds_task = "reg" if prior_cfg.prior_tasktype == "regression" else "cls"
        if task_type != "all" and ds_task != task_type:
            continue
        n = prior_cfg.n_val_datasets if prior_n_val else None
        try:
            datasets = load_prior_datasets(benchmark_dir, benchmark.name, prior_cfg.name, n=n)
        except FileNotFoundError as exc:
            print(f"[benchmark] Skipping prior '{prior_cfg.name}': {exc}")
            continue
        yield prior_cfg.name, ds_task, datasets

    # OpenML datasets (task_ids and suite_ids — may be cls or reg).
    # Task type is determined from the loaded data (y_train.dtype) rather than from
    # get_openml_task_type(), which can return stale "classification" values if its
    # cache was written by older code that compared task.task_type_id == 2 (always
    # False with newer openml-python where task_type_id is a TaskType enum).
    for task_id in _all_openml_task_ids(benchmark):
        try:
            if benchmark.openml_use_official_folds:
                split = load_openml_dataset_fold(task_id, fold=0, repeat=0)
            else:
                split = load_openml_dataset(task_id, benchmark.test_size, benchmark.seed)
        except Exception as exc:
            print(f"[benchmark] Skipping OpenML task {task_id}: {exc}")
            continue
        _, _, y_train, _ = split
        ds_task = "reg" if y_train.dtype == np.float32 else "cls"
        if task_type != "all" and ds_task != task_type:
            continue
        yield f"openml_{task_id}", ds_task, [split]

    # OpenML regression datasets resolved from dataset IDs (openml_reg_dataset_ids)
    if task_type in ("reg", "all") and benchmark.openml_reg_dataset_ids:
        try:
            resolved = get_task_ids_from_dataset_ids(
                benchmark.openml_reg_dataset_ids, task_type="regression"
            )
        except Exception as exc:
            print(f"[benchmark] WARNING: could not resolve regression dataset IDs: {exc}")
            resolved = []
        for dataset_id, task_id in resolved:
            try:
                split = load_openml_dataset(task_id, benchmark.test_size, benchmark.seed, force_regression=True)
            except Exception as exc:
                print(f"[benchmark] Skipping OpenML dataset {dataset_id} (task {task_id}): {exc}")
                continue
            yield f"openml_{task_id}", "reg", [split]

    # OpenML classification datasets resolved from dataset IDs (openml_cls_dataset_ids)
    if task_type in ("cls", "all") and benchmark.openml_cls_dataset_ids:
        try:
            resolved = get_task_ids_from_dataset_ids(
                benchmark.openml_cls_dataset_ids, task_type="classification"
            )
        except Exception as exc:
            print(f"[benchmark] WARNING: could not resolve classification dataset IDs: {exc}")
            resolved = []
        for dataset_id, task_id in resolved:
            try:
                split = load_openml_dataset(task_id, benchmark.test_size, benchmark.seed)
            except Exception as exc:
                print(f"[benchmark] Skipping OpenML dataset {dataset_id} (task {task_id}): {exc}")
                continue
            yield f"openml_{task_id}", "cls", [split]



# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------

def _priors_ready(benchmark_dir: str, benchmark_name: str, prior_cfg) -> bool:
    """Return True if prior datasets exist and are complete."""
    data_dir = _prior_dataset_dir(benchmark_dir, benchmark_name, prior_cfg.name)
    if not data_dir.exists():
        return False
    return len(list(data_dir.glob("*.npz"))) >= prior_cfg.n_datasets


def _baselines_ready(
    benchmark_dir: str,
    benchmark_name: str,
    dataset_names: list[str],
    baseline_models: list[str],
) -> bool:
    """Return True if per-model result files exist for all (dataset, baseline) pairs."""
    return all(
        _model_result_path(benchmark_dir, benchmark_name, ds, bm).exists()
        for ds in dataset_names
        for bm in baseline_models
    )


# ---------------------------------------------------------------------------
# Step 1: generate prior datasets
# ---------------------------------------------------------------------------

def generate_priors(
    benchmark: Benchmark,
    benchmark_dir: str,
    overwrite: bool = False,
    task_type: str = "all",
) -> None:
    """Generate and save prior datasets for every prior config in *benchmark*.

    Parameters
    ----------
    benchmark:
        Benchmark definition.
    benchmark_dir:
        Root output directory (e.g. ``"results/benchmarks/"``).
    overwrite:
        If True, regenerate datasets even if they already exist.
    task_type:
        ``"cls"``, ``"reg"``, or ``"all"`` — only generate priors matching this type.
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*nested tensors.*", category=UserWarning)

    from torch.utils.data import DataLoader

    _V1_TYPES = {"mlp_scm", "tree_scm", "mix_scm", "dummy"}
    _V2_TYPES = {"graph_scm"}

    for prior_cfg in benchmark.prior_datasets:
        ds_task = "reg" if prior_cfg.prior_tasktype == "regression" else "cls"
        if task_type != "all" and ds_task != task_type:
            continue
        out_dir = _prior_dataset_dir(benchmark_dir, benchmark.name, prior_cfg.name)

        if not overwrite and _priors_ready(benchmark_dir, benchmark.name, prior_cfg):
            print(
                f"[setup] Prior datasets for '{prior_cfg.name}' already complete "
                f"({prior_cfg.n_datasets} files). Skipping."
            )
            continue

        existing = len(list(out_dir.glob("*.npz"))) if out_dir.exists() else 0
        if overwrite:
            # Remove stale files
            for f in out_dir.glob("*.npz"):
                f.unlink()
            existing = 0

        start_idx = existing
        remaining = prior_cfg.n_datasets - start_idx
        print(
            f"[setup] Generating {remaining} prior datasets for '{prior_cfg.name}' → {out_dir}"
        )

        if prior_cfg.prior_type in _V2_TYPES:
            from loopicl.priors.tabicl_v2._dataset import PriorDataset as PriorDatasetV2
            from loopicl.priors.tabicl_v2.graph_lib._config import PriorConfig
            dataset = PriorDatasetV2(
                batch_size=1,
                batch_size_per_gp=1,
                regression=(prior_cfg.prior_tasktype == "regression"),
                prior_type=prior_cfg.prior_type,
                min_features=prior_cfg.min_features,
                max_features=prior_cfg.max_features,
                min_classes=prior_cfg.min_classes,
                max_classes=prior_cfg.max_classes,
                min_seq_len=prior_cfg.min_seq_len,
                max_seq_len=prior_cfg.max_seq_len,
                log_seq_len=prior_cfg.log_seq_len,
                seq_len_per_gp=prior_cfg.seq_len_per_gp,
                min_train_size=prior_cfg.min_train_size,
                max_train_size=prior_cfg.max_train_size,
                replay_small=prior_cfg.replay_small,
                config=PriorConfig(),
                device="cpu",
                n_jobs=1,
            )
        elif prior_cfg.prior_type in _V1_TYPES:
            from loopicl.priors.tabicl_v1.dataset import PriorDataset as PriorDatasetV1
            dataset = PriorDatasetV1(
                batch_size=1,
                batch_size_per_gp=1,
                prior_type=prior_cfg.prior_type,
                prior_tasktype=prior_cfg.prior_tasktype,
                min_features=prior_cfg.min_features,
                max_features=prior_cfg.max_features,
                min_classes=prior_cfg.min_classes,
                max_classes=prior_cfg.max_classes,
                min_seq_len=prior_cfg.min_seq_len,
                max_seq_len=prior_cfg.max_seq_len,
                log_seq_len=prior_cfg.log_seq_len,
                seq_len_per_gp=prior_cfg.seq_len_per_gp,
                min_train_size=prior_cfg.min_train_size,
                max_train_size=prior_cfg.max_train_size,
                replay_small=prior_cfg.replay_small,
                device="cpu",
                n_jobs=1,
            )
        else:
            raise ValueError(
                f"Unknown prior_type '{prior_cfg.prior_type}' for '{prior_cfg.name}'. "
                f"V1 types: {sorted(_V1_TYPES)}. V2 types: {sorted(_V2_TYPES)}."
            )
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)

        saved = 0
        for batch in dataloader:
            if saved >= remaining:
                break

            batch = [
                t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch
            ]
            X, y, d, seq_len_t, train_size_t = batch

            sl = int(seq_len_t[0].item())
            ts = int(train_size_t[0].item())
            n_feat = int(d[0].item())

            X_np = X[0, :sl, :n_feat].numpy()
            y_np = y[0, :sl].numpy()

            X_train, y_train = X_np[:ts], y_np[:ts]
            X_test, y_test = X_np[ts:], y_np[ts:]

            if len(X_train) == 0 or len(X_test) == 0:
                continue  # skip degenerate splits

            save_prior_dataset(
                benchmark_dir, benchmark.name, prior_cfg.name,
                start_idx + saved, X_train, X_test, y_train, y_test,
            )
            saved += 1
            print(f"  [{saved}/{remaining}]", end="\r", flush=True)

        print(f"\n[setup] '{prior_cfg.name}' done.")


# ---------------------------------------------------------------------------
# Step 2: evaluate baselines
# ---------------------------------------------------------------------------

# Hard class-count limits per baseline model (ignore_pretraining_limits does
# not bypass these for tabpfn class-count validation).
_BASELINE_MAX_CLASSES: dict[str, int] = {
    "tabpfn_v2":   10,
    "tabpfn_v2_5": 10,
    "tabpfn_v2_6": 10,
    "tabpfn_v3":     10,
    "tabpfn_v3-def": 10,   # same class-count limit; ensemble is model default
}


def _subsample_datasets(
    datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    max_train_size: int | None,
    max_test_size: int | None,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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


def _eval_on_datasets(
    clf,
    datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    task_type: str = "cls",
    max_classes: int | None = None,
) -> dict[str, float]:
    """Delegate to :func:`ticla.benchmarks.metrics.mean_metric_on_datasets`."""
    if max_classes is not None and task_type == "cls":
        datasets = [
            (Xtr, Xte, ytr, yte) for Xtr, Xte, ytr, yte in datasets
            if len(np.unique(ytr)) <= max_classes
        ]
        if not datasets:
            return {}
    return mean_metric_on_datasets(clf, datasets, task_type=task_type)


def log_baselines_to_wandb(
    benchmarks: list[Benchmark],
    benchmark_dir: str,
    wandb_project: str = None,  # unused, kept for backward compatibility
    max_steps: int | None = None,
) -> None:
    """Log all baseline metrics as flat reference lines into the current active wandb run.

    Reads each benchmark's ``baselines.json`` and logs every baseline model's
    average score using the same key format as candidates
    (``benchmark/{bench_name}/{baseline_name}``), so all values appear in the
    same wandb chart.  Values are logged at step 0 and at *max_steps* (when
    given) to produce flat horizontal reference lines alongside training curves.
    """
    try:
        import wandb as _wandb
    except ImportError:
        print("[setup] wandb not installed, skipping baseline wandb logging.")
        return

    log_dict: dict[str, float] = {}
    for benchmark in benchmarks:
        all_results = load_all_results(benchmark_dir, benchmark.name)
        if not all_results:
            continue

        all_model_names = sorted({m for ds_scores in all_results.values() for m in ds_scores})
        for bname in all_model_names:
            # Build summary_rows in the same format used by BenchmarkValidator
            # so that compute_summary_metrics produces identical calculations.
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
            log_dict.update(
                compute_summary_metrics(
                    b_rows,
                    candidate_name=bname,
                    benchmark_name=benchmark.name,
                    label=bname,
                    extra_baseline_keys=["tabpfn_v2_5"],
                )
            )

    if not log_dict:
        return

    # Log at step 0 (and max_steps) so baselines appear as flat reference lines
    # across the full training run, in the same chart as candidate metrics.
    _wandb.log(log_dict, step=0, commit=False)
    if max_steps is not None:
        _wandb.log(log_dict, step=max_steps, commit=True)
    else:
        _wandb.log(log_dict, commit=True)
    print(f"[setup] Logged {len(log_dict)} baseline metrics to current wandb run.")


def evaluate_baselines(
    benchmark: Benchmark,
    benchmark_dir: str,
    baseline_models: list[str],
    skip_existing: bool = True,
    task_type: str = "cls",
    n_ensemble: int | None = None,
    max_train_size: int | None = None,
    max_test_size: int | None = None,
) -> None:
    """Evaluate baseline models on all datasets in *benchmark* and save results.

    Results are stored as per-dataset/per-model JSON files under
    ``{benchmark_dir}/{benchmark.name}/{dataset_name}/{model_name}.json``.
    Already-evaluated pairs are skipped when *skip_existing* is True.

    Parameters
    ----------
    benchmark:
        Benchmark definition.
    benchmark_dir:
        Root directory (must already contain the generated prior datasets).
    baseline_models:
        List of baseline model names to evaluate.
    skip_existing:
        If True, skip pairs that already have a result on disk.
    task_type:
        Which dataset types to include: ``"cls"``, ``"reg"``, or ``"all"``.
    n_ensemble:
        If not None, passed as ``n_ensemble`` to TabICL baseline models.
    """
    for ds_key, ds_task, datasets in iter_benchmark_datasets(benchmark, benchmark_dir, task_type):
        datasets = _subsample_datasets(datasets, max_train_size, max_test_size)
        print(f"\n[setup] {ds_key}  ({ds_task},  {len(datasets)} dataset(s))")
        for bname in baseline_models:
            if skip_existing and _load_model_result(benchmark_dir, benchmark.name, ds_key, bname) is not None:
                existing = _load_model_result(benchmark_dir, benchmark.name, ds_key, bname)
                primary_key = next((k for k in _PRIMARY_METRIC_PRIORITY if k in existing), None)
                msg = f"{primary_key}={existing[primary_key]:.4f}" if primary_key else "done"
                print(f"  [{bname}] Already done ({msg}), skipping.")
                continue
            print(f"  [{bname}] Evaluating ...", end=" ", flush=True)
            try:
                scores = _eval_on_datasets(
                    load_baseline_clf(bname, ds_task, n_ensemble=n_ensemble),
                    datasets, ds_task,
                    max_classes=_BASELINE_MAX_CLASSES.get(bname.lower()),
                )
                save_scores = {k: v for k, v in scores.items() if not k.startswith("_")}
                _save_model_result(benchmark_dir, benchmark.name, ds_key, bname, save_scores)
                parts = "  ".join(f"{k}={v:.4f}" for k, v in scores.items() if not k.startswith("_"))
                print(parts)
            except Exception as exc:
                print(f"FAILED: {exc}")

    print(f"\n[setup] Results saved to: {Path(benchmark_dir) / benchmark.name}/")


# ---------------------------------------------------------------------------
# Combined: ensure a benchmark is fully ready
# ---------------------------------------------------------------------------

def ensure_benchmark_ready(
    benchmark: Benchmark,
    benchmark_dir: str,
    baseline_models: list[str],
    auto_setup: bool = False,
    force_priors: bool = False,
    force_baselines: bool = False,
    n_ensemble: int | None = None,
    task_type: str = "all",
    max_train_size: int | None = None,
    max_test_size: int | None = None,
) -> None:
    """Check whether benchmark data exists; run the missing steps when *auto_setup* is True.

    Called automatically by the trainer before validation begins.

    Parameters
    ----------
    benchmark:
        Benchmark definition.
    benchmark_dir:
        Root directory for all benchmark data.
    baseline_models:
        Which baselines to check (and optionally evaluate).
    auto_setup:
        If True, run any missing steps automatically.
    force_priors:
        If True, regenerate prior datasets even when they already exist.
    force_baselines:
        If True, re-evaluate all baselines even when results already exist.
    """
    _reg_ds_task_ids: list[int] = []
    if benchmark.openml_reg_dataset_ids:
        try:
            _reg_ds_task_ids = [tid for _, tid in get_task_ids_from_dataset_ids(
                benchmark.openml_reg_dataset_ids, task_type="regression"
            )]
        except Exception as exc:
            print(f"[setup] WARNING: could not resolve regression dataset IDs: {exc}")
    _cls_ds_task_ids: list[int] = []
    if benchmark.openml_cls_dataset_ids:
        try:
            _cls_ds_task_ids = [tid for _, tid in get_task_ids_from_dataset_ids(
                benchmark.openml_cls_dataset_ids, task_type="classification"
            )]
        except Exception as exc:
            print(f"[setup] WARNING: could not resolve classification dataset IDs: {exc}")

    def _prior_matches_task(p) -> bool:
        ds_task = "reg" if p.prior_tasktype == "regression" else "cls"
        return task_type == "all" or ds_task == task_type

    # Build the dataset names that would actually be evaluated for this task_type.
    # For OpenML tasks whose type is not yet cached, fall back to including all IDs
    # (conservative) so _baselines_ready returns False and evaluate_baselines runs.
    _openml_ids_for_check: list[int] = []
    for _tid in _all_openml_task_ids(benchmark):
        if task_type == "all":
            _openml_ids_for_check.append(_tid)
            continue
        try:
            _ott = get_openml_task_type(_tid)
            _ds_task = "reg" if _ott == "regression" else "cls"
            if _ds_task == task_type:
                _openml_ids_for_check.append(_tid)
        except Exception:
            _openml_ids_for_check.append(_tid)  # cache miss — include conservatively

    all_dataset_names = (
        [p.name for p in benchmark.prior_datasets if _prior_matches_task(p)]
        + [f"openml_{tid}" for tid in _openml_ids_for_check]
        + ([f"openml_{tid}" for tid in _reg_ds_task_ids] if task_type in ("reg", "all") else [])
        + ([f"openml_{tid}" for tid in _cls_ds_task_ids] if task_type in ("cls", "all") else [])
        + _get_talent_names(benchmark)
        + [f"tabularmath_{n}" for n in benchmark.tabularmath_dataset_names]
        + [f"multitab_{did}" for did in benchmark.multitab_dataset_ids]
        + [f"pmlb_{n}" for n in _get_pmlb_names(benchmark)]
        + [f"tabnetics_{n}" for n in _get_tabnetics_names(benchmark)]
        + [f"beyond_arena_{n}" for n in _get_beyond_arena_names(benchmark)]
        + ([f"tabular_reasoning_{t}" for t in benchmark.tabular_reasoning_task_names] if task_type in ("cls", "all") else [])
    )

    # Safety net: if the dataset list is empty but the benchmark has OpenML task IDs,
    # the task-type cache may be missing or wrong — force evaluation.
    if not all_dataset_names and _all_openml_task_ids(benchmark):
        all_dataset_names = [f"openml_{tid}" for tid in _all_openml_task_ids(benchmark)]

    priors_missing = [
        p for p in benchmark.prior_datasets
        if _prior_matches_task(p) and (force_priors or not _priors_ready(benchmark_dir, benchmark.name, p))
    ]

    print(f"[setup] Checking '{benchmark.name}' (task_type={task_type}): {len(all_dataset_names)} datasets ({len(_openml_ids_for_check)} openml), {len(baseline_models)} baselines")
    n_complete = sum(
        1 for ds in all_dataset_names for bm in baseline_models
        if _model_result_path(benchmark_dir, benchmark.name, ds, bm).exists()
    )
    n_total = len(all_dataset_names) * len(baseline_models)
    print(f"[setup]   baseline results: {n_complete}/{n_total} complete")

    baselines_missing = n_complete < n_total or force_baselines

    if not priors_missing and not baselines_missing:
        print(f"[setup] Benchmark '{benchmark.name}' is ready.")
        return

    if not auto_setup:
        msgs = []
        if priors_missing:
            names = [p.name for p in priors_missing]
            msgs.append(
                f"Prior datasets missing for: {names}.\n"
                f"  Set auto_setup: bool = True in the config for {benchmark.name}"
            )
        if baselines_missing:
            msgs.append(
                f"Baseline results incomplete.\n"
                f" Set auto_setup: bool = True in the config for {benchmark.name}"
            )
        raise RuntimeError(
            f"Benchmark '{benchmark.name}' is not ready.\n" + "\n".join(msgs)
        )

    # Auto-setup
    if priors_missing:
        generate_priors(benchmark, benchmark_dir, overwrite=force_priors, task_type=task_type)

    if baselines_missing:
        evaluate_baselines(
            benchmark,
            benchmark_dir,
            baseline_models,
            skip_existing=not force_baselines,
            task_type=task_type,
            n_ensemble=n_ensemble,
            max_train_size=max_train_size,
            max_test_size=max_test_size,
        )
