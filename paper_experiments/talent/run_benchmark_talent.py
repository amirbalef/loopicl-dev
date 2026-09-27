"""TALENT benchmark runner using the native TALENT Python API.

Follows the TALENT protocol exactly:
  - Native per-dataset train / val / test split (from the .npy files).
  - Train on train+val, evaluate on held-out test.
  - One or more seeds (default: 1, matching TALENT's default).
  - No additional cross-validation or random re-splitting.

Model name → TALENT registry name mapping
------------------------------------------
  TabICLv2        → tabicl_v2
  RealTabPFN-2.5  → tabpfn_v2_5   (alias for TabPFN-2.5; deduplicated)
  TabPFN-2.5      → tabpfn_v2_5   (PriorLabs Nov-2025)
  TabPFN-v3       → tabpfn_v3     (external tabpfn>=8.0.0; ~1M-row context)
  TabFM           → tabfm         (Google Research)
  LimiX           → limix
  TabPFNv2        → tabpfn_v2
  TabICL          → tabicl
  ModernNCA       → modernNCA
  XGBoost         → xgboost
  CatBoost        → catboost
  RealMLP         → realmlp
  LightGBM        → lightgbm
  FT-Transformer  → ftt
  MLP-PLR         → mlp_plr
  RandomForest    → RandomForest
  AutoInt         → autoint
  ExcelFormer     → excelformer
  ResNet          → resnet
  MLP             → mlp
  KNN             → knn

Usage examples
--------------
  uv run benchmark_eval/talent/run_benchmark_talent.py --dataset talent-lite --models LightGBM,CatBoost
  uv run benchmark_eval/talent/run_benchmark_talent.py --dataset talent

  # Multiple seeds (TALENT multi-seed protocol):
  uv run benchmark_eval/talent/run_benchmark_talent.py --dataset talent-lite --seed-num 5

  # Override raw TALENT folder:
  uv run benchmark_eval/talent/run_benchmark_talent.py --talent-folder /path/to/talent/data

  # Run LoopICL via a named config (loads configs/loopicl_configs.py):
  uv run benchmark_eval/talent/run_benchmark_talent.py \\
      --config loopicl_configs \\
      --dataset talent-lite --models LoopICL-base-v1-stage1

Output
------
  Per-(model, dataset) JSON files under --output-dir
  (default: results/talent-native/<dataset>/).
  A leaderboard CSV + console table printed after all runs.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure the project root (loopicl-dev/) is in sys.path so that
# `benchmark_eval` is importable regardless of how this script is invoked.
_project_root = str(_Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util as _iutil
_patches_spec = _iutil.spec_from_file_location("patches", Path(__file__).parent / "patches.py")
_patches_mod = _iutil.module_from_spec(_patches_spec)
_patches_spec.loader.exec_module(_patches_mod)
_patches_mod.apply_all()

# ---------------------------------------------------------------------------
# LoopICL config registry (populated at startup from --loopicl-config file)
# ---------------------------------------------------------------------------

# Maps friendly model name (e.g. "LoopICL-v1") → config dict from the config file.
# Populated by _load_loopicl_config() before the main run loop.
RICOTTA_CONFIGS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TALENT_FOLDER = Path(
    "results/benchmarks/datasets/talent/datasets-talent-default"
)

QUICK_DATASETS: list[str] = [
    "internet_firewall",
    "national-longitudinal-survey-binary",
    "jasmine",
]

_DATASET_PRESETS: dict[str, int] = {
    # preset name → default seed_num
    "quick":       1,
    "talent-lite": 1,
    "talent":      5,
}

# ---------------------------------------------------------------------------
# Model name → TALENT registry name
# ---------------------------------------------------------------------------
# Keys are the friendly names accepted by --models; values are the strings
# passed to TALENT.run().  Multiple keys may resolve to the same TALENT name
# (e.g. "TabPFN-2.5" resolves to tabpfn_v2_5; "RealTabPFN-2.5" to tabpfn_real).
# Deduplification is applied before running so each TALENT name runs once.

MODEL_MAP: dict[str, str] = {
    # Tabular ICL foundation models
    "TabICLv2":       "tabicl_v2",
    "TabICL":         "tabicl",
    # TabPFN family
    "RealTabPFN-2.5": "tabpfn_v2_5",   # alias for TabPFN-2.5; deduplicated automatically
    "TabPFN-2.5":     "tabpfn_v2_5",   # PriorLabs Nov-2025
    "TabPFN-v3":      "tabpfn_v3",     # external tabpfn>=8.0.0; ~1M-row context
    "TabFM":          "tabfm",         # Google Research
    "TabPFNv2":       "tabpfn_v2",
    # Other foundation / retrieval-augmented
    "LimiX":          "limix",
    "ModernNCA":      "modernNCA",
    # Classical tree-based
    "XGBoost":        "xgboost",
    "CatBoost":       "catboost",
    "LightGBM":       "lightgbm",
    "RandomForest":   "RandomForest",
    # Deep learning
    "RealMLP":        "realmlp",
    "FT-Transformer": "ftt",
    "MLP-PLR":        "mlp_plr",
    "AutoInt":        "autoint",
    "ExcelFormer":    "excelformer",
    "ResNet":         "resnet",
    "MLP":            "mlp",
    # Distance-based
    "KNN":            "knn",
    # External models (custom runners, not in TALENT registry)
    "EXAONE-Tabular": "exaone",
    # Not available in TALENT registry — will be skipped:
    # "TabRXGBoost": None,
}

# Ordered list of all supported friendly names (default order for --all-models)
DEFAULT_MODEL_NAMES: list[str] = [
    "TabICLv2",
    "RealTabPFN-2.5",
    "TabPFN-2.5",
    "TabPFN-v3",
    "TabFM",
    "TabPFNv2",
    "LimiX",
    "TabICL",
    "ModernNCA",
    "EXAONE-Tabular",
    "XGBoost",
    "CatBoost",
    "RealMLP",
    "LightGBM",
    "FT-Transformer",
    "MLP-PLR",
    "RandomForest",
    "AutoInt",
    "ExcelFormer",
    "ResNet",
    "MLP",
    "KNN",
]

# Models listed by the user that are not in the TALENT registry
_UNSUPPORTED: list[str] = ["TabRXGBoost"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def discover_talent_datasets(talent_folder: Path) -> list[str]:
    return sorted(
        d.name
        for d in talent_folder.iterdir()
        if d.is_dir() and (d / "info.json").exists()
    )


def load_talent_dataset(dataset_name: str, talent_folder: Path):
    """Load a TALENT dataset using the native TALENT data loader.

    Returns (train_val_data, test_data, info) exactly as get_dataset() does.
    """
    from TALENT.model.lib.data import get_dataset

    dataset_path = str(talent_folder.parent)
    folder_name = talent_folder.name + "/" + dataset_name
    # get_dataset joins dataset_path / dataset_name, so we point at the parent
    # and pass the sub-path as dataset_name.
    return get_dataset(folder_name, str(talent_folder.parent))


def _load_info(dataset_name: str, talent_folder: Path) -> dict:
    return json.loads((talent_folder / dataset_name / "info.json").read_text())


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def _result_path(output_dir: Path, model_talent_name: str, dataset_name: str) -> Path:
    return output_dir / "results" / model_talent_name / f"{dataset_name}.json"


def _save_result(
    path: Path,
    model_talent_name: str,
    friendly_name: str,
    dataset_name: str,
    result,
    info: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": friendly_name,
        "talent_name": model_talent_name,
        "dataset": dataset_name,
        "task_type": info.get("task_type", "unknown"),
        **result.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _collect_leaderboard(output_dir: Path) -> pd.DataFrame:
    """Walk result JSONs and build a tidy DataFrame."""
    records = []
    results_dir = output_dir / "results"
    if not results_dir.exists():
        return pd.DataFrame()

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for json_file in sorted(model_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
                row = {
                    "model": data.get("model", model_dir.name),
                    "talent_name": data.get("talent_name", model_dir.name),
                    "dataset": data.get("dataset", json_file.stem),
                    "task_type": data.get("task_type", ""),
                    "fit_time": data.get("fit_time", float("nan")),
                }
                # Primary metric: first entry in metrics_mean (if multi-seed)
                # or metrics dict
                metrics = data.get("metrics_mean") or data.get("metrics") or {}
                for k, v in metrics.items():
                    row[k] = v
                records.append(row)
            except Exception:
                pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def _print_leaderboard(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to display.")
        return

    # Identify metric columns (everything except bookkeeping cols)
    skip = {"model", "talent_name", "dataset", "task_type", "fit_time"}
    metric_cols = [c for c in df.columns if c not in skip]

    if not metric_cols:
        print(df.to_string(index=False))
        return

    # Per-dataset rank on the first metric column, then average rank
    primary = metric_cols[0]
    df = df.copy()
    df["rank"] = df.groupby("dataset")[primary].rank(ascending=False, method="average")
    summary = (
        df.groupby("model")[["rank"] + metric_cols]
        .mean()
        .sort_values("rank")
        .reset_index()
    )
    try:
        print(summary.to_markdown(index=False, floatfmt=".4f"))
    except Exception:
        print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def load_configs(name: str) -> list[dict]:
    """Load benchmark_eval/talent/configs/<name>.py and return a list of config dicts.

    Files may export either:
    - ``CONFIGS``: a list of dicts, each with a ``"name"`` key (preferred), or
    - ``CONFIG``:  a single dict with a ``"name"`` key (legacy/single-variant).
    """
    import importlib.util as _iu

    path = Path(__file__).parent / "configs" / Path(*name.split(".")).with_suffix(".py")
    if not path.exists():
        available = [p.stem for p in path.parent.glob("*.py") if p.stem != "__init__"]
        raise FileNotFoundError(
            f"Config '{name}' not found at {path}. Available: {available}"
        )
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if hasattr(mod, "CONFIGS"):
        return mod.CONFIGS
    cfg = mod.CONFIG
    if "name" not in cfg:
        cfg = {**cfg, "name": name}
    return [cfg]


def _register_configs(configs: list[dict]) -> None:
    """Register a list of LoopICL config dicts in RICOTTA_CONFIGS and MODEL_MAP."""
    for cfg in configs:
        name = cfg.get("name")
        if not name:
            print(f"  [warn] Config entry missing 'name', skipping: {cfg}")
            continue
        RICOTTA_CONFIGS[name] = cfg
        MODEL_MAP[name] = "loopicl"
    print(
        f"  [loopicl] Registered {len(RICOTTA_CONFIGS)} config(s): "
        f"{list(RICOTTA_CONFIGS)}"
    )


def _run_one_loopicl(
    friendly_name: str,
    dataset_name: str,
    talent_folder: Path,
    output_dir: Path,
    seed_num: int,
    overwrite: bool,
) -> bool:
    """Run a single (LoopICL config, dataset) pair. Returns True on success."""
    from benchmark_eval.talent.models.loopicl_runner import run_loopicl

    cfg = RICOTTA_CONFIGS.get(friendly_name)
    if cfg is None:
        print(f"    [error] No LoopICL config registered for {friendly_name!r}")
        return False

    # Use the friendly name (sanitised) as the results subdirectory so
    # different configs don't overwrite each other.
    talent_key = friendly_name.lower().replace(" ", "_")
    out_path = _result_path(output_dir, talent_key, dataset_name)
    if out_path.exists() and not overwrite:
        print(f"    [skip] Already done ({out_path})")
        return True

    info = _load_info(dataset_name, talent_folder)
    task_type = info.get("task_type", "")
    is_regression = task_type == "regression"

    if is_regression and not cfg.get("supports_regression", False):
        print(
            f"    [skip] {friendly_name!r} does not support regression "
            f"(set supports_regression=True in the config to enable) — "
            f"skipping {dataset_name}."
        )
        return False

    try:
        train_val_data, test_data, info = load_talent_dataset(dataset_name, talent_folder)
    except Exception as exc:
        print(f"    [error] Data loading failed: {exc}")
        traceback.print_exc()
        return False

    save_path = str(output_dir / "checkpoints" / talent_key / dataset_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    try:
        result = run_loopicl(
            cfg,
            train_val_data,
            test_data,
            info,
            seed_num=seed_num,
            save_path=save_path,
        )
    except Exception as exc:
        print(f"    [error] run_loopicl failed: {exc}")
        traceback.print_exc()
        return False

    if result is None:
        # Config does not support this task type (e.g. cls-only config on a
        # regression dataset) — skip silently.
        return True

    _save_result(out_path, talent_key, friendly_name, dataset_name, result, info)
    if seed_num > 1:
        primary = dict(zip(result.metric_names, result.metrics_mean))
    else:
        primary = dict(zip(result.metric_names, result.metrics))
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in primary.items())
    print(f"    [ok] {metric_str}  fit={result.fit_time:.1f}s")
    return True


def resolve_models(model_names: list[str]) -> list[tuple[str, str]]:
    """Map friendly names to TALENT registry names, dedup, warn about unknowns.

    Returns list of (friendly_name, talent_name) with unique talent names.
    """
    seen_talent: dict[str, str] = {}  # talent_name → first friendly name
    result: list[tuple[str, str]] = []
    unknown: list[str] = []

    for name in model_names:
        if name not in MODEL_MAP:
            if name in _UNSUPPORTED:
                print(f"  [warn] {name!r} is not available in the TALENT registry — skipped.")
            else:
                unknown.append(name)
            continue
        talent_name = MODEL_MAP[name]
        # LoopICL configs are distinguished by their friendly name (each has its
        # own config), so skip dedup for them.
        if talent_name != "loopicl":
            if talent_name in seen_talent:
                print(
                    f"  [dedup] {name!r} resolves to the same TALENT model as "
                    f"{seen_talent[talent_name]!r} ({talent_name!r}) — skipped."
                )
                continue
            seen_talent[talent_name] = name
        result.append((name, talent_name))

    if unknown:
        print(f"  [warn] Unrecognised model names (not in MODEL_MAP): {unknown}")

    return result


def _run_one_exaone(
    friendly_name: str,
    dataset_name: str,
    talent_folder: Path,
    output_dir: Path,
    seed_num: int,
    overwrite: bool,
) -> bool:
    """Run a single (EXAONE-Tabular, dataset) pair. Returns True on success."""
    from benchmark_eval.talent.models.exaone_runner import run_exaone

    talent_key = "exaone_tabular"
    out_path = _result_path(output_dir, talent_key, dataset_name)
    if out_path.exists() and not overwrite:
        print(f"    [skip] Already done ({out_path})")
        return True

    info = _load_info(dataset_name, talent_folder)

    try:
        train_val_data, test_data, info = load_talent_dataset(dataset_name, talent_folder)
    except Exception as exc:
        print(f"    [error] Data loading failed: {exc}")
        traceback.print_exc()
        return False

    save_path = str(output_dir / "checkpoints" / talent_key / dataset_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    try:
        result = run_exaone(
            train_val_data,
            test_data,
            info,
            seed_num=seed_num,
            save_path=save_path,
        )
    except Exception as exc:
        print(f"    [error] run_exaone failed: {exc}")
        traceback.print_exc()
        return False

    _save_result(out_path, talent_key, friendly_name, dataset_name, result, info)
    if seed_num > 1:
        primary = dict(zip(result.metric_names, result.metrics_mean))
    else:
        primary = dict(zip(result.metric_names, result.metrics))
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in primary.items())
    print(f"    [ok] {metric_str}  fit={result.fit_time:.1f}s")
    return True


def run_one(
    friendly_name: str,
    talent_name: str,
    dataset_name: str,
    talent_folder: Path,
    output_dir: Path,
    seed_num: int,
    overwrite: bool,
) -> bool:
    """Run a single (model, dataset) pair. Returns True on success."""
    # EXAONE-Tabular uses a custom runner (not the TALENT registry).
    if talent_name == "exaone":
        return _run_one_exaone(
            friendly_name=friendly_name,
            dataset_name=dataset_name,
            talent_folder=talent_folder,
            output_dir=output_dir,
            seed_num=seed_num,
            overwrite=overwrite,
        )

    # LoopICL models use a custom runner (not the TALENT registry).
    if talent_name == "loopicl":
        return _run_one_loopicl(
            friendly_name=friendly_name,
            dataset_name=dataset_name,
            talent_folder=talent_folder,
            output_dir=output_dir,
            seed_num=seed_num,
            overwrite=overwrite,
        )

    import TALENT

    out_path = _result_path(output_dir, talent_name, dataset_name)
    if out_path.exists() and not overwrite:
        print(f"    [skip] Already done ({out_path})")
        return True

    # Check task-type compatibility before loading data.
    info = _load_info(dataset_name, talent_folder)
    task_type = info.get("task_type", "")
    is_regression = task_type == "regression"

    try:
        spec = TALENT.get_method_spec(talent_name)
    except KeyError as e:
        print(f"    [skip] {e}")
        return False

    if is_regression and not spec.supports_regression:
        print(f"    [skip] {talent_name!r} does not support regression — skipping {dataset_name}.")
        return False
    if not is_regression and not spec.supports_classification:
        print(f"    [skip] {talent_name!r} does not support classification — skipping {dataset_name}.")
        return False

    # Load data using TALENT's native loader.
    try:
        train_val_data, test_data, info = load_talent_dataset(dataset_name, talent_folder)
    except Exception as exc:
        print(f"    [error] Data loading failed: {exc}")
        traceback.print_exc()
        return False

    # Save path for checkpoints (model-specific sub-dir).
    save_path = str(output_dir / "checkpoints" / talent_name / dataset_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    try:
        result = TALENT.run(
            talent_name,
            train_val_data,
            test_data,
            info,
            seed_num=seed_num,
            save_path=save_path,
        )
    except Exception as exc:
        if type(exc).__name__ == "TabPFNValidationError" or "exceeds the maximum number of classes" in str(exc):
            print(f"    [skip] {exc}")
            return False
        print(f"    [error] TALENT.run failed: {exc}")
        traceback.print_exc()
        return False

    _save_result(out_path, talent_name, friendly_name, dataset_name, result, info)
    if seed_num > 1:
        primary = dict(zip(result.metric_names, result.metrics_mean))
    else:
        primary = dict(zip(result.metric_names, result.metrics))
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in primary.items())
    print(f"    [ok] {metric_str}  fit={result.fit_time:.1f}s")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TALENT benchmark using the native TALENT Python API."
    )
    parser.add_argument(
        "--dataset",
        default="talent-lite",
        choices=list(_DATASET_PRESETS),
        help="Benchmark preset. (default: talent-lite)",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        metavar="NAME,NAME,...",
        help="Comma-separated dataset names (overrides --dataset task selection).",
    )
    parser.add_argument(
        "--talent-folder",
        default=None,
        help=(
            f"Path to the raw TALENT data folder (contains per-dataset subdirs "
            f"with info.json and .npy files). Default: {_DEFAULT_TALENT_FOLDER}"
        ),
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="NAME,NAME,...",
        help=(
            f"Comma-separated friendly model names to run. "
            f"Default: all supported models. "
            f"Available: {', '.join(sorted(MODEL_MAP))}. "
            f"Unsupported (no TALENT entry): {', '.join(_UNSUPPORTED)}."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all models in the default ordered list.",
    )
    parser.add_argument(
        "--seed-num",
        type=int,
        default=None,
        help=(
            "Number of seeds to run per (model, dataset). "
            "Default: 1 for quick/talent-lite, 5 for talent."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override default output directory (default: results/talent-native/<dataset>/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing result files (default: skip if already done).",
    )
    parser.add_argument(
        "--raise-on-failure",
        action="store_true",
        default=False,
        help="Re-raise exceptions instead of skipping failed (model, dataset) pairs.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="NAME,NAME,...",
        help=(
            "Comma-separated config names from benchmark_eval/talent/configs/*.py. "
            "Each config defines LoopICL model variants (name, weights_path, HPs). "
            "All named entries are registered and can be selected via --models. "
            "Example: --config loopicl_configs"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    # Load configs first so MODEL_MAP is up to date before model resolution.
    if args.config is not None:
        for cfg_name in [c.strip() for c in args.config.split(",") if c.strip()]:
            try:
                cfgs = load_configs(cfg_name)
            except FileNotFoundError as exc:
                print(f"  [skip] {exc}")
                continue
            _register_configs(cfgs)

    # Output directory
    if args.output_dir is not None:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(f"results/talent-native/{args.dataset}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Seed count
    seed_num = args.seed_num if args.seed_num is not None else _DATASET_PRESETS[args.dataset]

    # Talent data folder
    talent_folder = Path(args.talent_folder) if args.talent_folder else _DEFAULT_TALENT_FOLDER
    if not talent_folder.exists():
        raise FileNotFoundError(
            f"TALENT folder not found: {talent_folder}\n"
            f"Pass a valid path via --talent-folder."
        )

    # Dataset selection
    if args.datasets is not None:
        dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
        print(f"[custom] {len(dataset_names)} datasets.")
    elif args.dataset == "quick":
        dataset_names = QUICK_DATASETS
        print(f"[quick] {len(dataset_names)} datasets.")
    else:
        dataset_names = discover_talent_datasets(talent_folder)
        print(f"[{args.dataset}] {len(dataset_names)} datasets discovered.")

    if not dataset_names:
        print("No datasets found. Exiting.")
        return

    # Model selection
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
    elif RICOTTA_CONFIGS:
        requested = list(RICOTTA_CONFIGS)
    else:
        requested = DEFAULT_MODEL_NAMES

    print(f"\nResolving models: {requested}")
    models = resolve_models(requested)
    if not models:
        print("No valid models. Exiting.")
        return

    print(f"\nRunning {len(models)} model(s) × {len(dataset_names)} dataset(s)  "
          f"[seed_num={seed_num}]")
    print(f"Output: {base_dir}\n")

    # --- Main loop ---
    stats = {"ok": 0, "skip": 0, "error": 0}
    for friendly_name, talent_name in models:
        print(f"\n{'='*60}")
        print(f"  {friendly_name}  ({talent_name})")
        print(f"{'='*60}")
        for dataset_name in dataset_names:
            print(f"  {dataset_name}")
            try:
                ok = run_one(
                    friendly_name=friendly_name,
                    talent_name=talent_name,
                    dataset_name=dataset_name,
                    talent_folder=talent_folder,
                    output_dir=base_dir,
                    seed_num=seed_num,
                    overwrite=args.overwrite,
                )
                if ok:
                    stats["ok"] += 1
                else:
                    stats["skip"] += 1
            except Exception as exc:
                stats["error"] += 1
                print(f"    [error] {exc}")
                if args.raise_on_failure:
                    raise
                traceback.print_exc()

    # --- Leaderboard ---
    print(f"\n{'='*60}")
    print(f"  Summary: {stats['ok']} ok  |  {stats['skip']} skipped  |  {stats['error']} errors")
    print(f"{'='*60}")

    df = _collect_leaderboard(base_dir)
    if not df.empty:
        print("\n=== TALENT Leaderboard ===\n")
        _print_leaderboard(df)

        leaderboard_path = base_dir / "leaderboard.csv"
        df.to_csv(leaderboard_path, index=False)
        print(f"\nFull results CSV: {leaderboard_path}")

    print(f"\nDone. Results in: {base_dir}")


if __name__ == "__main__":
    run(parse_args())
