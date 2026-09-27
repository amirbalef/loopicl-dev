"""Unified TabArena benchmark runner.

Dataset modes (--dataset)
    quick               3 tasks, fold 0/repeat 0
    tabarena-lite       All 51 tasks, fold 0/repeat 0
    tabarena            All 51 tasks, full fold x repeat protocol
    tabarena-cls        Classification tasks only, full fold x repeat protocol

Usage examples
    uv run benchmark_eval/run_benchmark.py --dataset quick
    uv run benchmark_eval/run_benchmark.py --dataset tabarena-lite --models LightGBM,CatBoost,RealTabPFN-v2.5

Output
    results/<dataset>/
        <model_name>/               raw TabArena pkl files
    Run leaderboard.py on the results dir to build leaderboard.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import patches  # noqa: F401 — applies tabarena monkey-patches on import

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUICK_TASKS: list[int] = [363614, 359944, 168350]  # anneal, credit-g, diabetes

_DATASET_PRESETS: dict[str, tuple[str, str]] = {
    #                        tasks_mode  repetitions_mode
    "quick":                ("quick",    "TabArena-Lite"),
    "tabarena-lite":        ("all",      "TabArena-Lite"),
    "tabarena":             ("all",      "TabArena"),
    "tabarena-cls":         ("cls",      "TabArena"),
    "tabarena-cls-lite":    ("cls",      "TabArena-Lite"),
}

DEFAULT_MODEL_NAMES: list[str] = [
    "Dummy",
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "RandomForest",
    "TabICLv2",
    "RealTabPFN-v2.5",
    "LoopICL",
]

ALL_MODEL_NAMES: list[str] = [
    "Dummy",
    "LightGBM", "CatBoost", "XGBoost", "RandomForest", "ExtraTrees",
    "Linear", "KNN", "EBM", "RealMLP", "TabM", "ModernNCA",
    "TabICL", "TabICLv2", "RealTabPFN-v2", "RealTabPFN-v2.5",
    "TabDPT", "TabSTAR", "xRFM", "PerpetualBooster", "SAP-RPT-OSS",
    "FastaiMLP", "TorchMLP",
]

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def load_configs(name: str) -> list[dict]:
    """Load benchmark_eval/configs/<name>.py and return a list of config dicts.

    Files may export either:
    - ``CONFIGS``: a list of dicts, each with a ``"name"`` key (preferred), or
    - ``CONFIG``:  a single dict with a ``"name"`` key (legacy/single-variant).
    """
    import importlib.util
    path = Path(__file__).parent / "configs" / Path(*name.split(".")).with_suffix(".py")
    if not path.exists():
        available = [p.stem for p in path.parent.glob("*.py") if p.stem != "__init__"]
        raise FileNotFoundError(
            f"Config '{name}' not found at {path}. Available: {available}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "CONFIGS"):
        return mod.CONFIGS
    cfg = mod.CONFIG
    if "name" not in cfg:
        cfg = {**cfg, "name": name}
    return [cfg]


def build_experiments(
    model_names: list[str],
    configs: list[str] | None = None,
) -> list[tuple]:
    from tabarena.benchmark.experiment.experiment_constructor import AGModelExperiment
    from tabarena.models.utils import get_configs_generator_from_name
    from models.dummy_model import DummyModel
    from models.tabicl_v2_model import TabICLv2Model, TabICL2reg
    from models.tabpfn_v3_model import TabPFNv3Model
    from models.loopicl_model import LoopICLModel, LoopICLEarlyExitModel

    method_kwargs = {"fit_kwargs": {"refit_on_val": True}}

    _local = {
        "Dummy":               (DummyModel,               {}),
        "TabICL-v2":           (TabICLv2Model,            {}),
        "TabICL2reg":          (TabICL2reg,               {}),
        "TabPFN-v3":           (TabPFNv3Model,            {}),
        "LoopICL":             (LoopICLModel,             {}),
        "LoopICLEarlyExit": (LoopICLEarlyExitModel, {}),
    }

    experiments = []  # list of (AGModelExperiment, save_dir | None)

    for name in model_names:
        if name in _local:
            model_cls, default_hp = _local[name]
            experiments.append((AGModelExperiment(
                name=name,
                model_cls=model_cls,
                model_hyperparameters=default_hp,
                method_kwargs=method_kwargs,
            ), None))
            continue
        try:
            meta = get_configs_generator_from_name(name)
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
            continue
        default_hp = dict(meta.manual_configs[0]) if meta.manual_configs else {}
        experiments.append((AGModelExperiment(
            name=name,
            model_cls=meta.model_cls,
            model_hyperparameters=default_hp,
            method_kwargs=method_kwargs,
        ), None))

    # Config-file variants
    for cfg_file in (configs or []):
        try:
            cfgs = load_configs(cfg_file)
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue
        for cfg in cfgs:
            cfg_name   = cfg.get("name", cfg_file)
            model_name = cfg.get("model")
            extra_hp   = {k: v for k, v in cfg.items() if k not in ("name", "model")}
            save_dir   = extra_hp.pop("save_dir", None)
            if model_name in _local:
                model_cls, default_hp = _local[model_name]
                hp = {**default_hp, **extra_hp}
                experiments.append((AGModelExperiment(
                    name=cfg_name,
                    model_cls=model_cls,
                    model_hyperparameters=hp,
                    method_kwargs=method_kwargs,
                ), save_dir))
            else:
                print(f"  [skip] config '{cfg_name}': model '{model_name}' not in local models")

    return experiments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TabArena benchmark.")
    parser.add_argument(
        "--dataset",
        default="tabarena-lite",
        choices=list(_DATASET_PRESETS),
        help="Benchmark mode. (default: tabarena-lite)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        metavar="ID,ID,...",
        help="Comma-separated OpenML task IDs (overrides --dataset task selection).",
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="NAME,NAME,...",
        help=(
            f"Comma-separated model names. "
            f"Default: {','.join(DEFAULT_MODEL_NAMES)}."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all available TabArena models.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override default output directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="NAME,NAME,...",
        help=(
            "Comma-separated config names from benchmark_eval/configs/*.py. "
            "Each config defines a RepInject experiment (model, strategy, layers). "
            "Example: --config tabicl_v2_c1,tabpfn_v2_5_c1"
        ),
    )
    parser.add_argument(
        "--raise-on-failure",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_curated_task_metadata():
    """Import load_curated_task_metadata from whichever location this tabarena version uses."""
    try:
        from tabarena.nips2025_utils.fetch_metadata import load_curated_task_metadata
    except ImportError:
        from tabarena.benchmark.task.metadata.fetch_metadata import load_curated_task_metadata
    return load_curated_task_metadata()


def _run_experiment_new_api(
    *,
    output_dir: str,
    experiment,
    task_ids: list[int],
    repetitions_mode: str,
    tasks_metadata,
    raise_on_failure: bool,
):
    """Run one experiment using the newer tabarena ExperimentBatchRunner API.

    Bridges the old run_experiments_new(repetitions_mode=...) call convention
    to ExperimentBatchRunner.run(folds, repeats) / .run_all().
    """
    from tabarena.benchmark.experiment.experiment_utils import ExperimentBatchRunner
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection

    # Build a TaskMetadataCollection filtered to our task IDs.
    full_collection = TaskMetadataCollection.from_preset("TabArena-v0.1")
    # Keep only the requested tasks (subset by integer tid).
    all_datasets = full_collection.dataset_to_tid()  # {name: tid_str}
    wanted = {name for name, tid_str in all_datasets.items() if int(tid_str) in set(task_ids)}
    collection = full_collection.subset_tasks(dataset_names=list(wanted)) if wanted != set(all_datasets) else full_collection

    runner = ExperimentBatchRunner(
        expname=output_dir,
        task_metadata=collection,
        raise_on_failure=raise_on_failure,
        debug_mode=False,
    )

    datasets = [d for d in runner.datasets if d in wanted]

    if repetitions_mode == "TabArena-Lite":
        runner.run(methods=[experiment], datasets=datasets, folds=[0], repeats=[0])
    elif repetitions_mode == "TabArena":
        runner.run_all(methods=[experiment])
    else:
        raise ValueError(f"Unsupported repetitions_mode for new API: {repetitions_mode!r}")


def run(args: argparse.Namespace) -> None:
    _here = Path(__file__).parent.resolve()
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    # Detect which tabarena API generation is available.
    try:
        from tabarena.benchmark.experiment import run_experiments_new
        _use_new_api = False
    except ImportError:
        run_experiments_new = None
        _use_new_api = True

    tasks_mode, repetitions_mode = _DATASET_PRESETS[args.dataset]

    # Output dir: results/<dataset>/
    if args.output_dir is not None:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(f"results/{args.dataset}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Task selection
    if args.tasks is not None:
        tasks = [int(t.strip()) for t in args.tasks.split(",") if t.strip()]
        repetitions_mode = "TabArena-Lite"
        print(f"[custom] {len(tasks)} tasks, fold 0/repeat 0.")
    elif tasks_mode == "quick":
        tasks = QUICK_TASKS
        print(f"[quick] {len(tasks)} tasks, fold 0/repeat 0.")
    elif tasks_mode == "cls":
        try:
            import openml
            all_task_ids = list(openml.study.get_suite("tabarena-v0.1").tasks)
            tasks_df = openml.tasks.list_tasks(task_id=all_task_ids, output_format="dataframe")
            tasks = tasks_df[tasks_df["task_type"] == "Supervised Classification"]["tid"].tolist()
        except Exception as e:
            print(f"[WARNING] OpenML unavailable ({e}), falling back to local TaskMetadataCollection.")
            from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
            _col_df = TaskMetadataCollection.from_preset("TabArena-v0.1").to_dataframe()
            tasks = (
                _col_df[_col_df["problem_type"].isin(["binary", "multiclass"])][["dataset_name", "task_id_str"]]
                .drop_duplicates()["task_id_str"]
                .tolist()
            )
        print(f"[{args.dataset}] {len(tasks)} classification tasks, repetitions_mode={repetitions_mode}.")
    else:
        try:
            import openml
            tasks = list(openml.study.get_suite("tabarena-v0.1").tasks)
        except Exception as e:
            print(f"[WARNING] OpenML unavailable ({e}), falling back to local TaskMetadataCollection.")
            from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
            tasks = list(TaskMetadataCollection.from_preset("TabArena-v0.1").dataset_to_tid().values())
        print(f"[{args.dataset}] {len(tasks)} tasks, repetitions_mode={repetitions_mode}.")

    # Model selection
    if args.all_models:
        model_names = ALL_MODEL_NAMES
    elif args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.config:
        model_names = []  # config-only run: don't add default models
    else:
        model_names = DEFAULT_MODEL_NAMES

    configs = (
        [c.strip() for c in args.config.split(",") if c.strip()]
        if args.config else []
    )

    print(f"\nBuilding experiments for: {model_names}")
    if configs:
        print(f"Configs: {configs}")
    experiments = build_experiments(model_names, configs=configs)
    if not experiments:
        print("No valid experiments. Exiting.")
        return
    print(f"Valid: {[e.name for e, _ in experiments]}")
    print(f"Output: {base_dir}\n")

    # Run each experiment into its own subdir
    for experiment, save_dir_override in experiments:
        output_dir = Path(save_dir_override) if save_dir_override else base_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\n  {experiment.name}\n  → {output_dir}\n{'='*60}")

        if _use_new_api:
            _run_experiment_new_api(
                output_dir=str(output_dir),
                experiment=experiment,
                task_ids=tasks,
                repetitions_mode=repetitions_mode,
                tasks_metadata=None,
                raise_on_failure=args.raise_on_failure,
            )
        else:
            run_experiments_new(
                output_dir=str(output_dir),
                model_experiments=[experiment],
                tasks=tasks,
                repetitions_mode=repetitions_mode,
                tasks_metadata=_load_curated_task_metadata().reset_index(drop=True),
                raise_on_failure=args.raise_on_failure,
                debug_mode=False,
            )

    print(f"\nDone. Results in: {base_dir}")
    print(f"Build leaderboard: uv run benchmark_eval/leaderboard.py {base_dir}")

if __name__ == "__main__":
    run(parse_args())
