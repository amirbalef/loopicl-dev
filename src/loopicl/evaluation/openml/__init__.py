"""OpenML dataset loader: real-world tabular tasks from the OpenML repository."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Local cache directories so repeated calls (and offline runs)
# don't require a live network connection.
_CACHE_ROOT = Path(
    os.environ.get("LOOPICL_CACHE_DIR", str(Path.home() / ".cache" / "ticla"))
)
_SUITE_CACHE_DIR = _CACHE_ROOT / "openml_suites"
_DATASET_CACHE_DIR = _CACHE_ROOT / "openml_datasets"
_DATASET_TASK_CACHE_DIR = _CACHE_ROOT / "openml_dataset_tasks"


def get_openml_suite_task_ids(suite_id: int | str, use_cache: bool = True) -> list[int]:
    """Return the list of task IDs belonging to an OpenML benchmark suite.

    Results are cached on disk under ``~/.cache/loopicl/openml_suites/`` (or
    ``$LOOPICL_CACHE_DIR/openml_suites/``) so subsequent calls and offline runs
    do not require a network connection.

    Parameters
    ----------
    suite_id:
        OpenML suite/study identifier — numeric ID (e.g. 337 for CC18)
        or a string alias (e.g. ``"tabarena-v0.1"``).
    use_cache:
        If True (default), read from and write to the local cache.

    Raises
    ------
    ImportError
        If the ``openml`` package is not installed.
    """
    cache_key = str(suite_id).replace("/", "_").replace(".", "_")
    cache_file = _SUITE_CACHE_DIR / f"suite_{cache_key}.json"

    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    try:
        import openml
    except ImportError:
        raise ImportError(
            "openml is required for OpenML suite loading.  "
            "Install it with:  pip install openml"
        )
    suite = openml.study.get_suite(suite_id)
    task_ids = list(suite.tasks)

    if use_cache:
        _SUITE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(task_ids))

    return task_ids


def get_task_ids_from_dataset_ids(
    dataset_ids: list[int],
    task_type: str = "regression",
    use_cache: bool = True,
) -> list[tuple[int, int]]:
    """Resolve OpenML dataset IDs to their canonical task IDs.

    Returns ``(dataset_id, task_id)`` pairs in the same order as *dataset_ids*;
    datasets with no matching task are silently dropped.

    Results are cached under ``~/.cache/loopicl/openml_dataset_tasks/<task_type>.json``
    so that subsequent calls and offline runs don't require a network connection.

    Parameters
    ----------
    dataset_ids:
        OpenML dataset IDs to resolve.
    task_type:
        ``"regression"`` or ``"classification"``.
    use_cache:
        Whether to read/write the on-disk cache.
    """
    cache_file = _DATASET_TASK_CACHE_DIR / f"{task_type}.json"

    cached: dict[str, int | None] = {}
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text())

    missing = [did for did in dataset_ids if str(did) not in cached]
    if missing:
        try:
            import openml
        except ImportError:
            raise ImportError(
                "openml is required for OpenML dataset loading.  "
                "Install it with:  pip install openml"
            )
        from openml.tasks import TaskType
        tt = (
            TaskType.SUPERVISED_REGRESSION
            if task_type == "regression"
            else TaskType.SUPERVISED_CLASSIFICATION
        )
        df = openml.tasks.list_tasks(task_type=tt, output_format="dataframe")
        df_missing = df[df["did"].isin(missing)]
        # One task per dataset: keep the smallest (oldest) task ID.
        resolved: dict[int, int] = (
            df_missing.sort_values("tid")
            .drop_duplicates(subset="did", keep="first")
            .set_index("did")["tid"]
            .to_dict()
        )
        for did in missing:
            cached[str(did)] = int(resolved[did]) if did in resolved else None
        if use_cache:
            _DATASET_TASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cached))

    return [
        (did, int(cached[str(did)]))  # type: ignore[arg-type]
        for did in dataset_ids
        if cached.get(str(did)) is not None
    ]


_TASK_TYPE_CACHE_DIR = _CACHE_ROOT / "openml_task_types"


def get_openml_task_type(task_id: int) -> str:
    """Return ``'regression'`` or ``'classification'`` for an OpenML task.

    Results are cached under ``~/.cache/loopicl/openml_task_types/`` so
    repeated calls and offline runs don't require a network connection.

    Parameters
    ----------
    task_id:
        OpenML task identifier.
    """
    cache_file = _TASK_TYPE_CACHE_DIR / f"task_{task_id}.json"
    cached = json.loads(cache_file.read_text()) if cache_file.exists() else None

    # Infer from the already-cached dataset split if available (avoids network call).
    # Also used to self-correct a stale cache entry (e.g. written by the old buggy
    # task.task_type_id == 2 integer comparison that always mapped to "classification").
    for fold_file in _DATASET_CACHE_DIR.glob(f"task_{task_id}_fold*.npz"):
        data = np.load(fold_file, allow_pickle=False)
        task_type = "regression" if data["y_train"].dtype == np.float32 else "classification"
        if task_type != cached:  # write / correct the cache
            _TASK_TYPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(task_type))
        return task_type

    if cached is not None:
        return cached

    try:
        import openml
    except ImportError:
        raise ImportError(
            "openml is required for OpenML datasets.  "
            "Install it with:  pip install openml"
        )
    task = openml.tasks.get_task(task_id)
    # task_type_id is a TaskType enum in newer openml-python — compare by name not integer
    task_type = "regression" if "REGRESSION" in str(task.task_type_id).upper() else "classification"

    _TASK_TYPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(task_type))

    return task_type


def load_openml_dataset_fold(
    task_id: int,
    fold: int = 0,
    repeat: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an OpenML task using its **official** train/test fold indices.

    This replicates the exact split used by TabArena (fold=0, repeat=0), which
    reads the pre-defined split from the OpenML task rather than re-splitting
    randomly.  Results are cached to avoid repeated network calls.

    Parameters
    ----------
    task_id:
        OpenML task identifier.
    fold:
        Cross-validation fold index (TabArena uses fold=0).
    repeat:
        Repeat index for repeated CV tasks (TabArena uses repeat=0).
    """
    cache_file = _DATASET_CACHE_DIR / f"task_{task_id}_fold{fold}_rep{repeat}.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=False)
        return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    try:
        import openml
    except ImportError:
        raise ImportError(
            "openml is required for OpenML datasets.  "
            "Install it with:  pip install openml"
        )

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.impute import SimpleImputer

    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
            X[cat_cols] = enc.fit_transform(X[cat_cols])

    X = np.array(X, dtype=np.float32)
    if np.isnan(X).any():
        X = SimpleImputer(strategy="median").fit_transform(X)

    is_regression = "REGRESSION" in str(task.task_type_id).upper()

    train_indices, test_indices = task.get_train_test_split_indices(
        fold=fold, repeat=repeat
    )

    if is_regression:
        y = np.array(y, dtype=np.float32)
    else:
        if hasattr(y, "dtype") and (y.dtype == object or str(y.dtype) == "category"):
            y = LabelEncoder().fit_transform(y)
        y = np.array(y, dtype=np.int64)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    _DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    return X_train, X_test, y_train, y_test


def load_openml_dataset(
    task_id: int,
    test_size: float = 0.3,
    seed: int = 42,
    force_regression: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an OpenML task and return a fixed ``(X_train, X_test, y_train, y_test)`` split.

    Parameters
    ----------
    task_id:
        OpenML task identifier.
    test_size:
        Fraction of samples held out for testing.
    seed:
        Random seed for the train/test split (kept fixed across runs).
    force_regression:
        If True, treat the target as continuous regardless of the task type
        reported by OpenML (no stratification, float y).

    Raises
    ------
    ImportError
        If the ``openml`` package is not installed.
    """
    # Check local cache first (keyed by task_id + test_size + seed).
    _reg_suffix = "_reg" if force_regression else ""
    cache_file = _DATASET_CACHE_DIR / f"task_{task_id}_ts{test_size}_s{seed}{_reg_suffix}.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=False)
        return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    try:
        import openml
    except ImportError:
        raise ImportError(
            "openml is required for OpenML datasets.  "
            "Install it with:  pip install openml"
        )

    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
            X[cat_cols] = enc.fit_transform(X[cat_cols])

    X = np.array(X, dtype=np.float32)
    if np.isnan(X).any():
        X = SimpleImputer(strategy="median").fit_transform(X)

    is_regression = force_regression or "REGRESSION" in str(task.task_type_id).upper()
    if is_regression:
        y = np.array(y, dtype=np.float32)
        split = train_test_split(X, y, test_size=test_size, random_state=seed)
    else:
        if hasattr(y, "dtype") and (y.dtype == object or str(y.dtype) == "category"):
            y = LabelEncoder().fit_transform(y)
        y = np.array(y, dtype=np.int64)
        split = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    X_train, X_test, y_train, y_test = split
    _DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    return X_train, X_test, y_train, y_test
