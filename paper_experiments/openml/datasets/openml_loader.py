"""OpenML task loading with optional few-shot subsampling.

No tabarena or autogluon dependency — uses the openml library directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_task(task_id: int):
    """Load an OpenML task.

    Returns
    -------
    task        : openml task object
    X           : pd.DataFrame  (string column names, raw dtypes)
    y           : pd.Series
    problem_type: "binary" | "multiclass" | "regression"
    """
    import openml
    from openml.tasks import TaskType

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    if dataset.default_target_attribute is None:
        raise ValueError(f"Task {task_id}: dataset has no default_target_attribute")
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )
    if y is None:
        raise ValueError(f"Task {task_id}: target column '{dataset.default_target_attribute}' not found in dataset")
    X.columns = X.columns.astype(str)

    is_regression = task.task_type_id == TaskType.SUPERVISED_REGRESSION
    if is_regression:
        problem_type = "regression"
        y = y.astype(float)
    else:
        problem_type = "binary" if y.nunique() == 2 else "multiclass"

    return task, X, y, problem_type


def get_split(
    task,
    X: pd.DataFrame,
    y: pd.Series,
    fold: int = 0,
    repeat: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) for a given fold/repeat.

    If the OpenML task does not have the requested repeat stored, falls back to
    a random stratified split with the same test proportion as fold 0/repeat 0,
    seeded by (repeat * 1000 + fold).
    """
    try:
        train_idx, test_idx = task.get_train_test_split_indices(
            fold=fold, repeat=repeat, sample=0
        )
    except Exception as exc:
        if "not known" not in str(exc):
            raise
        # Infer test size from the canonical fold 0 / repeat 0 split
        base_train, base_test = task.get_train_test_split_indices(fold=0, repeat=0, sample=0)
        test_size = len(base_test) / (len(base_train) + len(base_test))
        seed = repeat * 1000 + fold
        from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            [(train_idx, test_idx)] = splitter.split(X, y)
        except Exception:
            splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            [(train_idx, test_idx)] = splitter.split(X)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Few-shot subsampling (same logic as benchmark_eval/datasets/fewshot_wrapper.py)
# ---------------------------------------------------------------------------

def stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Subsample to `size` rows, guaranteeing ≥1 sample per class."""
    classes, counts = np.unique(y, return_counts=True)

    # Clean stratified split when all classes are sufficiently represented
    if counts.min() >= 2 and all(int(c * size / len(y)) >= 1 for c in counts):
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=random_state)
        idx, _ = next(sss.split(X, y))
        return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    # Sparse minority classes: reserve one sample per class, fill the rest randomly
    rng = np.random.default_rng(random_state)
    reserved = [int(np.where(y.values == c)[0][0]) for c in classes]
    pool = [i for i in range(len(y)) if i not in reserved]
    n_fill = max(0, size - len(reserved))
    fill = rng.choice(pool, size=min(n_fill, len(pool)), replace=False).tolist()
    chosen = reserved + fill
    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)
