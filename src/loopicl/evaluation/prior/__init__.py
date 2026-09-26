"""Prior dataset loader: synthetic tabular datasets generated from the prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _prior_dataset_dir(benchmark_dir: str, benchmark_name: str, config_name: str) -> Path:
    return Path(benchmark_dir) / benchmark_name / "priors" / config_name


def load_prior_datasets(
    benchmark_dir: str,
    benchmark_name: str,
    config_name: str,
    n: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load saved prior datasets as ``(X_train, X_test, y_train, y_test)`` tuples.

    Parameters
    ----------
    benchmark_dir:
        Root directory containing benchmark data (e.g. ``results/benchmarks/``).
    benchmark_name:
        Name of the benchmark (e.g. ``"default"``).
    config_name:
        Name of the prior dataset config (e.g. ``"mix_scm_small"``).
    n:
        If given, only the first *n* files are loaded.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no ``.npz`` files.
    """
    data_dir = _prior_dataset_dir(benchmark_dir, benchmark_name, config_name)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Prior dataset directory not found: {data_dir}\n"
            f"Run  python scripts/generate_benchmark_priors.py --benchmark {benchmark_name}  first."
        )

    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if n is not None:
        files = files[:n]

    datasets = []
    for f in files:
        d = np.load(f)
        datasets.append((d["X_train"], d["X_test"], d["y_train"], d["y_test"]))
    return datasets


def save_prior_dataset(
    benchmark_dir: str,
    benchmark_name: str,
    config_name: str,
    idx: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Persist a single (X_train, X_test, y_train, y_test) tuple as a .npz file."""
    data_dir = _prior_dataset_dir(benchmark_dir, benchmark_name, config_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / f"{idx:04d}.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
