"""Benchmark dataclasses: named collections of datasets for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PriorDatasetConfig:
    """Config for one prior-generated dataset group within a benchmark.

    Parameters
    ----------
    name:
        Short identifier, used as the dataset name in metrics (e.g. ``mix_scm_small``).
    n_datasets:
        Total number of (X_train, X_test, y_train, y_test) tuples to generate
        and save to disk.  Run ``scripts/generate_benchmark_priors.py`` once to
        produce them.
    n_val_datasets:
        How many of the saved datasets to load during each validation step.
        Must be <= n_datasets.
    """

    name: str
    n_datasets: int = 100
    n_val_datasets: int = 20
    batch_size: int = 1
    prior_type: str = "graph_scm"
    prior_tasktype: str = "classification"
    min_features: int = 2
    max_features: int = 100
    min_classes: int = 2
    max_classes: int = 10
    min_seq_len: Optional[int] = None
    max_seq_len: int = 1024
    log_seq_len: bool = False
    seq_len_per_gp: bool = False
    min_train_size: float = 0.1
    max_train_size: float = 0.9
    replay_small: bool = False
    seed: int = 42


@dataclass
class Benchmark:
    """A named collection of datasets used for evaluation.

    A benchmark contains two dataset sources:

    * **Prior datasets** – synthetic tabular datasets generated from the prior.
      These are generated once and saved to disk (see
      ``scripts/generate_benchmark_priors.py``).  Each entry in
      ``prior_datasets`` corresponds to a different prior configuration (e.g.
      small vs. large feature space).

    * **OpenML datasets** – real-world tabular tasks referenced by their OpenML
      task ID.  These are loaded on demand from the OpenML repository and split
      with a fixed seed for reproducibility.

    Parameters
    ----------
    name:
        Unique identifier for the benchmark (e.g. ``"default"``).
    prior_datasets:
        List of prior dataset configurations included in this benchmark.
    openml_task_ids:
        OpenML task IDs included in this benchmark.
    seed:
        Random seed used for all train/test splits (both OpenML and, if
        applicable, any per-dataset re-splitting).
    test_size:
        Fraction of each dataset reserved for testing (OpenML splits).
    """

    name: str
    prior_datasets: list[PriorDatasetConfig] = field(default_factory=list)
    openml_task_ids: list[int] = field(default_factory=list)
    openml_suite_ids: list[int] = field(default_factory=list)
    openml_use_official_folds: bool = False  # use task.get_train_test_split_indices(fold=0) instead of random split
    # OpenML dataset IDs (not task IDs) resolved to canonical regression task IDs at runtime with caching.
    # Use for regression-only benchmarks where you have dataset IDs from dev_datasets.py or similar.
    openml_reg_dataset_ids: list[int] = field(default_factory=list)
    # OpenML dataset IDs (not task IDs) resolved to canonical classification task IDs at runtime with caching.
    # Use for classification-only benchmarks where you have dataset IDs from dev_datasets.py or similar.
    openml_cls_dataset_ids: list[int] = field(default_factory=list)
    seed: int = 42
    test_size: float = 0.3
