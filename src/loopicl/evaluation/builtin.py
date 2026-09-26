"""Pre-defined benchmark configurations."""

from __future__ import annotations

from loopicl.evaluation.benchmark import Benchmark, PriorDatasetConfig


BENCHMARKS: dict[str, Benchmark] = {

    # ------------------------------------------------------------------
    # "test" – fast iteration, minimal datasets
    # ------------------------------------------------------------------
    "test": Benchmark(
        name="test-priors",
        prior_datasets=[
            PriorDatasetConfig(
                name="mix_scm_small",
                n_datasets=3,
                n_val_datasets=3,
                prior_type="mix_scm",
                min_features=2,
                max_features=10,
                max_seq_len=128,
                log_seq_len=False,
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # "small" – fast iteration, minimal datasets
    # ------------------------------------------------------------------
    "small-priors": Benchmark(
        name="small-priors",
        prior_datasets=[
            PriorDatasetConfig(
                name="mix_scm_small",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="mix_scm",
                min_features=2,
                max_features=10,
                max_seq_len=512,
                log_seq_len=False,
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # "small" – fast iteration, minimal datasets
    # ------------------------------------------------------------------
    "medium-priors": Benchmark(
        name="medium-priors",
        prior_datasets=[
            PriorDatasetConfig(
                name="mix_scm_small",
                n_datasets=10,
                n_val_datasets=10,
                prior_type="mix_scm",
                min_features=2,
                max_features=30,
                min_seq_len = 512,
                max_seq_len=2048,
                min_train_size= 0.5,
                log_seq_len=False,
            ),
        ],
    ),
    "large-priors": Benchmark(
        name="large-priors",
        prior_datasets=[
            PriorDatasetConfig(
                name="mix_scm_small",
                n_datasets=5,
                n_val_datasets=5,
                prior_type="mix_scm",
                min_features=2,
                max_features=30,
                min_seq_len = 4096,
                max_seq_len=20000,
                min_train_size= 0.5,
                log_seq_len=True,
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # "tabicl-v1-priors" – small benchmark using tabicl v1 priors (mix_scm)
    # ------------------------------------------------------------------
    "tabicl-v1-priors": Benchmark(
        name="tabicl-v1-priors",
        prior_datasets=[
            PriorDatasetConfig(
                name="v1_mix_scm_small_binary",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="mix_scm",
                min_features=2,
                max_features=100,
                min_classes=2,
                max_classes=2,
                min_seq_len=512,
                max_seq_len=2048,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v1_mix_scm_small_multi",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="mix_scm",
                min_features=2,
                max_features=100,
                min_classes=3,
                max_classes=10,
                min_seq_len=512,
                max_seq_len=2048,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v1_mix_scm_large_binary",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="mix_scm",
                min_features=2,
                max_features=30,
                min_classes=2,
                max_classes=2,
                min_seq_len=4096,
                max_seq_len=20000,
                min_train_size=0.5,
                log_seq_len=True,
            ),
            PriorDatasetConfig(
                name="v1_mix_scm_large_multi",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="mix_scm",
                min_features=2,
                max_features=30,
                min_classes=3,
                max_classes=10,
                min_seq_len=4096,
                max_seq_len=20000,
                min_train_size=0.5,
                log_seq_len=True,
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # "tabicl-v2-priors" – benchmark using tabicl v2 priors (graph_scm)
    #   Includes both classification and regression prior configs so that
    #   BenchmarkValidatorConfig with task_type="cls" or task_type="reg" both work.
    # ------------------------------------------------------------------
    "tabicl-v2-priors": Benchmark(
        name="tabicl-v2-priors",
        prior_datasets=[
            # --- Classification ---
            PriorDatasetConfig(
                name="v2_graph_scm_small_binary",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="graph_scm",
                min_features=2,
                max_features=100,
                min_classes=2,
                max_classes=2,
                min_seq_len=512,
                max_seq_len=2048,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v2_graph_scm_small_multi",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="graph_scm",
                min_features=2,
                max_features=100,
                min_classes=3,
                max_classes=10,
                min_seq_len=512,
                max_seq_len=2048,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v2_graph_scm_medium_binary",
                n_datasets=10,
                n_val_datasets=10,
                prior_type="graph_scm",
                min_features=2,
                max_features=60,
                min_classes=2,
                max_classes=2,
                min_seq_len=2048,
                max_seq_len=4096,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v2_graph_scm_medium_multi",
                n_datasets=10,
                n_val_datasets=10,
                prior_type="graph_scm",
                min_features=2,
                max_features=60,
                min_classes=3,
                max_classes=10,
                min_seq_len=2048,
                max_seq_len=4096,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v2_graph_scm_large_binary",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="graph_scm",
                min_features=2,
                max_features=30,
                min_classes=2,
                max_classes=2,
                min_seq_len=4096,
                max_seq_len=20000,
                min_train_size=0.5,
                log_seq_len=True,
            ),
            PriorDatasetConfig(
                name="v2_graph_scm_large_multi",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="graph_scm",
                min_features=2,
                max_features=30,
                min_classes=3,
                max_classes=10,
                min_seq_len=4096,
                max_seq_len=20000,
                min_train_size=0.5,
                log_seq_len=True,
            ),
            # --- Regression ---
            PriorDatasetConfig(
                name="v2_graph_scm_small_reg",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="graph_scm",
                prior_tasktype="regression",
                min_features=2,
                max_features=100,
                min_seq_len=512,
                max_seq_len=2048,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v2_graph_scm_medium_reg",
                n_datasets=10,
                n_val_datasets=10,
                prior_type="graph_scm",
                prior_tasktype="regression",
                min_features=2,
                max_features=60,
                min_seq_len=2048,
                max_seq_len=4096,
                min_train_size=0.5,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="v2_graph_scm_large_reg",
                n_datasets=20,
                n_val_datasets=20,
                prior_type="graph_scm",
                prior_tasktype="regression",
                min_features=2,
                max_features=30,
                min_seq_len=4096,
                max_seq_len=20000,
                min_train_size=0.5,
                log_seq_len=True,
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # LeoGrin / tabular-benchmark suites
    #   Four OpenML suites from "Why do tree-based models still outperform
    #   deep learning on tabular data?" (Grinsztajn et al., 2022).
    #
    #   Classification suites work out of the box.
    #   Regression suites (297, 335) require the OpenML loader to support
    #   float targets before they can be used.
    # ------------------------------------------------------------------
    "tabular-benchmark-num-cls": Benchmark(
        name="tabular-benchmark-num-cls",
        openml_suite_ids=[298],  # numerical features, classification
    ),
    "tabular-benchmark-cat-cls": Benchmark(
        name="tabular-benchmark-cat-cls",
        openml_suite_ids=[300],  # numerical + categorical features, classification
    ),
    "tabular-benchmark-num-reg": Benchmark(
        name="tabular-benchmark-num-reg",
        openml_suite_ids=[297],  # numerical features, regression (needs loader update)
    ),
    "tabular-benchmark-cat-reg": Benchmark(
        name="tabular-benchmark-cat-reg",
        openml_suite_ids=[335],  # numerical + categorical features, regression (needs loader update)
    ),

    # ------------------------------------------------------------------
    # "tabzilla" – TabZilla benchmark suite (suite ID 379)
    #   36 curated classification tasks; task list fetched from OpenML
    #   at runtime (cached locally after first fetch).
    # ------------------------------------------------------------------
    "tabzilla": Benchmark(
        name="tabzilla",
        openml_suite_ids=[379],
    ),

    # ------------------------------------------------------------------
    # "opencc18" – OpenML-CC18 benchmark suite (suite ID 337)
    #   72 curated classification tasks; task list fetched from OpenML
    #   at runtime via openml.study.get_suite(337).
    # ------------------------------------------------------------------
    "opencc18": Benchmark(
        name="opencc18",
        openml_suite_ids=[337],
    ),

    # ------------------------------------------------------------------
    # "tabarena"            – 21 cls + 13 reg tasks from TabArena-v0.1
    # "tabarena-binary"     – 30 binary classification tasks
    # "tabarena-multiclass" – 8 multiclass classification tasks
    # "tabarena-reg"        – 13 regression tasks from TabArena-v0.1
    # All use fold=0/repeat=0 (TabArena-Lite protocol) for fair comparison.
    # Regression task IDs resolved via:
    #   openml.study.get_suite("tabarena-v0.1") filtered to Supervised Regression
    # ------------------------------------------------------------------
    "tabarena": Benchmark(
        name="tabarena",
        openml_task_ids=[
            # classification (original 21)
            363621, 363671, 363696, 363629, 363626, 363682, 363684, 363700,
            363674, 363694, 363619, 363623, 363689, 363706, 363624, 363614,
            363685, 363702, 363704, 363707, 363711,
            # regression (13 from tabarena-v0.1)
            363612, 363615, 363625, 363631, 363672, 363675, 363678,
            363686, 363693, 363697, 363698, 363705, 363708,
        ],
        openml_use_official_folds=True,
    ),
    "tabarena-reg": Benchmark(
        name="tabarena-reg",
        openml_task_ids=[
            363612, 363615, 363625, 363631, 363672, 363675, 363678,
            363686, 363693, 363697, 363698, 363705, 363708,
        ],
        openml_use_official_folds=True,
    ),
    "tabarena-binary": Benchmark(
        name="tabarena-binary",
        openml_task_ids=[
            363613, 363616, 363618, 363619, 363620, 363621, 363623, 363624,
            363626, 363627, 363628, 363629, 363630, 363632, 363671, 363673,
            363674, 363676, 363679, 363681, 363682, 363683, 363684, 363689,
            363691, 363694, 363696, 363700, 363706, 363712,
        ],
        openml_use_official_folds=True,
    ),
    "tabarena-multiclass": Benchmark(
        name="tabarena-multiclass",
        openml_task_ids=[
            363614, 363677, 363685, 363699, 363702, 363704, 363707, 363711,
        ],
        openml_use_official_folds=True,
    ),

    # ------------------------------------------------------------------
    # "dev-cls" – development classification datasets (Table 5 of the TabArena paper)
    #   OpenML dataset IDs from benchmark_eval/openml/datasets/dev_datasets.py::CLASSIFICATION_DATASET_IDS.
    #   Resolved to canonical OpenML task IDs at runtime (cached in ~/.cache/ticla/).
    #   Use with BenchmarkValidatorConfig(benchmark="dev-cls", task_type="cls").
    # ------------------------------------------------------------------
    "dev-cls": Benchmark(
        name="dev-cls",
        openml_cls_dataset_ids=[
             1037, 40707,   459,   465,  1060,  1064,   185, 40588,  1463,  1465,
              473, 40665, 40711, 42192,   468, 40669,   803,    35, 44130, 40646,
              338,    49,    51,    53, 45026,  1479,    59,   375, 40997, 40677,
             1056, 42532,  1071, 43892, 42793,    30, 42585, 42172,  1496, 44160,
              679,   312,  4153,   340,   377,  1115, 40474, 40497, 40704, 40705,
            45545,  1508,  1527,    56,    60,  1511, 40693,
        ],
    ),

    # ------------------------------------------------------------------
    # "dev-reg" – development regression datasets (Table 6 of the TabArena paper)
    #   OpenML dataset IDs from benchmark_eval/openml/datasets/dev_datasets.py::REGRESSION_DATASET_IDS.
    #   Resolved to canonical OpenML task IDs at runtime (cached in ~/.cache/ticla/).
    #   Use with BenchmarkValidatorConfig(benchmark="dev-reg", task_type="reg").
    # ------------------------------------------------------------------
    "dev-reg": Benchmark(
        name="dev-reg",
        openml_reg_dataset_ids=[
            43878, 43926,   555,   456,   518,   551, 44055,   482,   196,   207,
              512,   560,   543,   695,   689,   210,  1051, 41928,   534, 44132,
              227, 41968, 41969, 23516, 42360,   198, 42362,   232,  4544,  1070,
              567,   553, 42364, 41187,   566, 41938,   405,   511,   522,   529,
              308, 44028,   660,   663, 40601, 42366, 42110, 42545, 42367,   206,
            44056, 42368, 42369,   503,
        ],
    ),

    # ------------------------------------------------------------------
    # "default" – balanced coverage of prior configs and OpenML tasks
    # ------------------------------------------------------------------
    "default": Benchmark(
        name="default",
        prior_datasets=[
            PriorDatasetConfig(
                name="mix_scm_small",
                n_datasets=100,
                n_val_datasets=20,
                prior_type="mix_scm",
                min_features=2,
                max_features=20,
                max_seq_len=256,
                log_seq_len=False,
            ),
            PriorDatasetConfig(
                name="mix_scm_large",
                n_datasets=100,
                n_val_datasets=20,
                prior_type="mix_scm",
                min_features=2,
                max_features=100,
                max_seq_len=1024,
                log_seq_len=True,
            ),
        ],
        openml_task_ids=[
            # Small-to-medium classification tasks commonly used in TabICL/TabPFN papers
            11,    # balance-scale    (625 samples,  4 features, 3 classes)
            14,    # mfeat-fourier    (2000 samples, 76 features, 10 classes)
            15,    # breast-w         (569 samples,  9 features, 2 classes)
            37,    # diabetes         (768 samples,  8 features, 2 classes)
            54,    # vehicle          (846 samples, 18 features, 4 classes)
            188,   # eucalyptus       (736 samples, 19 features, 5 classes)
            1049,  # pc4              (1458 samples, 37 features, 2 classes)
            1111,  # KR-vs-KP        (3196 samples, 36 features, 2 classes)
            1468,  # cnae-9           (1080 samples, 856 features, 9 classes)
        ],
    ),

}


def get_benchmark(name: str) -> Benchmark:
    """Return a built-in benchmark by name."""
    if name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {name!r}. Available: {sorted(BENCHMARKS)}"
        )
    return BENCHMARKS[name]
