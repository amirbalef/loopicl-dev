# TALENT Benchmark Runner

Evaluates models on the [TALENT](https://github.com/qile2000/LAMDA-TALENT) benchmark suite using the native TALENT Python API.

## Installation

Install TALENT from source (run from the repo root):

```bash
cd ../temp/
git clone https://github.com/qile2000/LAMDA-TALENT
cd LAMDA-TALENT
uv pip install -r requirements.txt
uv pip install -e . --upgrade
```

Then copy the TALENT dataset files into the expected location (relative to the repo root):

```bash
cp -r /path/to/talent/datasets ../results/benchmarks/datasets/talent/datasets-talent-default
```

## Protocol

- Uses TALENT's native per-dataset train / val / test splits (from `.npy` files).
- Trains on train + val, evaluates on held-out test.
- Default: 1 seed for `quick` and `talent-lite`, 5 seeds for `talent`.
- No additional cross-validation or re-splitting.

## Usage

```bash
# Quick smoke-test (3 datasets, 1 seed)
uv run benchmark_eval/talent/run_benchmark_talent.py --dataset quick

# talent-lite suite with specific models
uv run benchmark_eval/talent/run_benchmark_talent.py --dataset talent-lite --models LightGBM,CatBoost

# Full TALENT suite (5 seeds)
uv run benchmark_eval/talent/run_benchmark_talent.py --dataset talent

# Specific datasets and models
uv run benchmark_eval/talent/run_benchmark_talent.py --datasets jasmine,adult --models TabICLv2,TabPFNv2

# Multiple seeds
uv run benchmark_eval/talent/run_benchmark_talent.py --dataset talent-lite --seed-num 5

# Custom TALENT data folder
uv run benchmark_eval/talent/run_benchmark_talent.py --talent-folder /path/to/talent/data
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `talent-lite` | Preset: `quick`, `talent-lite`, `talent` |
| `--datasets` | — | Comma-separated dataset names (overrides `--dataset`) |
| `--models` | all | Comma-separated model names to run |
| `--all-models` | — | Run all models in the default order |
| `--seed-num` | preset default | Seeds per (model, dataset) pair |
| `--talent-folder` | `results/benchmarks/datasets/talent/datasets-talent-default` | Path to raw TALENT data |
| `--output-dir` | `results/talent-native/<dataset>/` | Output directory |
| `--overwrite` | `False` | Re-run even if result already exists |
| `--raise-on-failure` | `False` | Raise exceptions instead of skipping failures |

## Supported Models

| Friendly name | TALENT registry name | Type |
|---|---|---|
| `TabICLv2` | `tabicl_v2` | Tabular ICL foundation |
| `TabICL` | `tabicl` | Tabular ICL foundation |
| `RealTabPFN-2.5` | `tabpfn_v2_5` | TabPFN family |
| `TabPFNv2` | `tabpfn_v2` | TabPFN family |
| `LimiX` | `limix` | Retrieval-augmented |
| `ModernNCA` | `modernNCA` | Retrieval-augmented |
| `CatBoost` | `catboost` | Tree-based |
| `LightGBM` | `lightgbm` | Tree-based |
| `RandomForest` | `RandomForest` | Tree-based |
| `RealMLP` | `realmlp` | Deep learning |
| `FT-Transformer` | `ftt` | Deep learning |
| `MLP-PLR` | `mlp_plr` | Deep learning |
| `AutoInt` | `autoint` | Deep learning |
| `ExcelFormer` | `excelformer` | Deep learning |
| `ResNet` | `resnet` | Deep learning |
| `MLP` | `mlp` | Deep learning |
| `KNN` | `knn` | Distance-based |

`TabPFN-2.5` is an alias for `RealTabPFN-2.5` (same TALENT entry); it is deduplicated automatically. `TabRXGBoost` is not in the TALENT registry and will be skipped with a warning.

## Output

```
results/talent-native/<dataset>/
  results/
    <talent_name>/
      <dataset_name>.json    # per-(model, dataset) metrics
  checkpoints/
    <talent_name>/
      <dataset_name>/        # model checkpoints
  leaderboard.csv            # aggregated results
```

Each JSON contains the model name, TALENT registry name, dataset name, task type, fit time, and all reported metrics. A ranked leaderboard table is printed to the console at the end of the run.
