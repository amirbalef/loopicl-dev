# Benchmarks

This module handles benchmark definition, dataset loading, baseline evaluation, and candidate model validation.

---

## Overview

A benchmark is a named collection of datasets grouped by source. Each dataset yields a fixed `(X_train, X_test, y_train, y_test)` split. The evaluation pipeline has two phases:

1. **Setup** (run once) — generate prior datasets and evaluate baseline models, saving results to disk.
2. **Validation** (run during training) — evaluate the candidate model and compare it against the pre-computed baselines.

---

## Dataset Sources

| Source | Class/Key | Task types | Notes |
|---|---|---|---|
| **Prior (synthetic)** | `PriorDatasetConfig` | cls / reg | Generated from SCM priors; saved as `.npz` files |
| **OpenML** | `openml_task_ids`, `openml_suite_ids` | cls / reg | Loaded on demand by task ID |
| **TALENT** | `talent_dataset_names`, `talent_data_dir` | cls / reg | Local directory of pre-split datasets |
| **TabularMath** | `tabularmath_dataset_names` | reg only | GSM8K / AIME tabular regression tasks |
| **MultiTab** | `multitab_dataset_ids` | cls / reg | 196 curated OpenML datasets |
| **PMLB** | `pmlb_dataset_names`, `pmlb_task_filter` | cls / reg | Penn ML Benchmarks; auto-discovered at runtime |
| **Tabnetics** | `tabnetics_dataset_names`, `tabnetics_dataset_set` | cls only | HDLSS biological datasets (transcriptomics, proteomics, metabolomics) |

---

## Train/Test Splitting

### Prior datasets

Generated synthetically from a Structural Causal Model (SCM) prior. Each dataset is a single draw from the prior; the train/test split is baked in at generation time via `min_train_size` / `max_train_size` (fractions of total sequence length).

Key generation parameters:

| Parameter | Description |
|---|---|
| `prior_type` | SCM family: `mix_scm`, `tree_scm`, `mlp_scm` (v1) or `graph_scm` (v2) |
| `min_features` / `max_features` | Feature count range |
| `min_seq_len` / `max_seq_len` | Total dataset size range |
| `log_seq_len` | Sample sequence length on a log scale if True |
| `min_train_size` / `max_train_size` | Train fraction range (default 0.1–0.9) |
| `max_classes` | Maximum number of classes for classification |
| `replay_small` | Oversample small datasets if True |
| `n_datasets` | Total datasets to generate and save to disk |
| `n_val_datasets` | Subset loaded during each validation step (`<= n_datasets`) |

### Real-world datasets (OpenML, TALENT, MultiTab, PMLB, Tabnetics)

Split with `sklearn.model_selection.train_test_split` using:
- `test_size` — fraction held out for testing (default **0.3**)
- `seed` — global random seed (default **42**), ensuring reproducibility across all runs

### TabularMath datasets

Support two split modes controlled by `tabularmath_split`:
- `"random"` — random train/test split
- `"ood"` — out-of-distribution split (held-out problem types)

---

## Metrics

Problem type is inferred automatically from `y_train`:
- `"binary"` — 2 unique classes
- `"multiclass"` — more than 2 unique classes
- `"regression"` — `task_type="reg"`

### Per-dataset metrics

| Problem type | Primary metric | Additional metrics |
|---|---|---|
| Binary classification | `roc_auc` (higher = better) | `pr_auc`, `mcc`, `logloss` |
| Multiclass classification | `balanced_accuracy` (higher = better) | `mcc`, `logloss` |
| Regression | `neg_rmse` (higher = better, i.e. negated RMSE) | `r2` |

- **`pr_auc`** (average precision / area under PR curve) — more informative than ROC-AUC on imbalanced binary datasets; not affected by the large number of true negatives that can inflate ROC-AUC.
- **`mcc`** (Matthews Correlation Coefficient, in `[-1, 1]`) — accounts for all four confusion matrix cells; reliable single metric for both binary and multiclass imbalanced problems.

### Aggregate metrics (across datasets)

After collecting per-dataset scores, several aggregate statistics are computed:

**Skill score** — normalizes each primary metric to `[0, 1]` where 0 = trivial baseline and 1 = perfect:

| Metric | Trivial baseline | Formula |
|---|---|---|
| `roc_auc` | 0.5 (random ranking) | `(score − 0.5) / 0.5` |
| `balanced_accuracy` | `1 / n_classes` (random guessing) | `(score − chance) / (1 − chance)` |
| `mcc` | 0.0 (random predictions) | `max(0, score)` (clips negatives to 0) |
| `r2` | 0.0 (predicting the mean) | `score` (already in [0, 1]) |
| `pr_auc`, `neg_rmse` | dataset-dependent | not normalizable; excluded from skill score |

**Summary statistics** (logged to wandb under `results/{benchmark_name}/`):

| Key | Description |
|---|---|
| `performance/{label}` | Mean primary metric across all datasets |
| `logloss/{label}` | Mean logloss across classification datasets |
| `r2/{label}` | Mean R² across regression datasets |
| `mean_skill_score/{label}` | Mean skill score (candidate only) |
| `sgm_skill_score/{label}` | Shifted geometric mean of skill scores |
| `mean_normalized_vs_random_forest/{label}` | Mean relative improvement of candidate over Random Forest skill score |
| `sgm_normalized_vs_random_forest/{label}` | SGM of relative improvements |
| `win_rate_vs_random_forest/{label}` | Fraction of datasets where candidate beats Random Forest |
| `improvability/{label}` | Mean improvability of the candidate (%) — average relative error gap to the best method per dataset |

**Improvability** measures how far the candidate is from the best-performing method on each dataset:

```
Improvability_i(m) = (err_i(m) − err*_i) / err_i(m) × 100%
```

where `err*_i = min_{m'} err_i(m')` is the error of the best method on dataset *i*.  Improvability is always in `[0%, 100%]` — 0% means the candidate is already optimal, 100% means the best method has zero error while the candidate does not.  Unlike win-rate, improvability is sensitive to the magnitude of performance differences.

---

## Baseline Models

Baseline models are evaluated once via `scripts/evaluate_baselines.py` and results are stored as individual JSON files at `{benchmark_dir}/{benchmark_name}/{dataset_name}/{model_name}.json`.

Available baseline model keys:

| Key | Model | Task support |
|---|---|---|
| `tabicl_v1` / `tabicl_1` | TabICL v1 classifier | cls only |
| `tabicl_v1_1` / `tabicl_1_1` | TabICL v1.1 classifier | cls only |
| `tabicl` / `tabicl_v2` / `tabicl_2` | TabICL v2 classifier/regressor | cls + reg |
| `tabpfn_v2` | TabPFN v2 | cls + reg |
| `tabpfn_v2_5` / `tabpfn_2_5` | TabPFN v2.5 | cls + reg |
| `tabpfn_v2_6` / `tabpfn_2_6` | TabPFN v2.6 | cls + reg |
| `tabpfn_v3` / `tabpfn_3` | TabPFN v3 | cls + reg |
| `limix` | LimiX classifier | cls only |
| `random_forest` / `rf` | sklearn RandomForest (100 trees, `random_state=42`) | cls + reg |

All TabPFN variants are loaded with `ignore_pretraining_limits = True`.

---

## Candidate Model Configuration

During validation the candidate model is wrapped in a `TabICLWrapper` and evaluated through the TabICL v2 interface (preprocessing + ensembling).

### Inference interface (`wrap_interface`)

The candidate is wrapped as a TabICL v2 estimator so it inherits TabICL's preprocessing pipeline (feature encoding, normalization, ensembling). The underlying transformer backbone is replaced with the model under training.

Key inference parameters:

| Parameter | Default | Description |
|---|---|---|
| `inference_batch_size` | 512 | Max test samples per forward pass |
| `inference_num_blocks` | `None` (model default) | Override number of ICL attention blocks at inference; useful for block-sweep evaluation |
| `dtype` | `"float32"` | Floating point precision during inference |

### Inference memory management (`InferenceConfig` / `MgrConfig`)

The model's forward pass is split across three managers — column attention (`COL_CONFIG`), row attention (`ROW_CONFIG`), and ICL attention (`ICL_CONFIG`) — each with independent memory offloading settings:

| Parameter | Default | Description |
|---|---|---|
| `use_amp` | `True` | Automatic mixed precision |
| `use_fa3` | `True` | Flash Attention 3 |
| `safety_factor` | `0.8` | Conservative memory usage multiplier for batch size estimation |
| `offload` | `"auto"` (COL) / `False` (ROW, ICL) | Where to store intermediate outputs: `"gpu"`, `"cpu"`, `"disk"`, or `"auto"` |
| `auto_offload_threshold` | `0.5` | GPU memory fraction above which auto-offloading activates |
| `max_pinned_memory_mb` | `32768` (32 GB) | Max output size for pinned CPU memory |
| `use_async` | `True` | Async CUDA streams for GPU-to-CPU transfers |
| `async_depth` | `4` | Max pending async copies before blocking |

---

## Built-in Benchmarks

Selected pre-defined benchmarks from `builtin.py`:

| Name | Dataset sources | Description |
|---|---|---|
| `test` | 3 prior datasets (mix_scm, ≤128 rows, ≤10 features) | Minimal sanity check |
| `tabicl-v1-priors` | 40 prior datasets (mix_scm, v1 regime) | TabICL v1 prior validation |
| `tabicl-v2-priors` | 40 prior datasets (graph_scm, v2 regime) | TabICL v2 prior validation |
| `tabicl-v2-priors-medium` | 10 prior datasets (graph_scm, 512–2048 rows, ≤30 features) | TabICL v2 medium-scale prior validation |
| `tabzilla` | OpenML suite 379 (36 cls tasks) | TabZilla benchmark |
| `opencc18` | OpenML suite 337 (72 cls tasks) | OpenML-CC18 benchmark |
| `tabarena` | 21 OpenML task IDs | TabArena benchmark |
| `multitab` | 196 MultiTab datasets (cls + reg) | MultiTab benchmark |
| `tabnetics-core` | All core Tabnetics datasets | HDLSS biological benchmark |
| `pmlb` | All PMLB datasets (cls + reg) | Penn ML Benchmarks |
| `tabularmath` | GSM8K + AIME tabular regression (random split) | Math reasoning benchmark |
| `talent-default` | All TALENT default datasets | TALENT benchmark |

---

## File Layout

```
{benchmark_dir}/
  {benchmark_name}/
    {dataset_name}/
      {model_name}.json        # {"performance": 0.87, "logloss": 0.31, "updated_at": "..."}
    prior_{prior_name}/
      0000.npz                 # X_train, X_test, y_train, y_test arrays
      0001.npz
      ...
```

---

## Setup Flow

```
scripts/generate_benchmark_priors.py   →  generate_priors()
scripts/evaluate_baselines.py          →  evaluate_baselines()
trainer (auto_setup=True)              →  ensure_benchmark_ready()  (runs both if missing)
```

During training, `BenchmarkValidator.validate()` loads prior datasets and real-world splits, evaluates the candidate model, loads pre-computed baseline results from disk, and logs all metrics and ELO ratings to wandb.
