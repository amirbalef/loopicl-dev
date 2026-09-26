"""LoopICL model configurations for the TALENT benchmark.

Each entry in CONFIGS defines one LoopICL variant to evaluate.
Add, remove, or duplicate entries to compare different checkpoints
or inference hyperparameters side-by-side.

Usage
-----
    uv run benchmark_eval/talent/run_benchmark_talent.py \\
        --ricotta-config benchmark_eval/talent/configs/ricotta_configs.py \\
        --dataset talent-lite

    # Run only specific LoopICL configs:
    uv run benchmark_eval/talent/run_benchmark_talent.py \\
        --ricotta-config benchmark_eval/talent/configs/ricotta_configs.py \\
        --models LoopICL-v1,LoopICL-v2 --dataset talent-lite

Fields
------
name                : str  — display name used as the model key (must be unique).
cls_weights_path    : str  — path to LoopICL classification checkpoint.
reg_weights_path    : str | None — path to LoopICLReg regression checkpoint.
                                   Required when supports_regression=True.
n_estimators        : int  — ensemble members (default 8).
batch_size          : int  — max ensemble members per forward call (default 8).
                             Reduce to save VRAM.
num_loops           : int | None — override ICL outer-loop count at inference.
                                   None = use the value the model was trained with.
subsample_samples   : int | None — cap training context to this many rows per
                                   estimator (None = all rows, default).
max_features        : int | None — cap features per estimator (TabFM style).
                                   None = all features (default).
test_chunk_size     : int | None — process test rows in chunks to reduce peak VRAM.
                                   None = all test rows in one call (default).
loop_residual_scaling_fn : str | None — override the residual scaling function at
                                   inference (e.g. "entropy_adaptive", "inv_n").
                                   None = use the value from the checkpoint (default).
device              : str | None — "cuda" / "cpu". None = auto-detect.
supports_regression : bool — if True, the config runs on BOTH classification and
                             regression datasets, loading cls_weights_path for cls
                             and reg_weights_path for regression. Default False
                             (classification only).

Notes
-----
- Classification (binclass / multiclass) is always run using cls_weights_path.
- Regression requires reg_weights_path and supports_regression=True.
- Threshold tuning on the validation split is applied automatically for
  binary classification (matches the TALENT evaluation protocol).
- Each seed reloads the model from disk and re-fits on train+val, so
  randomness comes only from the ensemble generator's random_state.
"""

# ---------------------------------------------------------------------------
# Edit the paths below to point at your checkpoints
# ---------------------------------------------------------------------------
_CKPT_BASE_= "../TICLA/results/workdir/ricotta_exp/one_layer_no_scaling_stage2/checkpoint/dir/step-30000.ckpt"
_CKPT_REG_= "../TICLA/results/workdir/ricotta_reg/no_scaling_stage2-long/checkpoint/dir/step-10000.ckpt"


num_loops_options = [8, 10, 6, 2, 4, 12, 14, 16, 1 , 3 , 5 , 7 ,  9, 11]

CONFIGS = []
for num_loops in num_loops_options:
    CONFIGS.append(
        {
            "name":                f"LoopICL-{num_loops}loops",
            "model":               "LoopICL",
            "weights_path":        _CKPT_BASE_,
            "reg_weights_path":    _CKPT_REG_    ,
            "supports_regression": True,
            "y_transform_methods": [None],
            "n_estimators":        8,
            "batch_size":          8,
            "num_loops":           num_loops,
        }
    )