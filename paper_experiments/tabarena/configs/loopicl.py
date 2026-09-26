"""LoopICL evaluation configs for TabArena.

Each entry in CONFIGS defines one LoopICL variant to evaluate.

Fields
------
name              : display name in leaderboard / output directory
model             : must be "LoopICL"
weights_path      : path to the .ckpt checkpoint saved by the LoopICL trainer
reg_weights_path  : (optional) path to a LoopICLReg regression checkpoint;
                    required when supports_regression=True
supports_regression : if True, run on regression datasets using reg_weights_path
                    (default False — classification only)
y_transform_methods : list of y-transform methods to cycle across estimators
                    (e.g. [None] to disable safepower, [None, "safepower"] to enable).
                    None = use regressor default.
n_estimators      : number of ensemble members (default 8)
batch_size        : max ensemble members per forward call (default 8);
                    reduce to save VRAM
num_loops         : (optional) override ICL outer-loop count at inference;
                    None = use the value the model was trained with
subsample_samples : (optional) cap training context to this many rows per
                    estimator (None = all rows, default)
max_features      : (optional) cap features per estimator (TabFM style);
                    None = all features (default)
test_chunk_size   : (optional) override test-row chunk size (default 4096)
row_chunk_size    : (optional) override train-row chunk size in cross-col + readout
                    (default 4096)
col_chunk_size    : (optional) override column chunk size in within-col set-transformer
                    (default 16)

Notes
-----
- GPU device is auto-detected from AutoGluon's resource allocation.
- Classification (binary / multiclass) is always supported.
- Regression requires reg_weights_path and supports_regression=True.

Run with
--------
    uv run benchmark_eval/tabarena/run_benchmark.py \\
        --dataset tabarena-cls --config ricotta_base_v1
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