"""LoopICL Early Exit evaluation configs for TabArena.

Evaluates LoopICLEarlyExit — a classification model with a test-set-agnostic
learned exit gate trained on top of the no-residual-scaling stage-2 checkpoint.

Gate (from loopicl_ee_cls-n training):
  - gate_delta_out_dim=8, gate_hidden_dim=64
  - gate_use_delta=True, gate_use_loop_index=True (16-dim), gate_history_len=3
  - gate_train_size_dim=8
  - 16 max loops; gate_only_stage (LM frozen)

exit_q controls the CDF quantile threshold for inference-time exit:
  1.0 = run all 16 loops (no early exit — upper-bound reference)
  0.7 = conservative early exit
  0.5 = balanced
  0.3 = aggressive early exit

Run with
--------
    uv run benchmark_eval/tabarena/run_benchmark.py \\
        --dataset tabarena-cls --config loopicl-ee
"""

_CKPT_EE_V2 = (
    "/cephfs/users/<username>/codes//TICLA/results/workdir/ricotta_exp/"
    "ricotta_ee_v2_cls-n/checkpoint/dir/step-4000.ckpt"
)

CONFIGS = [
    # {
    #     "name":         "LoopICL-EEv2-full",
    #     "model":        "LoopICLEarlyExitV2",
    #     "weights_path": _CKPT_EE_V2,
    #     "exit_q":       1.0,
    #     "n_estimators": 8,
    #     "batch_size":   8,
    # },
    {
        "name":         "LoopICL-EE-q70",
        "model":        "LoopICLEarlyExitV2",
        "weights_path": _CKPT_EE_V2,
        "exit_q":       0.7,
        "n_estimators": 8,
        "batch_size":   8,
    },
    {
        "name":         "LoopICL-EE-q50",
        "model":        "LoopICLEarlyExitV2",
        "weights_path": _CKPT_EE_V2,
        "exit_q":       0.5,
        "n_estimators": 8,
        "batch_size":   8,
    },
    {
        "name":         "LoopICL-EE-q30",
        "model":        "LoopICLEarlyExitV2",
        "weights_path": _CKPT_EE_V2,
        "exit_q":       0.3,
        "n_estimators": 8,
        "batch_size":   8,
    },
]
