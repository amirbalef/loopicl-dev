"""LoopICL evaluation configs for TabArena.

Run with
--------
    python benchmark_eval/tabarena/run_benchmark.py --dataset tabarena-cls --config loopicl
"""


_CKPT_BASE_= "results/weights/loopicl-cls-stage2-step-30000.ckpt"


num_loops_options = [1, 2, 4, 6, 8, 10, 12, 14, 16]

CONFIGS = []
for num_loops in num_loops_options:
    CONFIGS.append(
        {
            "name":                f"LoopICL-{num_loops}loops",
            "model":               "LoopICL",
            "weights_path":        _CKPT_BASE_,
            "reg_weights_path":    None,
            "supports_regression": False,
            "y_transform_methods": [None],
            "n_estimators":        8,
            "batch_size":          8,
            "num_loops":           num_loops,
        }
    )