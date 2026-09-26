"""All ricotta ablation models evaluation config — one entry per (model, checkpoint step).

Contains every model variant from src/ricotta/configs/ricotta/,
evaluated at each permanent checkpoint (every 5 000 steps up to 100 000).

Run with:
    python benchmark_eval/openml/run_benchmark.py --dataset dev-cls-lite --config ablations.ricotta_ablation_resample

"""

CHECKPOINT_DIR_ABLATION =  "../TICLA/results/workdir/ricotta_cls/"

CHECKPOINT_STEPS = list(range(0, 100_001, 5_000))

num_loops = 6

_BASE = {
    "save_mode":            "single_file",
    "num_loops":            num_loops,
    "interface":            "ricotta",
    "batch_size":           8,
    "n_estimators":         8,
    "multiclass_strategy":  "tabpfn",
    "subsample_samples": None,
    # "max_features":     100,
    "subsample_train":  2048,
    "subsample_test":   2048,
    "test_chunk_size":  None, 
}

def _abl(exp_name):
    return f"{CHECKPOINT_DIR_ABLATION}/{exp_name}/checkpoint/dir"


# (name, model_class, checkpoint_dir, steps)
# steps=None  →  use CHECKPOINT_STEPS (5 000 … 100 000)
_MODELS = [

    # # ------------------------------------------------------------------ #
    # # Base models                                                          #
    # # ------------------------------------------------------------------ #
    #("ricotta_base_v1",                "LoopICL",        _abl("ricotta_base_v1"),                None),
    ("ricotta_one_layer_v1",           "LoopICL",        _abl("ricotta_one_layer_v1"),           None),
    ("ricotta_one_layer_v1_no_scaling","LoopICL",        _abl("ricotta_one_layer_v1_no_scaling"),None),
    ("ricotta_one_layer_light_scaling","LoopICL",        _abl("ricotta_one_layer_light_scaling"),None),
    ("ricotta_adaptive_scaling_no_fe","LoopICL",        _abl("ricotta_adaptive_scaling_no_fe"),None, {"loop_residual_scaling_fn": "inv_n"}),
    # # ------------------------------------------------------------------ #
    ("ricotta_one_layer_unrolled",     "RicottaUnrolled",        _abl("ricotta_one_layer_unrolled"),   None, {"num_loops": 1}),
    # # # ------------------------------------------------------------------ #
]


# ------------------------------------------------------------------ #
# Baselines                                                            #
# ------------------------------------------------------------------ #
CONFIGS = [
    {
        "name":            "tabicl_v2",
        "model":           "TabICL-v2",
        "folder":          "tabicl_v2",
        "subsample_train": 2048,
        "subsample_test":  2048,
        "save_mode":       "single_file",
    },
    {
        "name":            "tabpfn_v3",
        "model":           "TabPFN-v3",
        "folder":          "tabpfn_v3",
        "subsample_train": 2048,
        "subsample_test":  2048,
        "save_mode":       "single_file",
    },
]

CONFIGS += [
    {
        **_BASE,
        **overrides,
        "name":         f"{name}_{overrides.get('num_loops', num_loops)}_step_{step}",
        "folder":       f"{name}_{overrides.get('num_loops', num_loops)}/step_{step}",
        "model":        model_cls,
        "weights_path": f"{ckpt_dir}/step-{step}.ckpt",
    }
    for name, model_cls, ckpt_dir, steps, *rest in _MODELS
    for overrides in [rest[0] if rest else {}]
    for step in (steps if steps is not None else CHECKPOINT_STEPS)
]
