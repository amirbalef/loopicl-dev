"""All ricotta ablation models evaluation config — one entry per (model, checkpoint step).

Contains every model variant from src/ricotta/configs/ricotta/,
evaluated at each permanent checkpoint (every 5 000 steps up to 100 000).

Run with:
    python benchmark_eval/openml/run_benchmark.py --dataset dev-cls-lite --config ablations.ricotta_ablation_resample

"""

CHECKPOINT_DIR_ABLATION =  "../TICLA/results/workdir/ricotta_cls/"
CHECKPOINT_DIR_STAGE2   =  "../TICLA/results/workdir/ricotta_exp/"
CHECKPOINT_NAME         =  "step-10000.ckpt"

CHECKPOINT_STEP = 100_000

SUBSAMPLE_SIZES = [256, 512, 1024, 2048, 4096, 8192, None]

num_loops = 6

_BASE = {
    "save_mode":            "single_file",
    "num_loops":            num_loops,
    "interface":            "ricotta",
    "batch_size":           1,
    "n_estimators":         1,
    "multiclass_strategy":  "tabpfn",
    "subsample_samples": None,
    # "max_features":     100,
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
    #("ricotta_adaptive_scaling_no_fe","LoopICL",        _abl("ricotta_adaptive_scaling_no_fe"),None, {"loop_residual_scaling_fn": "inv_n"}),
    # # ------------------------------------------------------------------ #
    ("ricotta_one_layer_unrolled",     "RicottaUnrolled",        _abl("ricotta_one_layer_unrolled"),   None, {"num_loops": 1}),
    # # # ------------------------------------------------------------------ #
]

# (name, model_class, overrides) — evaluated at full (no subsampling) only, step-10000
_MODELS_FULL_ONLY = [
    ("ricotta_one_layer_ablation_stage2",    "LoopICL",        {}),
    ("ricotta_no_scaling_ablation_stage2",   "LoopICL",        {}),
    ("ricotta_light_scaling_ablation_stage2","LoopICL",        {}),
    ("ricotta_unrolled_ablation_stage2",     "RicottaUnrolled",{"num_loops": 1}),
]


# ------------------------------------------------------------------ #
# Baselines                                                            #
# ------------------------------------------------------------------ #
_BASELINES = [
    ("tabicl_v2", "TabICL-v2"),
    ("tabpfn_v3", "TabPFN-v3"),
]

CONFIGS = [
    {
        "name":            f"{name}_sub_{subsample_n if subsample_n is not None else 'full'}",
        "model":           model_cls,
        "folder":          f"{name}/sub_{subsample_n if subsample_n is not None else 'full'}",
        "subsample_train": subsample_n,
        "subsample_test":  subsample_n if subsample_n is None else 2048,
        "save_mode":       "single_file",
    }
    for name, model_cls in _BASELINES
    for subsample_n in SUBSAMPLE_SIZES
]

CONFIGS += [
    {
        **_BASE,
        **overrides,
        "subsample_train":  subsample_n,
        "subsample_test":   subsample_n if subsample_n is None else 2048,
        "name":         f"{name}_{overrides.get('num_loops', num_loops)}_step_{step}_sub_{subsample_n if subsample_n is not None else 'full'}",
        "folder":       f"{name}_{overrides.get('num_loops', num_loops)}/step_{step}/sub_{subsample_n if subsample_n is not None else 'full'}",
        "model":        model_cls,
        "weights_path": f"{ckpt_dir}/step-{step}.ckpt",
    }
    for name, model_cls, ckpt_dir, steps, *rest in _MODELS
    for overrides in [rest[0] if rest else {}]
    for step in (steps if steps is not None else [CHECKPOINT_STEP])
    for subsample_n in SUBSAMPLE_SIZES
]

CONFIGS += [
    {
        **_BASE,
        **overrides,
        "subsample_train":  None,
        "subsample_test":   None,
        "name":         f"{name}_{overrides.get('num_loops', num_loops)}_step_10000_sub_full",
        "folder":       f"{name}_{overrides.get('num_loops', num_loops)}/step_10000/sub_full",
        "model":        model_cls,
        "weights_path": f"{CHECKPOINT_DIR_STAGE2}/{name}/checkpoint/dir/{CHECKPOINT_NAME}",
    }
    for name, model_cls, overrides in _MODELS_FULL_ONLY
]
