CHECKPOINT_DIR_ABLATION = "please_replace_with_actual_path"
CHECKPOINT_DIR_STAGE2   =  "please_replace_with_actual_path"
CHECKPOINT_NAME         =  "step-10000.ckpt"

CHECKPOINT_STEP = 100_000

SUBSAMPLE_SIZES = [256, 512, 1024, 2048, 4096, 8192, None]

num_loops = 6

_BASE = {
    "save_mode":            "single_file",
    "num_loops":            num_loops,
    "interface":            "loopicl",
    "batch_size":           1,
    "n_estimators":         1,
    "multiclass_strategy":  "prob_avg",
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
    #("loopicl_base_v1",                "LoopICL",        _abl("loopicl_base_v1"),                None),
    ("residual_scaling_one_over_L",           "LoopICL",        _abl("residual_scaling_one_over_L"),           None),
    ("no_residual_scaling","LoopICL",        _abl("no_residual_scaling"),None),
    ("residual_scaling_one_over_sqrt_L","LoopICL",        _abl("residual_scaling_one_over_sqrt_L"),None),
    #("no_dist_cond","LoopICL",        _abl("no_dist_cond"),None, {"loop_residual_scaling_fn": "inv_n"}),
    # # ------------------------------------------------------------------ #
    ("stackedblocks",     "LoopICLUnrolled",        _abl("stackedblocks"),   None, {"num_loops": 1}),
    # # # ------------------------------------------------------------------ #
]

# (name, model_class, overrides) — evaluated at full (no subsampling) only, step-10000
_MODELS_FULL_ONLY = [
    ("residual_scaling_one_over_L_stage2",    "LoopICL",        {}),
    ("no_residual_scaling_stage2",            "LoopICL",        {}),
    ("residual_scaling_one_over_sqrt_L_stage2","LoopICL",      {}),
    ("stackedblocks_stage2",                  "LoopICLUnrolled",{"num_loops": 1}),
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
