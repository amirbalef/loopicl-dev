CHECKPOINT_DIR_ABLATION =  "please_replace_with_actual_path"

CHECKPOINT_STEPS = list(range(0, 100_001, 5_000))

num_loops = 6

_BASE = {
    "save_mode":            "single_file",
    "num_loops":            num_loops,
    "interface":            "loopicl",
    "batch_size":           8,
    "n_estimators":         8,
    "multiclass_strategy":  "prob_avg",
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
    #("loopicl_base_v1",                "LoopICL",        _abl("loopicl_base_v1"),                None),
    ("residual_scaling_one_over_L",           "LoopICL",        _abl("residual_scaling_one_over_L"),           None),
    ("no_residual_scaling","LoopICL",        _abl("no_residual_scaling"),None),
    ("residual_scaling_one_over_sqrt_L","LoopICL",        _abl("residual_scaling_one_over_sqrt_L"),None),
    ("no_dist_cond","LoopICL",        _abl("no_dist_cond"),None, {"loop_residual_scaling_fn": "inv_n"}),
    # # ------------------------------------------------------------------ #
    ("stackedblocks",     "LoopICLUnrolled",        _abl("stackedblocks"),   None, {"num_loops": 1}),
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
