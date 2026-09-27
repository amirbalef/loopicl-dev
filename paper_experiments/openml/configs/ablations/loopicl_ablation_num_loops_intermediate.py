CHECKPOINT_DIR_ABLATION =  "please_replace_with_actual_path"

CHECKPOINT_STEP = 10_000
NUM_LOOPS_VALUES = list(range(1, 20+1))

_BASE = {
    "save_mode":                 "single_file",
    "interface":                 "loopicl",
    "batch_size":                1,
    "n_estimators":              1,
    "multiclass_strategy":       "prob_avg",
    "subsample_samples":         None,
    "subsample_train":           None,
    "subsample_test":            None,
    "test_chunk_size":           None,
    "capture_loop_predictions":  True,
    "capture_hidden_states":     False,
    "capture_layer_metrics":     False,
    "capture_layer_matrix":      False,
}


def _abl(exp_name):
    return f"{CHECKPOINT_DIR_ABLATION}/{exp_name}/checkpoint/dir"


# (display_name, checkpoint_name, extra_params)
_MODEL_SPECS = [
    ("residual_scaling_one_over_L_stage2",      "residual_scaling_one_over_L_stage2",      {}),
    ("no_residual_scaling_stage2",              "no_residual_scaling_stage2",              {}),
    ("residual_scaling_one_over_sqrt_L_stage2", "residual_scaling_one_over_sqrt_L_stage2", {}),
]

CONFIGS = [
    {
        **_BASE,
        **extra,
        "num_loops":    n_loops,
        "name":         f"{display_name}_loops{n_loops}_step_{CHECKPOINT_STEP}",
        "folder":       f"{display_name}_loops{n_loops}/step_{CHECKPOINT_STEP}",
        "model":        "LoopICL",
        "weights_path": f"{_abl(ckpt_name)}/step-{CHECKPOINT_STEP}.ckpt",
    }
    for display_name, ckpt_name, extra in _MODEL_SPECS
    for n_loops in NUM_LOOPS_VALUES
]
