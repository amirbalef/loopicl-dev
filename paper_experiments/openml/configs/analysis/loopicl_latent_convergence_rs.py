CHECKPOINT_DIR_ABLATION =  "please_replace_with_actual_path"
CHECKPOINT_STEP = 10_000

num_loops = 18

RS_VALUES = [1, 3, 6, 9]

# (display_name, checkpoint_name, loop_residual_scaling_fn)
_RS_MODEL_SPECS = [
    ("residual_scaling_one_over_L_stage2",      "residual_scaling_one_over_L_stage2",      "inv_n"),
    ("residual_scaling_one_over_sqrt_L_stage2", "residual_scaling_one_over_sqrt_L_stage2", "inv_sqrt_n"),
]

_NO_SCALING_MODEL_SPECS = [
    ("no_residual_scaling_stage2", "no_residual_scaling_stage2"),
]


def _abl(exp_name):
    return f"{CHECKPOINT_DIR_ABLATION}/{exp_name}/checkpoint/dir"


_BASE = {
    "exp_type":                      "latent_convergence_analysis",
    "capture_hidden_states":         True,
    "capture_test_embeddings":       True,
    "capture_layer_metrics":         False,
    "capture_layer_matrix":          False,
    "num_loops":                     num_loops,
    "model":                         "LoopICL",
    "interface":                     "loopicl",
    "multiclass_strategy":           "prob_avg",
    "batch_size":                    1,
    "n_estimators":                  1,
}

CONFIGS = [
    {
        **_BASE,
        "name":                        f"{display_name}_{num_loops}loops_rs{rs}_latent_convergence",
        "loop_residual_n_for_scaling": rs,
        "loop_residual_scaling_fn":    scaling_fn,
        "weights_path":                f"{_abl(ckpt_name)}/step-{CHECKPOINT_STEP}.ckpt",
    }
    for display_name, ckpt_name, scaling_fn in _RS_MODEL_SPECS
    for rs in RS_VALUES
] + [
    {
        **_BASE,
        "name":         f"{display_name}_{num_loops}loops_latent_convergence",
        "weights_path": f"{_abl(ckpt_name)}/step-{CHECKPOINT_STEP}.ckpt",
    }
    for display_name, ckpt_name in _NO_SCALING_MODEL_SPECS
] + [
    {
        **_BASE,
        "name":               "stackedblocks_stage2_1loops_latent_convergence",
        "num_loops":          1,
        "layers_as_loops":    True,
        "weights_path":       f"{_abl('stackedblocks_stage2')}/step-{CHECKPOINT_STEP}.ckpt",
    },
]
