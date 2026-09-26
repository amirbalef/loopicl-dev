"""num_loops ablation with lightweight intermediate predictions.

Same structure as ricotta_ablation_num_loops.py (one experiment per model per
loop count) but uses ``capture_loop_predictions`` instead of
``capture_hidden_states``.  This skips CKA, L2 deltas, and layer matrices,
so each experiment runs much faster while still storing per-loop logits,
loop_scores, and loop_loglosses in the result.

Run with:
    python benchmark_eval/openml/run_benchmark.py --dataset dev-cls  --config ablations.ricotta_ablation_num_loops_intermediate

"""

CHECKPOINT_DIR_ABLATION = "../TICLA/results/workdir/ricotta_exp/"

CHECKPOINT_STEP = 10_000
NUM_LOOPS_VALUES = list(range(1, 20+1))

_BASE = {
    "save_mode":                 "single_file",
    "interface":                 "ricotta",
    "batch_size":                1,
    "n_estimators":              1,
    "multiclass_strategy":       "tabpfn",
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
    ("ricotta_one_layer_ablation_stage2",    "ricotta_one_layer_ablation_stage2",    {}),
    ("ricotta_no_scaling_ablation_stage2",   "ricotta_no_scaling_ablation_stage2",   {}),
    ("ricotta_light_scaling_ablation_stage2","ricotta_light_scaling_ablation_stage2",{}),
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
