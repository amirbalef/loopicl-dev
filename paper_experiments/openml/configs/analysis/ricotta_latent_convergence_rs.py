"""Config for latent-space trajectory analysis with ablation models.

Captures test embeddings at every ICL loop for the three ablation models
from ricotta_ablation_num_loops_intermediate.py.

Run with:
    uv run benchmark_eval/openml/run_benchmark.py \
        --dataset dev-cls-lite \
        --config analysis.ricotta_latent_convergence_rs

Storage note: loop_embeddings are shape (n_loops+1, n_test_points, D) per split.
"""

CHECKPOINT_DIR_ABLATION = "../TICLA/results/workdir/ricotta_exp/"
CHECKPOINT_STEP = 10_000

num_loops = 18

RS_VALUES = [1, 3, 6, 9]

# (display_name, checkpoint_name, loop_residual_scaling_fn)
_RS_MODEL_SPECS = [
    ("ricotta_one_layer_ablation_stage2",     "ricotta_one_layer_ablation_stage2",     "inv_n"),
    ("ricotta_light_scaling_ablation_stage2", "ricotta_light_scaling_ablation_stage2", "inv_sqrt_n"),
]

_NO_SCALING_MODEL_SPECS = [
    ("ricotta_no_scaling_ablation_stage2", "ricotta_no_scaling_ablation_stage2"),
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
    "interface":                     "ricotta",
    "multiclass_strategy":           "tabpfn",
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
        "name":               "ricotta_unrolled_ablation_stage2_1loops_latent_convergence",
        "num_loops":          1,
        "layers_as_loops":    True,
        "weights_path":       f"{_abl('ricotta_unrolled_ablation_stage2')}/step-{CHECKPOINT_STEP}.ckpt",
    },
]
