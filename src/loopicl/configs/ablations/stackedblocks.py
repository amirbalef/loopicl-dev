"""loopicl/unrolled — unrolled version of one_layer.

Based on one_layer.py (LoopICL, one axial block looped 6 times).

Instead of running a single AxialEmbedderV3A in a loop (weight sharing), the
LoopICLUnrolled model uses axial_num_blocks=6 to allocate 6 independent block sets
and runs them exactly once (num_icl_loops=1).  There is no random-loop sampling
and no loop-specific scaling.

This measures the contribution of weight sharing in the looped architecture:
if the unrolled model matches the looped one, the shared weights are not important
beyond providing iteration depth; if it underperforms, weight sharing is a meaningful
inductive bias.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# -----------------------
# General Config
# -----------------------
@dataclass
class GeneralConfig:
    name: str = "loopicl"
    variant: str = "model"
    class_name: str = "LoopICLUnrolled"
    exp_name: str = "stackedblocks"
    weights_path: Optional[str] = None
    pretrained_weights_path: Optional[str] = None


# -----------------------
# Model Architecture Config
# -----------------------
@dataclass
class ModelConfig:
    max_classes: int = 10

    # Cell embedding
    embed_dim: int = 128
    feature_group_size: int = 3

    # Within-column attention
    dist_embed_num_heads: int = 8
    dist_embed_num_inducing_points: int = 128

    # Cross-column attention
    feat_agg_num_heads: int = 8
    feat_agg_num_cls_tokens: int = 4
    feat_agg_rope_base: float = 100_000.0

    # ICL transformer
    icl_num_heads: int = 8
    icl_num_kv_heads: Optional[int] = None
    icl_num_kv_heads_test: Optional[int] = None

    # Decoder
    decoder_head_dim: int = 64
    decoder_num_heads: int = 6
    decoder_use_softmax_scaling: bool = True
    use_class_normalized_decoder: bool = False

    # Shared
    ff_factor: int = 2
    dropout: float = 0.0
    softmax_scaling_mlp_hidden_dim: int = 64
    # SoftmaxScalingMLP v2 (softmax_scaling.py)
    use_v2_softmax_scaling: bool = True
    softmax_scaling_n_ref: int = 512
    attn_logit_softcap: Optional[float] = None

    layernorm_elementwise_affine: bool = True
    use_nan_indicators: bool = False
    # Sandwich normalisation on by default (the defining feature of LoopICL)
    use_sandwich_norm: bool = True

    # Axial architecture — 6 unrolled blocks, each with independent weights.
    # Equivalent to one_layer.py's axial_num_blocks=1 looped num_icl_loops=6 times.
    axial_num_blocks: int = 6
    num_within_per_round: int = 1
    num_cross_per_round: int = 1
    num_icl_per_round: int = 1

    # Outer loop — single pass (no looping / no weight sharing)
    num_icl_loops: int = 1
    icl_random_loops: bool = False
    icl_random_loops_mu: Optional[float] = None
    icl_random_loops_sigma: float = 0.5
    icl_random_loops_max: Optional[int] = None
    iter_embed: bool = False
    inject_only_first_loop: bool = False

    # Memory / speed
    loop_gradient_checkpointing: bool = False

    # Loop residual scaling — disabled (single pass)
    use_loop_residual_scaling: bool = False
    loop_residual_lambda: float = 1.0

    # One-way column→row attention
    use_oneway: bool = False

    # Long-context: v2 softmax scaling (bounded tanh input)
    softmax_scaling_use_tanh: bool = True

    # Long-context: discriminative histogram conditioner
    use_disc_hist_conditioning: bool = True
    disc_hist_chunk_size: int = 2048


# -----------------------
# Training Config
# -----------------------
@dataclass
class TrainingConfig:
    amp: bool = True
    model_compile: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"
    np_seed: int = 42
    torch_seed: int = 42

    max_steps: int = 100_000
    batch_size: int = 64
    micro_batch_size: int = 4
    optimizer: str = "tabicl_muon"
    lr: float = 8e-4
    muon_lr: float = 8e-4
    muon_momentum: float = 0.9
    muon_ns_steps: int = 5
    muon_adamw_lr: float = 3e-4
    muon_adamw_betas: Tuple[float, float] = (0.9, 0.95)
    muon_cautious_wd: bool = True
    scheduler: str = "wsd"
    wsd_decay_fraction: float = 0.0   # fraction of max_steps used for the decay phase
    wsd_lr_end: float = 0.0           # lr at end of decay
    wsd_decay_type: str = "cosine"    # "cosine" or "linear"
    warmup_proportion: float = 0.01
    warmup_steps: int = 5000
    sdpa_backend: str = "flash"
    gradient_clipping: float = 1.0
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    cosine_num_cycles: int = 1
    cosine_amplitude_decay: float = 1.0
    cosine_lr_end: float = 1e-7
    poly_decay_lr_end: float = 1e-7
    poly_decay_power: float = 1.0
    freeze_layers: list = field(default_factory=lambda: [])


# -----------------------
# Prior Dataset Config
# -----------------------
@dataclass
class PriorConfig:
    prior_dir: str = "./results/prior/tabicl_v2/data/"
    prior_type: str = "graph_scm"
    prior_device: str = "cpu"
    batch_size: int = 64
    batch_size_per_gp: int = 4
    num_batches: int = 200_000
    min_features: int = 2
    max_features: int = 100
    max_classes: int = 10
    min_seq_len: Optional[int] = None
    max_seq_len: int = 1024
    log_seq_len: bool = False
    seq_len_per_gp: bool = False
    min_train_size: float = 0.3
    max_train_size: float = 0.9
    replay_small: bool = False
    n_jobs: int = -1
    num_threads_per_generate: int = 1
    resume_from: int = 0
    delete_after_load: bool = False
    load_prior_start: int = 0
    save_dir: str = "./results/prior/tabicl_v2/data/"


# -----------------------
# Checkpoint Config
# -----------------------
@dataclass
class CheckpointConfig:
    dir: Optional[str] = None
    save_temp_every: int = 100
    save_perm_every: int = 1000
    max_checkpoints: int = 5
    path: Optional[str] = None
    only_load_model: bool = False


# -----------------------
# Validation Config
# -----------------------
@dataclass
class BenchmarkValidatorConfig:
    benchmark: str = ""
    benchmark_dir: str = "./results/loopicl_cls_benchmarks/"
    candidate_name: str = "loopicl_benchmarks"
    baseline_models: List[str] = field(
        default_factory=lambda: ["tabpfn_v3", "tabpfn_v3-def", "tabpfn_v2_6", "tabpfn_v2_5", "tabpfn_v2", "tabicl_v2", "tabicl_v2-def", "tabicl_v1_1", "tabicl_v1", "tabfm", "tabfm-def", "random_forest"]
    )
    improvability_exclude_models: List[str] = field(default_factory=list)
    task_type: str = "cls"
    inference_batch_size: int = 8
    n_ensemble: int = 1
    auto_setup: bool = True
    force_regenerate_priors: bool = False
    force_reevaluate_baselines: bool = False
    log_baselines_to_wandb: bool = True
    max_train_size: Optional[int] = 2048  # subsample train set to this size (None = no limit)
    max_test_size: Optional[int] = 2048   # subsample test set to this size (None = no limit)
    val_every_multiplier: int = 1  # run every val_every * multiplier steps (e.g. 3 = 3x less frequent)


@dataclass
class CandidateConfig:
    name: str = ""
    inference_num_blocks: Optional[int] = None
    mrl_dim: Optional[int] = None
    inference_batch_size: int = 1
    n_ensemble: int = 1


@dataclass
class ValidationConfig:
    enabled: bool = True
    val_every: int = 20_000
    save_dir: Optional[str] = None
    benchmarks: List[BenchmarkValidatorConfig] = field(
        default_factory=lambda: [
            BenchmarkValidatorConfig(benchmark="tabarena-binary", task_type="cls"),
            BenchmarkValidatorConfig(benchmark="tabarena-multiclass", task_type="cls"),
            BenchmarkValidatorConfig(benchmark="dev-cls", task_type="cls"),
            BenchmarkValidatorConfig(benchmark="talent-v2-selected", task_type="cls", val_every_multiplier=10),
            BenchmarkValidatorConfig(benchmark="beyond-arena-mini", task_type="cls", val_every_multiplier=10),
            BenchmarkValidatorConfig(benchmark="tabicl-v2-priors", task_type="cls", val_every_multiplier=10),
        ]
    )
    candidates: List[CandidateConfig] = field(
        default_factory=lambda: [
            CandidateConfig(name="{expname}"),
        ]
    )


# -----------------------
# Wandb Config
# -----------------------
@dataclass
class WandbConfig:
    log: bool = True
    project: str = "RICOTTA_CLS"
    name: Optional[str] = None
    id: Optional[str] = None
    dir: Optional[str] = "./results/wandb/loopicl_cls/"
    new_run: bool = False
    mode: str = "online"


# -----------------------
# Master Config
# -----------------------
@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self):
        n = self.general.exp_name
        if self.checkpoint.dir is None:
            self.checkpoint.dir = f"./results/workdir/loopicl_cls/{n}/checkpoint/dir"
        if self.validation.save_dir is None:
            self.validation.save_dir = f"./results/workdir/loopicl_cls/{n}/validation/"
        if self.wandb.name is None:
            self.wandb.name = n
        for b in self.validation.benchmarks:
            if b.candidate_name is None:
                b.candidate_name = n
        for c in self.validation.candidates:
            c.name = c.name.format(expname=n)
