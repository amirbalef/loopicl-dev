"""Early exit config — test-set-agnostic gate, task-agnostic trainer.

Standalone: no imports from other config files.

Key differences from previous early-exit configs
-------------------------------------------------
* Gate uses ONLY training-row embeddings — no test-set signals, no entropy
  from test predictions.  At inference the decoder is called exactly once
  (at the exit loop), not at every loop iteration.
* Task-agnostic: set prior_tasktype = "regression" and
  class_name = "LoopICLEarlyExit" to run on regression priors.
* Uses EarlyExitTrainerV2 which dispatches step losses by task type
  (cross-entropy for classification, pinball for regression).

Gate signals (all train-set only)
----------------------------------
  gate_row_aggregation  "mean_max_std"  [mean, max, std] of CLS embeddings
  gate_use_delta        True            ||x_t − x_{t-1}||₂ convergence
  gate_use_loop_index   True            sinusoidal loop position
  gate_history_len      3               past λ momentum window

Warm-start
----------
Loads a fine-tuned stage-2 classification checkpoint.  Gate-only training
(LM frozen from step 0) for 4 000 steps.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Checkpoint to warm-start from
# ---------------------------------------------------------------------------

_STAGE2_CKPT = (
    "./results/workdir/loopicl_exp/loopicl_stage2/checkpoint/dir/step-30000.ckpt"
)


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

@dataclass
class GeneralConfig:
    name:                   str           = "loopicl"
    variant:                str           = "model_early_exit"
    class_name:             str           = "LoopICLEarlyExit"
    exp_name:               str           = "loopicl_ee_v2_cls"
    weights_path:           Optional[str] = None
    pretrained_weights_path: str          = _STAGE2_CKPT
    # Trainer dispatch
    trainer_module: str = "loopicl.train.early_exit_trainer_v2"
    trainer_class:  str = "EarlyExitTrainerV2"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    max_classes: int = 10

    # Cell embedding
    embed_dim:           int   = 128
    feature_group_size:  int   = 3

    # Within-column attention
    dist_embed_num_heads:           int = 8
    dist_embed_num_inducing_points: int = 128

    # Cross-column attention
    feat_agg_num_heads:     int   = 8
    feat_agg_num_cls_tokens: int  = 4
    feat_agg_rope_base:     float = 100_000.0

    # ICL transformer
    icl_num_heads:          int           = 8
    icl_num_kv_heads:       Optional[int] = None
    icl_num_kv_heads_test:  Optional[int] = None

    # Decoder
    decoder_head_dim:            int  = 64
    decoder_num_heads:           int  = 6
    decoder_use_softmax_scaling: bool = True
    use_class_normalized_decoder: bool = False

    # Shared
    ff_factor:                    int           = 2
    dropout:                      float         = 0.0
    softmax_scaling_mlp_hidden_dim: int         = 64
    use_v2_softmax_scaling:       bool          = True
    softmax_scaling_n_ref:        int           = 512
    attn_logit_softcap:           Optional[float] = None
    layernorm_elementwise_affine: bool          = True
    use_nan_indicators:           bool          = False
    use_sandwich_norm:            bool          = True
    softmax_scaling_use_tanh:     bool          = True

    # Axial architecture
    axial_num_blocks:    int = 1
    num_within_per_round: int = 1
    num_cross_per_round:  int = 1
    num_icl_per_round:    int = 1

    # Outer loop — fixed 16 loops; gate decides when to exit
    num_icl_loops:        int           = 16
    icl_random_loops:     bool          = False
    icl_random_loops_mu:  Optional[float] = None
    icl_random_loops_sigma: float       = 0.5
    icl_random_loops_max: Optional[int] = None

    # TBPTT disabled: gradients flow through all loops
    tbptt_last_k: Optional[int] = None

    iter_embed:            bool = False
    inject_only_first_loop: bool = False

    # Memory / speed
    loop_gradient_checkpointing: bool = False

    # Loop residual scaling — off for the no-scaling variant
    use_loop_residual_scaling: bool  = False
    loop_residual_lambda:      float = 1.0

    use_oneway:               bool = False
    use_disc_hist_conditioning: bool = True
    disc_hist_chunk_size:     int  = 2048

    # ---------------------------------------------------------------------------
    # Gate hyperparameters (EarlyExitGate — test-set agnostic)
    # ---------------------------------------------------------------------------

    # Per-row MLP: D → hidden → compress_dim, mean-pooled → (B, compress_dim).
    # Gate input = [state(c), delta(c), loop_emb(16), history(3), train_size(8)] ≈ 43-dim.
    gate_delta_out_dim: int  = 8    # compress_dim

    # Hidden size of the decision MLP
    gate_hidden_dim: int = 64

    # Append (compressed_t − compressed_{t-1}) as extra convergence signal
    gate_use_delta:         bool = True

    # Sinusoidal embedding of log(num_train)
    gate_train_size_dim:    int  = 8   # must be even; 0 = disabled

    # Append sinusoidal embedding of the current loop index
    gate_use_loop_index:  bool = True
    gate_loop_index_dim:  int  = 16

    # Append last gate_history_len λ values as scalars (momentum signal)
    gate_history_len: int = 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    amp:           bool  = True
    model_compile: bool  = False
    device:        str   = "cuda"
    dtype:         str   = "bfloat16"
    np_seed:       int   = 42
    torch_seed:    int   = 42

    max_steps:       int   = 4_000
    batch_size:      int   = 64
    micro_batch_size: int  = 1
    optimizer:       str   = "tabicl_muon"

    lr:               float = 8e-4
    muon_lr:          float = 8e-4
    muon_momentum:    float = 0.9
    muon_ns_steps:    int   = 5
    muon_adamw_lr:    float = 3e-4
    muon_adamw_betas: Tuple[float, float] = (0.9, 0.95)
    muon_cautious_wd: bool  = True

    scheduler:          str   = "wsd"
    wsd_decay_fraction: float = 0.50
    wsd_lr_end:         float = 0.0
    wsd_decay_type:     str   = "cosine"
    warmup_proportion:  float = 0.05
    warmup_steps:       int   = 500

    sdpa_backend:      str   = "flash"
    gradient_clipping: float = 1.0
    weight_decay:      float = 0.01
    ema_decay:         float = 0.999

    cosine_num_cycles:       int   = 1
    cosine_amplitude_decay:  float = 1.0
    cosine_lr_end:           float = 1e-7
    poly_decay_lr_end:       float = 1e-7
    poly_decay_power:        float = 1.0

    freeze_layers: list = field(default_factory=list)

    # Early-exit specific
    # LM frozen from step 0; only gate params are trained
    gate_only_stage: bool = True

    # Entropy coefficient for joint LM + gate training
    early_exit_beta: float = 0.05

    # Gap penalty: pushes E[exit_loop] toward the oracle optimal loop
    exit_gap_loss_weight:      float = 0.02
    # Asymmetric multiplier: late exits penalised more than early exits
    exit_gap_late_multiplier:  float = 2.0


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

@dataclass
class PriorConfig:
    prior_source:   str = "tabicl_v2"
    prior_type:     str = "graph_scm"
    prior_tasktype: str = "classification"   # "classification" | "regression"
    regression:     bool = False
    prior_device:   str = "cpu"
    prior_dir:      Optional[str] = None     # None → generate on the fly

    # Generation
    batch_size:         int          = 64
    batch_size_per_gp:  int          = 4
    num_batches:        int          = 10_000
    min_features:       int          = 1
    max_features:       int          = 2000
    max_classes:        int          = 10
    min_seq_len:        Optional[int] = 400
    max_seq_len:        int          = 100_000
    feature_seq_budget: int          = 1_000_000
    log_seq_len:        bool         = False
    log_n_features:     bool         = False
    seq_len_per_gp:     bool         = True
    replay_small:       bool         = False

    # Edge fitting (regression only)
    n_edge_batches: int = 100

    # Workers
    n_jobs:                  int  = 60
    num_threads_per_worker:  int  = 1
    use_batched_gpu:         bool = True
    prefetch_batches:        int  = 2

    # Filtering
    filter_unpredictable_datasets: bool = True
    filter_unpredictable_graphs:   bool = True

    # Loading
    load_prior_start: int = 0

    # Graph SCM
    graph_noise:       bool  = False
    allow_act_warping: bool  = False
    min_n_nodes:       int   = 2
    max_n_nodes:       int   = 32
    cauchy_dag_offset: float = 0.0
    min_train_size:    float = 0.3
    max_train_size:    float = 0.9


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class CheckpointConfig:
    dir:             Optional[str] = None
    save_temp_every: int           = 100
    save_perm_every: int           = 1_000
    max_checkpoints: int           = 5
    path:            Optional[str] = None
    only_load_model: bool          = False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkValidatorConfig:
    benchmark:      str  = ""
    benchmark_dir:  str  = "./results/loopicl_benchmarks/"
    candidate_name: str  = "loopicl_benchmarks"
    baseline_models: List[str] = field(
        default_factory=lambda: [
            "tabpfn_v3", "tabpfn_v3-def", "tabpfn_v2_6", "tabpfn_v2_5",
            "tabpfn_v2", "tabicl_v2", "tabicl_v2-def", "tabicl_v1_1",
            "tabicl_v1", "tabfm", "tabfm-def", "random_forest",
        ]
    )
    improvability_exclude_models: List[str] = field(default_factory=list)
    task_type:             str           = "cls"
    inference_batch_size:  int           = 8
    n_ensemble:            int           = 1
    auto_setup:            bool          = True
    force_regenerate_priors:   bool      = False
    force_reevaluate_baselines: bool     = False
    log_baselines_to_wandb:    bool      = True
    max_train_size:        Optional[int] = None
    max_test_size:         Optional[int] = None
    val_every_multiplier:  int           = 1


@dataclass
class CandidateConfig:
    name:                 str           = ""
    inference_num_blocks: Optional[int] = None
    mrl_dim:              Optional[int] = None
    inference_batch_size: int           = 1
    n_ensemble:           int           = 1
    exit_q:               Optional[float] = None
    per_sample_exit:      bool          = False


@dataclass
class EarlyExitStatsConfig:
    """Controls EarlyExitStatsValidator added by EarlyExitTrainerV2."""
    exit_q:       float = 0.5
    max_datasets: int   = 20


@dataclass
class ValidationConfig:
    enabled:   bool         = True
    val_every: int          = 1_000
    save_dir:  Optional[str] = None

    benchmarks: List[BenchmarkValidatorConfig] = field(
        default_factory=lambda: [
            BenchmarkValidatorConfig(benchmark="tabarena-binary",    task_type="cls"),
            BenchmarkValidatorConfig(benchmark="tabarena-multiclass", task_type="cls"),
            BenchmarkValidatorConfig(benchmark="dev-cls",             task_type="cls"),
            BenchmarkValidatorConfig(benchmark="talent-v2-selected",  task_type="cls",
                                     val_every_multiplier=5),
            BenchmarkValidatorConfig(benchmark="beyond-arena-mini",   task_type="cls",
                                     val_every_multiplier=5),
            BenchmarkValidatorConfig(benchmark="tabicl-v2-priors",    task_type="cls",
                                     val_every_multiplier=5),
        ]
    )

    # Full-loop baseline + three early-exit operating points
    candidates: List[CandidateConfig] = field(
        default_factory=lambda: [
            CandidateConfig(name="{expname}_full",   exit_q=1.0),
            CandidateConfig(name="{expname}_ee_q30", exit_q=0.3),
            CandidateConfig(name="{expname}_ee_q50", exit_q=0.5),
            CandidateConfig(name="{expname}_ee_q70", exit_q=0.7),
        ]
    )

    early_exit_stats: EarlyExitStatsConfig = field(
        default_factory=EarlyExitStatsConfig
    )


# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------

@dataclass
class WandbConfig:
    log:     bool         = True
    project: str          = "LOOPICL_EXP"
    name:    Optional[str] = None
    id:      Optional[str] = None
    dir:     Optional[str] = "./results/wandb/loopicl_exp/"
    new_run: bool         = False
    mode:    str          = "online"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    general:    GeneralConfig    = field(default_factory=GeneralConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    wandb:      WandbConfig      = field(default_factory=WandbConfig)
    training:   TrainingConfig   = field(default_factory=TrainingConfig)
    prior:      PriorConfig      = field(default_factory=PriorConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self):
        n = self.general.exp_name
        if self.checkpoint.dir is None:
            self.checkpoint.dir = f"./results/workdir/loopicl_exp/{n}/checkpoint/dir"
        if self.validation.save_dir is None:
            self.validation.save_dir = f"./results/workdir/loopicl_exp/{n}/validation/"
        if self.wandb.name is None:
            self.wandb.name = n
        for b in self.validation.benchmarks:
            if b.candidate_name is None:
                b.candidate_name = n
        for c in self.validation.candidates:
            c.name = c.name.format(expname=n)
