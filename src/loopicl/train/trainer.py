from __future__ import annotations

import copy
import ctypes
import dataclasses
import ctypes.util
import gc
import os
import json
import math
import resource
import timeit

import warnings
import functools
from datetime import timedelta
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb
import trackio

from loopicl.priors.tabicl_v1.dataset import PriorDataset as TabICLV1PriorDataset
from loopicl.priors.tabicl_v1.genload import LoadPriorDataset as TabICLV1LoadPriorDataset
from loopicl.priors.tabicl_v2._dataset import PriorDataset as TabICLV2PriorDataset
from loopicl.priors.tabicl_v2._genload import LoadPriorDataset as TabICLV2LoadPriorDataset
from loopicl.priors.loopicl.dataset import PriorDataset as LoopICLPriorDataset
from loopicl.priors.loopicl.genload import LoadPriorDataset as LoopICLLoadPriorDataset
from loopicl.priors.loopicl.genload import LoadPriorDatasetDual as LoopICLLoadPriorDatasetDual
from loopicl.priors.loopicl.genload import LoadPriorDatasetSoftV2 as LoopICLLoadPriorDatasetSoftV2
from loopicl.priors.oprior import OPriorDataset
from loopicl.train.optimizers import get_optimizer, get_scheduler
from loopicl.train.checkpoint import get_latest_checkpoint, save_checkpoint, manage_checkpoints
from loopicl.models.utils import load_model_config, load_model_class
from loopicl.validation import BaseValidator, BenchmarkValidator, build_baseline_log_dict, create_baseline_wandb_runs
from loopicl.evaluation.builtin import get_benchmark
from loopicl.evaluation.setup import ensure_benchmark_ready

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)

# On Linux, malloc_trim(0) returns unused free-list pages back to the OS.
# This counters glibc's tendency to grow RSS monotonically even when Python
# and PyTorch have already freed the underlying memory.
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    _HAS_MALLOC_TRIM = hasattr(_libc, "malloc_trim")
except Exception:
    _HAS_MALLOC_TRIM = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False


def ddp_cleanup(func):
    """Decorator that calls ``destroy_process_group()`` after the wrapped method.

    Checks ``dist.is_initialized()`` before calling so that ``train()`` can call
    ``destroy_process_group()`` early (before the deferred checkpoint save) and
    this finally-block becomes a safe no-op.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp and dist.is_initialized():
                destroy_process_group()

    return wrapper


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Complete training lifecycle for TabICL.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to train.  Defaults to ``"looped_nanotabpfn"``.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "looped_nanotabpfn"
        self.config = load_model_config(self.model_name)

        self.configure_ddp()
        self.prepare_benchmarks()
        if self.ddp:
            dist.barrier()  # Wait for rank 0 to finish benchmark setup before all ranks proceed
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        self.configure_regression_edges()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        self.configure_validation()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure_ddp(self):
        """Set up distributed training and system configuration."""
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            # Disable the NCCL heartbeat monitor so it does not fire a false-
            # positive abort when the process is idle after training ends (the
            # monitor fires if no collective completes for TORCH_NCCL_HEARTBEAT_
            # TIMEOUT_SEC seconds, even when idle is expected).  Genuine hangs
            # are still caught by the per-collective timeout passed below.
            # setdefault lets the caller override via the environment if needed.
            os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            init_process_group(backend="nccl", device_id=torch.device(f"cuda:{self.ddp_local_rank}"), timeout=timedelta(minutes=60))
            self.master_process = self.ddp_rank == 0
            self.config.training.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.training.device)

            original_batch_size = self.config.training.batch_size
            self.config.training.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.training.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size "
                        f"({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.training.batch_size}.\n"
                        f"Effective batch size is {self.config.training.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0

        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.training.np_seed + seed_offset)
        torch.manual_seed(self.config.training.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up experiment tracking (wandb or trackio)."""
        # Load baseline data (needed regardless of wandb.log for panel setup).
        self._baseline_log_dict: dict[str, float] = {}
        if self._benchmarks_for_baseline_logging and self.master_process:
            benchmarks = [b for b, _ in self._benchmarks_for_baseline_logging]
            benchmark_dir = self._benchmarks_for_baseline_logging[0][1]
            self._baseline_log_dict = build_baseline_log_dict(benchmarks, benchmark_dir)

        backend = getattr(self.config.wandb, "backend", "wandb")

        if self.config.wandb.log and self.master_process:
            os.makedirs(self.config.checkpoint.dir, exist_ok=True)

            if backend == "trackio":
                self.wandb_run = trackio.init(
                    project=self.config.wandb.project,
                    name=self.config.wandb.name,
                    config=dataclasses.asdict(self.config),
                    resume="allow",
                )
            else:
                # Create one wandb run per baseline (skip if already created).
                create_baseline_wandb_runs(
                    baseline_log_dict=self._baseline_log_dict,
                    project=self.config.wandb.project,
                    max_steps=self.config.training.max_steps,
                    entity=getattr(self.config.wandb, "entity", None),
                    wandb_dir=self.config.wandb.dir,
                    wandb_mode=self.config.wandb.mode,
                )

                # Init the main training run.
                id_path = os.path.join(self.config.checkpoint.dir, "wand_id.txt")
                if not self.config.wandb.new_run and self.config.wandb.id is None:
                    if os.path.exists(id_path):
                        with open(id_path, "r") as f:
                            self.config.wandb.id = f.read().strip()

                self.wandb_run = wandb.init(
                    dir=self.config.wandb.dir,
                    project=self.config.wandb.project,
                    name=self.config.wandb.name,
                    id=self.config.wandb.id,
                    config=self.config,
                    resume="allow",
                    mode=self.config.wandb.mode,
                    settings=wandb.Settings(init_timeout=300),
                )
                with open(id_path, "w") as f:
                    f.write(self.wandb_run.id)

        else:
            self.wandb_run = None

    def build_model(self):
        """Instantiate, optionally compile, and wrap the model."""
        self.model_config = self.config.model.__dict__
        Model = load_model_class(self.config)
        model = Model(**self.model_config)
        model.to(device=self.config.training.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        self._load_pretrained_weights(model)

        for layer_name in self.config.training.freeze_layers:
            layer = getattr(model, layer_name)
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
            if self.master_process:
                print(f"Frozen layer: {layer_name}")

        if self.config.training.model_compile:
            if self.config.training.device == "cpu":
                model = torch.compile(model, backend="eager")
            else:
                model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
            # Pre-allocate persistent synchronisation tensors.
            # dist.barrier() calls `at::empty({1}, cuda_device)` internally on
            # every call → new GPU address each time → NCCL registers it for DMA
            # and caches the registration in CPU RAM.  Over thousands of steps
            # (3 barriers + 1 oom_flag allreduce per step) these registrations
            # accumulate linearly and are the DDP-specific RSS growth source.
            # Reusing the same tensor objects means NCCL always sees the same GPU
            # address and reuses its existing registration instead of adding new ones.
            _dev = self.config.training.device
            self._barrier_buf  = torch.zeros(1, dtype=torch.float32, device=_dev)
            self._oom_buf      = torch.zeros(1, dtype=torch.float32, device=_dev)
            self._has_nan_buf  = torch.zeros(1, dtype=torch.float32, device=_dev)
            self._allreduce_buf_cache: dict = {}
        else:
            self.model = model
            self.raw_model = model

        ema_decay = getattr(self.config.training, "ema_decay", 0.0)
        if ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self.raw_model)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            if self.master_process:
                print(f"EMA enabled with decay={ema_decay}")
        else:
            self.ema_model = None

    def _load_pretrained_weights(self, model):
        """Initialize model from a pretrained TabICL checkpoint (optional)."""
        pretrained_path = getattr(self.config.general, "pretrained_weights_path", None)
        if not pretrained_path:
            return

        print(f"Initializing from pretrained weights: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        src = ckpt["state_dict"]
        dst = model.state_dict()
        remapped = {}

        for key, val in src.items():
            if key.startswith("icl_predictor.tf_icl.blocks.0."):
                new_key = key.replace("icl_predictor.tf_icl.blocks.0.", "icl_predictor.tf_icl.block.")
                remapped[new_key] = val
            elif key.startswith("icl_predictor.tf_icl.blocks."):
                continue
            else:
                remapped[key] = val

        exclude_prefixes = tuple(getattr(self.config.general, "pretrained_weights_exclude_prefixes", []))
        if exclude_prefixes:
            excluded = [k for k in list(remapped) if k.startswith(exclude_prefixes)]
            for k in excluded:
                del remapped[k]
            print(f"  Excluded from pretrained load (random init): {excluded}")

        missing = [k for k in dst if k not in remapped]
        unexpected = [k for k in remapped if k not in dst]
        shape_mismatch = [k for k in remapped if k in dst and remapped[k].shape != dst[k].shape]
        if missing:
            print(f"  Keys missing in pretrained checkpoint (will use random init): {missing}")
        if unexpected:
            print(f"  Unexpected keys in pretrained checkpoint (ignored): {unexpected}")
        if shape_mismatch:
            print(f"  Keys with shape mismatch (skipped, will use random init): {shape_mismatch}")

        dst.update({k: v for k, v in remapped.items() if k in dst and v.shape == dst[k].shape})
        model.load_state_dict(dst)
        print("Pretrained weights loaded successfully.")

        axial_keys = [k for k in dst if k.startswith("axial_embedder.")]
        loaded_axial = [k for k in axial_keys if k in remapped and remapped[k].shape == dst[k].shape]
        print(f"  axial_embedder: {len(loaded_axial)}/{len(axial_keys)} keys loaded from checkpoint")

    def configure_prior(self):
        """Set up the prior dataset and dataloader."""
        prior_source = getattr(self.config.prior, "prior_source", "tabicl_v1")
        if self.config.prior.prior_dir is None:
            if prior_source == "oprior":
                dataset = OPriorDataset(
                    prior_type=self.config.prior.prior_type,
                    batch_size=self.config.training.batch_size,
                    min_features=self.config.prior.min_features,
                    max_features=self.config.prior.max_features,
                    max_classes=self.config.prior.max_classes,
                    min_seq_len=self.config.prior.min_seq_len,
                    max_seq_len=self.config.prior.max_seq_len,
                    log_seq_len=self.config.prior.log_seq_len,
                    min_train_size=self.config.prior.min_train_size,
                    max_train_size=self.config.prior.max_train_size,
                    device=self.config.prior.prior_device,
                    prefetch_batches=getattr(self.config.prior, "prefetch_batches", 2),
                )
            elif prior_source == "loopicl":
                dataset = LoopICLPriorDataset(
                    batch_size=self.config.training.batch_size,
                    batch_size_per_gp=self.config.prior.batch_size_per_gp,
                    min_features=self.config.prior.min_features,
                    max_features=self.config.prior.max_features,
                    max_classes=self.config.prior.max_classes,
                    min_seq_len=self.config.prior.min_seq_len,
                    max_seq_len=self.config.prior.max_seq_len,
                    log_seq_len=self.config.prior.log_seq_len,
                    seq_len_per_gp=self.config.prior.seq_len_per_gp,
                    min_train_size=self.config.prior.min_train_size,
                    max_train_size=self.config.prior.max_train_size,
                    replay_small=self.config.prior.replay_small,
                    prior_type=self.config.prior.prior_type,
                    n_jobs=self.config.prior.n_jobs,
                    num_threads_per_worker=getattr(self.config.prior, "num_threads_per_worker", 1),
                    use_batched_gpu=getattr(self.config.prior, "use_batched_gpu", False),
                    feature_seq_budget=getattr(self.config.prior, "feature_seq_budget", None),
                    prefetch_batches=getattr(self.config.prior, "prefetch_batches", 2),
                    device=self.config.prior.prior_device,
                    config=getattr(self.config.prior, "graph_config", None),
                )
            elif prior_source == "tabicl_v2":
                dataset = TabICLV2PriorDataset(
                    batch_size=self.config.training.batch_size,
                    batch_size_per_gp=self.config.prior.batch_size_per_gp,
                    min_features=self.config.prior.min_features,
                    max_features=self.config.prior.max_features,
                    max_classes=self.config.prior.max_classes,
                    min_seq_len=self.config.prior.min_seq_len,
                    max_seq_len=self.config.prior.max_seq_len,
                    log_seq_len=self.config.prior.log_seq_len,
                    log_n_features=getattr(self.config.prior, "log_n_features", False),
                    seq_len_per_gp=self.config.prior.seq_len_per_gp,
                    min_train_size=self.config.prior.min_train_size,
                    max_train_size=self.config.prior.max_train_size,
                    replay_small=self.config.prior.replay_small,
                    prior_type=self.config.prior.prior_type,
                    n_jobs=self.config.prior.n_jobs,
                    regression=getattr(self.config.prior, "regression", False),
                    device=self.config.prior.prior_device,
                    feature_seq_budget=getattr(self.config.prior, "feature_seq_budget", None),
                    prefetch_batches=getattr(self.config.prior, "prefetch_batches", 2),
                )
            else:
                dataset = TabICLV1PriorDataset(
                    batch_size=self.config.training.batch_size,
                    batch_size_per_gp=self.config.prior.batch_size_per_gp,
                    min_features=self.config.prior.min_features,
                    max_features=self.config.prior.max_features,
                    max_classes=self.config.prior.max_classes,
                    min_seq_len=self.config.prior.min_seq_len,
                    max_seq_len=self.config.prior.max_seq_len,
                    log_seq_len=self.config.prior.log_seq_len,
                    seq_len_per_gp=self.config.prior.seq_len_per_gp,
                    min_train_size=self.config.prior.min_train_size,
                    max_train_size=self.config.prior.max_train_size,
                    replay_small=self.config.prior.replay_small,
                    prior_type=self.config.prior.prior_type,
                    prior_tasktype=getattr(self.config.prior, "prior_tasktype", "classification"),
                    device=self.config.prior.prior_device,
                    n_jobs=self.config.prior.n_jobs,
                )
        else:
            if prior_source == "loopicl" and getattr(self.config.prior, "prior_type", "") == "graph_scm_soft_v2":
                dataset = LoopICLLoadPriorDatasetSoftV2(
                    data_dir=self.config.prior.prior_dir,
                    batch_size=self.config.training.batch_size,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    start_from=self.config.prior.load_prior_start,
                    max_batches=getattr(self.config.prior, "num_batches", None),
                    delete_after_load=getattr(self.config.prior, "delete_after_load", False),
                    device=self.config.prior.prior_device,
                    timeout=getattr(self.config.prior, "timeout", 30),
                )
            elif prior_source == "loopicl" and getattr(self.config.prior, "prior_type", "") == "graph_scm_dual":
                dataset = LoopICLLoadPriorDatasetDual(
                    data_dir=self.config.prior.prior_dir,
                    batch_size=self.config.training.batch_size,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    start_from=self.config.prior.load_prior_start,
                    max_batches=getattr(self.config.prior, "num_batches", None),
                    delete_after_load=getattr(self.config.prior, "delete_after_load", False),
                    device=self.config.prior.prior_device,
                    timeout=getattr(self.config.prior, "timeout", 30),
                )
            elif prior_source == "loopicl":
                dataset = LoopICLLoadPriorDataset(
                    data_dir=self.config.prior.prior_dir,
                    batch_size=self.config.training.batch_size,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    start_from=self.config.prior.load_prior_start,
                    max_batches=getattr(self.config.prior, "num_batches", None),
                    delete_after_load=getattr(self.config.prior, "delete_after_load", False),
                    device=self.config.prior.prior_device,
                    timeout=getattr(self.config.prior, "timeout", 30),
                )
            elif prior_source == "loopicl_graph_scm":
                dataset = LoopICLLoadPriorDataset(
                    data_dir=self.config.prior.prior_dir,
                    batch_size=self.config.training.batch_size,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    start_from=self.config.prior.load_prior_start,
                    max_batches=getattr(self.config.prior, "num_batches", None),
                    delete_after_load=getattr(self.config.prior, "delete_after_load", False),
                    device=self.config.prior.prior_device,
                    timeout=getattr(self.config.prior, "timeout", 30),
                )
            elif prior_source == "tabicl_v2":
                dataset = TabICLV2LoadPriorDataset(
                    data_dir=self.config.prior.prior_dir,
                    batch_size=self.config.training.batch_size,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    start_from=self.config.prior.load_prior_start,
                    max_batches=getattr(self.config.prior, "num_batches", None),
                    delete_after_load=getattr(self.config.prior, "delete_after_load", False),
                    device=self.config.prior.prior_device,
                    timeout=getattr(self.config.prior, "timeout", 30),
                )
            else:
                dataset = TabICLV1LoadPriorDataset(
                    data_dir=self.config.prior.prior_dir,
                    batch_size=self.config.training.batch_size,
                    ddp_world_size=self.ddp_world_size,
                    ddp_rank=self.ddp_rank,
                    start_from=self.config.prior.load_prior_start,
                    max_batches=getattr(self.config.prior, "num_batches", None),
                    delete_after_load=getattr(self.config.prior, "delete_after_load", False),
                    device=self.config.prior.prior_device,
                    timeout=getattr(self.config.prior, "timeout", 30),
                )

        if self.master_process:
            print(dataset)

        use_batched_gpu = getattr(self.config.prior, "use_batched_gpu", False)
        num_workers = 0 if use_batched_gpu else getattr(self.config.prior, "num_workers", 0)
        dl_kwargs = dict(batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=False)
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 4
        self.dataloader = DataLoader(dataset, **dl_kwargs)

    def configure_regression_edges(self):
        """Fit fixed global bar-distribution edges from the prior (regression only).

        Samples ``n_edge_batches`` batches from the prior, pools all y values,
        and calls :meth:`LoopICLReg.set_fixed_edges` so that every training
        batch uses the same bin borders.  This mirrors TabPFN v3's approach of
        pre-computing global bin edges from a large representative sample.

        Only runs when ``config.prior.prior_tasktype == "regression"`` and the
        model exposes ``set_fixed_edges`` (i.e. is a ``LoopICLReg``).

        Edges are cached to ``{checkpoint.dir}/global_edges.pt`` so that
        restarts skip the expensive sampling step.  In DDP mode only the master
        process samples and saves; all other ranks receive the edges via an
        all_reduce broadcast.
        """
        is_regression = getattr(self.config.prior, "prior_tasktype", "classification") == "regression"
        use_pinball = getattr(self.config.training, "use_pinball_loss", False)
        if not is_regression or not hasattr(self.raw_model, "set_fixed_edges") or use_pinball:
            return

        import os
        from loopicl.models.loopicl_ablation.bar_distribution import BarDistribution

        edges_path = os.path.join(self.config.checkpoint.dir, "global_edges.pt")
        n_edge_batches = getattr(self.config.prior, "n_edge_batches", 100)
        num_bars = self.raw_model.num_bars

        if self.master_process:
            if os.path.exists(edges_path):
                edges = torch.load(edges_path, map_location="cpu")
                print(f"[configure_regression_edges] Loaded cached edges from {edges_path}  "
                      f"edge range: [{edges.min():.4f}, {edges.max():.4f}]")
            else:
                print(f"[configure_regression_edges] Sampling {n_edge_batches} batches to fit global bin edges...")
                bar = BarDistribution(num_bars=num_bars)
                y_list = []
                dataset = self.dataloader.dataset
                dataset_iter = iter(dataset)
                for _ in range(n_edge_batches):
                    batch = next(dataset_iter)
                    # batch[1] is y_reg: (B, T) continuous float, may be a NestedTensor
                    y_raw = batch[1].float().cpu()
                    if y_raw.is_nested:
                        y_list.append(torch.cat([t.flatten() for t in y_raw.unbind()]))
                    else:
                        y_list.append(y_raw.flatten())
                all_y = torch.cat(y_list)
                edges = bar.fit_global_edges(all_y)
                os.makedirs(self.config.checkpoint.dir, exist_ok=True)
                torch.save(edges, edges_path)
                print(f"[configure_regression_edges] Done. Saved to {edges_path}  "
                      f"y range: [{all_y.min():.4f}, {all_y.max():.4f}]  "
                      f"edge range: [{edges.min():.4f}, {edges.max():.4f}]")

        # In DDP mode broadcast edges from master (rank 0) to all other ranks.
        if self.ddp:
            if self.master_process:
                edges_gpu = edges.to(self.config.training.device)
            else:
                edges_gpu = torch.zeros(1, num_bars + 1, device=self.config.training.device)
            dist.broadcast(edges_gpu, src=0)
            edges = edges_gpu.cpu()

        self.raw_model.set_fixed_edges(edges)

    def configure_optimizer(self):
        """Configure optimizer and scheduler from config."""
        self.optimizer = get_optimizer(self.config.training, self.raw_model.parameters())
        self.scheduler = get_scheduler(config=self.config.training, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP)."""
        on_cuda = "cuda" in self.config.training.device
        dtype_str = self.config.training.dtype
        _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        amp_dtype = _dtype_map.get(dtype_str, torch.bfloat16)
        # GradScaler only helps fp16; bf16 has sufficient dynamic range and doesn't need loss scaling
        use_scaler = self.config.training.amp and on_cuda and dtype_str == "float16"
        self.scaler = torch.GradScaler("cuda", enabled=use_scaler)
        if self.config.training.amp and on_cuda:
            if self.master_process:
                print("Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            self.amp_ctx = nullcontext()

        if on_cuda:
            sdpa_backend = getattr(self.config.training, "sdpa_backend", "flash")
            _has_cudnn = hasattr(torch.backends.cuda, "enable_cudnn_sdp")

            if sdpa_backend == "math":
                # Diagnostic mode: pure math attention — zero per-shape compilation
                # or host-side state.  Use for falsification runs (1–2k steps) to
                # confirm whether RssAnon growth is caused by SDPA backend caches.
                # If growth stops with math, the diagnosis is confirmed.
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                if _has_cudnn:
                    torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                _label = "math (diagnostic)"
            elif sdpa_backend == "cudnn":
                # Legacy: cuDNN flash attention.  Creates per-shape plan handles
                # that accumulate with varying seq_len (icl_random_loops=True).
                if _has_cudnn:
                    torch.backends.cuda.enable_cudnn_sdp(True)
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                if not _has_cudnn:
                    torch.backends.cuda.enable_math_sdp(True)
                _label = "cudnn" if _has_cudnn else "math (cudnn unavailable)"
            else:
                # Default "flash": Triton kernels compiled once per (head_dim, dtype),
                # not per seq_len — no unbounded accumulation with icl_random_loops.
                # mem-efficient as fallback; cuDNN disabled.
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                if _has_cudnn:
                    torch.backends.cuda.enable_cudnn_sdp(False)
                _label = "flash (mem-efficient fallback, cuDNN disabled)"

            if self.master_process:
                print(f"SDPA backend: {_label}")
                # --- GPU / Flash Attention diagnostics ---
                cap = torch.cuda.get_device_capability()
                device_name = torch.cuda.get_device_name()
                print(f"GPU: {device_name}  compute capability: {cap[0]}.{cap[1]}")
                if cap[0] >= 10:
                    print("  → Blackwell GPU detected (SM_100+)")
                elif cap[0] >= 9:
                    print("  → Hopper GPU detected (SM_90+)")
                try:
                    import flash_attn as _fa
                    _fa_ver = getattr(_fa, '__version__', 'version unknown')
                    _has_cute = hasattr(_fa, 'cute') or __import__('importlib').util.find_spec('flash_attn.cute') is not None
                    print(f"  flash_attn: {_fa_ver}  (FA4/cute: {'yes' if _has_cute else 'no'})")
                except ImportError:
                    print("  flash_attn: NOT installed")
                try:
                    import flash_attn_interface as _fai
                    _fai_ver = getattr(_fai, '__version__', 'version unknown')
                    print(f"  flash_attn_interface (FA3): {_fai_ver}")
                except ImportError:
                    print("  flash_attn_interface (FA3): NOT installed")
                try:
                    from loopicl.models.loopicl.layers import _HAS_FLASH_ATTN_FAST, _FLASH_ATTN_FAST_SOURCE
                    if _HAS_FLASH_ATTN_FAST:
                        print(f"  LoopICL FA fast path: ACTIVE via {_FLASH_ATTN_FAST_SOURCE}")
                    else:
                        # Re-run probes with verbose error to diagnose the failure.
                        import torch as _pt
                        _diag = []
                        for _pkg, _fn_name, _shape in [
                            ("flash_attn.cute", "flash_attn_func", "BSHD"),
                            ("flash_attn_interface", "flash_attn_varlen_func", "varlen"),
                            ("flash_attn", "flash_attn_varlen_func", "varlen"),
                        ]:
                            try:
                                _mod = __import__(_pkg, fromlist=[_fn_name])
                                _fn = getattr(_mod, _fn_name)
                                if _shape == "BSHD":
                                    _q = _pt.zeros(1, 4, 1, 64, dtype=_pt.bfloat16, device="cuda")
                                    _fn(_q, _q, _q)
                                else:
                                    _q = _pt.zeros(4, 1, 64, dtype=_pt.bfloat16, device="cuda")
                                    _cu = _pt.tensor([0, 4], dtype=_pt.int32, device="cuda")
                                    _fn(_q, _q, _q, _cu, _cu, 4, 4)
                                _pt.cuda.synchronize()
                                _diag.append(f"{_pkg}: OK")
                            except ImportError:
                                _diag.append(f"{_pkg}: not installed")
                            except Exception as _pe:
                                _diag.append(f"{_pkg}: {type(_pe).__name__}: {_pe}")
                        print(f"  LoopICL FA fast path: DISABLED — probe results:")
                        for _d in _diag:
                            print(f"    {_d}")
                except Exception as _e:
                    print(f"  LoopICL FA fast path: could not determine ({type(_e).__name__}: {_e})")

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model weights after an optimizer step."""
        decay = self.config.training.ema_decay
        for ema_p, p in zip(self.ema_model.parameters(), self.raw_model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for ema_b, b in zip(self.ema_model.buffers(), self.raw_model.buffers()):
            ema_b.copy_(b)

    def prepare_benchmarks(self):
        """Ensure benchmark data is ready; collect info for baseline wandb logging."""
        self._benchmarks_for_baseline_logging: list[tuple] = []  # (Benchmark, benchmark_dir)

        # Only rank 0 should set up benchmarks; other ranks wait at the barrier
        # in __init__ until rank 0 is done.
        if not self.master_process:
            return

        val_cfg = getattr(self.config, "validation", None)
        if val_cfg is None or not val_cfg.enabled:
            return

        for bcfg in val_cfg.benchmarks:
            bmark = get_benchmark(bcfg.benchmark)
            ensure_benchmark_ready(
                benchmark=bmark,
                benchmark_dir=bcfg.benchmark_dir,
                baseline_models=bcfg.baseline_models,
                auto_setup=bcfg.auto_setup,
                force_priors=bcfg.force_regenerate_priors,
                force_baselines=bcfg.force_reevaluate_baselines,
                n_ensemble=getattr(bcfg, "n_ensemble", None),
                task_type=getattr(bcfg, "task_type", "all"),
                max_train_size=getattr(bcfg, "max_train_size", None),
                max_test_size=getattr(bcfg, "max_test_size", None),
            )
            if getattr(bcfg, "log_baselines_to_wandb", False):
                self._benchmarks_for_baseline_logging.append((bmark, bcfg.benchmark_dir))

    def configure_validation(self):
        """Build the list of validators from ``config.validation`` (if present)."""
        self.validators: list[BaseValidator] = []
        val_cfg = getattr(self.config, "validation", None)
        if val_cfg is None or not val_cfg.enabled:
            return

        candidates = getattr(val_cfg, "candidates", None)
        use_two_phase = getattr(val_cfg, "use_two_phase_inference", False)
        for bcfg in val_cfg.benchmarks:
            bmark = get_benchmark(bcfg.benchmark)
            if candidates is not None:
                for cand in candidates:
                    # Per-candidate override takes priority; fall back to ValidationConfig flag.
                    cand_two_phase = getattr(cand, "use_two_phase_inference", None)
                    effective_two_phase = cand_two_phase if cand_two_phase is not None else use_two_phase
                    self.validators.append(
                        BenchmarkValidator(
                            benchmark=bmark,
                            benchmark_dir=bcfg.benchmark_dir,
                            inference_batch_size=cand.inference_batch_size,
                            candidate_name=cand.name,
                            inference_num_blocks=cand.inference_num_blocks,
                            n_ensemble=getattr(cand, "n_ensemble", None),
                            task_type=getattr(bcfg, "task_type", "all"),
                            improvability_exclude_keys=getattr(bcfg, "improvability_exclude_models", None),
                            mrl_dim=getattr(cand, "mrl_dim", None),
                            use_two_phase_inference=effective_two_phase,
                            max_train_size=getattr(bcfg, "max_train_size", None),
                            max_test_size=getattr(bcfg, "max_test_size", None),
                            val_every_multiplier=getattr(bcfg, "val_every_multiplier", 1),
                            loop_residual_scaling_fn=getattr(cand, "loop_residual_scaling_fn", None),
                            exit_q=getattr(cand, "exit_q", None),
                        )
                    )
            else:
                self.validators.append(
                    BenchmarkValidator(
                        benchmark=bmark,
                        benchmark_dir=bcfg.benchmark_dir,
                        inference_batch_size=bcfg.inference_batch_size,
                        candidate_name=bcfg.candidate_name,
                        task_type=getattr(bcfg, "task_type", "all"),
                        use_two_phase_inference=use_two_phase,
                        max_train_size=getattr(bcfg, "max_train_size", None),
                        max_test_size=getattr(bcfg, "max_test_size", None),
                        val_every_multiplier=getattr(bcfg, "val_every_multiplier", 1),
                    )
                )

        if self.master_process and self.validators:
            names = [v.name for v in self.validators]
            print(f"Validation enabled every {val_cfg.val_every} steps: {names}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def run_validation(self) -> dict[str, float]:
        """Run all configured validators and return combined metrics."""
        val_model = self.ema_model if self.ema_model is not None else self.raw_model
        val_cfg = getattr(self.config, "validation", None)
        return self._run_validation_on(val_model, self.curr_step, val_cfg, self.validators)

    def _broadcast_ema_weights(self) -> dict[str, torch.Tensor]:
        """Broadcast EMA weights from rank 0 to all ranks via raw_model.

        On rank 0, temporarily writes EMA params into raw_model so they can be
        broadcast; the original raw_model params are saved and returned so the
        caller can restore them after validation.  On non-rank-0 ranks the raw
        model is updated in-place with the broadcast EMA weights and an empty
        dict is returned.

        If there is no EMA model the raw_model params are already synced via
        DDP, so no broadcast is performed and an empty dict is returned.
        """
        if self.ema_model is None:
            return {}
        model = self.raw_model.module if hasattr(self.raw_model, "module") else self.raw_model
        saved: dict[str, torch.Tensor] = {}
        if self.master_process:
            with torch.no_grad():
                for (name, param), (_, ema_param) in zip(
                    model.named_parameters(), self.ema_model.named_parameters()
                ):
                    saved[name] = param.data.clone()
                    param.data.copy_(ema_param.data)
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        return saved

    def _restore_raw_params(self, saved: dict[str, torch.Tensor]) -> None:
        """Restore rank 0's raw_model params after EMA broadcast."""
        if not saved:
            return
        model = self.raw_model.module if hasattr(self.raw_model, "module") else self.raw_model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in saved:
                    param.data.copy_(saved[name])

    def _run_validation_distributed(self, step: int, val_cfg) -> dict[str, float]:
        """Run validators distributed across all DDP ranks.

        Each rank evaluates a disjoint slice of ``self.validators``
        (validators[rank::world_size]).  EMA weights are broadcast from rank 0
        so every rank evaluates the same model state.  Results are gathered back
        to rank 0 via ``dist.all_gather_object``.

        Falls back to single-rank behaviour when DDP is not active.
        """
        base_val_every = getattr(val_cfg, "val_every", 1)
        due_validators = [
            v for v in self.validators
            if step == 1 or step % (base_val_every * getattr(v, "val_every_multiplier", 1)) == 0
        ]

        if not self.ddp or self.ddp_world_size == 1:
            # Non-DDP path: only master runs validation (caller's responsibility).
            val_model = self.ema_model if self.ema_model is not None else self.raw_model
            return self._run_validation_on(val_model, step, val_cfg, due_validators)

        # --- DDP path ---
        # Step 1: broadcast EMA weights from rank 0 to all ranks.
        saved_params = self._broadcast_ema_weights()
        model = self.raw_model.module if hasattr(self.raw_model, "module") else self.raw_model

        # Step 2: each rank processes its slice of validators.
        my_validators = due_validators[self.ddp_rank::self.ddp_world_size]
        local_results = self._run_validation_on(model, step, val_cfg, my_validators)

        # Step 3: restore rank 0's raw_model params.
        self._restore_raw_params(saved_params)

        # Step 4: gather results from all ranks to rank 0.
        all_results: list[dict[str, float]] = [{}] * self.ddp_world_size
        dist.all_gather_object(all_results, local_results)

        if self.master_process:
            combined: dict[str, float] = {}
            for r in all_results:
                combined.update(r)
            # Remove run_once validators from the shared list (already removed
            # per-rank inside _run_validation_on; sync the master list here).
            self.validators = [v for v in self.validators if not v.run_once]
            return combined
        # Non-master ranks: run_once validators were already removed locally
        # inside _run_validation_on; ensure the master list stays consistent.
        self.validators = [v for v in self.validators if not v.run_once]
        return {}

    def _run_validation_on(
        self,
        val_model: nn.Module,
        step: int,
        val_cfg,
        validators: list[BaseValidator],
    ) -> dict[str, float]:
        """Run the given list of validators on val_model and return combined metrics."""
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np.random.seed(self.config.training.np_seed)
        torch.manual_seed(self.config.training.torch_seed)

        results: dict[str, float] = {}
        run_once_done = []
        for validator in validators:
            results.update(validator.validate(val_model, self.config.training.device, step=step, dtype=self.config.training.dtype))
            if validator.run_once:
                run_once_done.append(validator)
        for v in run_once_done:
            if v in self.validators:
                self.validators.remove(v)

        # Restore RNG state (only meaningful when called from the main thread).
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def load_checkpoint(self):
        """Load model and training state from the latest (or specified) checkpoint."""
        checkpoint_path = None
        if hasattr(self.config.checkpoint, "dir") and self.config.checkpoint.dir:
            checkpoint_path = get_latest_checkpoint(self.config.checkpoint.dir)
        if checkpoint_path is None and hasattr(self.config.checkpoint, "path") and self.config.checkpoint.path:
            checkpoint_path = self.config.checkpoint.path

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.training.device, weights_only=True)

        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        self.raw_model.load_state_dict(checkpoint["state_dict"])

        if self.ema_model is not None:
            if "ema_state_dict" in checkpoint:
                self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
                print("EMA weights loaded from checkpoint.")
            else:
                # No EMA state saved — initialize EMA from the loaded model weights.
                self.ema_model.load_state_dict(checkpoint["state_dict"])
                print("EMA weights initialized from model checkpoint (no EMA state found).")

        if self.config.checkpoint.only_load_model:
            print("Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")

    def _save_checkpoint(self, name: str):
        save_checkpoint(
            ckpt_dir=self.config.checkpoint.dir,
            name=name,
            model_config=self.model_config,
            raw_model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            curr_step=self.curr_step,
            ema_model=self.ema_model,
        )

    def _manage_checkpoints(self):
        manage_checkpoints(
            ckpt_dir=self.config.checkpoint.dir,
            limit=self.config.checkpoint.max_checkpoints,
            save_perm_every=self.config.checkpoint.save_perm_every,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @ddp_cleanup
    def train(self):
        """Main training loop."""
        # On the final training step in DDP mode, the checkpoint save is deferred
        # to AFTER destroy_process_group() (see below).  ProcessGroupNCCL's
        # teardown submits a collective that requires all ranks to participate.
        # If rank 0 is still writing a checkpoint to a network filesystem (e.g.
        # CephFS) when other ranks reach destroy_process_group(), they block in
        # that teardown collective until the NCCL heartbeat monitor fires (480 s).
        # Deferring the save ensures every rank calls destroy_process_group()
        # simultaneously — with no I/O delay on rank 0 — so teardown completes
        # in microseconds.  Intermediate-step saves are unaffected: the next
        # iteration's pre-batch _ddp_barrier() handles the re-sync naturally.
        _deferred_ckpt: str | None = None
        _deferred_is_temp: bool = False
        _deferred_is_perm: bool = False

        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.training.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.training.max_steps)

        dataloader = iter(self.dataloader)
        _log_buffer: list[dict] = []
        for step in step_progress:
            with Timer() as prior_timer:
                batch = next(dataloader)
            prior_time = prior_timer.elapsed

            # Sync all ranks after data loading so a slow rank doesn't cause
            # others to hang inside a collective inside run_batch.
            if self.ddp:
                self._ddp_barrier()

            if step % 100 == 0:
                _ds = self.dataloader.dataset
                _q = f" | Queue: {_ds.prefetch_queue_size}/{_ds.prefetch_queue_maxsize}" if hasattr(_ds, "prefetch_queue_size") else ""
                _gen = f" | Generated: {_ds.total_generated}" if hasattr(_ds, "total_generated") else ""
                print(f"Step {step+1}/{self.config.training.max_steps}: Prior generation time: {prior_time:.3f}s{_q}{_gen}")

            # Scheduled unfreeze: re-enable backbone layers frozen at model init.
            freeze_steps = getattr(self.config.training, "freeze_steps", 0)
            if freeze_steps > 0 and step == freeze_steps:
                for layer_name in getattr(self.config.training, "freeze_layers", []):
                    layer = getattr(self.raw_model, layer_name)
                    layer.train()
                    for param in layer.parameters():
                        param.requires_grad_(True)
                if self.master_process:
                    print(f"Step {step}: unfroze backbone layers: {getattr(self.config.training, 'freeze_layers', [])}")

            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed

            # Free CPU prior tensors immediately; the next iteration will allocate
            # a fresh batch anyway, so holding the reference serves no purpose.
            del batch

            # Periodic GC + heap trim every 10 steps.
            # malloc_trim(0) returns free glibc heap pages to the OS.
            #
            # DO NOT call torch.cuda.empty_cache() here.  Calling it forces the
            # CUDA allocator to release its cached pool blocks back to the OS.
            # The next training step then has to re-request those blocks, which
            # creates brand-new CUDA virtual-address reservations (anonymous
            # mmaps).  Because icl_random_loops=True produces different peak
            # sizes each step, the re-requested blocks don't land at the same
            # virtual addresses as the released ones, so the OS VMA table grows
            # by a few new entries every step.  The CUDA driver also rebuilds
            # its per-allocation bookkeeping structs in the glibc heap, which
            # fragments and grows at the same rate.  The net result is 6–8 MB/
            # step of CPU RSS growth in both RssAnon and [heap].
            #
            # Leaving the cache alone lets the allocator reuse its existing
            # pool at stable virtual addresses: no new VMAs, no heap churn.
            # empty_cache() is still called in the OOM handler and once before
            # validation to give that pass a clean slate.
            if step % 10 == 0:
                gc.collect()
                if _HAS_MALLOC_TRIM:
                    _libc.malloc_trim(0)
            # Log per-rank RSS + CUDA stats every 500 steps so memory growth is
            # visible across all ranks (not just master).  Linux ru_maxrss is KB.
            if step % 500 == 0:
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                cuda_alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
                cuda_reserv_mb = torch.cuda.memory_reserved() / 1024 / 1024
                n_gc_objs = len(gc.get_objects())
                n_allreduce_bufs = len(self._allreduce_buf_cache) if hasattr(self, "_allreduce_buf_cache") else 0
                # CUDA segment count: if this grows monotonically the allocator
                # is creating new pool blocks rather than reusing cached ones.
                try:
                    _mem_stats = torch.cuda.memory_stats()
                    _cuda_segments = _mem_stats.get("segment.current.all", -1)
                except Exception:
                    _cuda_segments = -1
                # Break down RSS by category via /proc/self/status:
                #   RssAnon  – anonymous pages (heap, CUDA driver mmaps, stack)
                #   RssFile  – file-backed pages (shared libs, mmapped .pt files)
                #   RssShmem – shared memory (tmpfs, /dev/shm)
                rss_anon_mb = rss_file_mb = rss_shmem_mb = -1
                try:
                    with open("/proc/self/status") as _f:
                        for _line in _f:
                            if _line.startswith("RssAnon:"):
                                rss_anon_mb = int(_line.split()[1]) / 1024
                            elif _line.startswith("RssFile:"):
                                rss_file_mb = int(_line.split()[1]) / 1024
                            elif _line.startswith("RssShmem:"):
                                rss_shmem_mb = int(_line.split()[1]) / 1024
                except Exception:
                    pass
                print(
                    f"[Rank {self.ddp_rank}] Step {self.curr_step + 1}: "
                    f"RSS={rss_mb:.0f} MB  "
                    f"RssAnon={rss_anon_mb:.0f} MB  RssFile={rss_file_mb:.0f} MB  RssShmem={rss_shmem_mb:.0f} MB  "
                    f"CUDA alloc={cuda_alloc_mb:.0f} MB  "
                    f"CUDA reserved={cuda_reserv_mb:.0f} MB  "
                    f"CUDA segments={_cuda_segments}  "
                    f"GC objects={n_gc_objs}  "
                    f"allreduce_buf_cache={n_allreduce_bufs}"
                )
                # Every 2000 steps, show top smaps regions and VMA count so we
                # can identify which library/mapping is responsible for growth.
                if step % 2000 == 0:
                    try:
                        _anon_regions: dict[str, int] = {}
                        _vma_count = 0
                        _anon_vma_count = 0
                        with open("/proc/self/smaps") as _sf:
                            _cur_name = "[anon]"
                            for _sl in _sf:
                                _sl = _sl.rstrip()
                                # VMA header: "addr-addr perms offset dev ino [path]"
                                # Addresses are hex; split on '-' to check first char
                                if _sl and not _sl[0] == ' ' and '-' in _sl[:20]:
                                    _vma_count += 1
                                    _parts = _sl.split()
                                    _cur_name = _parts[5] if len(_parts) > 5 else "[anon]"
                                    if _cur_name == "[anon]":
                                        _anon_vma_count += 1
                                elif _sl.startswith("Rss:"):
                                    _rss_kb = int(_sl.split()[1])
                                    if _rss_kb > 0:
                                        _anon_regions[_cur_name] = _anon_regions.get(_cur_name, 0) + _rss_kb
                        _top = sorted(_anon_regions.items(), key=lambda x: -x[1])[:10]
                        _top_str = "  ".join(f"{n.split('/')[-1]}:{v//1024}MB" for n, v in _top if v > 10240)
                        print(
                            f"[Rank {self.ddp_rank}] smaps: VMAs={_vma_count} anon_VMAs={_anon_vma_count}  "
                            f"top: {_top_str}"
                        )
                    except Exception:
                        pass
            self.curr_step = step + 1

            val_cfg = getattr(self.config, "validation", None)

            # All ranks sync here after the training step.
            if self.ddp:
                self._ddp_barrier()

            # Validation: distributed across all DDP ranks.  Each rank evaluates
            # a disjoint slice of validators; EMA weights are broadcast from
            # rank 0 so every rank uses the same model state.  Results are
            # gathered back to rank 0 via all_gather_object (which acts as the
            # inter-rank barrier, replacing the old explicit _ddp_barrier()).
            val_results: dict[str, float] = {}
            run_val = (
                bool(self.validators) and val_cfg is not None
                and (self.curr_step % val_cfg.val_every == 0 or self.curr_step == 1)
            )
            if run_val:
                # Release unused cached CUDA blocks on all ranks before validation.
                # This is the ONE legitimate use of empty_cache(): validation runs
                # once every val_every steps, so the VMA rebuild cost is negligible.
                torch.cuda.empty_cache()
                with torch.no_grad():
                    val_results = self._run_validation_distributed(self.curr_step, val_cfg)
                # Restore training mode on all ranks (validators call model.eval()).
                _raw = self.raw_model.module if hasattr(self.raw_model, "module") else self.raw_model
                _raw.train()
                if self.ema_model is not None:
                    self.ema_model.train()
                # Reclaim CPU heap from validation dataset arrays and cyclic
                # garbage left by numpy/openml/sklearn objects.
                # Do NOT call empty_cache() here: we already called it before
                # this validation pass; calling it again would force the first
                # training step post-validation to re-request GPU blocks at new
                # virtual addresses, restarting the VMA churn.
                gc.collect()
                if _HAS_MALLOC_TRIM:
                    _libc.malloc_trim(0)
                if self.master_process:
                    print(
                        f"Step {self.curr_step}: Validation results: "
                        + ", ".join(f"{k}={v:.4f}" for k, v in val_results.items())
                    )
                    if val_results and val_cfg.save_dir:
                        os.makedirs(val_cfg.save_dir, exist_ok=True)
                        save_path = os.path.join(val_cfg.save_dir, f"step_{self.curr_step}.json")
                        with open(save_path, "w") as f:
                            json.dump({"step": self.curr_step, "metrics": val_results}, f, indent=2)

            if self.master_process:
                results.update({"prior_time": prior_time, "train_time": train_time})
                step_progress.set_postfix(**{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()})

                is_temp_save = self.curr_step % self.config.checkpoint.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.checkpoint.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    if self.ddp and self.curr_step >= self.config.training.max_steps:
                        # Final DDP step: defer I/O to after destroy_process_group().
                        _deferred_ckpt = f"step-{self.curr_step}.ckpt"
                        _deferred_is_temp = is_temp_save
                        _deferred_is_perm = is_perm_save
                    else:
                        self._save_checkpoint(name=f"step-{self.curr_step}.ckpt")
                        if is_temp_save and not is_perm_save and self.config.checkpoint.max_checkpoints > 0:
                            self._manage_checkpoints()

            if self.master_process and self.wandb_run is not None:
                results["lr"] = self.scheduler.get_last_lr()[0]
                _log_buffer.append(results)
                _backend = getattr(self.config.wandb, "backend", "wandb")
                _log = trackio.log if _backend == "trackio" else wandb.log
                if self.curr_step % 500 == 0:
                    avg = {k: sum(d[k] for d in _log_buffer if k in d) / sum(1 for d in _log_buffer if k in d) for k in _log_buffer[-1]}
                    _log(avg, step=self.curr_step)
                    _log_buffer.clear()
                if val_results:
                    _log(val_results, step=self.curr_step)

        # All ranks reach here at the same time because rank 0 no longer does
        # checkpoint I/O inside the loop on the last step (deferred above).
        # Calling destroy_process_group() here ensures the ProcessGroupNCCL
        # teardown collective runs while all ranks are simultaneously present.
        # The @ddp_cleanup decorator's finally-block checks dist.is_initialized()
        # and becomes a no-op after this explicit call.
        if self.ddp:
            destroy_process_group()

        # Save the deferred final checkpoint after NCCL teardown (rank 0 only).
        if _deferred_ckpt is not None and self.master_process:
            self._save_checkpoint(name=_deferred_ckpt)
            if _deferred_is_temp and not _deferred_is_perm and self.config.checkpoint.max_checkpoints > 0:
                self._manage_checkpoints()

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _ddp_barrier(self) -> None:
        """Barrier using a pre-allocated tensor to avoid NCCL registration growth.

        dist.barrier() allocates `at::empty({1}, cuda)` on every call, producing
        a fresh GPU address that NCCL must register for DMA.  Those registrations
        are cached in CPU-side tables and never evicted, so calling barrier() N
        times leaks O(N) CPU memory.  Using a persistent tensor keeps the GPU
        address stable so NCCL reuses the same registration every step.
        """
        self._barrier_buf.zero_()
        dist.all_reduce(self._barrier_buf, op=dist.ReduceOp.SUM)

    def _allreduce_bucket(self, bucket: list[torch.Tensor]) -> None:
        """Allreduce a list of gradient tensors using a cached flat buffer.

        Using a pre-allocated, reused buffer (keyed by total element count,
        dtype, and device) ensures NCCL always sees the same GPU virtual
        address.  Allocating a fresh tensor every step via torch.cat causes
        NCCL to register a new address each time; those registrations
        accumulate in a CPU-side cache and are the root cause of the linear
        RSS growth observed in DDP training.
        """
        total_numel = sum(g.numel() for g in bucket)
        key = (total_numel, bucket[0].dtype, bucket[0].device.index)
        if key not in self._allreduce_buf_cache:
            self._allreduce_buf_cache[key] = torch.empty(
                total_numel, dtype=bucket[0].dtype, device=bucket[0].device
            )
        flat = self._allreduce_buf_cache[key]
        # Pack
        offset = 0
        for g in bucket:
            numel = g.numel()
            flat[offset:offset + numel].copy_(g.flatten())
            offset += numel
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.ddp_world_size)
        # Unpack
        offset = 0
        for g in bucket:
            numel = g.numel()
            g.copy_(flat[offset:offset + numel].view_as(g))
            offset += numel

    def run_batch(self, batch):
        """Train on one batch with gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        # Sort by train_size (last element) so each micro-batch shares a common train_size.
        # This is needed when the loader concatenates file batches with different train_sizes.
        sort_idx = torch.argsort(batch[-1])
        batch = [t[sort_idx] for t in batch]

        # Adapt dual/soft-v2 batch to either a 6-element KD batch (hard + soft) or a 5-element standard batch.
        # 7-tuple: (X, y_reg, y_cls_hard, y_cls_soft, d, seq_lens, train_sizes)          — graph_scm_dual
        # 8-tuple: (X, y_reg, y_cls_hard, y_cls_soft, y_reg_soft, d, seq_lens, train_sizes) — graph_scm_soft_v2
        if len(batch) in (7, 8):
            _X, _y_reg, _y_cls_hard, _y_cls_soft = batch[0], batch[1], batch[2], batch[3]
            _d, _seq_lens, _train_sizes = batch[-3], batch[-2], batch[-1]
            _use_kd = getattr(self.config.training, "use_kd_loss", False)
            _use_soft = getattr(self.config.training, "use_soft_labels", False)
            if _use_kd:
                batch = [_X, _y_cls_hard, _y_cls_soft, _d, _seq_lens, _train_sizes]
            else:
                batch = [_X, _y_cls_soft if _use_soft else _y_cls_hard, _d, _seq_lens, _train_sizes]
        num_micro_batches = math.ceil(self.config.training.batch_size / self.config.training.micro_batch_size)
        micro_batches = list(zip(*[torch.split(t, self.config.training.micro_batch_size, dim=0) for t in batch]))

        is_regression = getattr(self.config.prior, "prior_tasktype", "classification") == "regression"
        use_pinball = is_regression and getattr(self.config.training, "use_pinball_loss", False)
        if use_pinball:
            results = {"pinball": 0.0}
        elif is_regression:
            results = {"nll": 0.0}
        else:
            results = {"ce": 0.0, "accuracy": 0.0}
        failed_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                micro_results = self._run_micro_batch(micro_batch, idx, num_micro_batches)
                for k, v in micro_results.items():
                    results[k] = results.get(k, 0.0) + v
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                retried = False
                if (hasattr(self.raw_model, "loop_gradient_checkpointing")
                        and not self.raw_model.loop_gradient_checkpointing):
                    print(f"Warning: OOM at step {self.curr_step}, enabling loop_gradient_checkpointing and retrying.")
                    self.raw_model.loop_gradient_checkpointing = True
                    try:
                        micro_results = self._run_micro_batch(micro_batch, idx, num_micro_batches, skip_nan_check=True)
                        for k, v in micro_results.items():
                            results[k] = results.get(k, 0.0) + v
                        retried = True
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                    finally:
                        self.raw_model.loop_gradient_checkpointing = False
                if not retried:
                    print(f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping.")
                    failed_batches += 1

        if self.ddp:
            # Coordinate OOM status across all ranks BEFORE gradient sync.
            # With require_backward_grad_sync=False on all micro-batches, the DDP
            # all-reduce has NOT happened yet, so this all_reduce is safe.
            # Reuse self._oom_buf (pre-allocated) to avoid creating a new GPU
            # tensor every step — new tensors → new NCCL registrations → RSS growth.
            self._oom_buf.fill_(float(failed_batches > 0))
            dist.all_reduce(self._oom_buf, op=dist.ReduceOp.MAX)
            if self._oom_buf.item() > 0:
                # At least one rank OOM'd — skip the optimizer step on all ranks.
                print(f"Warning: OOM on at least one rank at step {self.curr_step}. Skipping optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.scheduler.step()
                return results
            # All ranks succeeded — manually sync gradients (replaces DDP auto-sync).
            # Bucketed allreduce: pack gradients into ~25 MB flat buffers and allreduce
            # each bucket.  This matches DDP's default strategy and avoids two problems
            # from per-param allreduce:
            #   1. One allreduce per param = O(N_params) NCCL calls; bucketing reduces
            #      this to O(model_size / 25 MB).
            #   2. Allocating a new flat tensor every step (via torch.cat) gives NCCL a
            #      new GPU address to register each step; these registrations are cached
            #      by NCCL and grow linearly → CPU RSS leak in DDP mode.  Fixed by
            #      reusing the same pre-allocated flat buffer each step so NCCL always
            #      sees the same address and never adds new registrations.
            # Only requires_grad params are synced; frozen params are identical on all
            # ranks after the synchronous stage transition so they need no allreduce.
            # Params with no gradient (e.g. a skipped loop) are zero-filled so every
            # rank participates in exactly the same set of allreduce calls.
            _BUCKET_BYTES = 25 * 1024 * 1024  # 25 MB, matching DDP's default
            params = [p for p in self.model.parameters()
                      if p.requires_grad and not (p.grad is not None and p.grad.is_sparse)]
            # _allreduce_buf_cache (pre-allocated in build_model) maps
            # (total_numel, dtype, device_index) → flat GPU tensor.  Reusing the
            # same tensor keeps the GPU address stable so NCCL can reuse its memory
            # registrations instead of accumulating new ones each step.
            bucket: list[torch.Tensor] = []
            bucket_bytes = 0
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                bucket.append(p.grad)
                bucket_bytes += p.grad.numel() * p.grad.element_size()
                if bucket_bytes >= _BUCKET_BYTES:
                    self._allreduce_bucket(bucket)
                    bucket.clear()
                    bucket_bytes = 0
            if bucket:  # flush the last (possibly partial) bucket
                self._allreduce_bucket(bucket)
        else:
            failure_ratio = failed_batches / num_micro_batches
            if failure_ratio > 0.1:
                raise RuntimeError(
                    f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                    "Please check configuration to reduce memory consumption."
                )

        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.gradient_clipping if self.config.training.gradient_clipping > 0 else float("inf"),
        )
        results["grad_norm"] = grad_norm.item()

        if torch.isfinite(grad_norm):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema_model is not None:
                self._update_ema()
        else:
            print(f"Warning: Non-finite gradient norm ({grad_norm.item()}) at step {self.curr_step}. Skipping optimizer step.")
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results

    def _run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches, skip_nan_check=False):
        """Forward + backward pass for one micro-batch."""
        # KD batch: (X, y_hard, y_soft, d, seq_len, train_size) — 6 elements.
        # Standard batch: (X, y, d, seq_len, train_size) — 5 elements.
        is_kd = len(micro_batch) == 6
        if is_kd:
            micro_X, micro_y_hard, micro_y_soft, micro_d, micro_seq_len, micro_train_size = micro_batch
        else:
            micro_X, micro_y_soft, micro_d, micro_seq_len, micro_train_size = micro_batch
            micro_y_hard = None

        seq_len, train_size = _validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y_soft = _align_micro_batch(micro_X, micro_y_soft, micro_d, seq_len)
        if is_kd and micro_y_hard.shape[1] > seq_len:
            micro_y_hard = micro_y_hard[:, :seq_len]

        # NaN check runs on CPU before .to(device) so the all_reduce is guaranteed
        # to happen on all ranks before any OOM-prone GPU allocation. If this check
        # were placed after .to(device), an OOM on one rank would cause it to bail
        # via the try/except in run_batch while other ranks block on the all_reduce.
        # skip_nan_check=True on OOM retries: the NaN check already passed on the
        # first attempt, and re-running the all_reduce would be asymmetric (only
        # the OOM'd rank retries), causing an NCCL sequence-number mismatch.
        has_nan = False
        if not skip_nan_check:
            has_nan = not torch.isfinite(micro_X).all()
            if self.ddp:
                self._has_nan_buf.fill_(float(has_nan))
                dist.all_reduce(self._has_nan_buf, op=dist.ReduceOp.MAX)
                has_nan = bool(self._has_nan_buf.item())
        if has_nan:
            print(f"Warning: NaN/Inf in input X at micro-batch {micro_batch_idx}. Skipping.")
            return {"nll": 0.0} if getattr(self.config.prior, "prior_tasktype", "classification") == "regression" else {"ce": 0.0, "accuracy": 0.0}

        micro_X = micro_X.to(self.config.training.device)
        micro_y_soft = micro_y_soft.to(self.config.training.device)
        micro_d = micro_d.to(self.config.training.device)
        if is_kd:
            micro_y_hard = micro_y_hard.to(self.config.training.device)

        # Soft labels: y_soft is (B, T, C) float; hard labels: y is (B, T) long.
        is_soft_labels = micro_y_soft.dim() == 3

        y_train = micro_y_soft[:, :train_size]
        y_test = micro_y_soft[:, train_size:]

        if is_kd:
            # KD: context uses actual observed hard labels; loss target is the soft distribution.
            y_train_for_model = micro_y_hard[:, :train_size].long()
            y_test_hard = micro_y_hard[:, train_size:]
        else:
            # Standard: convert soft labels to argmax for context, or use hard labels directly.
            y_train_for_model = y_train.argmax(dim=-1) if is_soft_labels else y_train
            y_test_hard = None

        if self.ddp:
            # Always suppress DDP's automatic gradient sync during backward.
            # Gradient sync is done explicitly in run_batch after OOM coordination.
            self.model.require_backward_grad_sync = False

        is_regression = getattr(self.config.prior, "prior_tasktype", "classification") == "regression"
        use_pinball = is_regression and getattr(self.config.training, "use_pinball_loss", False)
        use_aux_loss = getattr(self.config.training, "icl_aux_loss", False)
        aux_loss_weight = getattr(self.config.training, "icl_aux_loss_weight", 0.3)
        use_support_loss = getattr(self.config.training, "support_set_loss", False) and not is_regression
        support_loss_weight = getattr(self.config.training, "support_set_loss_weight", 0.3)

        # Normalise CE by log(num_classes) so batches with different num_classes
        # contribute equal gradient signal (tabicl_v2 prior saves one num_classes
        # per batch file, so this is a per-batch scalar, not per-sample).
        normalize_ce = getattr(self.config.training, "normalize_ce_by_log_classes", False)
        kd_soft_weight = getattr(self.config.training, "kd_soft_weight", 0.5)

        # Infer actual num_classes: use hard labels when available (exact),
        # otherwise count non-zero columns in soft labels (padding is exactly 0).
        if is_kd:
            _actual_C = int(micro_y_hard.max().item()) + 1
        elif is_soft_labels:
            _actual_C = int((micro_y_soft > 0).any(dim=0).any(dim=0).sum().item())
        else:
            _actual_C = None

        if normalize_ce and not is_regression:
            _k = _actual_C if _actual_C is not None else int(y_train_for_model.max().item()) + 1
            _ce_norm = math.log(max(_k, 2))
        else:
            _ce_norm = 1.0

        # Precompute flattened hard test targets for KD path.
        y_test_hard_flat = y_test_hard.long().flatten() if is_kd else None

        def _cls_loss(logits_flat, y_soft_tgt):
            """Cross-entropy loss supporting hard, soft, and KD (hard + soft) targets.

            For KD: loss = CE(logits, y_hard) + kd_soft_weight * CE(logits, y_soft).
            For soft-only: loss = CE(logits, y_soft).
            Both slice logits to the actual num_classes (not max_classes) so the
            softmax normalises over the correct C classes and padded logits are
            excluded. When normalize_ce_by_log_classes=True the loss is divided by
            log(actual_num_classes) so every batch contributes equal gradient signal.
            """
            if is_kd:
                C = _actual_C
                hard_ce = F.cross_entropy(logits_flat[:, :C], y_test_hard_flat) / _ce_norm
                soft_ce = F.cross_entropy(logits_flat[:, :C], y_soft_tgt[:, :C]) / _ce_norm
                return hard_ce + kd_soft_weight * soft_ce
            if is_soft_labels:
                C = _actual_C
                return F.cross_entropy(logits_flat[:, :C], y_soft_tgt[:, :C]) / _ce_norm
            return F.cross_entropy(logits_flat, y_soft_tgt) / _ce_norm

        use_two_phase_forward = getattr(self.config.training, "use_two_phase_forward", False)
        two_phase_forward_fn  = getattr(self.config.training, "two_phase_forward_fn", "forward_two_phase")

        with self.amp_ctx:
            if use_two_phase_forward:
                _forward_fn = getattr(self.raw_model, two_phase_forward_fn)
                pred = _forward_fn(
                    micro_X, y_train_for_model, train_size,
                    return_all_logits=use_aux_loss,
                    return_train_logits=use_support_loss,
                )
            else:
                pred = self.model(micro_X, y_train_for_model, train_size,
                                  return_all_logits=use_aux_loss,
                                  return_train_logits=use_support_loss)

            # Unpack support-set (train-row LOO) logits when requested.
            train_pred_for_support = None
            if use_support_loss and isinstance(pred, tuple):
                pred, train_pred_for_support = pred

            if isinstance(pred, list):
                # Per-iteration auxiliary loss.
                # pred[-1] is the final iteration; pred[:-1] are the aux iterations.
                # Loss = final_loss + (aux_loss_weight / N_aux) * Σ aux_losses
                # This keeps the final iteration's gradient scale identical to baseline.
                n_aux = len(pred) - 1
                if use_pinball:
                    # pred[-1]: (M, B, Q); y_test: (B, M) → (M, B, 1)
                    def _pinball(p):
                        p = p.sort(dim=-1).values  # enforce monotonicity during training
                        Q = p.shape[-1]
                        alphas = torch.linspace(0.0, 1.0, Q + 2, device=p.device, dtype=p.dtype)[1:-1]
                        alphas = alphas.view(1, 1, -1)
                        errors = y_test.float().T.unsqueeze(-1) - p
                        return torch.maximum(alphas * errors, (alphas - 1) * errors).mean()
                    final_loss = _pinball(pred[-1])
                    if n_aux > 0:
                        aux_loss = sum(_pinball(p) for p in pred[:-1])
                        loss = final_loss + (aux_loss_weight / n_aux) * aux_loss
                    else:
                        aux_loss = None
                        loss = final_loss
                    final_pred = pred[-1]
                elif is_regression:
                    from loopicl.models.loopicl_ablation.bar_distribution import BarDistribution
                    _bar = BarDistribution(num_bars=pred[-1].shape[-1])
                    _edges = self.raw_model.fixed_edges.expand(pred[-1].shape[1], -1)  # (B, num_bars+1)
                    _y_test_idx = _bar.to_indices(y_test.float(), _edges)              # (B, M)
                    final_loss = _bar.nll_loss(pred[-1], _y_test_idx)
                    if n_aux > 0:
                        aux_losses = [_bar.nll_loss(p, _y_test_idx) for p in pred[:-1]]
                        aux_loss = sum(aux_losses)
                        loss = final_loss + (aux_loss_weight / n_aux) * aux_loss
                    else:
                        aux_loss = None
                        loss = final_loss
                    final_pred = pred[-1]  # (M, B, num_bars) — kept for metrics
                else:
                    if is_soft_labels:
                        true = y_test.reshape(-1, y_test.shape[-1]).float()
                    else:
                        true = y_test.long().flatten()
                    final_pred = pred[-1].flatten(end_dim=-2)
                    final_loss = _cls_loss(final_pred, true)
                    if n_aux > 0:
                        aux_losses = [_cls_loss(p.flatten(end_dim=-2), true) for p in pred[:-1]]
                        aux_loss = sum(aux_losses)
                        loss = final_loss + (aux_loss_weight / n_aux) * aux_loss
                    else:
                        aux_loss = None
                        loss = final_loss
                    pred = final_pred
            else:
                if use_pinball:
                    # pred: (M, B, Q); y_test: (B, M) → (M, B, 1)
                    pred = pred.sort(dim=-1).values  # enforce monotonicity during training
                    Q = pred.shape[-1]
                    alphas = torch.linspace(0.0, 1.0, Q + 2, device=pred.device, dtype=pred.dtype)[1:-1]
                    alphas = alphas.view(1, 1, -1)
                    errors = y_test.float().T.unsqueeze(-1) - pred
                    loss = torch.maximum(alphas * errors, (alphas - 1) * errors).mean()
                elif is_regression:
                    from loopicl.models.loopicl_ablation.bar_distribution import BarDistribution
                    _bar = BarDistribution(num_bars=pred.shape[-1])
                    _edges = self.raw_model.fixed_edges.expand(pred.shape[1], -1)  # (B, num_bars+1)
                    _y_test_idx = _bar.to_indices(y_test.float(), _edges)          # (B, M)
                    loss = _bar.nll_loss(pred, _y_test_idx)
                else:
                    pred = pred.flatten(end_dim=-2)
                    if is_soft_labels:
                        true = y_test.reshape(-1, y_test.shape[-1]).float()
                    else:
                        true = y_test.long().flatten()
                    loss = _cls_loss(pred, true)
                final_loss = loss
                aux_loss = None
                n_aux = 0

            # Support-set loss: LOO cross-entropy on train rows.
            support_loss = None
            if train_pred_for_support is not None:
                true_train = y_train_for_model.long().flatten()
                support_loss = F.cross_entropy(train_pred_for_support.flatten(end_dim=-2), true_train) / _ce_norm
                loss = loss + support_loss_weight * support_loss

        if not torch.isfinite(loss):
            print(f"Warning: NaN/Inf loss at micro-batch {micro_batch_idx} (step {self.curr_step}). Skipping.")
            if use_pinball:
                return {"pinball": 0.0}
            return {"nll": 0.0} if is_regression else {"ce": 0.0, "accuracy": 0.0}

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            if use_pinball:
                metrics = {
                    "pinball": final_loss.item() / num_micro_batches,
                }
                if aux_loss is not None:
                    metrics["aux_pinball"] = (aux_loss.item() / n_aux) / num_micro_batches
            elif is_regression:
                metrics = {
                    "nll": final_loss.item() / num_micro_batches,
                }
                if aux_loss is not None:
                    metrics["aux_nll"] = (aux_loss.item() / n_aux) / num_micro_batches
            else:
                if is_kd:
                    # KD accuracy: compare predicted class against the actual observed hard label.
                    accuracy = (pred[:, :_actual_C].argmax(dim=1) == y_test_hard_flat).sum() / pred.shape[0]
                elif is_soft_labels:
                    # Accuracy: compare predicted class against argmax of soft target.
                    accuracy = (pred[:, :_actual_C].argmax(dim=1) == true.argmax(dim=1)).sum() / pred.shape[0]
                else:
                    accuracy = (pred.argmax(dim=1) == true).sum() / len(true)
                metrics = {
                    "ce": final_loss.item() / num_micro_batches,
                    "accuracy": accuracy.item() / num_micro_batches,
                }
                if aux_loss is not None:
                    metrics["aux_ce"] = (aux_loss.item() / n_aux) / num_micro_batches
                if support_loss is not None:
                    metrics["support_ce"] = support_loss.item() / num_micro_batches
            return metrics


# ---------------------------------------------------------------------------
# Pure helpers (no Trainer state needed)
# ---------------------------------------------------------------------------


def _validate_micro_batch(micro_seq_len, micro_train_size):
    """Use min seq_len and min train_size when datasets in the micro batch differ.

    When prior data is generated with micro_batch_size=1 (one sample per file),
    different files can have different seq_lens.  Using the minimum seq_len is safe
    because seq_len = train_size + test_size, so only test rows beyond the minimum
    are dropped from longer samples — no train rows are lost.  Likewise, the minimum
    train_size is used so extra train rows become additional test rows, consistent
    with the loopicl_graph_scm loading adapter.
    """
    return micro_seq_len.min().item(), micro_train_size.min().item()


def _align_micro_batch(micro_X, micro_y, micro_d, seq_len):
    """Truncate micro batch tensors to (seq_len, max_active_features)."""
    if micro_X.shape[1] > seq_len:
        micro_X = micro_X[:, :seq_len]
    if micro_y.shape[1] > seq_len:
        micro_y = micro_y[:, :seq_len]
    max_features = micro_d.max().item()
    if micro_X.shape[-1] > max_features:
        micro_X = micro_X[..., :max_features]
    return micro_X, micro_y
