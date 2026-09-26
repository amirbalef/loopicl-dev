"""LoopICLReg — LoopICL adapted for bar-distribution regression.

Differences from the base ``LoopICL`` (classification) model
-------------------------------------------------------------
1. ``reg_y_encoder``  : ``nn.Linear(1, E)`` instead of a discrete
   ``TrainableOrthogonalEmbedding``.  Continuous target values are
   projected directly into embedding space.

2. ``discriminative_hist_conditioner``: replaced by
   ``DiscriminativeHistConditionerReg``, which computes the conditional
   mean of ``y`` in each feature histogram bin (the regression analogue
   of per-class histogram residuals).

3. ``reg_head``       : ``nn.Linear(D, num_bars)`` replaces
   ``ClassNormalizedManyClassDecoder``.  It maps test-row embeddings
   directly to bar-distribution logits.

Architecture integration
------------------------
The model is used with ``BarDistribution``:

    bar = BarDistribution(num_bars=5000)
    edges    = bar.compute_edges(y_train)       # (B, 5001)
    y_tr_idx = bar.to_indices(y_train, edges)   # (B, N_train) long
    y_te_idx = bar.to_indices(y_test,  edges)   # (B, M_test)  long
    logits   = model(X, y_train, train_size)    # (M, B, 5000)
    loss     = bar.nll_loss(logits, y_te_idx)

Note: ``y_train`` passed to the model is the *raw continuous* targets
(float), NOT bin indices.  Bin indices are only used for the loss and
for converting test predictions.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import (
    _N_FOURIER,
    _N_HIST_BINS,
    _adaptive_bin_geometry,
    _compute_moments,
    _soft_log_histogram,
)
from .layers import _safe_log_seqlen
from .model import LoopICL, LoopICLConfig

__all__ = ["LoopICLReg", "DiscriminativeHistConditionerReg", "RankLabelEncoder"]


# ---------------------------------------------------------------------------
# Ordinal bar decoder
# ---------------------------------------------------------------------------


def _sinusoidal_pe(num_bars: int, dim: int) -> torch.Tensor:
    """Returns (num_bars, dim) sinusoidal positional encodings."""
    position = torch.arange(num_bars).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(num_bars, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: dim // 2])
    return pe


class OrdinalBarDecoder(nn.Module):
    """Ordinal-aware multi-head bar distribution decoder.

    Replaces the flat ``nn.Linear(D, num_bars)`` regression head with a
    multi-head dot-product decoder whose per-bar query vectors are biased by
    sinusoidal positional encodings.

    Two expressiveness improvements over the single-head version:

    1. **Multi-head**: ``num_heads`` independent projections of the backbone
       embedding, each contributing its own bar logits.  The full ``D``-dim
       embedding participates via ``num_heads`` different subspaces instead of
       being squeezed to a single ``head_dim`` vector.  By generating PE over
       ``num_heads * head_dim`` dims and reshaping, each head covers a different
       sinusoidal frequency band — head 0 captures large-scale ordinal structure,
       head ``H-1`` captures fine-grained structure.

    2. **MLP projection**: a two-layer ``D → 2·H·d → H·d`` MLP with GELU
       replaces the linear projection, allowing non-linear feature extraction
       before the bar dot products.

    Forward interface is identical to ``nn.Linear(D, num_bars)``:
        (B, M, D) → (B, M, num_bars)

    Parameters
    ----------
    in_dim    : D — input dimension from the backbone (= Cl * embed_dim)
    num_bars  : number of regression bins
    head_dim  : dimension per head in the bar query / key space
    num_heads : number of parallel heads
    """

    def __init__(
        self,
        in_dim: int,
        num_bars: int,
        head_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        D_head = num_heads * head_dim

        # Learned per-bar query vectors: (num_bars, num_heads, head_dim)
        self.bar_embed = nn.Parameter(torch.empty(num_bars, num_heads, head_dim))
        nn.init.normal_(self.bar_embed, std=0.02)

        # Sinusoidal PE over num_heads * head_dim dims, reshaped to give each
        # head a distinct frequency band: (num_bars, num_heads, head_dim)
        pe = _sinusoidal_pe(num_bars, D_head).view(num_bars, num_heads, head_dim)
        self.register_buffer("bar_pos", pe)

        # MLP projection: D → 2·D_head → D_head (non-linear)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, D_head * 2),
            nn.GELU(),
            nn.Linear(D_head * 2, D_head, bias=False),
        )

        # Per-bar output bias
        self.bar_bias = nn.Parameter(torch.zeros(num_bars))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, M, D)

        Returns
        -------
        logits : (B, M, num_bars)
        """
        B, M, _ = x.shape

        # Bar queries with per-head ordinal positional bias: (num_bars, num_heads, head_dim)
        q = self.bar_embed + self.bar_pos

        # MLP projection + reshape into heads: (B, M, num_heads, head_dim)
        k = self.proj(x).view(B, M, self.num_heads, self.head_dim) * self.scale

        # Multi-head dot-product, summed over heads: (B, M, num_bars)
        return torch.einsum("bmhd,nhd->bmn", k, q) + self.bar_bias


# ---------------------------------------------------------------------------
# Regression discriminative conditioner
# ---------------------------------------------------------------------------


class DiscriminativeHistConditionerReg(nn.Module):
    """Regression analogue of ``DiscriminativeHistConditioner``.

    Instead of per-class histogram residuals, computes the *conditional
    mean of y* in each feature histogram bin, normalised by the global
    y standard deviation and zero-centred per column.

    This gives the model a discriminative signal proportional to the
    feature–target correlation without requiring class labels.

    Output: ``(B, C, E)`` — same shape as the classification version.
    """

    def __init__(
        self,
        embed_dim: int,
        n_hidden: int = 64,
        n_bins: int = _N_HIST_BINS,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.mlp = nn.Sequential(
            nn.Linear(n_bins, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, embed_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        x_train_BCN: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        log_hist_BCK: torch.Tensor,
        y_BN: torch.Tensor,
        num_train: int,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        B, C, N = x_train_BCN.shape
        y_float = y_BN.float()  # (B, N)

        if chunk_size is None or chunk_size >= N:
            # Full materialisation: (B, C, N, K)
            dist     = (x_train_BCN.unsqueeze(-1) - centers.unsqueeze(2)) / widths.unsqueeze(2)
            soft_ind = (1.0 - dist.abs()).clamp(min=0.0)
            y_view   = y_float.unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
            y_sum    = (soft_ind * y_view).sum(2)            # (B, C, K)
            w_sum    = soft_ind.sum(2).clamp(min=0.5)
        else:
            # Chunked accumulation over N to avoid (B,C,N,K) materialisation.
            K = centers.shape[-1]
            y_sum   = x_train_BCN.new_zeros(B, C, K)
            w_sum   = x_train_BCN.new_zeros(B, C, K)
            for start in range(0, N, chunk_size):
                end        = min(start + chunk_size, N)
                x_chunk    = x_train_BCN[:, :, start:end]                          # (B, C, chunk)
                dist_chunk = (x_chunk.unsqueeze(-1) - centers.unsqueeze(2)) / widths.unsqueeze(2)
                si_chunk   = (1.0 - dist_chunk.abs()).clamp(min=0.0)               # (B, C, chunk, K)
                y_chunk    = y_float[:, start:end].unsqueeze(1).unsqueeze(-1)      # (B, 1, chunk, 1)
                y_sum     += (si_chunk * y_chunk).sum(2)
                w_sum     += si_chunk.sum(2)
            w_sum = w_sum.clamp(min=0.5)

        # Conditional mean: E[y | x in bin k]
        cond_mean = y_sum / w_sum                            # (B, C, K)

        # Scale invariance: divide by per-batch y std.
        # Clamp to 0.1 (not 1e-6) so near-constant y_train subsets don't
        # produce huge cond_mean values that destabilize cell embeddings.
        y_std   = y_float.std(dim=-1).clamp(min=0.1)        # (B,)
        cond_mean = cond_mean / y_std.view(B, 1, 1)

        # Zero-centre per column (mirrors log-hist zero-mean)
        cond_mean = cond_mean - cond_mean.mean(dim=-1, keepdim=True)

        return self.mlp(cond_mean)                           # (B, C, E)


# ---------------------------------------------------------------------------
# Rank label encoder
# ---------------------------------------------------------------------------


class RankLabelEncoder(nn.Module):
    """Encode the empirical rank of each training label within the training set.

    For each y value in ``y_BN`` the rank is computed as its position in the
    sorted training-label sequence, normalised to [0, 1].  The rank is then
    encoded with a Fourier basis (same frequency schedule as
    ``FourierQuantileEncoderWithOOD`` used for features) and projected to the
    model embedding dimension via a small MLP.

    The output is meant to be *added* to the raw continuous y embedding from
    ``reg_y_encoder``, giving the model both the absolute scale and the
    relative ordinal position of every training label.

    Parameters
    ----------
    embed_dim : int
        Target embedding dimension (E).
    n_fourier : int
        Number of Fourier frequency bands (default: same as feature encoder).
    n_hidden : int
        Hidden width of the two-layer MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        n_fourier: int = _N_FOURIER,
        n_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_fourier = n_fourier
        # Input: 2*n_fourier Fourier rank features + 1 normalised y value.
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_fourier + 1, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, embed_dim),
        )
        # Zero-init output so that at the start of training the rank signal
        # contributes nothing and the model can ramp it up gradually.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, y_BN: torch.Tensor, y_sorted_BN: torch.Tensor) -> torch.Tensor:
        """Compute rank+value embedding for training labels.

        Parameters
        ----------
        y_BN : Tensor (B, N)
            Raw training targets (float).
        y_sorted_BN : Tensor (B, N)
            ``y_BN`` sorted along the N dimension (ascending).

        Returns
        -------
        Tensor (B, N, embed_dim)
        """
        N = y_BN.shape[1]
        rank_idx = torch.searchsorted(
            y_sorted_BN.float().contiguous(),
            y_BN.float().contiguous(),
        )
        ranks = rank_idx.to(y_BN.dtype) / N  # (B, N) in [0, 1]
        k = torch.arange(self.n_fourier, device=y_BN.device, dtype=y_BN.dtype)
        freqs = (2.0 ** k) * math.pi
        angles = ranks.unsqueeze(-1) * freqs  # (B, N, n_fourier)
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, N, 2*n_fourier)

        # Normalise y to zero mean / unit std so its scale matches the Fourier
        # features regardless of the target's original units.
        y_std = y_BN.std(dim=-1, keepdim=True).clamp(min=0.1)
        y_norm = ((y_BN - y_BN.mean(dim=-1, keepdim=True)) / y_std).unsqueeze(-1)  # (B, N, 1)

        features = torch.cat([fourier, y_norm], dim=-1)  # (B, N, 2*n_fourier + 1)
        return self.mlp(features)  # (B, N, embed_dim)


# ---------------------------------------------------------------------------
# LoopICLReg
# ---------------------------------------------------------------------------


class LoopICLReg(LoopICL):
    """LoopICL for pinball-loss regression with a 2-layer MLP quantile head.

    Predicts ``num_quantiles`` evenly-spaced quantile levels directly
    (TabICL v2 / EXAONE-Tabular style), trained with pinball loss.

    Parameters
    ----------
    num_quantiles : int, default 999
        Number of quantile levels to predict.
        α_i = i / (num_quantiles + 1) for i = 1, …, num_quantiles.
    mlp_hidden_dim : int, default 512
        Hidden dimension of the 2-layer MLP decoder head.
    use_rank_label_repr : bool, default False
        If True, add an empirical-rank Fourier embedding to the y encoder.
    **kwargs :
        All other keyword arguments are forwarded to ``LoopICL``
        (i.e. every ``ModelConfig`` field).
    """

    def __init__(
        self,
        num_quantiles: int = 999,
        mlp_hidden_dim: int = 512,
        use_rank_label_repr: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("num_bars", None)
        super().__init__(**{**kwargs, "max_num_classes": 2})

        self.num_quantiles = num_quantiles
        E   = self.cfg.embed_dim
        Cl  = self.cfg.feat_agg_num_cls_tokens
        D   = Cl * E
        n_h = self.cfg.softmax_scaling_mlp_hidden_dim

        # Continuous y encoder
        self.reg_y_encoder = nn.Linear(1, E)

        # Optional rank label encoder
        self.use_rank_label_repr = use_rank_label_repr
        if use_rank_label_repr:
            self.rank_label_encoder = RankLabelEncoder(E)

        # Regression discriminative histogram conditioner
        # self.use_disc_hist_conditioning is already set by LoopICL.__init__ from cfg
        if self.use_disc_hist_conditioning:
            self.discriminative_hist_conditioner = DiscriminativeHistConditionerReg(
                E, n_h, _N_HIST_BINS
            )

        # 2-layer MLP quantile head
        self.reg_head = nn.Sequential(
            nn.Linear(D, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_quantiles),
        )

    # set_fixed_edges is intentionally absent — signals trainer to use pinball loss.

    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        single_eval_pos: int | None = None,
        train_size: int | None = None,
        inference_num_blocks: int | None = None,
        num_loops: int | None = None,
        return_all_logits: bool = False,
        return_embedding: bool = False,
        **_kwargs,
    ) -> torch.Tensor | List[torch.Tensor]:
        """Multi-view looped forward pass for regression.

        Parameters
        ----------
        X         : Tensor (B, R, C)
        y_train   : Tensor (B, R)  — continuous float targets
        train_size : int
            Number of training rows; rows[train_size:] are test rows.
        return_all_logits : bool
            If True return a list of per-loop logits (aux-loss training).

        Returns
        -------
        Tensor (M, B, num_quantiles) or List thereof
        """
        x_RBC = X.transpose(0, 1)
        y_NB  = y_train.transpose(0, 1)
        num_train = (
            int(single_eval_pos) if single_eval_pos is not None
            else int(train_size) if train_size is not None
            else int(y_NB.shape[0])
        )

        # Stage 0: preprocess + feature grouping
        x_BRC, nan_ind_BRC = self._preprocess(x_RBC, num_train)
        x_BRCG = self._group_features(x_BRC, nan_ind_BRC)

        x_train_BCN  = x_BRC[:, :num_train].permute(0, 2, 1)
        x_sorted_BCN = x_train_BCN.sort(dim=-1).values

        # Stage 1: cell embedding + label injection
        x_BRCE = self.x_embed(x_BRCG)
        y_BN   = y_NB.transpose(0, 1)[:, :num_train]  # (B, N_train) float

        # Continuous y encoder: (B, N, 1) → (B, N, E)
        y_col_emb = self.reg_y_encoder(y_BN.float().unsqueeze(-1))

        # Optional rank label encoder: add empirical-rank Fourier embedding.
        if self.use_rank_label_repr:
            y_sorted_BN = y_BN.float().sort(dim=-1).values
            y_col_emb = y_col_emb + self.rank_label_encoder(y_BN.float(), y_sorted_BN)

        x_BRCE[:, :num_train] = x_BRCE[:, :num_train] + y_col_emb.unsqueeze(2)

        if self.use_log_hist_conditioning:
            centers, widths = _adaptive_bin_geometry(x_sorted_BCN, self._n_bins)
            log_hist_BCK    = _soft_log_histogram(x_train_BCN, centers, widths, num_train)
            moments_BC4     = _compute_moments(x_train_BCN)

            x_BRCE = x_BRCE + self.log_hist_conditioner(log_hist_BCK, moments_BC4).unsqueeze(1)
            if self.use_disc_hist_conditioning:
                x_BRCE = x_BRCE + self.discriminative_hist_conditioner(
                    x_train_BCN, centers, widths, log_hist_BCK, y_BN, num_train,
                    chunk_size=self._disc_hist_chunk_size,
                ).unsqueeze(1)
            x_BRCE = x_BRCE + self.fourier_quantile_encoder(x_BRC, x_sorted_BCN, num_train)

        # Stage 2: axial conditioning (no AdaLN)
        B = x_BRCE.shape[0]
        num_cross = self.axial_embedder.num_axial_blocks
        num_icl   = sum(1 for _ in self.axial_embedder.icl_blocks)
        axial_conds = [None] * num_cross
        icl_conds   = [None] * num_icl

        # Stage 3: outer loop
        B, R, _, E = x_BRCE.shape
        Cl = self.axial_embedder.num_cls_tokens
        D  = Cl * E
        x_BRD = self._init_x_brd(B, R, x_BRCE, Cl, E)

        n_loops = (
            num_loops if num_loops is not None
            else inference_num_blocks if inference_num_blocks is not None
            else self._sample_num_loops() if (self.training and self.icl_random_loops)
            else self.num_icl_loops
        )

        all_logits: List[torch.Tensor] = []

        loop_scale: float | None = None
        if self.use_loop_residual_scaling:
            loop_scale = self.loop_residual_lambda / (
                n_loops * math.sqrt(self._layers_per_loop)
            )

        alpha_layer = beta_layer = None
        x_BRCE_init = None

        for loop_i in range(n_loops):

            skip_inject = self.inject_only_first_loop and loop_i > 0
            need_pre    = loop_scale is not None
            if need_pre:
                x_BRCE_pre = x_BRCE
                x_BRD_pre  = x_BRD

            if self.loop_gradient_checkpointing and self.training:
                def _step(_xe, _xd, _skip=skip_inject, _alpha=alpha_layer, _beta=beta_layer):
                    return self._axial_step(
                        _xe, _xd, num_train, y_col_emb,
                        skip_cell_label_inject=_skip,
                        residual_alpha=_alpha,
                        residual_beta=_beta,
                    )
                x_BRCE, x_BRD = torch.utils.checkpoint.checkpoint(
                    _step, x_BRCE, x_BRD, use_reentrant=False)
            else:
                x_BRCE, x_BRD = self._axial_step(
                    x_BRCE, x_BRD, num_train, y_col_emb,
                    skip_cell_label_inject=skip_inject,
                    residual_alpha=alpha_layer,
                    residual_beta=beta_layer,
                )

            if loop_scale is not None:
                x_BRD  = x_BRD_pre  + loop_scale * (x_BRD  - x_BRD_pre)
                x_BRCE = x_BRCE_pre + loop_scale * (x_BRCE - x_BRCE_pre)

            if return_all_logits:
                normed   = self.output_norm(x_BRD)
                test_emb = normed[:, num_train:]        # (B, M, D)
                logits   = self.reg_head(test_emb)      # (B, M, num_quantiles)
                all_logits.append(torch.nan_to_num(logits.transpose(0, 1), nan=0.0))

        if return_all_logits:
            return all_logits

        if return_embedding:
            return x_BRD, y_BN, num_train

        # Stage 4: regression head
        x_BRD    = self.output_norm(x_BRD)
        test_emb = x_BRD[:, num_train:]          # (B, M, D)
        test_out = self.reg_head(test_emb)        # (B, M, num_quantiles)
        return torch.nan_to_num(test_out.transpose(0, 1), nan=0.0)  # (M, B, num_quantiles)

    def _safe_log_seqlen_cached(
        self, num_train: int, ref: torch.Tensor
    ) -> torch.Tensor:
        return _safe_log_seqlen(num_train, ref.device, ref.dtype)
