"""KV-cache low-rank analysis across loop iterations for LoopICL.

For each ICL attention block, we capture the key and value projections at
every outer loop iteration and analyse how much of the resulting variation
lives in a low-dimensional subspace.

The central object is the *loop-evolution matrix*:

    M_b ∈ R^{n_loops × D},   D = N_train × H × d_head

where each row is the (batch-averaged, flattened) key (or value) tensor
produced by block b at loop iteration i.  After centering the rows we
compute the thin SVD of M_b and report:

    stable_rank   = ‖M‖_F² / σ₁²         ∈ [1, n_loops]
    singular_values (normalized by σ₁)    shape (n_loops,)
    evr           cumulative explained variance ratio  shape (n_loops,)
    n_components_{90|95|99}   min k s.t. EVR@k ≥ threshold

A stable_rank close to 1 means the KV cache barely changes direction
across loops (rank-1 evolution).  n_components_99 ≪ n_loops is the
headline number for the paper.

We also compute per-loop *spatial* rank: the stable rank of the key
matrix K_b^{(i)} ∈ R^{N_train × (H × d_head)} itself, which tells us
how many independent "key patterns" exist over training tokens at each
loop.

Usage
-----
with torch.no_grad():
    logits, kv_metrics = forward_with_kv_rank_metrics(
        model, X, y_full, single_eval_pos=n_train, num_loops=10
    )

# headline: stable rank of K across loops, averaged over blocks
kv_metrics["K_temporal_stable_rank"]          # (n_loops_blocks,) — per block
kv_metrics["K_temporal_stable_rank_mean"]     # scalar
kv_metrics["K_temporal_evr"]                  # (n_blocks, n_loops)
kv_metrics["K_temporal_singular_values_norm"] # (n_blocks, n_loops)
kv_metrics["K_spatial_stable_rank"]           # (n_loops, n_blocks) per-loop per-block
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from loopicl.models.loopicl.model import LoopICL


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------

def _stable_rank(M: torch.Tensor) -> float:
    """Stable rank = ‖M‖_F² / σ_max².  M is (m, n) float32.

    Uses power iteration to estimate σ_max instead of full SVD —
    much faster when M is large (e.g. N_train × H*d_head).
    """
    frob_sq = float((M * M).sum())
    # Power iteration: estimate σ_max² = largest eigenvalue of M^T M
    v = M.new_empty(M.shape[1]).normal_()
    v = v / (v.norm() + 1e-10)
    for _ in range(20):
        v = M.T @ (M @ v)
        norm = float(v.norm())
        if norm < 1e-10:
            break
        v = v / norm
    sigma_max_sq = float((M @ v).pow(2).sum())
    return frob_sq / (sigma_max_sq + 1e-30)


def _svd_metrics(M: torch.Tensor) -> dict[str, np.ndarray | float]:
    """Full SVD-based rank metrics for a (m, n) float32 matrix.

    Centers along the row dimension (removes mean KV across loops so we
    measure *variation* only).

    Returns a flat dict; all arrays are numpy, scalars are Python floats.
    """
    M = M.float()
    # center: subtract per-feature mean across loops → focus on variation
    M_c = M - M.mean(dim=0, keepdim=True)

    # Gram matrix trick: when m (n_loops) << n (N*H*d), M_c @ M_c.T is (m,m)
    # and its eigenvalues are σ_i² — much cheaper than full SVD of (m, n).
    G = M_c @ M_c.T                                      # (n_loops, n_loops)
    eigvals = torch.linalg.eigvalsh(G)                   # ascending, (n_loops,)
    sv = eigvals.flip(0).clamp(min=0).sqrt()             # descending singular values
    sv2 = sv ** 2
    total = sv2.sum() + 1e-30

    sv_np  = sv.cpu().numpy()
    evr    = (sv2.cumsum(0) / total).cpu().numpy()   # (n_loops,)
    sr     = float(total / (sv[0] ** 2 + 1e-30))

    def _n_for(thresh):
        idx = np.searchsorted(evr, thresh)
        return int(min(idx + 1, len(evr)))

    return {
        "stable_rank":        sr,
        "singular_values":    sv_np,                      # (n_loops,)
        "singular_values_norm": sv_np / (sv_np[0] + 1e-30),
        "evr":                evr,                        # (n_loops,)
        "n_components_90":    _n_for(0.90),
        "n_components_95":    _n_for(0.95),
        "n_components_99":    _n_for(0.99),
        "total_variance":     float(total),
        "sigma_1_fraction":   float(sv2[0] / total),     # % variance in top component
    }


# ---------------------------------------------------------------------------
# Hook-based KV capture
# ---------------------------------------------------------------------------

class _KVCapture:
    """Registers forward hooks on every ICL attention layer to capture K, V.

    For RicottaSandwich, also hooks first_axial_embedder and last_axial_embedder
    so their single passes are prepended/appended to the middle-loop temporal sequence,
    giving (1 + n_middle + 1) = n_loops entries per block.
    """

    def __init__(self, model: "LoopICL", num_train: int):
        self._num_train = num_train
        self._hooks: list = []
        # per block: list of tensors, one per loop call
        self._K: dict[int, list[torch.Tensor]] = {}
        self._V: dict[int, list[torch.Tensor]] = {}
        # Sandwich: per-block-idx lists for first/last embedder captures
        self._prefix_K: dict[int, list[torch.Tensor]] = {}
        self._suffix_K: dict[int, list[torch.Tensor]] = {}
        self._prefix_V: dict[int, list[torch.Tensor]] = {}
        self._suffix_V: dict[int, list[torch.Tensor]] = {}

        emb = model.axial_embedder
        for idx, block in enumerate(emb.icl_blocks):
            self._K[idx] = []
            self._V[idx] = []
            hook = block.icl_attention.register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)

        # Sandwich: hook first and last embedders if present
        for emb_attr, store_K, store_V in [
            ('first_axial_embedder', self._prefix_K, self._prefix_V),
            ('last_axial_embedder',  self._suffix_K, self._suffix_V),
        ]:
            emb_extra = getattr(model, emb_attr, None)
            if emb_extra is not None:
                for idx, block in enumerate(emb_extra.icl_blocks):
                    store_K[idx] = []
                    store_V[idx] = []
                    hook = block.icl_attention.register_forward_hook(
                        self._make_hook_into(store_K[idx], store_V[idx])
                    )
                    self._hooks.append(hook)

    def _make_hook(self, block_idx: int):
        num_train = self._num_train

        def hook(module, input, output):
            # input[0] = layernorm(x_BRE), shape (B, R, E)
            # single_eval_pos passed as kwarg → not in input tuple; use closure
            x = input[0]  # (B, R, E)
            B = x.shape[0]
            N = num_train
            with torch.no_grad():
                k = module.k_projection(x[:, :N]).view(
                    B, N, module.num_kv_heads, module.head_dim
                )
                v = module.v_projection(x[:, :N]).view(
                    B, N, module.num_kv_heads, module.head_dim
                )
            # average over batch, keep on CPU
            self._K[block_idx].append(k.detach().float().mean(0).cpu())  # (N, H, d)
            self._V[block_idx].append(v.detach().float().mean(0).cpu())

        return hook

    def _make_hook_into(self, K_list: list, V_list: list):
        num_train = self._num_train

        def hook(module, input, output):
            x = input[0]
            B = x.shape[0]
            N = num_train
            with torch.no_grad():
                k = module.k_projection(x[:, :N]).view(
                    B, N, module.num_kv_heads, module.head_dim
                )
                v = module.v_projection(x[:, :N]).view(
                    B, N, module.num_kv_heads, module.head_dim
                )
            K_list.append(k.detach().float().mean(0).cpu())
            V_list.append(v.detach().float().mean(0).cpu())

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_loop_matrices(self) -> dict[int, dict[str, torch.Tensor]]:
        """Return per-block dict with 'K' and 'V' of shape (n_loops, N, H, d).

        For Sandwich: prepends first-block entries and appends last-block entries
        so the temporal dimension covers the full forward pass.
        """
        result = {}
        for idx in self._K:
            K_list = (
                self._prefix_K.get(idx, [])
                + self._K[idx]
                + self._suffix_K.get(idx, [])
            )
            V_list = (
                self._prefix_V.get(idx, [])
                + self._V[idx]
                + self._suffix_V.get(idx, [])
            )
            if not K_list:
                continue
            result[idx] = {
                "K": torch.stack(K_list, dim=0),  # (n_loops, N, H, d)
                "V": torch.stack(V_list, dim=0),
            }
        return result


# ---------------------------------------------------------------------------
# Rank metric computation
# ---------------------------------------------------------------------------

def _compute_temporal_rank(
    kv_loop: torch.Tensor,   # (n_loops, N_train, H, d_head)
) -> dict[str, np.ndarray | float]:
    """Analyse KV rank in the *loop* dimension.

    Flatten spatial dims → M ∈ R^{n_loops × (N*H*d)}.
    Rank analysis measures how many independent directions KV traverses
    across loop iterations.
    """
    n_loops = kv_loop.shape[0]
    D = math.prod(kv_loop.shape[1:])
    M = kv_loop.reshape(n_loops, D).float()   # (n_loops, D)
    return _svd_metrics(M)


def _compute_per_dim_temporal_variance(
    kv_loop: torch.Tensor,   # (n_loops, N_train, H, d_head)
) -> dict[str, np.ndarray | float]:
    """Per-feature-dimension temporal variance analysis.

    For each of the H*d_head dimensions, compute how much that dimension
    varies across loop iterations (averaged over tokens).

    This answers: are there a few dimensions that carry all the "loop clock"
    information while the rest are temporally stable?

    Strategy
    --------
    1. Average over N_train tokens: K_avg ∈ R^{n_loops × (H*d_head)}.
    2. For each dimension j: temporal_var[j] = Var_loops(K_avg[:, j]).
    3. Normalise: dim_var_frac[j] = temporal_var[j] / sum(temporal_var).
    4. Sort descending → cumulative EVR over feature dimensions.

    Returns
    -------
    per_dim_var_frac        (H*d_head,) sorted descending — each dim's share
                            of total temporal variance.
    per_dim_var_evr         (H*d_head,) cumulative EVR sorted descending.
    per_dim_n_dims_50       min k s.t. top-k dims capture ≥ 50% of temporal var
    per_dim_n_dims_90
    per_dim_n_dims_95
    per_dim_n_dims_99
    per_dim_top1_frac       variance fraction of the single most variable dim
    per_dim_stable_rank     stable rank = 1 / sum(frac²)  (H*d_head analogue)
    """
    n_loops, N, H, d = kv_loop.shape
    D = H * d
    # (n_loops, D): token-averaged keys
    K_avg = kv_loop.float().reshape(n_loops, N, D).mean(dim=1)   # (n_loops, D)
    # Centre across loops
    K_c = K_avg - K_avg.mean(dim=0, keepdim=True)                # (n_loops, D)
    # Per-dim variance across loops
    temporal_var = (K_c ** 2).mean(dim=0)                         # (D,)

    total = float(temporal_var.sum()) + 1e-30
    frac = (temporal_var / total).cpu().numpy()                   # (D,) unsorted
    frac_sorted = np.sort(frac)[::-1]                             # descending
    evr = np.cumsum(frac_sorted)

    def _n_for(thresh):
        idx = np.searchsorted(evr, thresh)
        return int(min(idx + 1, D))

    # Stable rank in feature-dimension space: 1 / Σ frac²
    # equals H*d_head when all dims have equal variance, 1 when one dim dominates
    sr = float(1.0 / ((frac_sorted ** 2).sum() + 1e-30))

    return {
        "per_dim_var_frac":   frac_sorted.astype(np.float32),    # (H*d_head,)
        "per_dim_var_evr":    evr.astype(np.float32),            # (H*d_head,)
        "per_dim_n_dims_50":  _n_for(0.50),
        "per_dim_n_dims_90":  _n_for(0.90),
        "per_dim_n_dims_95":  _n_for(0.95),
        "per_dim_n_dims_99":  _n_for(0.99),
        "per_dim_top1_frac":  float(frac_sorted[0]),
        "per_dim_stable_rank": sr,
    }


def _compute_spatial_rank_per_loop(
    kv_loop: torch.Tensor,   # (n_loops, N_train, H, d_head)
) -> np.ndarray:
    """At each loop, compute stable rank of K ∈ R^{N_train × (H*d_head)}.

    Returns array of shape (n_loops,).
    """
    n_loops, N, H, d = kv_loop.shape
    sr_list = []
    for i in range(n_loops):
        M_i = kv_loop[i].reshape(N, H * d).float()   # (N, H*d)
        sr_list.append(_stable_rank(M_i))
    return np.array(sr_list, dtype=np.float32)


def _compute_delta_rank(
    kv_loop: torch.Tensor,   # (n_loops, N_train, H, d_head)
) -> dict[str, np.ndarray | float]:
    """Rank of the *change* matrix: stack K^{(i)} - K^{(i-1)} across loops.

    This answers: how many independent directions does the KV cache *update*
    in?  If low rank, cache updates are even more constrained than the cache
    values themselves.
    """
    deltas = kv_loop[1:] - kv_loop[:-1]   # (n_loops-1, N, H, d)
    n_deltas = deltas.shape[0]
    D = math.prod(deltas.shape[1:])
    M = deltas.reshape(n_deltas, D).float()
    return _svd_metrics(M)


def _compute_delta_from_first_rank(
    kv_loop: torch.Tensor,   # (n_loops, N_train, H, d_head)
) -> np.ndarray:
    """Spatial stable rank of (KV_i − KV_0) for each step i = 1..n_loops-1.

    KV_0 is treated as the cached reference (what a looped model could reuse).
    For each subsequent step, we measure the rank of the deviation from this
    reference.

      low rank(KV_i − KV_0)  →  the update is a low-rank correction → compressible
      high rank(KV_i − KV_0) →  the update is full-rank → not compressible

    Returns array of shape (n_loops-1,).
    """
    N, H, d = kv_loop.shape[1], kv_loop.shape[2], kv_loop.shape[3]
    kv0 = kv_loop[0]                          # (N, H, d)
    ranks = []
    for i in range(1, kv_loop.shape[0]):
        delta = (kv_loop[i] - kv0).reshape(N, H * d).float()
        ranks.append(_stable_rank(delta))
    return np.array(ranks, dtype=np.float32)


def _compute_delta_from_first_magnitude(
    kv_loop: torch.Tensor,   # (n_loops, N_train, H, d_head)
) -> dict[str, np.ndarray]:
    """Per-token delta magnitude, cosine similarity, and approximation error vs KV_0.

      rel_norm_mean       mean over tokens of ‖KV_i_tok − KV_0_tok‖ / ‖KV_0_tok‖
                          overall magnitude of change relative to the base.
                          Looped: consistently smaller than unrolled.

      approx_error        ‖KV_i − KV_0‖_F / ‖KV_i‖_F
                          relative error of using KV_0 as a proxy for KV_i.
                          Looped: should stay low (KV_0 is a good approximation).
                          Unrolled: ≈ 1 (KV_0 tells you nothing about KV_i).

      cos_sim_with_first  mean per-token cosine similarity between KV_i and KV_0.
                          Looped: KV converges → should stay high / rise across loops.
                          Unrolled: ≈ 0 (independent weight matrices, random alignment).

      seq_delta_rel_norm  ‖KV_i − KV_{i-1}‖_F / ‖KV_{i-1}‖_F (sequential change).
                          Looped: should shrink across loops (convergence).
                          Unrolled: no trend.

      gini                Gini coefficient of per-token ‖KV_i_tok − KV_0_tok‖ norms.
                          Higher = change more concentrated on a few tokens.

    Returns dict with arrays of shape (n_loops-1,).
    """
    N, H, d = kv_loop.shape[1], kv_loop.shape[2], kv_loop.shape[3]
    kv0_flat = kv_loop[0].float().reshape(N, H * d)   # (N, D)
    base_tok_norm = kv0_flat.norm(dim=1)               # (N,)
    kv0_norm_F    = float(kv0_flat.norm())

    rel_norm_mean_list, approx_error_list = [], []
    cos_sim_list, seq_delta_list, gini_list = [], [], []

    prev_flat = kv0_flat
    for i in range(1, kv_loop.shape[0]):
        curr_flat = kv_loop[i].float().reshape(N, H * d)   # (N, D)

        # -- delta from first --
        delta_flat = curr_flat - kv0_flat                   # (N, D)
        tok_delta_norm = delta_flat.norm(dim=1)             # (N,)
        rel_norm = (tok_delta_norm / (base_tok_norm + 1e-30)).cpu().numpy()
        rel_norm_mean_list.append(float(rel_norm.mean()))

        # -- approx error: how good is KV_0 as a proxy for KV_i? --
        curr_norm_F = float(curr_flat.norm())
        approx_error_list.append(float(delta_flat.norm()) / (curr_norm_F + 1e-30))

        # -- cosine similarity with KV_0, per token, then averaged --
        cos = torch.nn.functional.cosine_similarity(curr_flat, kv0_flat, dim=1)  # (N,)
        cos_sim_list.append(float(cos.mean()))

        # -- sequential delta: how much did KV change since previous loop? --
        seq_delta = (curr_flat - prev_flat).norm()
        seq_delta_list.append(float(seq_delta) / (float(prev_flat.norm()) + 1e-30))
        prev_flat = curr_flat

        # -- Gini of per-token delta norms --
        s = np.sort(rel_norm)
        n = len(s)
        gini = float((2 * (np.arange(1, n + 1) * s).sum() / (n * s.sum() + 1e-30)) - (n + 1) / n)
        gini_list.append(gini)

    return {
        "rel_norm_mean":      np.array(rel_norm_mean_list, dtype=np.float32),
        "approx_error":       np.array(approx_error_list,  dtype=np.float32),
        "cos_sim_with_first": np.array(cos_sim_list,        dtype=np.float32),
        "seq_delta_rel_norm": np.array(seq_delta_list,      dtype=np.float32),
        "gini":               np.array(gini_list,           dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# KV compression helpers
# ---------------------------------------------------------------------------


def _low_rank_approx(M: torch.Tensor, rank: int) -> torch.Tensor:
    """Return rank-k SVD approximation of M ∈ R^{m × n} (float32 internally)."""
    r = min(rank, M.shape[0], M.shape[1])
    M_f = M.float()
    U, S, Vh = torch.linalg.svd(M_f, full_matrices=False)
    return ((U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r]).to(M.dtype)


class _KVCompressionHooks:
    """Registers output hooks on every ICL attention k/v_projection to apply
    rank-k truncated SVD before the projections are used in attention.

    Covers the main axial_embedder and, for Sandwich, first/last embedders.
    """

    def __init__(self, rank: int):
        self._rank = rank
        self._hooks: list = []

    def _make_hook(self):
        rank = self._rank

        def hook(module, input, output):
            # output: (B, N_train, D_kv)
            B, N, D = output.shape
            approx = torch.stack([_low_rank_approx(output[b], rank) for b in range(B)])
            return approx

        return hook

    def register(self, model: "LoopICL") -> None:
        for emb_attr in ("axial_embedder", "first_axial_embedder", "last_axial_embedder"):
            emb = getattr(model, emb_attr, None)
            if emb is None:
                continue
            for block in emb.icl_blocks:
                attn = block.icl_attention
                self._hooks.append(attn.k_projection.register_forward_hook(self._make_hook()))
                self._hooks.append(attn.v_projection.register_forward_hook(self._make_hook()))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class _KVReuseHooks:
    """Freeze K/V at the first call (loop 1 / block 0) and replay it for all
    subsequent calls on the same projection module.

    For a looped model each projection fires n_loops times → loop 1's output
    is cached and returned for loops 2-N.
    For an unrolled model each projection fires once per block → block 0's
    output is injected into every subsequent block via separate hooks.

    This lets us test whether the KV produced at the first pass is a good
    approximation for later passes.  For a looped model (shared weights) it
    should be; for an unrolled model (independent weights) it should not.
    """

    def __init__(self):
        self._hooks: list = []
        self._cache: dict[int, torch.Tensor] = {}   # module id → cached output
        self._call_count: dict[int, int] = {}        # module id → call counter

    def _make_hook(self, mod_id: int):
        def hook(module, input, output):
            count = self._call_count.get(mod_id, 0)
            if count == 0:
                self._cache[mod_id] = output.detach()
            self._call_count[mod_id] = count + 1
            return self._cache[mod_id]
        return hook

    def _make_inject_hook(self, src_mod_id: int):
        """Injects block-0's cached output into a later block (unrolled case)."""
        def hook(module, input, output):
            if src_mod_id in self._cache:
                return self._cache[src_mod_id]
            return output
        return hook

    def register(self, model: "LoopICL") -> None:
        for emb_attr in ("axial_embedder", "first_axial_embedder", "last_axial_embedder"):
            emb = getattr(model, emb_attr, None)
            if emb is None:
                continue
            blocks = list(emb.icl_blocks)
            for b_idx, block in enumerate(blocks):
                attn = block.icl_attention
                for proj in (attn.k_projection, attn.v_projection):
                    mod_id = id(proj)
                    if b_idx == 0:
                        # Block 0 / loop 1: capture on first call, replay after.
                        # Covers both looped (same module called n_loops times)
                        # and unrolled (called once, cached for injection below).
                        h = proj.register_forward_hook(self._make_hook(mod_id))
                        self._hooks.append(h)
                    else:
                        # Unrolled: inject block-0's cached K/V into this block.
                        b0_proj = (
                            blocks[0].icl_attention.k_projection
                            if proj is attn.k_projection
                            else blocks[0].icl_attention.v_projection
                        )
                        h = proj.register_forward_hook(self._make_inject_hook(id(b0_proj)))
                        self._hooks.append(h)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._cache.clear()
        self._call_count.clear()


def forward_with_kv_reuse(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int,
    num_loops: int | None = None,
) -> torch.Tensor:
    """Forward pass where every loop/block reuses the K/V from the first pass.

    For a looped model (shared weights) loop 1's K/V is a meaningful
    approximation of later loops', so accuracy should degrade only slightly.
    For an unrolled model (independent weights per block) block 0's K/V is
    semantically unrelated to block 4's, so accuracy should collapse.

    This contrast demonstrates the looped model's structural KV-reuse advantage.
    """
    assert not model.training, "forward_with_kv_reuse is for inference only"
    n_loops = num_loops if num_loops is not None else model.num_icl_loops

    hooks = _KVReuseHooks()
    hooks.register(model)
    try:
        logits = model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)
    finally:
        hooks.remove()
    return logits


class _KVCollectHooks:
    """Collect K/V projections at every call without modifying them.

    Used in the first pass of forward_with_kv_mean_reuse to accumulate
    per-module outputs so the cross-loop/block global mean can be computed
    before the injection pass.
    """

    def __init__(self):
        self._hooks: list = []
        self._outputs: dict[int, list[torch.Tensor]] = {}
        self._k_mod_ids: list[int] = []
        self._v_mod_ids: list[int] = []

    def _make_hook(self, mod_id: int):
        outputs = self._outputs
        def hook(module, input, output):
            outputs[mod_id].append(output.detach())
        return hook

    def register(self, model: "LoopICL") -> None:
        for emb_attr in ("axial_embedder", "first_axial_embedder", "last_axial_embedder"):
            emb = getattr(model, emb_attr, None)
            if emb is None:
                continue
            for block in emb.icl_blocks:
                attn = block.icl_attention
                for proj, mod_id_list in (
                    (attn.k_projection, self._k_mod_ids),
                    (attn.v_projection, self._v_mod_ids),
                ):
                    mod_id = id(proj)
                    if mod_id not in self._outputs:
                        self._outputs[mod_id] = []
                        mod_id_list.append(mod_id)
                        self._hooks.append(proj.register_forward_hook(self._make_hook(mod_id)))

    def global_mean_kv(self) -> dict[int, torch.Tensor]:
        """Return global mean K/V: every k_proj mod_id → mean over all K calls
        from all blocks/loops, and similarly for V.

        For the looped model (1 shared block called N times) this is the mean
        of K_1…K_N.  For the unrolled model (N independent blocks, each called
        once) this averages all 8 blocks' K/V into a single tensor, so every
        block receives the same global mean rather than its own output.
        """
        all_k = [t for mid in self._k_mod_ids for t in self._outputs[mid]]
        all_v = [t for mid in self._v_mod_ids for t in self._outputs[mid]]
        gm_k = torch.stack(all_k).mean(0) if all_k else None
        gm_v = torch.stack(all_v).mean(0) if all_v else None
        result: dict[int, torch.Tensor] = {}
        for mid in self._k_mod_ids:
            result[mid] = gm_k
        for mid in self._v_mod_ids:
            result[mid] = gm_v
        return result

    def last_kv(self) -> dict[int, torch.Tensor]:
        """Return the last captured K/V for every module.

        For the looped model (1 shared block called N times) this is K_N — the
        most refined K/V after all loops have run.  For the unrolled model
        (N independent blocks each called once) this is identical to the
        per-block output, so it behaves the same as global_mean_kv when each
        block fires only once.
        """
        all_k_last = [self._outputs[mid][-1] for mid in self._k_mod_ids if self._outputs[mid]]
        all_v_last = [self._outputs[mid][-1] for mid in self._v_mod_ids if self._outputs[mid]]
        # global last: for looped model this is K_N; for unrolled we average
        # the single-call outputs of all blocks (same as global_mean for unrolled)
        lk = torch.stack(all_k_last).mean(0) if all_k_last else None
        lv = torch.stack(all_v_last).mean(0) if all_v_last else None
        result: dict[int, torch.Tensor] = {}
        for mid in self._k_mod_ids:
            result[mid] = lk
        for mid in self._v_mod_ids:
            result[mid] = lv
        return result

    def remove(self) -> None:
        """Remove all registered hooks. Collected data is preserved for global_mean_kv()."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class _KVInjectHooks:
    """Inject a fixed precomputed tensor for every call to a given module."""

    def __init__(self, fixed: dict[int, torch.Tensor]):
        self._fixed = fixed
        self._hooks: list = []

    def _make_hook(self, mod_id: int):
        fixed = self._fixed
        def hook(module, input, output):
            return fixed[mod_id]
        return hook

    def register(self, model: "LoopICL") -> None:
        for emb_attr in ("axial_embedder", "first_axial_embedder", "last_axial_embedder"):
            emb = getattr(model, emb_attr, None)
            if emb is None:
                continue
            for block in emb.icl_blocks:
                attn = block.icl_attention
                for proj in (attn.k_projection, attn.v_projection):
                    mod_id = id(proj)
                    if mod_id in self._fixed:
                        self._hooks.append(proj.register_forward_hook(self._make_hook(mod_id)))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def forward_with_kv_mean_reuse(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int,
    num_loops: int | None = None,
) -> torch.Tensor:
    """Forward pass where every loop/block uses the global mean K/V.

    Pass 1 (collect): run the full forward and record every K/V projection
    output across all blocks and loops.  Compute a single global mean K tensor
    (mean over all k_projection calls, all blocks, all loops) and a global
    mean V tensor similarly.

    Pass 2 (inject): run the full forward again, replacing every k_projection
    and v_projection output with the corresponding global mean.

    This is a fair comparison across model types:
    - Looped model (1 shared block × N loops): global mean = mean of K_1…K_N.
    - Unrolled model (N independent blocks × 1 call each): global mean = mean
      of all 8 blocks' K/V — each block now receives a foreign average rather
      than its own output, so the drop is non-trivial.
    """
    assert not model.training, "forward_with_kv_mean_reuse is for inference only"
    n_loops = num_loops if num_loops is not None else model.num_icl_loops

    # Pass 1: collect
    collector = _KVCollectHooks()
    collector.register(model)
    try:
        with torch.no_grad():
            model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)
    finally:
        collector.remove()

    mean_kv = collector.global_mean_kv()

    # Pass 2: inject mean
    injector = _KVInjectHooks(mean_kv)
    injector.register(model)
    try:
        with torch.no_grad():
            logits = model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)
    finally:
        injector.remove()

    return logits


class _KVDimSplitHooks:
    """Two-pass dim-split KV reuse with a *global* reference.

    The reference is always the **first** K (or V) projection call in the
    entire forward pass — loop 1 for the looped model, block 0 for the
    unrolled model.  All subsequent calls receive the reference's stable-dim
    values, regardless of which module fired them.

    This makes the comparison fair across architectures:

    * Looped  (1 block × N loops): reference = loop-1.  Stable dims are those
      that barely change across the 8 loop iterations of the *same* weights.
      These dims truly converge, so freezing them is cheap.

    * Unrolled (N blocks × 1 call): reference = block-0.  Stable dims are
      identified from cross-block variance.  Blocks 1-7 get block-0's stable-
      dim values injected.  Because blocks have independent weights the
      "stable dims" are only superficially similar, so the drop will be larger.

    Pass 1 (collect): record ALL K/V projection outputs (globally ordered).
    Compute stable mask from variance of each call relative to the first call.

    Pass 2 (inject): first global call is cached; every subsequent call gets
    its stable dims replaced with the cached first-call value.
    """

    def __init__(self, stable_frac: float = 0.5):
        self._stable_frac = stable_frac
        self._hooks: list = []
        # collect mode: accumulate all outputs in call order
        self._k_outputs: list[torch.Tensor] = []
        self._v_outputs: list[torch.Tensor] = []
        # inject mode: global counters + reference + mask
        self._k_call_count: int = 0
        self._v_call_count: int = 0
        self._k_first: torch.Tensor | None = None   # (B, N, D_k)
        self._v_first: torch.Tensor | None = None
        self._k_stable_mask: torch.Tensor | None = None  # (D_k,) bool
        self._v_stable_mask: torch.Tensor | None = None
        # dedup: only register one hook per unique module
        self._registered_ids: set[int] = set()
        self._mode: str = "collect"

    def _make_k_hook(self):
        def hook(module, input, output):
            if self._mode == "collect":
                self._k_outputs.append(output.detach())
                return None
            # inject mode
            if self._k_call_count == 0:
                self._k_first = output.detach()
                self._k_call_count += 1
                return None
            self._k_call_count += 1
            if self._k_stable_mask is None:
                return None
            result = output.clone()
            result[:, :, self._k_stable_mask] = self._k_first[:, :, self._k_stable_mask]
            return result
        return hook

    def _make_v_hook(self):
        def hook(module, input, output):
            if self._mode == "collect":
                self._v_outputs.append(output.detach())
                return None
            if self._v_call_count == 0:
                self._v_first = output.detach()
                self._v_call_count += 1
                return None
            self._v_call_count += 1
            if self._v_stable_mask is None:
                return None
            result = output.clone()
            result[:, :, self._v_stable_mask] = self._v_first[:, :, self._v_stable_mask]
            return result
        return hook

    def compute_stable_masks(self) -> None:
        """Derive global stable masks from variance relative to the first call."""
        for outputs, attr in (
            (self._k_outputs, "_k_stable_mask"),
            (self._v_outputs, "_v_stable_mask"),
        ):
            if len(outputs) < 2:
                continue
            stacked = torch.stack(outputs).float()   # (n_calls, B, N, D)
            ref = stacked[0]                          # first call (loop-1 / block-0)
            deltas = stacked[1:] - ref.unsqueeze(0)  # (n_calls-1, B, N, D)
            # mean squared deviation from the reference, averaged over calls, B, N
            var = (deltas ** 2).mean(dim=[0, 1, 2])  # (D,)
            thresh = float(torch.quantile(var, self._stable_frac))
            setattr(self, attr, var <= thresh)

    def switch_to_inject(self) -> None:
        self._mode = "inject"
        self._k_call_count = 0
        self._v_call_count = 0

    def register(self, model: "LoopICL") -> None:
        for emb_attr in ("axial_embedder", "first_axial_embedder", "last_axial_embedder"):
            emb = getattr(model, emb_attr, None)
            if emb is None:
                continue
            for block in emb.icl_blocks:
                attn = block.icl_attention
                if id(attn.k_projection) not in self._registered_ids:
                    self._registered_ids.add(id(attn.k_projection))
                    self._hooks.append(
                        attn.k_projection.register_forward_hook(self._make_k_hook())
                    )
                if id(attn.v_projection) not in self._registered_ids:
                    self._registered_ids.add(id(attn.v_projection))
                    self._hooks.append(
                        attn.v_projection.register_forward_hook(self._make_v_hook())
                    )

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def forward_with_kv_dim_split_reuse(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int,
    stable_frac: float = 0.5,
    num_loops: int | None = None,
) -> torch.Tensor:
    """Forward where the stable fraction of K/V dims are frozen after loop 1.

    Pass 1 (profile): full forward, record K/V outputs to find which dims
    change little across loops (temporal variance below stable_frac quantile).

    Pass 2 (inject): run the full forward again.  At each loop, the stable
    dims of K/V are replaced with loop-1's values; dynamic dims are computed
    normally.

    stable_frac=0.5 → freeze the 50% least-varying K/V dims.
    stable_frac=0.0 → no reuse (identical to normal forward).
    stable_frac=1.0 → freeze all dims (identical to kv_reuse).
    """
    assert not model.training
    n_loops = num_loops if num_loops is not None else model.num_icl_loops

    hooks = _KVDimSplitHooks(stable_frac=stable_frac)
    hooks.register(model)

    # Pass 1: collect
    with torch.no_grad():
        model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)

    hooks.compute_stable_masks()
    hooks.switch_to_inject()

    # Pass 2: inject
    with torch.no_grad():
        logits = model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)

    hooks.remove()
    return logits


def forward_with_kv_last_reuse(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int,
    num_loops: int | None = None,
) -> torch.Tensor:
    """Forward pass where every loop/block uses the last-loop K/V.

    Pass 1 (collect): run the full forward and record every K/V projection
    output.  Take the *last* captured output for each unique module — for the
    looped model this is K_N (the most refined K/V after all loops), for the
    unrolled model it is each block's single output (same as the mean since
    each fires once).

    Pass 2 (inject): run the full forward again replacing every k_projection
    and v_projection with the last-loop K/V from pass 1.

    Rationale: K_N is the "converged" K/V state.  If the model learns to
    refine K/V across loops, K_N should be a better proxy for all loops than
    K_1 (used by kv_reuse) or the mean (used by kv_mean_reuse).
    """
    assert not model.training, "forward_with_kv_last_reuse is for inference only"
    n_loops = num_loops if num_loops is not None else model.num_icl_loops

    # Pass 1: collect
    collector = _KVCollectHooks()
    collector.register(model)
    try:
        with torch.no_grad():
            model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)
    finally:
        collector.remove()

    last_kv = collector.last_kv()

    # Pass 2: inject last K/V
    injector = _KVInjectHooks(last_kv)
    injector.register(model)
    try:
        with torch.no_grad():
            logits = model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)
    finally:
        injector.remove()

    return logits


def forward_with_kv_compression(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int,
    rank_k: int,
    num_loops: int | None = None,
) -> torch.Tensor:
    """Standard LoopICL forward pass with rank-k compressed K and V.

    At every ICL attention call, K and V are replaced by their rank-k
    truncated-SVD approximation before the dot-product attention.  All other
    model behaviour (residuals, MLP, layer norm, etc.) is unchanged.

    Parameters
    ----------
    model           : trained LoopICL in eval() mode
    X               : (B, R, C)
    y_train         : (B, R)
    single_eval_pos : number of training rows
    rank_k          : SVD truncation rank applied to every K and V matrix
    num_loops       : overrides model default

    Returns
    -------
    logits : (B, n_test, n_classes) — same shape as uncompressed forward
    """
    assert not model.training, "forward_with_kv_compression is for inference only"
    n_loops = num_loops if num_loops is not None else model.num_icl_loops

    hooks = _KVCompressionHooks(rank_k)
    hooks.register(model)
    try:
        logits = model(X, y_train, single_eval_pos=single_eval_pos, num_loops=n_loops)
    finally:
        hooks.remove()
    return logits


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def forward_with_kv_rank_metrics(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int | None = None,
    train_size: int | None = None,
    num_loops: int | None = None,
) -> tuple[torch.Tensor, dict[str, np.ndarray | float]]:
    """LoopICL forward pass that captures KV projections at each loop and
    computes low-rank statistics.

    Parameters
    ----------
    model           : trained LoopICL in eval() mode
    X               : (B, R, C)
    y_train         : (B, R)
    single_eval_pos : number of training rows
    num_loops       : overrides model default

    Returns
    -------
    logits          : (B, T, n_classes) — identical to normal forward
    metrics         : dict of rank statistics (see module docstring)

    Metric keys
    -----------
    K_temporal_{block}_{key}              per-block temporal rank metrics
    V_temporal_{block}_{key}
    K_delta_{block}_{key}                 rank of loop-to-loop *changes* in K
    V_delta_{block}_{key}
    K_spatial_stable_rank                 (n_loops, n_blocks) — spatial rank per loop
    V_spatial_stable_rank                 (n_loops, n_blocks)
    K_delta_from_first_stable_rank_mean   (n_loops-1,) — rank(KV_i − KV_0) per step
    V_delta_from_first_stable_rank_mean   (n_loops-1,)
    *_mean                                aggregated (mean) over blocks
    """
    assert not model.training, "forward_with_kv_rank_metrics is for inference only"

    x_RBC  = X.transpose(0, 1)
    y_NB   = y_train.transpose(0, 1)
    num_train = (
        int(single_eval_pos) if single_eval_pos is not None
        else int(train_size)  if train_size  is not None
        else int(y_NB.shape[0])
    )
    n_loops = num_loops if num_loops is not None else model.num_icl_loops

    import time as _time

    # Register KV capture hooks
    _t = _time.time()
    capture = _KVCapture(model, num_train)
    print(f"          [kv_rank] hook register: {_time.time()-_t:.3f}s  n_blocks={len(capture._K)}")

    try:
        # Run standard forward pass — hooks fire inside the loops
        _t = _time.time()
        logits = model(
            X, y_train,
            single_eval_pos=num_train,
            num_loops=n_loops,
        )
        print(f"          [kv_rank] model forward: {_time.time()-_t:.3f}s")
    finally:
        capture.remove()

    _t = _time.time()
    kv_by_block = capture.get_loop_matrices()   # {block_idx: {"K": ..., "V": ...}}
    print(f"          [kv_rank] get_loop_matrices: {_time.time()-_t:.3f}s")

    # --- Unrolled: stack blocks as temporal dimension ---
    # Detect: multiple blocks each with exactly 1 temporal entry → unrolled model.
    # Treat block index as the "loop" dimension: block_0, block_1, ... → (n_blocks, N, H, d).
    if kv_by_block:
        n_temporal_per_block = [v["K"].shape[0] for v in kv_by_block.values()]
        if max(n_temporal_per_block) == 1 and len(kv_by_block) > 1:
            K_cat = torch.cat([kv_by_block[i]["K"] for i in sorted(kv_by_block)], dim=0)
            V_cat = torch.cat([kv_by_block[i]["V"] for i in sorted(kv_by_block)], dim=0)
            kv_by_block = {0: {"K": K_cat, "V": V_cat}}
            print(f"          [kv_rank] unrolled mode: stacked {len(n_temporal_per_block)} blocks → temporal {K_cat.shape[0]}")

    # ------------------------------------------------------------------
    # Compute rank metrics
    # ------------------------------------------------------------------
    metrics: dict[str, np.ndarray | float] = {}

    n_blocks = len(kv_by_block)

    # Per-tensor-type aggregation buffers
    K_temporal_sr:   list[float]        = []
    V_temporal_sr:   list[float]        = []
    K_temporal_evr:  list[np.ndarray]   = []
    V_temporal_evr:  list[np.ndarray]   = []
    K_temporal_sv:   list[np.ndarray]   = []
    V_temporal_sv:   list[np.ndarray]   = []
    K_delta_sr:      list[float]        = []
    V_delta_sr:      list[float]        = []
    K_spatial_sr_per_loop: list[np.ndarray] = []
    V_spatial_sr_per_loop: list[np.ndarray] = []
    K_delta_from_first_sr: list[np.ndarray] = []
    V_delta_from_first_sr: list[np.ndarray] = []
    K_delta_from_first_gini: list[np.ndarray] = []
    V_delta_from_first_gini: list[np.ndarray] = []
    K_delta_from_first_rel_norm: list[np.ndarray] = []
    V_delta_from_first_rel_norm: list[np.ndarray] = []
    K_delta_from_first_approx_error: list[np.ndarray] = []
    V_delta_from_first_approx_error: list[np.ndarray] = []
    K_delta_from_first_cos_sim: list[np.ndarray] = []
    V_delta_from_first_cos_sim: list[np.ndarray] = []
    K_delta_from_first_seq_delta: list[np.ndarray] = []
    V_delta_from_first_seq_delta: list[np.ndarray] = []
    K_per_dim_sr:  list[float] = []
    V_per_dim_sr:  list[float] = []
    K_per_dim_evr: list[np.ndarray] = []
    V_per_dim_evr: list[np.ndarray] = []
    K_per_dim_n90: list[int] = []
    V_per_dim_n90: list[int] = []
    K_per_dim_top1: list[float] = []
    V_per_dim_top1: list[float] = []

    _t = _time.time()
    for block_idx, tensors in sorted(kv_by_block.items()):
        for name, kv_loop in tensors.items():   # "K" or "V"
            prefix = f"{name}_b{block_idx}_"

            # --- temporal rank ---
            t_metrics = _compute_temporal_rank(kv_loop)
            for k, v in t_metrics.items():
                metrics[f"{prefix}temporal_{k}"] = v

            # --- per-dimension temporal variance ---
            pd_metrics = _compute_per_dim_temporal_variance(kv_loop)
            for k, v in pd_metrics.items():
                metrics[f"{prefix}{k}"] = v

            # --- delta rank ---
            if kv_loop.shape[0] > 1:
                d_metrics = _compute_delta_rank(kv_loop)
                for k, v in d_metrics.items():
                    metrics[f"{prefix}delta_{k}"] = v
                if name == "K":
                    K_delta_sr.append(d_metrics["stable_rank"])
                else:
                    V_delta_sr.append(d_metrics["stable_rank"])

                # --- delta-from-first rank: rank(KV_i - KV_0) per step ---
                dfr = _compute_delta_from_first_rank(kv_loop)
                metrics[f"{prefix}delta_from_first_stable_rank"] = dfr  # (n_loops-1,)

                # --- delta-from-first magnitude: how large are the changes? ---
                dfm = _compute_delta_from_first_magnitude(kv_loop)
                for k, v in dfm.items():
                    metrics[f"{prefix}delta_from_first_{k}"] = v          # (n_loops-1,)

                if name == "K":
                    K_delta_from_first_sr.append(dfr)
                    K_delta_from_first_gini.append(dfm["gini"])
                    K_delta_from_first_rel_norm.append(dfm["rel_norm_mean"])
                    K_delta_from_first_approx_error.append(dfm["approx_error"])
                    K_delta_from_first_cos_sim.append(dfm["cos_sim_with_first"])
                    K_delta_from_first_seq_delta.append(dfm["seq_delta_rel_norm"])
                else:
                    V_delta_from_first_sr.append(dfr)
                    V_delta_from_first_gini.append(dfm["gini"])
                    V_delta_from_first_rel_norm.append(dfm["rel_norm_mean"])
                    V_delta_from_first_approx_error.append(dfm["approx_error"])
                    V_delta_from_first_cos_sim.append(dfm["cos_sim_with_first"])
                    V_delta_from_first_seq_delta.append(dfm["seq_delta_rel_norm"])

            # --- spatial rank per loop ---
            sr_per_loop = _compute_spatial_rank_per_loop(kv_loop)
            metrics[f"{prefix}spatial_stable_rank"] = sr_per_loop

            # aggregation
            if name == "K":
                K_temporal_sr.append(t_metrics["stable_rank"])
                K_temporal_evr.append(t_metrics["evr"])
                K_temporal_sv.append(t_metrics["singular_values_norm"])
                K_spatial_sr_per_loop.append(sr_per_loop)
                K_per_dim_sr.append(pd_metrics["per_dim_stable_rank"])
                K_per_dim_evr.append(pd_metrics["per_dim_var_evr"])
                K_per_dim_n90.append(pd_metrics["per_dim_n_dims_90"])
                K_per_dim_top1.append(pd_metrics["per_dim_top1_frac"])
            else:
                V_temporal_sr.append(t_metrics["stable_rank"])
                V_temporal_evr.append(t_metrics["evr"])
                V_temporal_sv.append(t_metrics["singular_values_norm"])
                V_spatial_sr_per_loop.append(sr_per_loop)
                V_per_dim_sr.append(pd_metrics["per_dim_stable_rank"])
                V_per_dim_evr.append(pd_metrics["per_dim_var_evr"])
                V_per_dim_n90.append(pd_metrics["per_dim_n_dims_90"])
                V_per_dim_top1.append(pd_metrics["per_dim_top1_frac"])

    print(f"          [kv_rank] SVD metrics:    {_time.time()-_t:.3f}s  n_blocks={n_blocks}")

    # ------------------------------------------------------------------
    # Cross-block aggregates (shape = (n_blocks,) or scalar means)
    # ------------------------------------------------------------------
    if K_temporal_sr:
        metrics["K_temporal_stable_rank"] = np.array(K_temporal_sr)
        metrics["K_temporal_stable_rank_mean"] = float(np.mean(K_temporal_sr))
        metrics["K_temporal_evr"] = np.stack(K_temporal_evr)          # (n_blocks, n_loops)
        metrics["K_temporal_evr_mean"] = np.stack(K_temporal_evr).mean(0)  # (n_loops,)
        metrics["K_temporal_singular_values_norm"] = np.stack(K_temporal_sv)
        metrics["K_spatial_stable_rank"] = np.stack(K_spatial_sr_per_loop).T  # (n_loops, n_blocks)
        metrics["K_spatial_stable_rank_mean"] = np.stack(K_spatial_sr_per_loop).T.mean(1)  # (n_loops,)

    if V_temporal_sr:
        metrics["V_temporal_stable_rank"] = np.array(V_temporal_sr)
        metrics["V_temporal_stable_rank_mean"] = float(np.mean(V_temporal_sr))
        metrics["V_temporal_evr"] = np.stack(V_temporal_evr)
        metrics["V_temporal_evr_mean"] = np.stack(V_temporal_evr).mean(0)
        metrics["V_temporal_singular_values_norm"] = np.stack(V_temporal_sv)
        metrics["V_spatial_stable_rank"] = np.stack(V_spatial_sr_per_loop).T
        metrics["V_spatial_stable_rank_mean"] = np.stack(V_spatial_sr_per_loop).T.mean(1)

    if K_delta_sr:
        metrics["K_delta_stable_rank"] = np.array(K_delta_sr)
        metrics["K_delta_stable_rank_mean"] = float(np.mean(K_delta_sr))
    if V_delta_sr:
        metrics["V_delta_stable_rank"] = np.array(V_delta_sr)
        metrics["V_delta_stable_rank_mean"] = float(np.mean(V_delta_sr))

    # delta-from-first: mean across blocks → (n_loops-1,)
    if K_delta_from_first_sr:
        metrics["K_delta_from_first_stable_rank_mean"] = np.stack(K_delta_from_first_sr).mean(0)
        metrics["K_delta_from_first_gini_mean"]        = np.stack(K_delta_from_first_gini).mean(0)
        metrics["K_delta_from_first_rel_norm_mean"]    = np.stack(K_delta_from_first_rel_norm).mean(0)
        metrics["K_delta_from_first_approx_error_mean"]    = np.stack(K_delta_from_first_approx_error).mean(0)
        metrics["K_delta_from_first_cos_sim_mean"]         = np.stack(K_delta_from_first_cos_sim).mean(0)
        metrics["K_delta_from_first_seq_delta_rel_norm_mean"] = np.stack(K_delta_from_first_seq_delta).mean(0)
    if V_delta_from_first_sr:
        metrics["V_delta_from_first_stable_rank_mean"] = np.stack(V_delta_from_first_sr).mean(0)
        metrics["V_delta_from_first_gini_mean"]        = np.stack(V_delta_from_first_gini).mean(0)
        metrics["V_delta_from_first_rel_norm_mean"]    = np.stack(V_delta_from_first_rel_norm).mean(0)
        metrics["V_delta_from_first_approx_error_mean"]    = np.stack(V_delta_from_first_approx_error).mean(0)
        metrics["V_delta_from_first_cos_sim_mean"]         = np.stack(V_delta_from_first_cos_sim).mean(0)
        metrics["V_delta_from_first_seq_delta_rel_norm_mean"] = np.stack(V_delta_from_first_seq_delta).mean(0)

    if K_per_dim_sr:
        metrics["K_per_dim_stable_rank"]      = np.array(K_per_dim_sr)
        metrics["K_per_dim_stable_rank_mean"] = float(np.mean(K_per_dim_sr))
        metrics["K_per_dim_var_evr"]          = np.stack(K_per_dim_evr)   # (n_blocks, D)
        metrics["K_per_dim_var_evr_mean"]     = np.stack(K_per_dim_evr).mean(0)  # (D,)
        metrics["K_per_dim_n_dims_90"]        = np.array(K_per_dim_n90)
        metrics["K_per_dim_n_dims_90_mean"]   = float(np.mean(K_per_dim_n90))
        metrics["K_per_dim_top1_frac"]        = np.array(K_per_dim_top1)
        metrics["K_per_dim_top1_frac_mean"]   = float(np.mean(K_per_dim_top1))
    if V_per_dim_sr:
        metrics["V_per_dim_stable_rank"]      = np.array(V_per_dim_sr)
        metrics["V_per_dim_stable_rank_mean"] = float(np.mean(V_per_dim_sr))
        metrics["V_per_dim_var_evr"]          = np.stack(V_per_dim_evr)
        metrics["V_per_dim_var_evr_mean"]     = np.stack(V_per_dim_evr).mean(0)
        metrics["V_per_dim_n_dims_90"]        = np.array(V_per_dim_n90)
        metrics["V_per_dim_n_dims_90_mean"]   = float(np.mean(V_per_dim_n90))
        metrics["V_per_dim_top1_frac"]        = np.array(V_per_dim_top1)
        metrics["V_per_dim_top1_frac_mean"]   = float(np.mean(V_per_dim_top1))

    metrics["n_loops"] = n_loops
    metrics["n_icl_blocks"] = n_blocks
    metrics["num_train"] = num_train

    return logits, metrics
