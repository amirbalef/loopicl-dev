"""Custom LoopICL forward pass that computes hidden-state metrics and
per-loop logits in a single forward pass.

Two hidden-state streams × two splits (train / test) are tracked per loop:
  row stream   — x_BRD  (B, n, D)
  cell stream  — x_BRCE (B, n, C, E), column-mean-pooled to (B, n, E)

Only two extra tensors per stream×split (current + initial snapshot) are
kept alive between iterations; the previous snapshot is freed immediately
after metric computation.  No full hidden-state tensors are accumulated.

Metric keys  (prefix = {train|test}_{row|cell}_)
-------------------------------------------------
{p}cos_sim            cosine similarity between step i and step i-1
{p}l2_delta           L2 norm of the hidden-state delta
{p}norm_residual      l2_delta / norm(current state)   [convergence proxy]
{p}l2_from_init       L2 drift from the pre-loop initial state

CKA similarity (scalars, shape (n_loops,)):
{p}cka_prev           linear CKA between step i-1 and step i
{p}cka_init           linear CKA between initial state and step i

Train-only separation gap (prefix = train_{row|cell}_):
{p}sep_gap_euclidean  between_class - (within_class + 1e-3)  [euclidean]
{p}sep_gap_cosine     between_class - (within_class + 1e-3)  [cosine]
{p}between_class_euclidean / within_class_euclidean  (raw components)
{p}between_class_cosine    / within_class_cosine

Per-point arrays have shape (n_loops, n_points).
Aggregate "_mean" keys have shape (n_loops,).
Separation gap keys are scalars of shape (n_loops,).

Usage
-----
with torch.no_grad():
    logits, loop_logits, metrics = forward_with_hs_metrics(
        model, X, y_full, single_eval_pos=n_train, num_loops=6,
    )

# per-test-point:  metrics["test_row_l2_delta"]       shape (6, n_test)
# per-train-point: metrics["train_row_l2_delta"]       shape (6, n_train)
# aggregates:      metrics["test_row_l2_delta_mean"]   shape (6,)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from loopicl.models.loopicl.model import LoopICL


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _linear_cka(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    """Linear Centered Kernel Alignment between two (n, D) matrices.

    Returns a scalar in [0, 1]; higher = more similar representations.
    Reference: Kornblith et al., ICML 2019.
    """
    # Cast to float64 BEFORE any arithmetic: float32 mean on inf arrays gives nan.
    # Bail early if inputs are non-finite (hidden states have exploded).
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        return float("nan")
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    # Normalise so matrix products stay bounded — CKA is scale-invariant.
    a_norm = np.linalg.norm(A)
    b_norm = np.linalg.norm(B)
    if a_norm == 0 or b_norm == 0:
        return float("nan")
    A = A / a_norm
    B = B / b_norm
    numerator   = np.sum((A.T @ B) ** 2)
    denominator = (
        np.linalg.norm(A.T @ A, ord="fro") * np.linalg.norm(B.T @ B, ord="fro") + eps
    )
    return float(numerator / denominator)


def _stream_metrics(
    prev: torch.Tensor,   # (B, n, D) fp32
    curr: torch.Tensor,   # (B, n, D) fp32
    init: torch.Tensor,   # (B, n, D) fp32
    prefix: str,
) -> dict[str, np.ndarray]:
    """Per-point metrics between two consecutive hidden states.

    Averages over the batch dimension (B), keeping the point dimension (n).
    Returns numpy arrays of shape (n,) plus scalar "_mean" aggregates,
    plus CKA scalars (prev↔curr and init↔curr).
    """
    delta = curr - prev                                        # (B, n, D)

    l2_delta  = delta.norm(dim=-1).mean(dim=0).cpu().numpy()  # (n,)
    curr_norm = curr.norm(dim=-1).mean(dim=0).cpu().numpy()   # (n,)
    safe_norm = np.where(np.isfinite(curr_norm) & (curr_norm > 0), curr_norm, np.nan)
    with np.errstate(invalid='ignore'):
        norm_res = l2_delta / safe_norm
    cos       = F.cosine_similarity(prev, curr, dim=-1).mean(dim=0).cpu().numpy()  # (n,)
    drift     = (curr - init).norm(dim=-1).mean(dim=0).cpu().numpy()               # (n,)

    # CKA: batch-averaged (n, D) matrices
    prev_np = prev.mean(dim=0).cpu().numpy().astype(np.float32)
    curr_np = curr.mean(dim=0).cpu().numpy().astype(np.float32)
    init_np = init.mean(dim=0).cpu().numpy().astype(np.float32)
    cka_prev = _linear_cka(prev_np, curr_np)
    cka_init = _linear_cka(init_np, curr_np)

    return {
        f"{prefix}cos_sim":            cos,
        f"{prefix}cos_sim_mean":       cos.mean(),
        f"{prefix}l2_delta":           l2_delta,
        f"{prefix}l2_delta_mean":      l2_delta.mean(),
        f"{prefix}norm_residual":      norm_res,
        f"{prefix}norm_residual_mean": norm_res.mean(),
        f"{prefix}l2_from_init":       drift,
        f"{prefix}l2_from_init_mean":  drift.mean(),
        f"{prefix}cka_prev":           cka_prev,
        f"{prefix}cka_init":           cka_init,
    }


# ---------------------------------------------------------------------------
# Separation gap helper
# ---------------------------------------------------------------------------

def _separation_gap(
    emb: torch.Tensor,   # (B, n_train, D) fp32
    y_int: np.ndarray,   # (n_train,) integer labels
    max_samples: int = 200,
    prefix: str = "train_row_",
) -> dict[str, float]:
    """Separation gap between class clusters in embedding space.

    Computes ``between_class − (within_class + 1e-3)`` for both euclidean
    and cosine distances, following the definition used in
    ``util._pairwise_class_distances``.  Works for binary and multiclass.

    Parameters
    ----------
    emb : (B, n_train, D)
        Train hidden states (batch-averaged before distance computation).
    y_int : (n_train,) int
        Integer class labels aligned to the train rows.
    max_samples : int
        Per-class subsample cap (avoids O(n²) cost on large datasets).
    prefix : str
        Key prefix, e.g. ``"train_row_"`` or ``"train_cell_"``.

    Returns
    -------
    dict of scalars (one gap + two raw components, per distance metric)
    """
    from sklearn.metrics.pairwise import pairwise_distances as _pdist

    # Average over batch dimension → (n_train, D).
    # Use float64: float32 values can be finite yet their squares overflow in norm().
    emb_np = emb.mean(dim=0).cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(emb_np)):
        nan = float("nan")
        return {
            f"{prefix}sep_gap_euclidean":      nan, f"{prefix}between_class_euclidean": nan,
            f"{prefix}within_class_euclidean": nan, f"{prefix}sep_gap_cosine":          nan,
            f"{prefix}between_class_cosine":   nan, f"{prefix}within_class_cosine":     nan,
        }
    classes = np.unique(y_int)

    result: dict[str, float] = {}
    for metric in ("euclidean", "cosine"):
        if metric == "cosine":
            norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-12
            X = emb_np / norms
        else:
            X = emb_np

        # Per-class subsampled embeddings
        class_embs = []
        rng = np.random.default_rng(0)
        for c in classes:
            idx = np.where(y_int == c)[0]
            if len(idx) > max_samples:
                idx = rng.choice(idx, max_samples, replace=False)
            class_embs.append(X[idx])

        # Within-class distances
        within_vals = []
        for Xc in class_embs:
            if len(Xc) < 2:
                continue
            D = _pdist(Xc, metric=metric)
            within_vals.append(D[np.triu_indices_from(D, k=1)].mean())
        within = float(np.mean(within_vals)) if within_vals else 0.0

        # Between-class distances (all pairs of classes)
        n_cls = len(classes)
        between_vals = []
        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                D = _pdist(class_embs[i], class_embs[j], metric=metric)
                between_vals.append(D.mean())
        between = float(np.mean(between_vals)) if between_vals else 0.0

        result[f"{prefix}sep_gap_{metric}"]          = between - (within + 1e-3)
        result[f"{prefix}between_class_{metric}"]    = between
        result[f"{prefix}within_class_{metric}"]     = within

    return result


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _layer_sim_metrics(
    brd_states: list,        # list of n_blocks (B, R, D) CPU float tensors
    brce_states: list,       # list of n_blocks (B, R, E) CPU float tensors (col-pooled)
    loop_start_brd: torch.Tensor,   # (B, R, D) CPU float — state before this loop's axial step
    loop_start_brce: torch.Tensor,  # (B, R, E) CPU float
    num_train: int,
) -> dict[str, np.ndarray]:
    """Cosine similarity and CKA between consecutive axial-block hidden states.

    For each of the four streams (train/test × row/cell), computes per-block:
      layer_cos_sim  — cosine similarity vs. the previous block (block 0 vs. loop start)
      layer_cka_prev — linear CKA vs. the previous block
      layer_cka_init — linear CKA vs. the loop-start state

    Returns arrays of shape (n_blocks,).
    """
    results: dict[str, np.ndarray] = {}
    streams = [
        ("train_row_",  brd_states,  loop_start_brd,  0,         num_train),
        ("test_row_",   brd_states,  loop_start_brd,  num_train, None),
        ("train_cell_", brce_states, loop_start_brce, 0,         num_train),
        ("test_cell_",  brce_states, loop_start_brce, num_train, None),
    ]
    for prefix, states, loop_start, start, end in streams:
        cos_sims: list[float] = []
        cka_prevs: list[float] = []
        cka_inits: list[float] = []

        init_slice = loop_start[:, start:end]           # (B, n, D)
        init_np    = init_slice.mean(dim=0).numpy().astype(np.float32)
        prev       = init_slice

        for state in states:
            curr    = state[:, start:end]               # (B, n, D)
            cos     = F.cosine_similarity(prev, curr, dim=-1).mean(dim=0).numpy()
            cos_sims.append(float(cos.mean()))

            prev_np = prev.mean(dim=0).numpy().astype(np.float32)
            curr_np = curr.mean(dim=0).numpy().astype(np.float32)
            cka_prevs.append(_linear_cka(prev_np, curr_np))
            cka_inits.append(_linear_cka(init_np,  curr_np))
            prev = curr

        results[f"{prefix}layer_cos_sim"]  = np.array(cos_sims,  dtype=np.float32)
        results[f"{prefix}layer_cka_prev"] = np.array(cka_prevs, dtype=np.float32)
        results[f"{prefix}layer_cka_init"] = np.array(cka_inits, dtype=np.float32)

    return results


def _pairwise_sim_matrices(
    traj: list[np.ndarray],  # list of N states, each (n_points, D) float32
    max_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise cosine similarity, linear CKA, and RMS-L2 matrices.

    Fully vectorized — no Python loop over pairs.

    Parameters
    ----------
    traj        : list of N hidden-state snapshots, each (n_points, D).
                  Typically: [init, loop1_block1, loop1_block2, ..., loopN_blockM].
    max_samples : subsample cap to bound memory usage.

    Returns
    -------
    cos_mat : (N, N) float32 — mean per-point cosine similarity
    cka_mat : (N, N) float32 — linear CKA  (Kornblith et al., ICML 2019)
    l2_mat  : (N, N) float32 — RMS per-point L2 distance
    """
    n = len(traj)
    n_pts = traj[0].shape[0]
    if n_pts > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_pts, max_samples, replace=False)
        traj = [t[idx].astype(np.float32) for t in traj]
    else:
        traj = [t.astype(np.float32) for t in traj]

    n_s = traj[0].shape[0]   # effective sample count after subsampling

    # ── CKA via Gram matrices (vectorized) ────────────────────────────────
    # For centered A, B:  ||A.T @ B||_F^2  =  <G_A, G_B>_F
    #   where G_X = X_c @ X_c.T  (n_s × n_s, symmetric).
    # CKA(A, B) = <G_A, G_B>_F / (||G_A||_F * ||G_B||_F)
    #           = cosine sim between flattened Gram matrices.
    traj_c  = [t - t.mean(axis=0, keepdims=True) for t in traj]
    G_flat  = np.stack([c @ c.T for c in traj_c], axis=0).reshape(n, n_s * n_s)
    G_norms = np.linalg.norm(G_flat, axis=-1, keepdims=True) + 1e-12
    cka_mat = ((G_flat / G_norms) @ (G_flat / G_norms).T).astype(np.float32)

    # ── Cosine similarity (vectorized) ────────────────────────────────────
    # cos[i,j] = mean_k  hat_t_i[k] · hat_t_j[k]
    #          = (flat_hat_i · flat_hat_j) / n_s
    row_n   = [np.linalg.norm(t, axis=-1, keepdims=True) + 1e-12 for t in traj]
    flat_n  = np.stack([(t / rn).reshape(-1) for t, rn in zip(traj, row_n)], axis=0)
    cos_mat = (flat_n @ flat_n.T / n_s).astype(np.float32)

    # ── RMS-L2 distance (vectorized) ──────────────────────────────────────
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>   (expanded over all points)
    flat     = np.stack([t.reshape(-1) for t in traj], axis=0)
    sq_norms = (flat ** 2).sum(axis=-1)
    l2_sq    = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (flat @ flat.T)
    l2_mat   = np.sqrt(np.maximum(l2_sq, 0.0) / n_s).astype(np.float32)

    return cos_mat, cka_mat, l2_mat


def _compute_lyapunov_spectrum(
    model: "LoopICL",
    x_BRCE_0: torch.Tensor,
    x_BRD_0: torch.Tensor,
    num_train: int,
    y_col_emb: torch.Tensor,
    n_loops: int,
    n_exponents: int = 5,
    offload_x_brce: bool = False,
    test_chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Estimate the k leading Lyapunov exponents of the ICL loop dynamics.

    Applies the Benettin QR algorithm with exact forward-mode autograd
    Jacobian-vector products to characterise how infinitesimal perturbations
    to the row hidden-state x_BRD grow (or shrink) across ICL loops.

    Dynamical system analysed
    -------------------------
    The map at loop t is h_t: x_BRD → x_BRD_new, where

        x_BRD_film = film_t(x_BRD)                      [if loop_film is set]
        x_BRD_out  = axial_step(x_BRCE_nom, x_BRD_film) [x_BRCE on nominal traj]
        x_BRD_new  = x_BRD_film + s_t * (x_BRD_out − x_BRD_film)  [if LRS]
                   = x_BRD_out                                       [else]

    Perturbations are propagated via exact forward-mode AD (JVP):

        Jv = d/dε h_t(x + ε·v)|_{ε=0}

    Algorithm
    ---------
    1. Initialise Q ∈ ℝ^{(R·D) × k} — k orthonormal perturbation vectors.
    2. Each loop: propagate all k vectors through h_t via autograd JVP.
    3. QR-decompose the evolved matrix Z = J·Q = Q_new · R_mat.
    4. Accumulate log |diag(R_mat)| for each exponent.
    5. LE_i = (1/n_loops) · Σ_t log|R_mat_t[i,i]|.

    Interpretation
    --------------
    • All LEs < 0  → contracting dynamics (state converges to attractor).
    • Max LE ≈ 0   → neutral (e.g., random walk or marginal stability).
    • Max LE > 0   → chaotic / divergent.

    The Kaplan-Yorke dimension and KS entropy can be derived from the spectrum
    in the notebook.

    Returns
    -------
    "lyapunov_spectrum"   : (k,) float32 — final LE estimates [nats/step]
    "lyapunov_running"    : (n_loops, k) float32 — running LE averages
    "lyapunov_log_growth" : (n_loops, k) float32 — per-step log |diag(R)|
    """
    device  = x_BRD_0.device
    B, R, D = x_BRD_0.shape
    k = min(n_exponents, R * D)
    use_lrs = model.use_loop_residual_scaling

    # Orthonormal perturbation basis (CPU float32 for numerical stability)
    Q_raw = torch.randn(B * R * D, k, dtype=torch.float32)
    Q, _  = torch.linalg.qr(Q_raw)   # (B*R*D, k)

    log_sum:     np.ndarray         = np.zeros(k, dtype=np.float64)
    log_growths: list[np.ndarray]   = []
    running_les: list[np.ndarray]   = []

    x_BRCE = x_BRCE_0.clone()
    x_BRD  = x_BRD_0.clone()

    for loop_i in range(n_loops):
        skip_inject = model.inject_only_first_loop and loop_i > 0

        # ── Apply film (save pre-film state) ─────────────────────────────────
        x_BRD_pre_film = x_BRD.clone()
        if model.loop_film is not None:
            with torch.no_grad():
                x_BRD = model._apply_film(x_BRD, loop_i)
        x_BRD_post_film = x_BRD  # LRS residual base

        # ── Advance nominal trajectory ────────────────────────────────────────
        with torch.no_grad():
            x_BRCE_new, x_BRD_out = model._axial_step_chunked(
                x_BRCE, x_BRD, num_train, y_col_emb,
                skip_cell_label_inject=skip_inject,
                test_chunk_size=test_chunk_size,
                offload_x_brce=offload_x_brce,
            )

        if use_lrs:
            loop_scale = float(model._compute_loop_scale(n_loops, loop_i))
            x_BRD_new  = x_BRD_post_film + loop_scale * (x_BRD_out  - x_BRD_post_film)
            x_BRCE_new = x_BRCE          + loop_scale * (x_BRCE_new - x_BRCE)
        else:
            loop_scale = 1.0
            x_BRD_new  = x_BRD_out

        # ── Propagate perturbations via forward-mode autograd (exact JVPs) ───
        # Full map h_t: x_BRD_pre_film → x_BRD_new (film + axial_step + LRS).
        # Uses _axial_step (non-chunked) so the computation graph is intact.
        # x_BRCE is captured from the nominal trajectory and treated as fixed.
        def _h_t(x_pre: torch.Tensor) -> torch.Tensor:
            if model.loop_film is not None:
                x_film = model._apply_film(x_pre, loop_i)
            else:
                x_film = x_pre
            _, x_out = model._axial_step(
                x_BRCE, x_film, num_train, y_col_emb,
                skip_cell_label_inject=skip_inject,
            )
            if use_lrs:
                return x_film + loop_scale * (x_out - x_film)
            return x_out

        Z = torch.zeros(B * R * D, k, dtype=torch.float32)
        x0 = x_BRD_pre_film.detach()

        from loopicl.models.loopicl.layers import force_math_sdpa
        for i in range(k):
            v = Q[:, i].reshape(B, R, D).to(device=device, dtype=x0.dtype)
            with force_math_sdpa():
                _, jvp_out = torch.func.jvp(_h_t, (x0,), (v,))
            Z[:, i] = jvp_out.float().detach().reshape(-1).cpu()

        # ── QR decompose → updated Q and growth rates ─────────────────────────
        Q, R_mat = torch.linalg.qr(Z)   # Q: (B*R*D, k), R_mat: (k, k)
        log_growth = torch.log(torch.abs(torch.diagonal(R_mat)) + 1e-30).cpu().numpy()
        log_sum   += log_growth
        log_growths.append(log_growth.astype(np.float32))
        running_les.append((log_sum / (loop_i + 1)).astype(np.float32))

        x_BRCE = x_BRCE_new
        x_BRD  = x_BRD_new

    return {
        "lyapunov_spectrum":   (log_sum / n_loops).astype(np.float32),
        "lyapunov_running":    np.stack(running_les,  axis=0),   # (n_loops, k)
        "lyapunov_log_growth": np.stack(log_growths,  axis=0),   # (n_loops, k)
    }


def _compute_lyapunov_spectrum_per_block(
    model: "LoopICL",
    x_BRCE_0: torch.Tensor,
    x_BRD_0: torch.Tensor,
    num_train: int,
    y_col_emb: torch.Tensor,
    n_exponents: int = 5,
    offload_x_brce: bool = False,
) -> dict[str, np.ndarray]:
    """Estimate leading Lyapunov exponents using per-block stepping.

    Designed for unrolled models (e.g. RicottaUnrolled) where
    ``axial_num_blocks > 1`` and ``num_icl_loops=1``.  Each axial block is
    treated as one Benettin step rather than one full ICL loop, yielding
    ``n_blocks`` steps through the depth of the network.  This is directly
    comparable to a recurrent model run for ``n_blocks`` loops.

    The block-wise map at step b is h_b: x_BRD → x_BRD_new, applied via
    ``model._axial_block_i`` with the current nominal x_BRCE.

    Returns
    -------
    "lyapunov_spectrum"   : (k,) float32 — final LE estimates [nats/step]
    "lyapunov_running"    : (n_blocks, k) float32 — running LE averages
    "lyapunov_log_growth" : (n_blocks, k) float32 — per-step log |diag(R)|
    """
    emb      = model.axial_embedder
    n_blocks = emb.num_axial_blocks
    device   = x_BRD_0.device
    B, R, D  = x_BRD_0.shape
    k = min(n_exponents, R * D)

    Q_raw = torch.randn(B * R * D, k, dtype=torch.float32)
    Q, _  = torch.linalg.qr(Q_raw)

    log_sum:     np.ndarray       = np.zeros(k, dtype=np.float64)
    log_growths: list[np.ndarray] = []
    running_les: list[np.ndarray] = []

    x_BRCE = x_BRCE_0.clone()
    x_BRD  = x_BRD_0.clone()

    for block_i in range(n_blocks):
        within_start = block_i * emb.num_within_per_round
        cross_start  = block_i * emb.num_cross_per_round
        icl_start    = block_i * emb.num_icl_per_round

        # ── Advance nominal trajectory one block ──────────────────────────────
        with torch.no_grad():
            x_BRCE_new, x_BRD_new = model._axial_block_i(
                x_BRCE, x_BRD, block_i, within_start, cross_start, icl_start,
                num_train, y_col_emb,
                skip_cell_label_inject=False,
                offload_x_brce=offload_x_brce,
            )

        # ── Propagate perturbations via forward-mode autograd (exact JVPs) ───
        def _h_b(x: torch.Tensor) -> torch.Tensor:
            _, x_out = model._axial_block_i(
                x_BRCE, x, block_i, within_start, cross_start, icl_start,
                num_train, y_col_emb, skip_cell_label_inject=False,
            )
            return x_out

        Z = torch.zeros(B * R * D, k, dtype=torch.float32)
        x0 = x_BRD.detach()

        from loopicl.models.loopicl.layers import force_math_sdpa
        for i in range(k):
            v = Q[:, i].reshape(B, R, D).to(device=device, dtype=x0.dtype)
            with force_math_sdpa():
                _, jvp_out = torch.func.jvp(_h_b, (x0,), (v,))
            Z[:, i] = jvp_out.float().detach().reshape(-1).cpu()

        # ── QR decompose → updated Q and growth rates ─────────────────────────
        Q, R_mat   = torch.linalg.qr(Z)
        log_growth = torch.log(torch.abs(torch.diagonal(R_mat)) + 1e-30).cpu().numpy()
        log_sum   += log_growth
        log_growths.append(log_growth.astype(np.float32))
        running_les.append((log_sum / (block_i + 1)).astype(np.float32))

        x_BRCE = x_BRCE_new
        x_BRD  = x_BRD_new

    return {
        "lyapunov_spectrum":   (log_sum / n_blocks).astype(np.float32),
        "lyapunov_running":    np.stack(running_les,  axis=0),   # (n_blocks, k)
        "lyapunov_log_growth": np.stack(log_growths,  axis=0),   # (n_blocks, k)
    }


# ---------------------------------------------------------------------------
# Lightweight loop-predictions-only forward
# ---------------------------------------------------------------------------

def forward_loop_predictions_only(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int | None = None,
    train_size: int | None = None,
    num_loops: int | None = None,
    test_chunk_size: int | None = None,
    offload_x_brce: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Lightweight LoopICL forward that captures per-loop logits only.

    Mirrors the loop body of ``forward_with_hs_metrics`` exactly but skips
    all hidden-state metric computation (no CKA, no L2 deltas, no layer
    matrices).  Use this when you only need ``loop_logits`` / ``loop_scores``
    and want the minimum overhead on top of a normal forward pass.

    Parameters
    ----------
    model            : trained LoopICL in eval() mode
    X                : (B, R, C)
    y_train          : (B, R)
    single_eval_pos  : number of training rows
    num_loops        : overrides model default if given
    test_chunk_size  : enables chunked axial step (saves memory)

    Returns
    -------
    logits          : (B, n_test, max_classes) — final logits
    all_loop_logits : list[(B, n_test, max_classes)] — one per loop (index 0 = pre-loop init)
    """
    assert not model.training, "forward_loop_predictions_only is for inference only"

    from loopicl.models.loopicl.conditioning import (
        _adaptive_bin_geometry,
        _compute_moments,
        _soft_log_histogram,
    )

    x_RBC = X.transpose(0, 1)
    y_NB  = y_train.transpose(0, 1)
    num_train = (
        int(single_eval_pos) if single_eval_pos is not None
        else int(train_size)  if train_size  is not None
        else int(y_NB.shape[0])
    )

    x_BRC, nan_ind_BRC = model._preprocess(x_RBC, num_train)
    x_BRCG = model._group_features(x_BRC, nan_ind_BRC)

    x_train_BCN  = x_BRC[:, :num_train].permute(0, 2, 1)
    x_sorted_BCN = x_train_BCN.sort(dim=-1).values

    centers, widths = _adaptive_bin_geometry(x_sorted_BCN, model._n_bins)
    log_hist_BCK    = _soft_log_histogram(x_train_BCN, centers, widths, num_train)
    moments_BC4     = _compute_moments(x_train_BCN)

    x_BRCE    = model.x_embed(x_BRCG)
    y_BN      = y_NB.transpose(0, 1)[:, :num_train]
    y_col_emb = model.col_y_encoder(y_BN)
    x_BRCE[:, :num_train] = x_BRCE[:, :num_train] + y_col_emb.unsqueeze(2)

    if model.use_log_hist_conditioning:
        x_BRCE = x_BRCE + model.log_hist_conditioner(log_hist_BCK, moments_BC4).unsqueeze(1)
    if model.use_disc_hist_conditioning:
        x_BRCE = x_BRCE + model.discriminative_hist_conditioner(
            x_train_BCN, centers, widths, log_hist_BCK, y_BN, num_train,
            chunk_size=model._disc_hist_chunk_size,
        ).unsqueeze(1)
    if model.use_fourier_ood:
        x_BRCE = x_BRCE + model.fourier_quantile_encoder(x_BRC, x_sorted_BCN, num_train)

    B, R, _, E = x_BRCE.shape
    Cl = model.axial_embedder.num_cls_tokens
    x_BRD = model._init_x_brd(B, R, x_BRCE, Cl, E)

    n_loops  = num_loops if num_loops is not None else model.num_icl_loops
    use_lrs  = model.use_loop_residual_scaling
    _chunk   = test_chunk_size if test_chunk_size is not None else 4096

    # Initial logits (index 0 = before any loop)
    _normed_init = model.output_norm(x_BRD)
    _init_logits = model.many_class_decoder(
        _normed_init[:, :num_train], _normed_init[:, num_train:], y_BN
    )
    all_loop_logits: list[torch.Tensor] = [
        torch.nan_to_num(_init_logits.transpose(0, 1), nan=0.0)
    ]

    for loop_i in range(n_loops):
        if model.loop_film is not None:
            x_BRD = model._apply_film(x_BRD, loop_i)

        skip_inject = model.inject_only_first_loop and loop_i > 0

        if use_lrs:
            x_BRCE_pre = x_BRCE
            x_BRD_pre  = x_BRD.clone()

        x_BRCE, x_BRD = model._axial_step_chunked(
            x_BRCE, x_BRD, num_train, y_col_emb,
            skip_cell_label_inject=skip_inject,
            test_chunk_size=_chunk,
            offload_x_brce=offload_x_brce,
        )

        if use_lrs:
            loop_scale = model._compute_loop_scale(n_loops, loop_i)
            x_BRD  = x_BRD_pre  + loop_scale * (x_BRD  - x_BRD_pre)
            x_BRCE = x_BRCE_pre + loop_scale * (x_BRCE - x_BRCE_pre)

        normed      = model.output_norm(x_BRD)
        loop_logits = model.many_class_decoder(normed[:, :num_train], normed[:, num_train:], y_BN)
        all_loop_logits.append(torch.nan_to_num(loop_logits.transpose(0, 1), nan=0.0))

    logits = all_loop_logits[-1]
    return logits, all_loop_logits


def _snap_row(x_BRD: torch.Tensor, start: int, end: int | None) -> torch.Tensor:
    """Detached fp32 slice of x_BRD along the row dimension → (B, n, D)."""
    return x_BRD[:, start:end].detach().float()


def _snap_cell(x_BRCE: torch.Tensor, start: int, end: int | None) -> torch.Tensor:
    """Detached fp32 column-mean-pooled slice of x_BRCE → (B, n, E).

    Pooling over C keeps memory proportional to E rather than C×E.
    """
    return x_BRCE[:, start:end].detach().float().mean(dim=2)


# ---------------------------------------------------------------------------
# Custom forward
# ---------------------------------------------------------------------------

def forward_with_hs_metrics(
    model: "LoopICL",
    X: torch.Tensor,
    y_train: torch.Tensor,
    single_eval_pos: int | None = None,
    train_size: int | None = None,
    num_loops: int | None = None,
    test_chunk_size: int | None = None,
    offload_x_brce: bool = False,
    y_int_train: np.ndarray | None = None,
    sep_gap_max_samples: int = 200,
    capture_embeddings: bool = False,
    capture_test_embeddings: bool = False,
    capture_layer_metrics: bool = False,
    capture_layer_matrix: bool = False,
    layer_matrix_max_samples: int = 200,
    layers_as_loops: bool = False,
    capture_lyapunov: bool = False,
    lyapunov_n_exponents: int = 5,
    lyapunov_per_block: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """LoopICL forward with per-loop logits and on-the-fly hidden-state metrics.

    Tracks both the train and test row slices of x_BRD and x_BRCE.
    Mirrors ``LoopICL.forward()`` exactly for the computation.

    Parameters
    ----------
    model               : trained LoopICL in eval() mode
    X                   : (B, R, C)
    y_train             : (B, R)
    single_eval_pos     : number of training rows
    num_loops           : overrides model default if given
    test_chunk_size     : enables memory-efficient chunked axial step
    y_int_train         : (n_train,) integer class labels.  When provided,
                          separation gap metrics are computed each loop on
                          both the row and cell train streams.
    sep_gap_max_samples : per-class subsample cap for pairwise distance
                          computation (avoids O(n²) cost).  Default 200.
    capture_embeddings  : if True, also return raw row-stream hidden states at
                          every loop (including the pre-loop init snapshot) as a
                          separate dict with keys "train_row" and "test_row",
                          each shape (n_loops+1, n_points, D).  Loop index 0 is
                          the initial state before any loop.

    Returns
    -------
    logits          : (M, B, T) — final logits (identical to normal forward)
    all_loop_logits : list[(M, B, T)] — one per loop iteration
    metrics         : dict mapping metric name → np.ndarray
                      Per-point keys: shape (n_loops, n_train) or (n_loops, n_test)
                      Aggregate "_mean" keys: shape (n_loops,)
    embeddings      : dict{"train_row", "test_row"} → np.ndarray (n_loops+1, n, D)
                      Only present (not None) when capture_embeddings=True.
    """
    assert not model.training, "forward_with_hs_metrics is for inference only"

    from loopicl.models.loopicl.conditioning import (
        _adaptive_bin_geometry,
        _compute_moments,
        _soft_log_histogram,
    )

    # ------------------------------------------------------------------
    # Stage 0: preprocessing
    # ------------------------------------------------------------------
    x_RBC = X.transpose(0, 1)
    y_NB  = y_train.transpose(0, 1)
    num_train = (
        int(single_eval_pos) if single_eval_pos is not None
        else int(train_size)  if train_size  is not None
        else int(y_NB.shape[0])
    )

    x_BRC, nan_ind_BRC = model._preprocess(x_RBC, num_train)
    x_BRCG = model._group_features(x_BRC, nan_ind_BRC)

    x_train_BCN  = x_BRC[:, :num_train].permute(0, 2, 1)
    x_sorted_BCN = x_train_BCN.sort(dim=-1).values

    centers, widths = _adaptive_bin_geometry(x_sorted_BCN, model._n_bins)
    log_hist_BCK    = _soft_log_histogram(x_train_BCN, centers, widths, num_train)
    moments_BC4     = _compute_moments(x_train_BCN)

    # ------------------------------------------------------------------
    # Stage 1: cell embedding + conditioning
    # ------------------------------------------------------------------
    x_BRCE    = model.x_embed(x_BRCG)
    y_BN      = y_NB.transpose(0, 1)[:, :num_train]
    _is_reg   = hasattr(model, "reg_y_encoder")
    if _is_reg:
        y_col_emb = model.reg_y_encoder(y_BN.float().unsqueeze(-1))  # (B, N, E)
    else:
        y_col_emb = model.col_y_encoder(y_BN)
    x_BRCE[:, :num_train] = x_BRCE[:, :num_train] + y_col_emb.unsqueeze(2)

    if model.use_log_hist_conditioning:
        x_BRCE = x_BRCE + model.log_hist_conditioner(log_hist_BCK, moments_BC4).unsqueeze(1)
    if model.use_disc_hist_conditioning:
        x_BRCE = x_BRCE + model.discriminative_hist_conditioner(
            x_train_BCN, centers, widths, log_hist_BCK, y_BN, num_train,
            chunk_size=model._disc_hist_chunk_size,
        ).unsqueeze(1)
    if model.use_fourier_ood:
        x_BRCE = x_BRCE + model.fourier_quantile_encoder(x_BRC, x_sorted_BCN, num_train)

    # ------------------------------------------------------------------
    # Stage 2: outer loop
    # ------------------------------------------------------------------
    B, R, _, E = x_BRCE.shape
    Cl = model.axial_embedder.num_cls_tokens
    D  = Cl * E
    x_BRD = model._init_x_brd(B, R, x_BRCE, Cl, E)

    n_loops = num_loops if num_loops is not None else model.num_icl_loops

    # Residual scaling (identical to LoopICL.forward)
    use_lrs = model.use_loop_residual_scaling

    # ------------------------------------------------------------------
    # Initial snapshots: train split and test split, both streams
    # ------------------------------------------------------------------
    tr_row_init  = _snap_row(x_BRD,   0,         num_train)  # (B, n_train, D)
    te_row_init  = _snap_row(x_BRD,   num_train,  None)       # (B, n_test,  D)
    tr_cell_init = _snap_cell(x_BRCE, 0,         num_train)  # (B, n_train, E)
    te_cell_init = _snap_cell(x_BRCE, num_train,  None)       # (B, n_test,  E)

    tr_row_prev, te_row_prev   = tr_row_init.clone(),  te_row_init.clone()
    tr_cell_prev, te_cell_prev = tr_cell_init.clone(), te_cell_init.clone()

    metrics_lists: dict[str, list[np.ndarray]] = {}
    all_loop_logits: list[torch.Tensor] = []

    # Embedding trajectory: index 0 = init state (before any loop)
    emb_tr_rows: list[np.ndarray] = []
    emb_te_rows: list[np.ndarray] = []
    if capture_embeddings:
        emb_tr_rows.append(tr_row_init.mean(dim=0).cpu().numpy())   # (n_train, D)
        emb_te_rows.append(te_row_init.mean(dim=0).cpu().numpy())   # (n_test,  D)
    if capture_test_embeddings:
        emb_te_rows.append(te_row_init.mean(dim=0).cpu().numpy())   # (n_test,  D)

    # Lyapunov: save initial states (pre-loop x_BRCE and x_BRD)
    _lya_x_BRCE_0: torch.Tensor | None = None
    _lya_x_BRD_0:  torch.Tensor | None = None
    _lyapunov_test_chunk = test_chunk_size if test_chunk_size is not None else 4096
    if capture_lyapunov:
        _lya_x_BRCE_0 = x_BRCE.clone()
        _lya_x_BRD_0  = x_BRD.clone()

    # Layer-matrix trajectory: index 0 = init, then one entry per axial block per loop
    _traj_row_tr:  list[np.ndarray] = []
    _traj_row_te:  list[np.ndarray] = []
    _traj_cell_tr: list[np.ndarray] = []
    _traj_cell_te: list[np.ndarray] = []
    if capture_layer_matrix:
        _traj_row_tr.append(tr_row_init.mean(dim=0).cpu().numpy().astype(np.float32))
        _traj_row_te.append(te_row_init.mean(dim=0).cpu().numpy().astype(np.float32))
        _traj_cell_tr.append(tr_cell_init.mean(dim=0).cpu().numpy().astype(np.float32))
        _traj_cell_te.append(te_cell_init.mean(dim=0).cpu().numpy().astype(np.float32))

    # Initial logits (index 0 = before any loop, mirrors embedding trajectory)
    _normed_init = model.output_norm(x_BRD)
    if _is_reg:
        _init_logits = model.reg_head(_normed_init[:, num_train:])  # (B, M, num_quantiles)
        all_loop_logits.append(torch.nan_to_num(_init_logits, nan=0.0))
    else:
        _init_logits = model.many_class_decoder(
            _normed_init[:, :num_train], _normed_init[:, num_train:], y_BN
        )
        all_loop_logits.append(torch.nan_to_num(_init_logits.transpose(0, 1), nan=0.0))

    for loop_i in range(n_loops):

        if model.loop_film is not None:
            x_BRD = model._apply_film(x_BRD, loop_i)

        skip_inject = model.inject_only_first_loop and loop_i > 0

        if use_lrs:
            x_BRCE_pre = x_BRCE
            x_BRD_pre  = x_BRD.clone()

        _test_chunk = test_chunk_size if test_chunk_size is not None else 4096
        _need_layer_states = capture_layer_metrics or capture_layer_matrix or layers_as_loops
        if _need_layer_states:
            _loop_start_brd  = x_BRD.detach().cpu().float()
            _loop_start_brce = x_BRCE.detach().cpu().float().mean(dim=2)
            x_BRCE, x_BRD, _brd_states, _brce_states, _brd_pre_icl = model._axial_step_chunked(
                x_BRCE, x_BRD, num_train, y_col_emb,
                skip_cell_label_inject=skip_inject,
                test_chunk_size=_test_chunk,
                offload_x_brce=offload_x_brce,
                return_layer_states=True,
            )
        else:
            x_BRCE, x_BRD = model._axial_step_chunked(
                x_BRCE, x_BRD, num_train, y_col_emb,
                skip_cell_label_inject=skip_inject,
                test_chunk_size=_test_chunk,
                offload_x_brce=offload_x_brce,
            )

        if use_lrs:
            loop_scale = model._compute_loop_scale(n_loops, loop_i)
            x_BRD  = x_BRD_pre  + loop_scale * (x_BRD  - x_BRD_pre)
            x_BRCE = x_BRCE_pre + loop_scale * (x_BRCE - x_BRCE_pre)

        # ------ per-loop logits ------
        normed    = model.output_norm(x_BRD)
        train_emb = normed[:, :num_train]
        test_emb  = normed[:, num_train:]
        if _is_reg:
            loop_logits = model.reg_head(test_emb)  # (B, M, num_quantiles)
            all_loop_logits.append(torch.nan_to_num(loop_logits, nan=0.0))
        else:
            loop_logits = model.many_class_decoder(train_emb, test_emb, y_BN)
            all_loop_logits.append(
                torch.nan_to_num(loop_logits.transpose(0, 1), nan=0.0)
            )

        # ------ hidden-state metrics: train and test, both streams ------
        tr_row_curr  = _snap_row(x_BRD,   0,         num_train)
        te_row_curr  = _snap_row(x_BRD,   num_train,  None)
        tr_cell_curr = _snap_cell(x_BRCE, 0,         num_train)
        te_cell_curr = _snap_cell(x_BRCE, num_train,  None)

        step: dict[str, np.ndarray] = {}
        step.update(_stream_metrics(tr_row_prev,  tr_row_curr,  tr_row_init,  "train_row_"))
        step.update(_stream_metrics(te_row_prev,  te_row_curr,  te_row_init,  "test_row_"))
        step.update(_stream_metrics(tr_cell_prev, tr_cell_curr, tr_cell_init, "train_cell_"))
        step.update(_stream_metrics(te_cell_prev, te_cell_curr, te_cell_init, "test_cell_"))

        if y_int_train is not None:
            step.update(_separation_gap(tr_row_curr,  y_int_train, sep_gap_max_samples, "train_row_"))
            step.update(_separation_gap(tr_cell_curr, y_int_train, sep_gap_max_samples, "train_cell_"))

        if capture_embeddings:
            emb_tr_rows.append(tr_row_curr.mean(dim=0).cpu().numpy())   # (n_train, D)
            emb_te_rows.append(te_row_curr.mean(dim=0).cpu().numpy())   # (n_test,  D)
        if capture_test_embeddings:
            emb_te_rows.append(te_row_curr.mean(dim=0).cpu().numpy())   # (n_test,  D)

        for k, v in step.items():
            metrics_lists.setdefault(k, []).append(np.asarray(v))

        if capture_layer_metrics:
            layer_step = _layer_sim_metrics(
                _brd_states, _brce_states,
                _loop_start_brd, _loop_start_brce,
                num_train,
            )
            for k, v in layer_step.items():
                metrics_lists.setdefault(k, []).append(np.asarray(v))

        if capture_layer_matrix:
            # Interleave pre-ICL and post-ICL states for each axial block:
            # [..., pre_icl_i, post_icl_i, ...]
            for pre, post, brce in zip(_brd_pre_icl, _brd_states, _brce_states):
                _traj_row_tr.append(pre[:,  :num_train].mean(dim=0).numpy().astype(np.float32))
                _traj_row_te.append(pre[:,  num_train:].mean(dim=0).numpy().astype(np.float32))
                _traj_row_tr.append(post[:, :num_train].mean(dim=0).numpy().astype(np.float32))
                _traj_row_te.append(post[:, num_train:].mean(dim=0).numpy().astype(np.float32))
                _traj_cell_tr.append(brce[:, :num_train].mean(dim=0).numpy().astype(np.float32))
                _traj_cell_te.append(brce[:, num_train:].mean(dim=0).numpy().astype(np.float32))
                _traj_cell_tr.append(brce[:, :num_train].mean(dim=0).numpy().astype(np.float32))
                _traj_cell_te.append(brce[:, num_train:].mean(dim=0).numpy().astype(np.float32))

        del tr_row_prev, te_row_prev, tr_cell_prev, te_cell_prev
        tr_row_prev, te_row_prev   = tr_row_curr,  te_row_curr
        tr_cell_prev, te_cell_prev = tr_cell_curr, te_cell_curr

    del tr_row_prev, te_row_prev, tr_cell_prev, te_cell_prev
    del tr_row_init, te_row_init, tr_cell_init, te_cell_init

    # Treat each axial block (layer) as a loop iteration, replacing per-loop metrics
    # with per-layer metrics.  Intended for unrolled models with n_loops=1 where the
    # 6 internal layers should be compared against multi-loop convergence curves.
    if layers_as_loops:
        metrics_lists.clear()

        # _loop_start_brd / _loop_start_brce are (B, n_total, D/E) CPU float tensors
        # _brd_states[i]  / _brce_states[i]  are the same after axial block i
        _lay_tr_row_init  = _loop_start_brd[:,  :num_train]
        _lay_te_row_init  = _loop_start_brd[:,  num_train:]
        _lay_tr_cell_init = _loop_start_brce[:, :num_train]
        _lay_te_cell_init = _loop_start_brce[:, num_train:]

        _lay_tr_row_prev  = _lay_tr_row_init
        _lay_te_row_prev  = _lay_te_row_init
        _lay_tr_cell_prev = _lay_tr_cell_init
        _lay_te_cell_prev = _lay_te_cell_init

        for _brd_s, _brce_s in zip(_brd_states, _brce_states):
            _lay_tr_row_curr  = _brd_s[:,  :num_train]
            _lay_te_row_curr  = _brd_s[:,  num_train:]
            _lay_tr_cell_curr = _brce_s[:, :num_train]
            _lay_te_cell_curr = _brce_s[:, num_train:]

            lay_step: dict[str, np.ndarray] = {}
            lay_step.update(_stream_metrics(_lay_tr_row_prev,  _lay_tr_row_curr,  _lay_tr_row_init,  "train_row_"))
            lay_step.update(_stream_metrics(_lay_te_row_prev,  _lay_te_row_curr,  _lay_te_row_init,  "test_row_"))
            lay_step.update(_stream_metrics(_lay_tr_cell_prev, _lay_tr_cell_curr, _lay_tr_cell_init, "train_cell_"))
            lay_step.update(_stream_metrics(_lay_te_cell_prev, _lay_te_cell_curr, _lay_te_cell_init, "test_cell_"))

            if y_int_train is not None:
                lay_step.update(_separation_gap(_lay_tr_row_curr,  y_int_train, sep_gap_max_samples, "train_row_"))
                lay_step.update(_separation_gap(_lay_tr_cell_curr, y_int_train, sep_gap_max_samples, "train_cell_"))

            for k, v in lay_step.items():
                metrics_lists.setdefault(k, []).append(np.asarray(v))

            _lay_tr_row_prev  = _lay_tr_row_curr
            _lay_te_row_prev  = _lay_te_row_curr
            _lay_tr_cell_prev = _lay_tr_cell_curr
            _lay_te_cell_prev = _lay_te_cell_curr

        # Rebuild embeddings per layer (init + one per layer)
        if capture_test_embeddings:
            emb_te_rows.clear()
            emb_te_rows.append(_loop_start_brd[:, num_train:].mean(dim=0).cpu().numpy())
            for _brd_s in _brd_states:
                emb_te_rows.append(_brd_s[:, num_train:].mean(dim=0).cpu().numpy())
        if capture_embeddings:
            emb_tr_rows.clear()
            emb_te_rows.clear()
            emb_tr_rows.append(_loop_start_brd[:, :num_train].mean(dim=0).cpu().numpy())
            emb_te_rows.append(_loop_start_brd[:, num_train:].mean(dim=0).cpu().numpy())
            for _brd_s in _brd_states:
                emb_tr_rows.append(_brd_s[:, :num_train].mean(dim=0).cpu().numpy())
                emb_te_rows.append(_brd_s[:, num_train:].mean(dim=0).cpu().numpy())

    # Stack per-loop rows → (n_loops, n_points) or (n_loops,) for _mean keys
    metrics: dict[str, np.ndarray] = {
        k: np.stack(v, axis=0) for k, v in metrics_lists.items()
    }

    # Pairwise similarity matrices across the full layer trajectory
    if capture_layer_matrix:
        for prefix, traj in [
            ("train_row_",  _traj_row_tr),
            ("test_row_",   _traj_row_te),
            ("train_cell_", _traj_cell_tr),
            ("test_cell_",  _traj_cell_te),
        ]:
            cos_mat, cka_mat, l2_mat = _pairwise_sim_matrices(
                traj, max_samples=layer_matrix_max_samples,
            )
            metrics[f"{prefix}layer_matrix_cos"] = cos_mat
            metrics[f"{prefix}layer_matrix_cka"] = cka_mat
            metrics[f"{prefix}layer_matrix_l2"]  = l2_mat

    # Lyapunov spectrum — separate re-run of the nominal trajectory + perturbed passes
    if capture_lyapunov and _lya_x_BRCE_0 is not None and _lya_x_BRD_0 is not None:
        if lyapunov_per_block:
            lya = _compute_lyapunov_spectrum_per_block(
                model=model,
                x_BRCE_0=_lya_x_BRCE_0,
                x_BRD_0=_lya_x_BRD_0,
                num_train=num_train,
                y_col_emb=y_col_emb,
                n_exponents=lyapunov_n_exponents,
                offload_x_brce=offload_x_brce,
            )
        else:
            lya = _compute_lyapunov_spectrum(
                model=model,
                x_BRCE_0=_lya_x_BRCE_0,
                x_BRD_0=_lya_x_BRD_0,
                num_train=num_train,
                y_col_emb=y_col_emb,
                n_loops=n_loops,
                n_exponents=lyapunov_n_exponents,
                offload_x_brce=offload_x_brce,
                test_chunk_size=_lyapunov_test_chunk,
            )
        metrics.update(lya)

    embeddings: dict[str, np.ndarray] | None = None
    if capture_embeddings:
        embeddings = {
            "train_row": np.stack(emb_tr_rows, axis=0),  # (n_loops+1, n_train, D)
            "test_row":  np.stack(emb_te_rows, axis=0),  # (n_loops+1, n_test,  D)
        }
    elif capture_test_embeddings:
        embeddings = {
            "test_row": np.stack(emb_te_rows, axis=0),   # (n_loops+1, n_test,  D)
        }

    logits = all_loop_logits[-1]
    return logits, all_loop_logits, metrics, embeddings
