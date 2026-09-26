"""Shared metric utilities for benchmark evaluation.

Used by both :mod:`ticla.benchmarks.setup` (baseline evaluation) and
:mod:`ticla.validation.benchmark_val` (candidate model evaluation) so that
both pipelines compute the same metric for a given problem type.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer


# Ordered list used to identify the primary (comparison) metric from a stored
# result dict.  The first key present in the dict wins.
_PRIMARY_METRIC_PRIORITY = ["roc_auc", "balanced_accuracy", "neg_rmse"]


def _pick_primary(cand_scores: dict) -> str | None:
    """Select the primary metric key from *cand_scores*.

    Rules (applied in order):
    - Multiclass classification (``balanced_accuracy`` or ``_n_classes`` present):
        ``logloss`` — lower is better.
    - Binary classification or regression: first key from *_PRIMARY_METRIC_PRIORITY*
        present in *cand_scores* (``roc_auc`` → ``balanced_accuracy`` → ``neg_rmse``).
    - Fallback: first non-metadata, non-logloss key.
    """
    if ("balanced_accuracy" in cand_scores or "_n_classes" in cand_scores) and "logloss" in cand_scores:
        return "logloss"
    return next(
        (k for k in _PRIMARY_METRIC_PRIORITY if k in cand_scores),
        next((k for k in cand_scores if not k.startswith("_") and k != "logloss"), None),
    )


def _lower_is_better(metric: str) -> bool:
    """Return True if lower values are better for *metric*."""
    return metric in ("logloss", "rmse", "nrmse")


def infer_problem_type(y: np.ndarray, task_type: str) -> str:
    """Return ``"regression"``, ``"binary"``, or ``"multiclass"``."""
    if task_type == "reg":
        return "regression"
    return "binary" if len(np.unique(y)) == 2 else "multiclass"


def _to_numpy(x) -> np.ndarray:
    """Convert a PyTorch tensor (any device) or array-like to a numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metric(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
) -> dict[str, float]:
    """Return ``{metric_name: score, ...}`` for *clf* evaluated on the test split.

    Classification metrics always include ``"logloss"``.  Regression does not.

    Metrics
    -------
    binary classification
        ``roc_auc`` — higher is better; ``pr_auc`` — higher is better (area
        under precision-recall curve, more informative than ROC-AUC on
        imbalanced data); ``mcc`` — higher is better (Matthews Correlation
        Coefficient, in [-1, 1]); ``logloss`` — lower is better.
    multiclass classification
        ``balanced_accuracy`` — higher is better; ``mcc`` — higher is better;
        ``logloss`` — lower is better.
    regression
        ``neg_rmse`` — higher is better (negated RMSE); ``rmse`` — lower is
        better (raw RMSE); ``nrmse`` — lower is better (RMSE / std(y_test));
        ``r2`` — higher is better.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        matthews_corrcoef,
        log_loss,
        mean_squared_error,
        r2_score,
        balanced_accuracy_score,
    )

    if problem_type == "binary":
        clf_classes = getattr(clf, "classes_", None)
        proba = _to_numpy(clf.predict_proba(X_test))
        if proba.ndim == 2:
            proba = proba[:, 1]
        nan_mask = ~np.isfinite(proba)
        if nan_mask.any():
            proba = proba.copy()
            proba[nan_mask] = 0.5
        pred = _to_numpy(clf.predict(X_test))
        pos_label = clf_classes[1] if clf_classes is not None else 1
        ll_labels = clf_classes if clf_classes is not None else None
        return {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba, pos_label=pos_label)),
            "mcc": float(matthews_corrcoef(y_test, pred)),
            "logloss": float(log_loss(y_test, proba, labels=ll_labels)),
        }

    if problem_type == "multiclass":
        proba = _to_numpy(clf.predict_proba(X_test))
        # Use the classifier's own class ordering so that labels matches proba.shape[1].
        clf_classes = getattr(clf, "classes_", None)
        nan_mask = ~np.isfinite(proba)
        if nan_mask.any():
            proba = proba.copy()
            proba[nan_mask] = 1.0 / proba.shape[1]
        pred = _to_numpy(clf.predict(X_test))
        # log_loss requires every y_true value to appear in labels.  When train/test
        # class sets differ (e.g. TALENT datasets with large integer labels), restrict
        # to samples whose true label the classifier actually knows.
        if clf_classes is not None:
            known = np.isin(y_test, clf_classes)
            y_ll, p_ll = (y_test[known], proba[known]) if not known.all() else (y_test, proba)
        else:
            y_ll, p_ll = y_test, proba
        n_classes = len(clf_classes) if clf_classes is not None else len(np.unique(y_test))
        out = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "mcc": float(matthews_corrcoef(y_test, pred)),
            "logloss": float(log_loss(y_ll, p_ll, labels=clf_classes)),
            "_n_classes": float(n_classes),
        }
        # OvR macro-averaged ROC AUC — requires all y_ll classes to appear in labels.
        try:
            out["roc_auc"] = float(
                roc_auc_score(y_ll, p_ll, multi_class="ovr", average="macro", labels=clf_classes)
            )
        except Exception:
            pass  # skip if classes missing from y_ll or other degenerate case
        return out

    # regression
    pred = _to_numpy(clf.predict(X_test))
    nan_mask = ~np.isfinite(pred)
    if nan_mask.any():
        pred = pred.copy()
        pred[nan_mask] = np.nanmean(pred) if not nan_mask.all() else 0.0
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    y_std = float(np.std(y_test))
    nrmse = rmse / y_std if y_std > 1e-9 else float("nan")
    return {
        "neg_rmse": -rmse,
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": float(r2_score(y_test, pred)),
    }


def _run_elo_round(
    score_list: list[dict[str, float]],
    all_models: list[str],
    k: float,
    initial_elo: float = 1500.0,
) -> dict[str, float]:
    """Run one ELO tournament over a sequence of per-dataset score dicts.

    Uses the standard formula ``E = 1 / (1 + 10^(Δ/400))``, so a 400-point
    gap corresponds to a 10:1 (≈91%) expected win rate.
    """
    elo = {m: initial_elo for m in all_models}
    for scores in score_list:
        models = [m for m in scores if m in elo]
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                s1, s2 = scores[m1], scores[m2]
                e1 = 1.0 / (1.0 + 10 ** ((elo[m2] - elo[m1]) / 400))
                if s1 > s2:
                    a1, a2 = 1.0, 0.0
                elif s1 < s2:
                    a1, a2 = 0.0, 1.0
                else:
                    a1, a2 = 0.5, 0.5
                elo[m1] += k * (a1 - e1)
                elo[m2] += k * (a2 - (1.0 - e1))
    return elo


def compute_elo(
    results: dict[str, dict[str, float]],
    k: float = 32.0,
    n_bootstrap: int = 200,
    rf_key: str = "random_forest",
    rf_target_elo: float = 1000.0,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrapped ELO ratings from {dataset_name: {model_name: score}}.

    Runs *n_bootstrap* rounds, each sampling datasets with replacement.
    Calibrates scores so that *rf_key* anchors to *rf_target_elo* (default: RF = 1000).

    Parameters
    ----------
    results:
        ``{dataset_name: {model_name: primary_score}}``
    k:
        ELO K-factor.
    n_bootstrap:
        Number of bootstrap resampling rounds.
    rf_key:
        Model name used as the calibration anchor (default: ``"random_forest"``).
    rf_target_elo:
        ELO value assigned to *rf_key* after calibration (default: ``1000``).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        ``{model_name: (mean_elo, ci_lower, ci_upper)}`` where the CI is 95%.
    """
    rng = np.random.default_rng(seed)
    dataset_scores = list(results.values())
    n = len(dataset_scores)

    all_models: list[str] = sorted({m for scores in dataset_scores for m in scores})
    bootstrap_elos: dict[str, list[float]] = {m: [] for m in all_models}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample = [dataset_scores[i] for i in indices]
        round_elo = _run_elo_round(sample, all_models, k)
        shift = (rf_target_elo - round_elo[rf_key]) if rf_key in round_elo else 0.0
        for m in all_models:
            bootstrap_elos[m].append(round_elo[m] + shift)

    out: dict[str, tuple[float, float, float]] = {}
    for m in all_models:
        vals = np.array(bootstrap_elos[m])
        out[m] = (
            float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
        )
    return out


def _to_error(score: float, metric: str) -> float:
    """Convert a score to a non-negative error value.

    ``neg_rmse`` is stored as a negative number (negated RMSE), so the error
    is its absolute value.  ``logloss`` is already an error (lower is better).
    All other metrics are bounded in [0, 1] so the error is ``1 − score``.
    """
    if metric == "neg_rmse":
        return -score  # neg_rmse = -RMSE  →  error = RMSE
    if metric in ("logloss", "rmse", "nrmse"):
        return score   # already an error; lower is better
    return 1.0 - score  # roc_auc, balanced_accuracy: error = 1 − score


def compute_improvability(
    dataset_entries: list[tuple[str, dict[str, float]]],
    candidate_key: str,
) -> float | None:
    """Compute mean Improvability for *candidate_key* across datasets.

    Improvability measures the relative error gap between a method and the
    best-performing method on each dataset::

        Improvability_i(m) = (err_i(m) − err*_i) / err_i(m) × 100%

    where ``err*_i = min_{m'≠m} err_i(m')`` is the error of the best *baseline*
    on dataset *i* (the candidate itself is excluded).  Improvability is always
    in ``[0%, 100%]`` — 0% means the candidate matches or beats the best
    baseline, 100% means the best baseline has zero error while the candidate
    does not.

    Parameters
    ----------
    dataset_entries:
        ``[(metric_name, {model_name: score}), ...]``
        *metric_name* determines how scores are converted to errors (see
        :func:`_to_error`).  Scores are higher-is-better.
    candidate_key:
        Name of the candidate model in each score dict.

    Returns
    -------
    float or None
        Mean improvability in percent, or ``None`` if *candidate_key* is absent
        from every entry.
    """
    values: list[float] = []
    for metric, scores in dataset_entries:
        if candidate_key not in scores:
            continue
        errors = {m: _to_error(s, metric) for m, s in scores.items()}
        err_m = errors[candidate_key]
        baseline_errors = {m: e for m, e in errors.items() if m != candidate_key}
        if not baseline_errors:
            continue
        err_star = min(baseline_errors.values())
        if err_m <= 0.0:
            # Candidate has zero (or negative) error — already optimal.
            imp = 0.0
        else:
            imp = (err_m - err_star) / err_m * 100.0
        values.append(imp)
    return float(np.mean(values)) if values else None


def mean_metric_on_datasets(
    clf,
    datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    task_type: str = "cls",
    skip_on_error: bool = False,
) -> dict[str, float]:
    """Fit *clf* and score on each dataset; return mean scores across all datasets.

    Parameters
    ----------
    clf:
        Any sklearn-compatible estimator.
    datasets:
        List of ``(X_train, X_test, y_train, y_test)`` tuples.
    task_type:
        ``"cls"`` for classification or ``"reg"`` for regression.
    skip_on_error:
        If ``True``, catch exceptions during fit/predict and skip the failing
        dataset with a warning instead of propagating the error.

    Returns
    -------
    dict[str, float]
        ``{metric_name: mean_score}`` for all metrics (e.g. ``roc_auc``,
        ``logloss``, ``balanced_accuracy``, ``neg_rmse``).
        Returns ``{}`` if no datasets produced valid scores.
    """
    all_scores: dict[str, list[float]] = {}

    for X_train, X_test, y_train, y_test in datasets:
        problem_type = infer_problem_type(y_train, task_type)
        try:
            clf.fit(X_train, y_train)
            result = compute_metric(clf, X_test, y_test, problem_type)
        except Exception as exc:
            if skip_on_error:
                print(f"[mean_metric_on_datasets] WARNING: {type(exc).__name__}: {exc}. Skipping dataset.")
                # After a CUDA OOM the allocator may be fragmented; clearing the
                # cache before the next dataset avoids a cascade of OOM failures.
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
        for k, v in result.items():
            all_scores.setdefault(k, []).append(v)

    return {k: float(np.mean(v)) for k, v in all_scores.items()}


def compute_summary_metrics(
    summary_rows: list[tuple[str, dict[str, float], dict[str, float]]],
    candidate_name: str,
    benchmark_name: str,
    label: str,
    extra_baseline_keys: list[str] | None = None,
    improvability_exclude_keys: list[str] | None = None,
) -> dict[str, float]:
    """Compute wandb-ready summary metrics from evaluation results.

    This is the single source of truth for all aggregate metrics (skill score,
    improvability, win rates, etc.).  Both the candidate model (via
    :class:`BenchmarkValidator`) and baseline models (via
    :func:`log_baselines_to_wandb`) must call this function so that the same
    formulas are always used.

    Parameters
    ----------
    summary_rows:
        ``[(ds_name, cand_scores, baselines), ...]``
        *cand_scores* — full ``{metric: value}`` dict for the model being evaluated.
        *baselines*   — ``{model_name: primary_score}`` for comparison models.
    candidate_name:
        Name of the model being evaluated (used as the key in improvability).
    benchmark_name:
        Benchmark identifier (used to build wandb key prefixes).
    label:
        Suffix appended to every wandb key — e.g. ``"default_blocks"`` for a
        candidate or ``"tabpfn_v2_5"`` for a baseline.
    extra_baseline_keys:
        Additional baseline names for which win-rate metrics are computed.
    improvability_exclude_keys:
        Baseline model names to exclude from the improvability computation.
        These models still appear in results but do not affect ``err*``.

    Returns
    -------
    dict[str, float]
        Flat ``{wandb_key: value}`` dict ready to be logged.
    """
    prefix = f"results/{benchmark_name}"
    more_prefix = f"more_results/{benchmark_name}"
    out: dict[str, float] = {}

    # Primary performance metric (excludes logloss which is tracked separately)
    primary_scores = [
        cand_scores[_pick_primary(cand_scores)]
        for _, cand_scores, _ in summary_rows
        if cand_scores and _pick_primary(cand_scores) not in (None, "logloss")
    ]
    if primary_scores:
        out[f"{more_prefix}/performance/{label}"] = float(np.mean(primary_scores))

    # Per-metric averages
    for metric, metric_prefix in (
        ("logloss", prefix),
        ("pr_auc", more_prefix),
        ("mcc", more_prefix),
        ("r2", prefix),
        ("rmse", prefix),
        ("nrmse", prefix),
    ):
        vals = [cs[metric] for _, cs, _ in summary_rows if metric in cs and np.isfinite(cs[metric])]
        if vals:
            out[f"{metric_prefix}/{metric}/{label}"] = float(np.mean(vals))

    # Improvability
    _imp_exclude = set(improvability_exclude_keys or [])
    imp_entries: list[tuple[str, dict[str, float]]] = []
    for _, cand_scores, baselines in summary_rows:
        if not cand_scores:
            continue
        primary = _pick_primary(cand_scores)
        if primary is None:
            continue
        row: dict[str, float] = {candidate_name: cand_scores[primary]}
        row.update({k: v for k, v in baselines.items() if k not in _imp_exclude})
        imp_entries.append((primary, row))
    imp = compute_improvability(imp_entries, candidate_name)
    if imp is not None:
        out[f"{prefix}/improvability/{label}"] = imp

    # Aggregated stats: skill score, normalized score, win rates
    _more_stat_keys = {"mean_skill_score"}
    agg = compute_aggregated_stats(summary_rows, extra_baseline_keys=extra_baseline_keys)
    for stat_name, stat_val in agg.items():
        p = more_prefix if stat_name in _more_stat_keys else prefix
        out[f"{p}/{stat_name}/{label}"] = stat_val

    return out


def shifted_geometric_mean(values: list[float], shift: float = 1.0) -> float:
    """Shifted geometric mean of *values*.

    Avoids the instability of the plain geometric mean near zero by adding a
    constant *shift* before taking logarithms, then subtracting it afterwards::

        SGM_s = exp( mean( log(v_i + s) ) ) − s

    With ``shift=1`` and values in [0, 1] the inputs to ``log`` lie in [1, 2],
    so the result is always in [0, 1] and well-defined.

    Parameters
    ----------
    values:
        Sequence of non-negative floats (e.g. skill scores in [0, 1]).
    shift:
        Additive shift before log.  Default ``1.0`` is standard for [0, 1]
        inputs.

    Returns
    -------
    float
        SGM value, or ``0.0`` if *values* is empty.
    """
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(np.maximum(arr + shift, 1e-9)))) - shift)


def skill_score(score: float, metric: str, n_classes: int = 2) -> float:
    """Normalize *score* to [0, 1] where 0 = trivial baseline, 1 = perfect.

    Trivial baselines
    -----------------
    ``roc_auc``
        0.5 — random ranking.
    ``balanced_accuracy``
        1 / n_classes — random guessing.
    ``mcc``
        0.0 — random predictions yield MCC ≈ 0.
    ``r2``
        0.0 — predicting the mean.
    ``logloss``
        Trivial baseline = ``log(n_classes)`` (uniform prediction); perfect = 0.
        Skill score = ``1 − logloss / log(n_classes)``, clipped to [0, 1].
    ``pr_auc``, ``neg_rmse``
        Not normalizable without dataset-specific bounds; returns 0.0.

    Returns
    -------
    float
        Skill score clipped to [0, 1].
    """
    if metric == "roc_auc":
        return max(0.0, min(1.0, (score - 0.5) / 0.5))
    if metric == "balanced_accuracy":
        chance = 1.0 / max(n_classes, 2)
        return max(0.0, min(1.0, (score - chance) / (1.0 - chance)))
    if metric == "logloss":
        trivial = np.log(max(n_classes, 2))  # log(N) for N-class uniform prediction
        return max(0.0, min(1.0, 1.0 - score / trivial))
    if metric == "mcc":
        # MCC is in [-1, 1]; map [0, 1] to skill score (negative = worse than random)
        return max(0.0, min(1.0, score))
    if metric == "r2":
        return max(0.0, min(1.0, score))
    return 0.0  # pr_auc, neg_rmse and unknowns: not normalizable without bounds


def compute_aggregated_stats(
    summary_rows: list[tuple[str, dict[str, float], dict[str, float]]],
    baseline_key: str = "random_forest",
    extra_baseline_keys: list[str] | None = None,
) -> dict[str, float]:
    """Compute aggregate statistics across all datasets in *summary_rows*.

    For each dataset row the *primary* metric is the first non-logloss,
    non-metadata (``_``-prefixed) key in ``cand_scores``.  Metadata key
    ``_n_classes`` is used for ``balanced_accuracy`` skill score computation.
    For regression rows that contain both ``neg_rmse`` and ``r2``, the skill
    score is computed on ``r2`` (which has a meaningful trivial baseline of 0)
    even though ``neg_rmse`` remains the primary for ELO / comparison.

    Parameters
    ----------
    summary_rows:
        ``[(ds_name, cand_scores, baselines), ...]``
        *cand_scores* — ``{metric: value, "_n_classes": float (optional)}``
        *baselines*   — ``{model_name: primary_score}``
    baseline_key:
        Which baseline model to use for relative normalization and win-rate.
    extra_baseline_keys:
        Additional baseline model names for which only ``win_rate_vs_{key}``
        is computed (no normalised score).

    Returns
    -------
    dict[str, float]
        ``mean_skill_score``
            Average skill score across all datasets (candidate only).
        ``mean_normalized_vs_{baseline_key}``
            Average relative improvement of the candidate's skill score over
            the baseline's skill score, normalised by (1 − baseline skill).
        ``win_rate_vs_{baseline_key}``
            Fraction of datasets where the candidate beats the baseline on
            the primary metric (higher is better for roc_auc/balanced_accuracy/
            neg_rmse; lower is better for logloss).
        ``win_rate_vs_{k}`` (for each *k* in *extra_baseline_keys*)
            Same win-rate statistic for each extra baseline.
    """
    skill_scores: list[float] = []
    rel_scores: list[float] = []
    wins: list[float] = []

    for _, cand_scores, baselines in summary_rows:
        primary = _pick_primary(cand_scores)
        if primary is None:
            continue

        cand_val = cand_scores[primary]
        n_classes = int(round(cand_scores.get("_n_classes", 2.0)))

        # Use r2 for skill score when available (regression case), as neg_rmse
        # has no meaningful trivial baseline without dataset-specific bounds.
        skill_metric = "r2" if ("r2" in cand_scores and primary == "neg_rmse") else primary
        skill_val = cand_scores.get(skill_metric, cand_val)
        ss = skill_score(skill_val, skill_metric, n_classes)
        skill_scores.append(ss)

        if baseline_key in baselines:
            bl_val = baselines[baseline_key]
            bl_ss = skill_score(bl_val, skill_metric, n_classes)
            denom = 1.0 - bl_ss
            rel = (ss - bl_ss) / denom if denom > 1e-9 else 0.0
            rel_scores.append(float(rel))
            cand_wins = cand_val < bl_val if _lower_is_better(primary) else cand_val > bl_val
            wins.append(1.0 if cand_wins else 0.0)

    out: dict[str, float] = {}
    if skill_scores:
        out["mean_skill_score"] = float(np.mean(skill_scores))
        errors = [1.0 - ss for ss in skill_scores]
        out["sgm_skill_score"] = 1.0 - shifted_geometric_mean(errors, shift=0.01)
    if rel_scores:
        out[f"mean_normalized_vs_{baseline_key}"] = float(np.mean(rel_scores))
        clipped = [max(0.0, min(1.0, r)) for r in rel_scores]
        rel_errors = [1.0 - r for r in clipped]
        out[f"sgm_normalized_vs_{baseline_key}"] = 1.0 - shifted_geometric_mean(rel_errors, shift=0.01)
    if wins:
        out[f"win_rate_vs_{baseline_key}"] = float(np.mean(wins))
    for extra_key in (extra_baseline_keys or []):
        extra_wins: list[float] = []
        for _, cand_scores, baselines in summary_rows:
            primary = _pick_primary(cand_scores)
            if primary is None or extra_key not in baselines:
                continue
            cand_wins = cand_scores[primary] < baselines[extra_key] if _lower_is_better(primary) else cand_scores[primary] > baselines[extra_key]
            extra_wins.append(1.0 if cand_wins else 0.0)
        if extra_wins:
            out[f"win_rate_vs_{extra_key}"] = float(np.mean(extra_wins))
    return out
