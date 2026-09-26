"""Build a leaderboard CSV from benchmark_no_tabarena results.

Usage
-----
    uv run benchmark_no_tabarena/leaderboard.py results/tabarena-lite/
    uv run benchmark_no_tabarena/leaderboard.py results/tabarena-lite/ --output leaderboard.csv
    uv run benchmark_no_tabarena/leaderboard.py results/tabarena-lite/ --reference-model RandomForest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _build_match_list(
    task_scores: pd.DataFrame,
    task_ids: np.ndarray,
) -> list[tuple[str, str, float, float]]:
    """Return pairwise matches for the given (possibly repeated) task_ids."""
    matches: list[tuple[str, str, float, float]] = []
    scores_by_task = {
        tid: grp.set_index("model")["mean_score"]
        for tid, grp in task_scores.groupby("task_id")
    }
    for tid in task_ids:
        rows = scores_by_task[tid]
        model_list = rows.index.tolist()
        for i, ma in enumerate(model_list):
            for mb in model_list[i + 1 :]:
                matches.append((ma, mb, float(rows[ma]), float(rows[mb])))
    return matches


def _run_elo(
    matches: list[tuple[str, str, float, float]],
    models: np.ndarray,
    k: float,
    base_rating: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One randomised pass of ELO updates over all matches."""
    ratings: dict[str, float] = {m: base_rating for m in models}
    for i in rng.permutation(len(matches)):
        ma, mb, sa, sb = matches[i]
        ra, rb = ratings[ma], ratings[mb]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        wa = 1.0 if sa > sb else (0.0 if sa < sb else 0.5)
        wb = 1.0 - wa
        ratings[ma] = ra + k * (wa - ea)
        ratings[mb] = rb + k * (wb - (1.0 - ea))
    return ratings


def compute_elo(
    task_scores: pd.DataFrame,
    k: float = 32.0,
    reference_model: str | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute ELO ratings with 95 % bootstrap confidence intervals.

    Parameters
    ----------
    task_scores:
        DataFrame with columns ``model``, ``task_id``, ``mean_score``.
    k:
        ELO update factor (default 32).
    reference_model:
        If given, all ratings are shifted so this model sits at exactly 1000.
        Otherwise ratings are centred on 1000 at initialisation.
    n_bootstrap:
        Number of bootstrap rounds (resample tasks with replacement).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    DataFrame indexed by ``model`` with columns ``elo``, ``elo_lower``,
    ``elo_upper`` (2.5th / 97.5th percentile over bootstrap samples).

    Notes
    -----
    A 400-point ELO gap corresponds to a 10:1 (~91 %) expected win rate,
    matching the standard ELO formula.  Calibration and bootstrap follow
    Chiang et al. (Chatbot Arena, 2024).
    """
    rng = np.random.default_rng(seed)
    models = task_scores["model"].unique()
    task_ids = task_scores["task_id"].unique()

    def _calibrate(ratings: dict[str, float]) -> dict[str, float]:
        if reference_model is not None and reference_model in ratings:
            shift = 1000.0 - ratings[reference_model]
            return {m: v + shift for m, v in ratings.items()}
        return ratings

    # Point estimate — single pass over all tasks
    all_matches = _build_match_list(task_scores, task_ids)
    point = _calibrate(_run_elo(all_matches, models, k, 1000.0, rng))

    # Bootstrap: resample tasks with replacement
    boot_records: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        boot_matches = _build_match_list(task_scores, sampled)
        boot_records.append(_calibrate(_run_elo(boot_matches, models, k, 1000.0, rng)))

    boot_df = pd.DataFrame(boot_records)  # shape: (n_bootstrap, n_models)

    result = pd.DataFrame(
        {
            "elo": pd.Series(point).round(1),
            "elo_lower": boot_df.quantile(0.025).round(1),
            "elo_upper": boot_df.quantile(0.975).round(1),
        }
    )
    result.index.name = "model"
    return result


def build_leaderboard(
    results_dir: Path,
    output_path: Path | None = None,
    reference_model: str | None = None,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    """Aggregate all per-model results.csv files into a single leaderboard.

    Returns a DataFrame with one row per model, ranked by ELO.
    Higher is always better (roc_auc / neg_log_loss / neg_rmse).
    """
    results_dir = Path(results_dir)
    dfs = []
    for csv_path in sorted(results_dir.glob("*/results.csv")):
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception as exc:
            print(f"  [skip] {csv_path}: {exc}")

    if not dfs:
        print(f"No results.csv found under {results_dir}")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Average over folds/repeats per (model, task)
    task_scores = (
        df.groupby(["model", "task_id", "dataset", "problem_type", "metric"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "mean_score"})
    )

    # Overall mean per model (across tasks)
    overall = (
        task_scores.groupby("model")["mean_score"]
        .mean()
        .reset_index()
        .rename(columns={"mean_score": "mean_score_overall"})
    )

    # Pivot: rows=model, columns=task_id
    pivot = task_scores.pivot_table(
        index="model", columns="task_id", values="mean_score", aggfunc="mean"
    )

    elo_df = compute_elo(
        task_scores, reference_model=reference_model, n_bootstrap=n_bootstrap
    )
    leaderboard = (
        overall.set_index("model")
        .join(elo_df)
        .join(pivot)
        .reset_index()
        .sort_values("elo", ascending=False)
    )

    # Print summary
    n_tasks = task_scores["task_id"].nunique()
    ref_note = f", reference={reference_model}" if reference_model else ""
    print(f"\nLeaderboard ({len(overall)} models, {n_tasks} tasks{ref_note})")
    summary = leaderboard[["model", "mean_score_overall", "elo", "elo_lower", "elo_upper"]].copy()
    summary["elo_ci"] = (
        summary["elo"].astype(str)
        + " ["
        + summary["elo_lower"].astype(str)
        + ", "
        + summary["elo_upper"].astype(str)
        + "]"
    )
    print(summary[["model", "mean_score_overall", "elo_ci"]].to_string(index=False))

    if output_path is None:
        output_path = results_dir / "leaderboard.csv"
    leaderboard.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return leaderboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing per-model result folders.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument(
        "--reference-model",
        default=None,
        help="Model to anchor at ELO 1000 (e.g. RandomForest).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap rounds for 95%% CI (default 200).",
    )
    args = parser.parse_args()
    build_leaderboard(
        Path(args.results_dir),
        Path(args.output) if args.output else None,
        reference_model=args.reference_model,
        n_bootstrap=args.n_bootstrap,
    )
