"""Core experiment runner — no tabarena, no autogluon.

Results are saved as pickle files, one per (task_id, fold, repeat):

    <output_dir>/<model_name>/<task_id>/<fold>_<repeat>.pkl

Each pickle contains a single dict with keys:
    task_id, dataset, problem_type, fold, repeat, model,
    metric, score, logloss, fit_time, predict_time,
    n_train, n_test,
    predictions, predict_probas, true_labels,
    hs_metrics,      # dict[str, np.ndarray] or None
    loop_logits,     # list[np.ndarray] or None
    loop_loglosses,  # list[float] or None
"""
from __future__ import annotations

import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd  # used in type hints for compute_metric


# ---------------------------------------------------------------------------
# Experiment descriptor
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    factory: Callable  # (problem_type: str) -> model instance
    exp_type: str | None = None  # optional subfolder (e.g. "hidden_states_analysis")
    folder: str | None = None   # override output subfolder (from config "folder" key)
    subsample_train: int | None = None  # cap train set size before model.fit
    subsample_test: int | None = None   # cap test set size before scoring
    save_mode: str = "per_task"  # "per_task": one pkl per (task,fold,repeat); "single_file": one results.pkl per model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metric(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> tuple[str, float, np.ndarray, np.ndarray | None]:
    """Returns (metric_name, score, predictions, predict_probas).

    predict_probas is None for regression tasks.
    """
    from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

    if problem_type == "binary":
        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)
        if proba.ndim == 2:
            proba_score = proba[:, 1]
        else:
            proba_score = proba
        score = roc_auc_score(y_test, proba_score)
        return "roc_auc", float(score), pred, proba

    elif problem_type == "multiclass":
        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        else:
            classes = sorted(y_test.unique())
        score = -log_loss(y_test, proba, labels=classes)
        return "neg_log_loss", float(score), pred, proba

    else:  # regression
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        return "neg_rmse", float(-rmse), pred, None


# ---------------------------------------------------------------------------
# Per-loop scoring
# ---------------------------------------------------------------------------

def _compute_logloss(
    predict_probas: np.ndarray,
    y_test: pd.Series,
    problem_type: str,
) -> float | None:
    """Return log loss for classification tasks, None for regression."""
    if problem_type not in ("binary", "multiclass"):
        return None
    from sklearn.metrics import log_loss
    classes = sorted(y_test.unique())
    return float(log_loss(y_test.to_numpy(), predict_probas, labels=classes))


def _score_per_loop(
    loop_probas: list[np.ndarray],
    y_test: pd.Series,
    problem_type: str,
) -> list[float]:
    """Compute the task metric for every loop's predicted probabilities.

    Parameters
    ----------
    loop_probas : list of (n_test, n_classes) arrays — one per loop
    y_test      : ground-truth labels
    problem_type: "binary" | "multiclass" | "regression"

    Returns
    -------
    list of float, length = n_loops
    """
    from sklearn.metrics import roc_auc_score, log_loss

    scores = []
    y_np = y_test.to_numpy()

    for proba in loop_probas:
        if problem_type == "binary":
            score = roc_auc_score(y_np, proba[:, 1])
        elif problem_type == "multiclass":
            classes = sorted(y_test.unique())
            score = -log_loss(y_np, proba, labels=classes)
        else:
            score = float("nan")
        scores.append(float(score))

    return scores


def _logloss_per_loop(
    loop_probas: list[np.ndarray],
    y_test: pd.Series,
    problem_type: str,
) -> list[float] | None:
    """Return log loss at each loop step for classification tasks, None for regression."""
    if problem_type not in ("binary", "multiclass"):
        return None
    return [
        _compute_logloss(proba, y_test, problem_type)
        for proba in loop_probas
    ]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiments(
    experiments: list[Experiment],
    tasks: list[int],
    output_dir: Path,
    repetitions: str = "lite",  # "lite" = (fold=0, repeat=0) only; "folds3" = folds 0-2 of repeat 0; "full" = all
    raise_on_failure: bool = False,
) -> None:
    """Run each experiment on each task and save one pickle per (fold, repeat).

    Output layout:
        <output_dir>/<model_name>/<task_id>/<fold>_<repeat>.pkl

    Parameters
    ----------
    repetitions : "lite" | "folds3" | "full"
        "lite"    → only fold=0, repeat=0 per task.
        "folds3"  → folds 0, 1, 2 of repeat 0.
        "full"    → all (repeat, fold) pairs from the OpenML task definition.
    """
    from datasets.openml_loader import load_task, get_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_id in tasks:
        print(f"\n  Loading task {task_id} …")
        try:
            task, X, y, problem_type = load_task(task_id)
            dataset_name = task.get_dataset().name
        except Exception as exc:
            print(f"  [ERROR] Failed to load task {task_id}: {exc}")
            if raise_on_failure:
                raise
            continue

        if repetitions == "lite":
            folds_repeats = [(0, 0)]
        elif repetitions == "folds3":
            folds_repeats = [(r, 0) for r in range(3)]
        else:
            n_repeats, n_folds, _ = task.get_split_dimensions()
            folds_repeats = [(f, r) for r in range(n_repeats) for f in range(n_folds)]

        print(f"  Task {task_id} ({dataset_name}): {problem_type}, "
              f"{len(folds_repeats)} split(s)")

        for experiment in experiments:
            exp_base = (
                output_dir / experiment.exp_type / experiment.folder
                if experiment.folder and experiment.exp_type else
                output_dir / experiment.folder
                if experiment.folder else
                output_dir / experiment.exp_type / experiment.name
                if experiment.exp_type else
                output_dir / experiment.name
            )
            exp_base.mkdir(parents=True, exist_ok=True)

            # single_file mode: load existing results.pkl for resuming
            if experiment.save_mode == "single_file":
                out_file = exp_base / "results.pkl"
                if out_file.exists():
                    with open(out_file, "rb") as f:
                        _results: list[dict] = pickle.load(f)
                else:
                    _results = []
                _done = {(r["task_id"], r["fold"], r["repeat"]) for r in _results}

            for fold, repeat in folds_repeats:
                if experiment.save_mode == "single_file":
                    if (task_id, fold, repeat) in _done:
                        continue
                else:
                    task_dir = exp_base / str(task_id)
                    task_dir.mkdir(parents=True, exist_ok=True)
                    out_file = task_dir / f"{fold}_{repeat}.pkl"
                    if out_file.exists():
                        try:
                            with open(out_file, "rb") as f:
                                pickle.load(f)
                            continue  # valid file — skip
                        except Exception:
                            pass  # truncated/corrupt — re-run

                try:
                    X_train, y_train, X_test, y_test = get_split(
                        task, X, y, fold=fold, repeat=repeat
                    )
                except Exception as exc:
                    print(f"    [skip] task={task_id} fold={fold} rep={repeat}: {exc}")
                    continue

                try:
                    seed = fold * 100 + repeat
                    rng = np.random.default_rng(seed=seed)
                    if (experiment.subsample_train is not None
                            and len(X_train) > experiment.subsample_train):
                        if problem_type in ("binary", "multiclass"):
                            from sklearn.model_selection import StratifiedShuffleSplit
                            sss = StratifiedShuffleSplit(n_splits=1, train_size=experiment.subsample_train, random_state=seed)
                            idx, _ = next(sss.split(X_train, y_train))
                        else:
                            idx = rng.choice(len(X_train), experiment.subsample_train, replace=False)
                        X_train = X_train.iloc[idx].reset_index(drop=True)
                        y_train = y_train.iloc[idx].reset_index(drop=True)
                    if (experiment.subsample_test is not None
                            and len(X_test) > experiment.subsample_test):
                        if problem_type in ("binary", "multiclass"):
                            from sklearn.model_selection import StratifiedShuffleSplit
                            sss = StratifiedShuffleSplit(n_splits=1, train_size=experiment.subsample_test, random_state=seed)
                            idx, _ = next(sss.split(X_test, y_test))
                        else:
                            idx = rng.choice(len(X_test), experiment.subsample_test, replace=False)
                        X_test = X_test.iloc[idx].reset_index(drop=True)
                        y_test = y_test.iloc[idx].reset_index(drop=True)

                    model = experiment.factory(problem_type)

                    t0 = time.time()
                    model.fit(X_train, y_train, problem_type)
                    fit_time = time.time() - t0

                    t0 = time.time()
                    metric_name, score, predictions, predict_probas = compute_metric(
                        model, X_test, y_test, problem_type
                    )
                    predict_time = time.time() - t0

                    result = {
                        "task_id":        task_id,
                        "dataset":        dataset_name,
                        "problem_type":   problem_type,
                        "fold":           fold,
                        "repeat":         repeat,
                        "model":          experiment.name,
                        "metric":         metric_name,
                        "score":          score,
                        "logloss":        _compute_logloss(predict_probas, y_test, problem_type),
                        "fit_time":       round(fit_time, 3),
                        "predict_time":   round(predict_time, 3),
                        "n_train":        len(X_train),
                        "n_test":         len(X_test),
                        "predictions":    predictions,
                        "predict_probas": predict_probas,
                        "true_labels":    y_test.to_numpy(),
                        "train_labels":   None,
                        "train_predictions": None,
                        "hs_metrics":     None,
                        "loop_logits":    None,
                        "loop_scores":    None,
                        "loop_loglosses": None,
                        "loop_embeddings": None,
                        "kv_rank_metrics": None,
                        "kv_compression_metrics": None,
                        "quantization_metrics": None,
                        "attention_head_metrics": None,
                        "exit_gate_values": None,
                    }

                    if getattr(model, "save_train_data", False):
                        result["train_labels"]      = y_train.to_numpy()
                        result["train_predictions"] = model.predict_proba(X_train)

                    if getattr(model, "capture_loop_predictions", False):
                        model.compute_loop_predictions(X_train, y_train, X_test)
                        result["loop_logits"]    = model.loop_logits
                        result["loop_scores"]    = _score_per_loop(
                            model.loop_logits, y_test, problem_type
                        )
                        result["loop_loglosses"] = _logloss_per_loop(
                            model.loop_logits, y_test, problem_type
                        )

                    if getattr(model, "capture_hidden_states", False) or getattr(model, "capture_lyapunov", False):
                        capture_emb = getattr(model, "capture_embeddings", False)
                        capture_test_emb = getattr(model, "capture_test_embeddings", False)
                        model.compute_hs_metrics(X_train, y_train, X_test,
                                                 capture_embeddings=capture_emb,
                                                 capture_test_embeddings=capture_test_emb)
                        result["hs_metrics"]      = model.hs_metrics
                        result["loop_logits"]     = model.loop_logits
                        result["loop_embeddings"] = model.loop_embeddings
                        result["loop_scores"]     = _score_per_loop(
                            model.loop_logits, y_test, problem_type
                        )
                        result["loop_loglosses"]  = _logloss_per_loop(
                            model.loop_logits, y_test, problem_type
                        )

                    if getattr(model, "capture_exit_steps", False):
                        model.compute_exit_gate_values(X_train, y_train, X_test)
                        result["exit_gate_values"] = model.exit_gate_values
                        result["loop_logits"]      = model.loop_logits
                        result["loop_scores"]      = _score_per_loop(
                            model.loop_logits, y_test, problem_type
                        )
                        result["loop_loglosses"]   = _logloss_per_loop(
                            model.loop_logits, y_test, problem_type
                        )

                    if getattr(model, "capture_kv_rank", False):
                        t0 = time.time()
                        model.compute_kv_rank_metrics(X_train, y_train, X_test)
                        result["kv_rank_metrics"] = model.kv_rank_metrics
                        print(f"      [kv_rank total]  {time.time()-t0:.2f}s")

                    if getattr(model, "capture_kv_compression", False):
                        t0 = time.time()
                        model.compute_kv_compression_metrics(
                            X_train, y_train, X_test, y_test, problem_type,
                            n_timing_repeats=getattr(model, "_n_timing_repeats", 3),
                        )
                        result["kv_compression_metrics"] = model.kv_compression_metrics
                        print(f"      [kv_compress total]  {time.time()-t0:.2f}s")

                    if getattr(model, "capture_quantization", False):
                        t0 = time.time()
                        model.compute_quantization_metrics(
                            X_train, y_train, X_test, y_test, problem_type,
                            n_timing_repeats=getattr(model, "_n_timing_repeats", 5),
                        )
                        result["quantization_metrics"] = model.quantization_metrics
                        print(f"      [quantization total]  {time.time()-t0:.2f}s")

                    if (getattr(model, "capture_attention_heads", False)
                            or getattr(model, "capture_head_ablation", False)):
                        t0 = time.time()
                        model.compute_attention_head_metrics(
                            X_train, y_train, X_test, y_test
                        )
                        result["attention_head_metrics"] = model.attention_head_metrics
                        print(f"      [attention_heads total]  {time.time()-t0:.2f}s")

                    if experiment.save_mode == "single_file":
                        _results.append(result)
                        _done.add((task_id, fold, repeat))
                        tmp = out_file.with_suffix(".tmp")
                        with open(tmp, "wb") as f:
                            pickle.dump(_results, f)
                        tmp.replace(out_file)
                    else:
                        tmp = out_file.with_suffix(".tmp")
                        with open(tmp, "wb") as f:
                            pickle.dump(result, f)
                        tmp.replace(out_file)

                    print(f"    {experiment.name:40s} task={task_id} "
                          f"fold={fold} rep={repeat} "
                          f"{metric_name}={score:.4f}  [{fit_time:.1f}s]  → {out_file}")

                except FileNotFoundError as exc:
                    print(f"    [skip] {experiment.name} | "
                          f"task={task_id} fold={fold} rep={repeat}: {exc}")
                    if raise_on_failure:
                        raise
                except Exception as exc:
                    print(f"    [ERROR] {experiment.name} | "
                          f"task={task_id} fold={fold} rep={repeat}: {exc}")
                    traceback.print_exc()
                    if raise_on_failure:
                        raise
                finally:
                    try:
                        import gc
                        import torch
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Synthetic-prior runner
# ---------------------------------------------------------------------------


def run_synthetic_experiments(
    experiments: list[Experiment],
    prior_dir: Path,
    output_dir: Path,
    raise_on_failure: bool = False,
) -> None:
    """Run experiments on pre-generated synthetic prior datasets.

    Expects ``prior_dir`` to contain files named ``{idx:06d}.pkl`` as written
    by ``generate_synthetic_priors.py``.

    Output layout mirrors the OpenML runner:
        <output_dir>/<model_name>/<dataset_id>/0_0.pkl
    """
    prior_dir = Path(prior_dir)
    dataset_files = sorted(prior_dir.glob("[0-9]" * 6 + ".pkl"))
    if not dataset_files:
        print(f"[ERROR] No dataset .pkl files found in {prior_dir}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_file in dataset_files:
        dataset_id = int(ds_file.stem)
        print(f"\n  Loading synthetic dataset {dataset_id} …")
        try:
            with open(ds_file, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {ds_file}: {exc}")
            continue

        X_arr       = data["X"]           # (seq_len, n_features)
        y_arr       = data["y"]           # (seq_len,)
        train_size  = data["train_size"]
        problem_type = data["problem_type"]
        dataset_name = f"synthetic_{dataset_id:06d}"

        X_df     = pd.DataFrame(X_arr, columns=[f"f{j}" for j in range(X_arr.shape[1])])
        y_series = pd.Series(y_arr)

        X_train = X_df.iloc[:train_size].reset_index(drop=True)
        y_train = y_series.iloc[:train_size].reset_index(drop=True)
        X_test  = X_df.iloc[train_size:].reset_index(drop=True)
        y_test  = y_series.iloc[train_size:].reset_index(drop=True)

        print(f"  Dataset {dataset_id} ({dataset_name}): {problem_type}, "
              f"train={len(X_train)}, test={len(X_test)}")

        for experiment in experiments:
            exp_base = (
                output_dir / experiment.exp_type / experiment.folder
                if experiment.folder and experiment.exp_type else
                output_dir / experiment.folder
                if experiment.folder else
                output_dir / experiment.exp_type / experiment.name
                if experiment.exp_type else
                output_dir / experiment.name
            )
            exp_base.mkdir(parents=True, exist_ok=True)

            # single_file mode: load existing results.pkl for resuming
            if experiment.save_mode == "single_file":
                out_file = exp_base / "results.pkl"
                if out_file.exists():
                    with open(out_file, "rb") as f:
                        _results: list[dict] = pickle.load(f)
                else:
                    _results = []
                _done = {r["task_id"] for r in _results}
                if dataset_id in _done:
                    continue
            else:
                task_dir = exp_base / str(dataset_id)
                task_dir.mkdir(parents=True, exist_ok=True)
                out_file = task_dir / "0_0.pkl"
                if out_file.exists():
                    try:
                        with open(out_file, "rb") as f:
                            pickle.load(f)
                        continue  # valid file — skip
                    except Exception:
                        pass  # truncated/corrupt — re-run

            try:
                model = experiment.factory(problem_type)

                t0 = time.time()
                model.fit(X_train, y_train, problem_type)
                fit_time = time.time() - t0

                t0 = time.time()
                metric_name, score, predictions, predict_probas = compute_metric(
                    model, X_test, y_test, problem_type
                )
                predict_time = time.time() - t0

                result = {
                    "task_id":         dataset_id,
                    "dataset":         dataset_name,
                    "problem_type":    problem_type,
                    "fold":            0,
                    "repeat":          0,
                    "model":           experiment.name,
                    "metric":          metric_name,
                    "score":           score,
                    "logloss":         _compute_logloss(predict_probas, y_test, problem_type),
                    "fit_time":        round(fit_time, 3),
                    "predict_time":    round(predict_time, 3),
                    "n_train":         len(X_train),
                    "n_test":          len(X_test),
                    "predictions":     predictions,
                    "predict_probas":  predict_probas,
                    "true_labels":     y_test.to_numpy(),
                    "hs_metrics":      None,
                    "loop_logits":     None,
                    "loop_scores":     None,
                    "loop_loglosses":  None,
                    "kv_rank_metrics": None,
                }

                if experiment.save_mode == "single_file":
                    _results.append(result)
                    _done.add(dataset_id)
                    tmp = out_file.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        pickle.dump(_results, f)
                    tmp.replace(out_file)
                else:
                    tmp = out_file.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        pickle.dump(result, f)
                    tmp.replace(out_file)

                print(f"    {experiment.name:40s} ds={dataset_id} "
                      f"{metric_name}={score:.4f}  [{fit_time:.1f}s]  → {out_file}")

            except Exception as exc:
                print(f"    [ERROR] {experiment.name} | ds={dataset_id}: {exc}")
                traceback.print_exc()
                if raise_on_failure:
                    raise
