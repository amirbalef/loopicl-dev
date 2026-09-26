"""Monkey-patches for upstream tabarena bugs.

Import this module early (before tabarena is used) to apply all fixes.
"""


def _fix_parse_repetitions_mode():
    """Fix a bug in tabarena's _parse_repetitions_mode_and_args.

    The upstream code does:
        metadata_task_ids = tasks_metadata["task_id"].astype(str).tolist()
        ...
        task_meta = tasks_metadata[metadata_task_ids == t_id].iloc[0]

    .tolist() converts the Series to a plain Python list.  Comparing a list to
    a scalar with == returns a single bool (False), which pandas then treats as
    a column key → KeyError: False.

    Fix: keep metadata_task_ids as a pandas Series so element-wise == works.

    This patch is a no-op on newer tabarena versions that have already fixed or
    removed _parse_repetitions_mode_and_args.
    """
    import tabarena.benchmark.experiment.experiment_runner_api as _mod

    # Newer tabarena versions removed this function entirely — nothing to patch.
    if not hasattr(_mod, "_parse_repetitions_mode_and_args"):
        return

    # UserTask moved from experiment_runner_api to benchmark.task.user_task in
    # newer releases; try both locations and fall back to None so the isinstance
    # check below is simply always False (all tasks are plain ints in that case).
    try:
        from tabarena.benchmark.experiment.experiment_runner_api import UserTask
    except ImportError:
        try:
            from tabarena.benchmark.task.user_task import UserTask
        except ImportError:
            UserTask = None  # type: ignore[assignment,misc]

    _original = _mod._parse_repetitions_mode_and_args

    def _patched(
        *,
        repetitions_mode,
        repetitions_mode_args=None,
        tasks,
        tasks_metadata=None,
    ):
        if repetitions_mode == "TabArena":
            if tasks_metadata is None:
                from tabarena.nips2025_utils.fetch_metadata import load_curated_task_metadata
                tasks_metadata = load_curated_task_metadata()
            else:
                req_columns = ["tabarena_num_repeats", "num_folds", "task_id"]
                for col in req_columns:
                    assert col in tasks_metadata.columns, (
                        f"`tasks_metadata` must contain the column '{col}' "
                        f"when `repetitions_mode` is 'TabArena'"
                    )
            # Keep as Series (not list) so == does element-wise comparison.
            metadata_task_ids = tasks_metadata["task_id"].astype(str)
            fold_repeat_pairs_per_task = []
            for task in tasks:
                t_id = task.task_id_str if (UserTask is not None and isinstance(task, UserTask)) else str(task)
                assert t_id in metadata_task_ids.values, (
                    f"Task ID '{t_id}' from `tasks` not found in `tasks_metadata`"
                )
                task_meta = tasks_metadata[metadata_task_ids == t_id].iloc[0]
                n_folds = int(task_meta["num_folds"])
                n_repeats = int(task_meta["tabarena_num_repeats"])
                fold_repeat_pairs = [
                    (f, r) for r in range(n_repeats) for f in range(n_folds)
                ]
                fold_repeat_pairs_per_task.append(fold_repeat_pairs)
            return fold_repeat_pairs_per_task

        # All other modes: delegate to the original.
        return _original(
            repetitions_mode=repetitions_mode,
            repetitions_mode_args=repetitions_mode_args,
            tasks=tasks,
            tasks_metadata=tasks_metadata,
        )

    _mod._parse_repetitions_mode_and_args = _patched


def _fix_agsingle_wrapper_refit():
    """Add refit_on_val support to AGSingleWrapper.

    Without this patch, AutoGluon holds out ~10-20% of train_data internally
    (for validation scoring), so the model only trains on 80-90% of the
    available data.

    When fit_kwargs contains refit_on_val=True, this patch provides an explicit
    tuning_data (20% subset of X) to AutoGluon. This prevents AutoGluon from
    creating its own internal holdout, so train_data (100% of X) is fully used
    for training. For foundation models like TabICL (no early stopping),
    tuning_data only affects validation scoring, not training.

    Usage: pass method_kwargs={"fit_kwargs": {"refit_on_val": True}} to
    AGModelExperiment.
    """
    try:
        import tabarena.benchmark.models.wrapper.AutoGluon_class as _ag_mod
    except ModuleNotFoundError:
        try:
            import tabarena.benchmark.exec_models.autogluon as _ag_mod
        except ModuleNotFoundError:
            return

    _original_fit = _ag_mod.AGSingleWrapper._fit

    def _patched_fit(self, X, y, X_val=None, y_val=None, **kwargs):
        # Pop refit_on_val before _build_predictor_args deep-copies self.fit_kwargs.
        refit_on_val = self.fit_kwargs.pop("refit_on_val", False)
        if hasattr(self, "fit_kwargs_extra"):
            self.fit_kwargs_extra.pop("refit_on_val", None)

        if refit_on_val and X_val is None:
            # Provide an explicit tuning split so AutoGluon skips its internal
            # holdout and trains on 100% of train_data.
            from sklearn.model_selection import train_test_split
            try:
                _, X_val, _, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except Exception:
                X_val, y_val = X, y
            # Stratified split can silently put all minority samples in train
            # when the class is very rare — fall back to full data in that case.
            if len(set(y_val)) < len(set(y)):
                X_val, y_val = X, y

        return _original_fit(self, X, y, X_val=X_val, y_val=y_val, **kwargs)

    _ag_mod.AGSingleWrapper._fit = _patched_fit


def _fix_autogluon_fillna_downcast():
    """Remove the unsupported `downcast` kwarg from autogluon's FillNaFeatureGenerator.

    Pandas removed the `downcast` parameter from DataFrame.fillna() in pandas 2.2+.
    AutoGluon's fillna.py still passes it, causing a TypeError at fit time.
    """
    try:
        import autogluon.features.generators.fillna as _mod
    except ImportError:
        return

    _original_transform = _mod.FillNaFeatureGenerator._transform

    def _patched_transform(self, X):
        import pandas as pd
        # pandas >= 2.2 removed the `downcast` kwarg; call without it.
        X.fillna(self._fillna_feature_map, inplace=True)
        return X

    _mod.FillNaFeatureGenerator._transform = _patched_transform


_fix_parse_repetitions_mode()
_fix_agsingle_wrapper_refit()
_fix_autogluon_fillna_downcast()
