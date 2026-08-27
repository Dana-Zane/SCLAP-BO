from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable

import numpy as np
import optuna

from Optimization_algorithm.optimization_history import save_history


def save_optuna_history(
    study: optuna.Study,
    *,
    algorithm: str,
    output_path: str | Path,
    variable_names: Iterable[str],
    physical_bounds: Any | None = None,
    x_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = list(variable_names)
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    x_history = np.array(
        [[trial.params[name] for name in names] for trial in completed_trials],
        dtype=float,
    )
    if x_transform:
        x_history = np.asarray(x_transform(x_history), dtype=float)
    fom_history = np.array([trial.value for trial in completed_trials], dtype=float)
    return save_history(
        algorithm=algorithm,
        x_history=x_history,
        fom_history=fom_history,
        output_path=output_path,
        variable_names=names,
        physical_bounds=physical_bounds,
        extra=extra,
    )


def make_optuna_history_callback(
    *,
    algorithm: str,
    output_path: str | Path,
    variable_names: Iterable[str],
    physical_bounds: Any | None = None,
    extra: dict[str, Any] | None = None,
):
    def callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        save_optuna_history(
            study,
            algorithm=algorithm,
            output_path=output_path,
            variable_names=variable_names,
            physical_bounds=physical_bounds,
            extra=extra,
        )

    return callback
