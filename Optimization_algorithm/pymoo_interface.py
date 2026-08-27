from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np

from Optimization_algorithm.optimization_history import HistoryRecorder
from Optimization_algorithm.optimization_history import normalize_bounds
from Optimization_algorithm.optimization_history import save_history


class PymooOptimizationProblem:
    def __new__(
        cls,
        *,
        objective: Callable[[np.ndarray], float],
        bounds: Any,
        recorder: HistoryRecorder,
        decode_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        vtype: Any | None = None,
    ):
        try:
            from pymoo.core.problem import ElementwiseProblem
        except ImportError as exc:
            raise ImportError("pymoo is required. Install it with: pip install pymoo") from exc

        class _Problem(ElementwiseProblem):
            def __init__(self) -> None:
                problem_bounds = normalize_bounds(bounds)
                super().__init__(
                    n_var=problem_bounds.shape[0],
                    n_obj=1,
                    xl=problem_bounds[:, 0],
                    xu=problem_bounds[:, 1],
                    vtype=vtype,
                )

            def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
                x_array = np.asarray(x, dtype=float)
                x_eval = decode_fn(x_array) if decode_fn else x_array
                fom = float(objective(x_eval))
                recorder.record(x_eval, fom)
                out["F"] = -fom

        return _Problem()


def run_pymoo(
    *,
    algorithm: Any,
    algorithm_name: str,
    objective: Callable[[np.ndarray], float],
    bounds: Any,
    max_evals: int,
    seed: int | None,
    history_file: str | Path,
    variable_names: list[str] | None = None,
    physical_bounds: Any | None = None,
    decode_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    vtype: Any | None = None,
    extra: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    try:
        from pymoo.optimize import minimize
    except ImportError as exc:
        raise ImportError("pymoo is required. Install it with: pip install pymoo") from exc

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    recorder = HistoryRecorder()
    problem = PymooOptimizationProblem(
        objective=objective,
        bounds=bounds,
        recorder=recorder,
        decode_fn=decode_fn,
        vtype=vtype,
    )

    start_time = time.time()
    result = minimize(
        problem,
        algorithm,
        termination=("n_eval", max_evals),
        seed=seed,
        verbose=verbose,
        save_history=False,
    )
    runtime_seconds = time.time() - start_time

    history_extra = {
        "seed": seed,
        "requested_max_evals": max_evals,
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60.0,
    }
    history_extra.update(extra or {})

    payload = save_history(
        algorithm=algorithm_name,
        x_history=recorder.x_history,
        fom_history=recorder.fom_history,
        output_path=history_file,
        variable_names=variable_names,
        physical_bounds=physical_bounds,
        extra=history_extra,
    )

    best_x = decode_fn(result.X) if decode_fn else result.X
    return {
        "best_x": best_x,
        "best_fom": -float(np.asarray(result.F).reshape(-1)[0]),
        "history": payload,
        "raw_result": result,
        "runtime_seconds": runtime_seconds,
    }
