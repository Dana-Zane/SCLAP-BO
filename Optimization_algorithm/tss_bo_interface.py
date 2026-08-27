from __future__ import annotations

import random
import time
from typing import Any
from typing import Dict

import numpy as np
import torch

from Optimization_algorithm.optimization_history import HistoryRecorder
from Optimization_algorithm.optimization_history import save_history


def run_tss_bo_from_dict(config: Dict[str, Any]) -> None:
    return run_tss_bo(config)


def run_tss_bo(config: Dict[str, Any]) -> None:
    cfg = dict(config)
    seed = cfg.pop("seed", None)
    if seed is not None:
        _set_random_seed(seed)

    try:
        from .tSS_BO.src.main import main_solver
    except ImportError:
        from tSS_BO.src.main import main_solver

    objective = cfg.pop("objective") if "objective" in cfg else cfg.pop("funct")
    direction = _direction(cfg.pop("direction", "min"))
    input_format = cfg.pop("objective_input", "tensor")
    recorder = HistoryRecorder()

    funct = _wrap_objective(
        objective,
        input_format=input_format,
        direction=direction,
        recorder=recorder,
    )
    dim = int(cfg.pop("dim"))
    bounds = _bounds(cfg.pop("bounds"), dim)

    if direction == "max" and cfg.get("init_y") is not None:
        cfg["init_y"] = (-torch.as_tensor(cfg["init_y"])).tolist()

    save_history_enabled = bool(cfg.pop("save_history", True))
    history_file = cfg.pop("history_file", "tss_bo_history.json")
    variable_names = cfg.pop("history_variable_names", None)
    physical_bounds = cfg.pop("history_physical_bounds", None)

    start_time = time.time()
    main_solver(funct, dim, bounds, **cfg)
    runtime_seconds = time.time() - start_time

    if save_history_enabled:
        save_history(
            algorithm="tss_bo",
            x_history=recorder.x_history,
            fom_history=recorder.fom_history,
            output_path=history_file,
            variable_names=variable_names,
            physical_bounds=physical_bounds,
            extra={
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": runtime_seconds / 60.0,
            },
        )


def _wrap_objective(
    objective: Any,
    input_format: str,
    direction: str,
    recorder: HistoryRecorder,
):
    multiplier = -1.0 if direction == "max" else 1.0

    def wrapped(X: Any) -> Any:
        X_tensor = X if torch.is_tensor(X) else torch.as_tensor(X)
        if X_tensor.ndim == 1:
            return _evaluate_one(objective, X_tensor, input_format, multiplier, recorder)
        return [
            _evaluate_one(objective, row, input_format, multiplier, recorder)
            for row in X_tensor
        ]

    return wrapped


def _evaluate_one(
    objective: Any,
    x: Any,
    input_format: str,
    multiplier: float,
    recorder: HistoryRecorder,
) -> float:
    x_value = _convert_vector(x, input_format)
    value = objective(x_value)
    recorder.record(x_value, value)
    return float(torch.as_tensor(value, dtype=torch.float64).reshape(-1)[0]) * multiplier


def _convert_vector(x: Any, input_format: str) -> Any:
    input_format = str(input_format).lower()
    if input_format == "tensor":
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x.tolist() if input_format == "list" else np.asarray(x)


def _bounds(bounds: Any, dim: int) -> Any:
    bounds_tensor = torch.as_tensor(bounds)
    return bounds_tensor.t().tolist() if tuple(bounds_tensor.shape) == (2, dim) else bounds


def _direction(direction: Any) -> str:
    direction = str(direction).lower()
    return "max" if direction in {"max", "maximize"} else "min"


def _set_random_seed(seed: Any) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
