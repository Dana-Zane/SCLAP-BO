from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Any
from typing import Dict

import numpy as np
import torch

from Optimization_algorithm.optimization_history import save_history


def run_turbo_from_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    result = run_turbo(config)
    print("TuRBO finished. Compact result:")
    pprint(_compact_result(result))
    return result


def run_turbo(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)
    seed = cfg.get("seed")
    if seed is not None:
        _set_random_seed(seed)

    direction = _direction(cfg.get("direction", "min"))
    bounds = _bounds(cfg["bounds"], cfg.get("dim"))
    objective = _wrap_objective(
        cfg["objective"],
        input_format=cfg.get("objective_input", "numpy"),
        direction=direction,
    )

    Turbo1, TurboM = _load_turbo_classes()
    n_trust_regions = int(cfg.get("n_trust_regions", 1))
    optimizer_cls = TurboM if n_trust_regions > 1 else Turbo1

    optimizer_kwargs = {
        "f": objective,
        "lb": bounds[:, 0],
        "ub": bounds[:, 1],
        "n_init": int(cfg["n_init"]),
        "max_evals": int(cfg["max_evals"]),
        "batch_size": int(cfg.get("batch_size", 1)),
        "verbose": bool(cfg.get("verbose", True)),
        "use_ard": bool(cfg.get("use_ard", True)),
        "max_cholesky_size": int(cfg.get("max_cholesky_size", 2000)),
        "n_training_steps": int(cfg.get("n_training_steps", 50)),
        "min_cuda": int(cfg.get("min_cuda", 1024)),
        "device": str(cfg.get("device", "cpu")),
        "dtype": str(cfg.get("dtype", "float64")),
    }
    if n_trust_regions > 1:
        optimizer_kwargs["n_trust_regions"] = n_trust_regions

    optimizer = optimizer_cls(**optimizer_kwargs)
    start_time = time.time()
    optimizer.optimize()
    runtime_seconds = time.time() - start_time

    minimized_values = optimizer.fX.reshape(-1)
    values = minimized_values if direction == "min" else -minimized_values
    best_index = int(np.argmin(minimized_values))
    result = {
        "mode": "turbo_m" if n_trust_regions > 1 else "turbo_1",
        "dim": int(optimizer.dim),
        "n_evals": int(optimizer.n_evals),
        "best_x": optimizer.X[best_index].tolist(),
        "best_value": float(values[best_index]),
        "best_minimized_value": float(minimized_values[best_index]),
        "X_history": optimizer.X,
        "values_history": values,
        "minimized_values_history": minimized_values,
        "optimizer": optimizer,
        "runtime_seconds": runtime_seconds,
    }

    if bool(cfg.get("save_history", True)):
        save_history(
            algorithm="turbo",
            x_history=result["X_history"],
            fom_history=result["values_history"],
            output_path=cfg.get("history_file", "turbo_history.json"),
            variable_names=cfg.get("history_variable_names"),
            physical_bounds=cfg.get("history_physical_bounds"),
            extra={
                "mode": result["mode"],
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": runtime_seconds / 60.0,
            },
        )
    return result


def _wrap_objective(objective: Any, input_format: str, direction: str):
    multiplier = -1.0 if direction == "max" else 1.0

    def wrapped(x: Any) -> float:
        value = objective(_convert_vector(x, input_format))
        return float(np.asarray(value, dtype=float).reshape(-1)[0]) * multiplier

    return wrapped


def _convert_vector(x: Any, input_format: str) -> Any:
    input_format = str(input_format).lower()
    if input_format == "tensor":
        return torch.as_tensor(x)
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.tolist() if input_format == "list" else x


def _bounds(bounds: Any, dim: Any = None) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape[0] == 2 and (bounds.shape[1] != 2 or dim == bounds.shape[1]):
        bounds = bounds.T
    return bounds


def _direction(direction: Any) -> str:
    direction = str(direction).lower()
    return "max" if direction in {"max", "maximize"} else "min"


def _load_turbo_classes() -> Any:
    turbo_root = Path(__file__).resolve().parent / "TuRBO"
    if str(turbo_root) not in sys.path:
        sys.path.insert(0, str(turbo_root))
    from turbo import Turbo1, TurboM

    return Turbo1, TurboM


def _set_random_seed(seed: Any) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": result["mode"],
        "dim": result["dim"],
        "n_evals": result["n_evals"],
        "best_value": result["best_value"],
        "best_x": result["best_x"],
        "runtime_seconds": result["runtime_seconds"],
    }
