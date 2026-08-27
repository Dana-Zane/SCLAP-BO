from __future__ import annotations

import importlib
from pprint import pprint
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
import torch
from morbo.run_one_replication import run_one_replication
from morbo.morbo_native_optimize import run_native_morbo



def run_morbo_from_dict(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    config_dir = config_dir
    result = run_morbo(config=config, config_dir=config_dir)
    print("MORBO finished. Compact result:")
    pprint(_compact_result(result))
    #return run_morbo(config=config, config_dir=config_dir)


def run_morbo(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    config_dir = config_dir or Path.cwd()
    if "objective" in config:
        native_config = _prepare_native_morbo_config(config=config)
        return run_native_morbo(config=native_config, config_dir=config_dir)
    if "evalfn" in config:
        return run_builtin_benchmark(config=config, config_dir=config_dir)
    raise ValueError("Config must contain either 'objective' or 'evalfn'.")


def run_builtin_benchmark(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    last_output = {"value": None}

    def save_callback(output: Dict[str, Any]) -> None:
        last_output["value"] = output

    run_one_replication(
        seed=int(config.get("seed", 0)),
        label="morbo",
        max_evals=int(config["max_evals"]),
        evalfn=str(config["evalfn"]),
        batch_size=int(config["batch_size"]),
        dim=int(config["dim"]),
        n_initial_points=int(config["n_initial_points"]),
        min_tr_size=int(config["min_tr_size"]),
        max_reference_point=list(config["max_reference_point"]),
        verbose=bool(config.get("verbose", False)),
        save_callback=save_callback,
        save_during_opt=True if config.get("save_during_opt", True) else None,
    )

    output = last_output["value"]
    summary = {
        "mode": "builtin_benchmark",
        "evalfn": config["evalfn"],
        "dim": int(config["dim"]),
        "max_evals": int(config["max_evals"]),
    }
    if output is not None:
        summary["n_evals"] = output["n_evals"][-1] if output["n_evals"] else 0
        summary["final_hv"] = output["true_hv"][-1] if output["true_hv"] else 0.0
    return summary


def _prepare_native_morbo_config(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    native_config = dict(config)
    missing = [
        key
        for key in ("objective", "bounds", "max_evals", "max_reference_point")
        if key not in native_config
    ]
    if missing:
        raise ValueError(
            "Custom MORBO config is missing required keys: "
            + ", ".join(f"'{key}'" for key in missing)
            + "."
        )

    objective = native_config["objective"]
    if not callable(objective) and not _is_objective_path(objective):
        raise TypeError("config['objective'] must be a callable or 'module:function'.")
    native_config["objective"] = _adapt_objective_input(
        objective,
        input_format=native_config.get("objective_input", "tensor"),
    )

    reference_point = list(native_config["max_reference_point"])
    directions = native_config.get("directions")
    if directions is None:
        directions = ["minimize"] * len(reference_point)
        native_config["directions"] = directions
    native_config["directions"] = [_normalize_direction(direction) for direction in directions]

    if len(native_config["directions"]) != len(reference_point):
        raise ValueError(
            "The length of config['directions'] must match "
            "config['max_reference_point']."
        )

    return native_config


def _adapt_objective_input(
    objective: Any,
    input_format: str = "tensor",
) -> Callable[[torch.Tensor], Any]:
    input_format = _normalize_objective_input(input_format)
    objective_fn = _load_objective(objective)
    if input_format == "tensor":
        return objective_fn

    def adapted_objective(X: torch.Tensor) -> Any:
        if torch.is_tensor(X) and X.ndim > 1:
            return [objective_fn(_convert_vector(row, input_format)) for row in X]
        return objective_fn(_convert_vector(X, input_format))

    return adapted_objective


def _load_objective(objective: Any) -> Callable[[Any], Any]:
    if callable(objective):
        return objective
    module_name, function_name = objective.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _normalize_objective_input(input_format: Any) -> str:
    normalized = str(input_format).lower()
    if normalized in {"tensor", "numpy", "list"}:
        return normalized
    raise ValueError("config['objective_input'] must be 'tensor', 'numpy', or 'list'.")


def _convert_vector(x: Any, input_format: str) -> Any:
    if input_format == "tensor":
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return x.numpy() if input_format == "numpy" else x.tolist()
    if input_format == "list" and hasattr(x, "tolist"):
        return x.tolist()
    return x


def _is_objective_path(objective: Any) -> bool:
    return isinstance(objective, str) and ":" in objective


def _normalize_direction(direction: Any) -> str:
    normalized = str(direction).lower()
    if normalized in {"min", "minimize"}:
        return "min"
    if normalized in {"max", "maximize"}:
        return "max"
    raise ValueError(
        "Each direction must be 'min', 'minimize', 'max', or 'maximize'. "
        f"Got {direction!r}."
    )


def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(result)
    for key in ("X_history", "values_history", "pareto_X", "pareto_values"):
        if key in compact:
            compact[f"{key}_count"] = len(compact[key])
            del compact[key]
    return compact

