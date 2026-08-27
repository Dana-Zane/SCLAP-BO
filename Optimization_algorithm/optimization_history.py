from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np


def normalize_bounds(bounds: Any) -> np.ndarray:
    bounds_array = np.asarray(bounds, dtype=float)
    if bounds_array.ndim != 2:
        raise ValueError("bounds must be a 2D array-like object.")
    if bounds_array.shape[1] == 2:
        return bounds_array
    if bounds_array.shape[0] == 2:
        return bounds_array.T
    raise ValueError(f"bounds must have shape (dim, 2) or (2, dim), got {bounds_array.shape}.")


def denormalize(x_norm: Any, physical_bounds: Any | None) -> list[float] | None:
    if physical_bounds is None:
        return None
    bounds = normalize_bounds(physical_bounds)
    x = np.asarray(x_norm, dtype=float).reshape(-1)
    return (x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]).tolist()


class HistoryRecorder:
    def __init__(self) -> None:
        self.x_history: list[list[float]] = []
        self.fom_history: list[float] = []

    def record(self, x: Any, fom: Any) -> None:
        x_array = np.asarray(x, dtype=float).reshape(-1)
        fom_value = float(np.asarray(fom, dtype=float).reshape(-1)[0])
        self.x_history.append(x_array.tolist())
        self.fom_history.append(fom_value)

    def extend(self, xs: Any, foms: Any) -> None:
        x_array = np.asarray(xs, dtype=float)
        fom_array = np.asarray(foms, dtype=float).reshape(-1)
        if x_array.ndim == 1:
            self.record(x_array, fom_array[0])
            return
        for x, fom in zip(x_array, fom_array):
            self.record(x, fom)


def save_history(
    *,
    algorithm: str,
    x_history: Any,
    fom_history: Any,
    output_path: str | Path,
    variable_names: Iterable[str] | None = None,
    physical_bounds: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_array = np.asarray(x_history, dtype=float)
    fom_array = np.asarray(fom_history, dtype=float).reshape(-1)
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)
    if x_array.shape[0] != fom_array.shape[0]:
        raise ValueError("x_history and fom_history must contain the same number of rows.")

    names = list(variable_names) if variable_names is not None else [f"x{i}" for i in range(x_array.shape[1])]
    if len(names) != x_array.shape[1]:
        names = [f"x{i}" for i in range(x_array.shape[1])]

    best_running = np.maximum.accumulate(fom_array)
    best_index = int(np.argmax(fom_array)) if fom_array.size else -1
    best_x_norm = x_array[best_index].tolist() if best_index >= 0 else []
    best_x = denormalize(best_x_norm, physical_bounds)

    records = []
    for idx, (x_row, fom, best_fom) in enumerate(zip(x_array, fom_array, best_running), start=1):
        record = {
            "eval": idx,
            "fom": float(fom),
            "best_fom": float(best_fom),
            "x_norm": x_row.tolist(),
        }
        x_physical = denormalize(x_row, physical_bounds)
        if x_physical is not None:
            record["x"] = x_physical
        records.append(record)

    payload = {
        "algorithm": algorithm,
        "n_evals": int(fom_array.size),
        "variable_names": names,
        "best": {
            "eval": best_index + 1 if best_index >= 0 else None,
            "fom": float(fom_array[best_index]) if best_index >= 0 else None,
            "x_norm": best_x_norm,
            "x": best_x,
        },
        "history": records,
        "extra": extra or {},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_path.with_suffix(".csv"), records, names)
    return payload


def _write_csv(path: Path, records: list[dict[str, Any]], variable_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["eval", "fom", "best_fom"]
    fieldnames.extend(f"x_norm_{name}" for name in variable_names)
    has_physical = any("x" in record for record in records)
    if has_physical:
        fieldnames.extend(f"x_{name}" for name in variable_names)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "eval": record["eval"],
                "fom": record["fom"],
                "best_fom": record["best_fom"],
            }
            for name, value in zip(variable_names, record["x_norm"]):
                row[f"x_norm_{name}"] = value
            if has_physical:
                for name, value in zip(variable_names, record.get("x", [])):
                    row[f"x_{name}"] = value
            writer.writerow(row)
