"""Circuit simulation launcher built on the Ngspice_Benchmark simulator.

This module intentionally contains no optimizer or plotting interfaces.  It
loads a circuit setup, maps sizing values to the configured design variables,
runs ngspice, and can optionally evaluate the configured objectives and
constraints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import ngspice


OBJECTIVE_DIRECTIONS = {"MIN": 1, "MAX": -1}


def parse_objectives(objective_defs: Sequence[str] | None) -> list[tuple[str, int]]:
    objectives: list[tuple[str, int]] = []
    for obj_str in objective_defs or []:
        token = obj_str.upper().split()
        if len(token) == 2 and token[0] in OBJECTIVE_DIRECTIONS:
            objectives.append((token[1], OBJECTIVE_DIRECTIONS[token[0]]))
            continue
        raise ValueError(
            f"Found {obj_str!r} in objective definition, expected 'min meas' or 'max meas'"
        )
    return objectives


def parse_constraints(
    constraint_defs: Sequence[str] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    constraints: dict[str, dict[str, float]] = {"<": {}, ">": {}}

    for cstr_str in constraint_defs or []:
        token = cstr_str.upper().split()
        if len(token) < 3:
            raise ValueError(
                f"Found {cstr_str!r} in constraint definition, expected 'meas > value' or 'meas < value'"
            )

        value = float(token[-1])
        operator = token[-2]
        if operator not in constraints:
            raise ValueError(
                f"Found {cstr_str!r} in constraint definition, expected 'meas > value' or 'meas < value'"
            )
        for meas in token[:-2]:
            constraints[operator][meas] = value

    return constraints["<"], constraints[">"]


class SimulationTarget:
    """Evaluate simulation measures against setup objectives and constraints."""

    def __init__(
        self,
        objectives: Sequence[str] | None = None,
        constraints: Sequence[str] | None = None,
    ) -> None:
        self.lt, self.gt = parse_constraints(constraints)
        self.objectives = parse_objectives(objectives)

    def evaluate_corner(
        self, measures: Mapping[str, Any]
    ) -> tuple[list[float], float, dict[str, Any]]:
        log: dict[str, Any] = {}
        gsum = 0.0

        for meas, limit in self.lt.items():
            if meas not in measures or measures[meas] is None:
                log[f"{meas}_lt"] = (limit, None)
                gsum += -1000.0
            elif measures[meas] > limit:
                violation = (limit - measures[meas]) / abs(limit)
                log[f"{meas}_lt"] = (limit, measures[meas], violation)
                gsum += violation

        for meas, limit in self.gt.items():
            if meas not in measures or measures[meas] is None:
                log[f"{meas}_gt"] = (limit, None)
                gsum += -1000.0
            elif measures[meas] < limit:
                violation = (-limit + measures[meas]) / abs(limit)
                log[f"{meas}_gt"] = (limit, measures[meas], violation)
                gsum += violation

        obj = [
            measures[item[0]] * item[1]
            if item[0] in measures and measures[item[0]] is not None
            else math.inf
            for item in self.objectives
        ]
        return obj, gsum, log

    def evaluate(
        self, measures: Mapping[str, Mapping[str, Any]]
    ) -> tuple[list[float], float, dict[str, Any]]:
        log: dict[str, Any] = {}
        gsum = 0.0
        obj = [-math.inf] * len(self.objectives)

        for corner, corner_measures in measures.items():
            obj_corner, gsum_corner, log_corner = self.evaluate_corner(corner_measures)
            log[corner] = log_corner
            gsum += gsum_corner
            obj = [max(o1, o2) for o1, o2 in zip(obj, obj_corner)]

        return obj, gsum, log

    def evaluate_single(
        self, measures: Mapping[str, Mapping[str, Any]]
    ) -> tuple[float, bool, dict[str, Any]]:
        obj, gsum, log = self.evaluate(measures)
        denom = len(self.gt) + len(self.lt)
        penalty = gsum / denom if denom else 0.0
        return sum(obj) + penalty, gsum == 0, log


class Circuit:
    """Load one circuit folder and run its configured ngspice testbenches."""

    def __init__(
        self,
        folder: str | Path,
        setup_file: str = "circuit_setup.json",
        corners: str | None = "corners.inc",
        num_threads: int = 1,
    ) -> None:
        self.folder = Path(folder).expanduser().resolve()
        self.setup_file = self.folder / setup_file

        with self.setup_file.open() as file:
            setup = json.load(file)

        self.testbenches = list(setup["testbenches"])
        self.parameters: list[str] = []
        self.ranges: list[list[float]] = []

        for range_def in setup["ranges"]:
            for param in range_def["params"]:
                self.parameters.append(param)
                self.ranges.append(
                    [float(range_def["min"]), float(range_def["max"]), float(range_def["grid"])]
                )

        self.ranges_array = np.array(self.ranges, dtype=float)
        self.target = SimulationTarget(setup.get("objectives"), setup.get("constraints"))
        self.ngspice = ngspice.Ngspice(
            str(self.folder),
            self.testbenches,
            self.parameters,
            corners=corners,
            num_threads=num_threads,
        )

    def __str__(self) -> str:
        return (
            f"Running folder: {self.folder}\n"
            f"Testbenches: {self.testbenches}\n"
            f"Parameters: {self.parameters}\n"
            f"Ranges: {self.ranges_array}"
        )

    def range_min(self) -> np.ndarray:
        return self.ranges_array[:, 0]

    def range_max(self) -> np.ndarray:
        return self.ranges_array[:, 1]

    def range_grid(self) -> np.ndarray:
        return self.ranges_array[:, 2]

    def sizing_to_values(self, sizing: Mapping[str, float]) -> np.ndarray:
        missing = [param for param in self.parameters if param not in sizing]
        if missing:
            raise KeyError(f"Sizing file is missing parameters: {', '.join(missing)}")
        return np.array([sizing[param] for param in self.parameters], dtype=float)

    def normalize_values(self, values: np.ndarray) -> np.ndarray:
        return (values - self.range_min()) / (self.range_max() - self.range_min())

    def denormalize_values(self, values: np.ndarray) -> np.ndarray:
        return self.range_min() + values * (self.range_max() - self.range_min())

    def snap_to_grid(self, values: np.ndarray) -> np.ndarray:
        grid = self.range_grid()
        snapped = np.array(values, copy=True)
        grid_mask = grid > 0
        if np.any(grid_mask):
            snapped[..., grid_mask] = (
                np.round(snapped[..., grid_mask] / grid[grid_mask]) * grid[grid_mask]
            )
        return np.fmin(np.fmax(snapped, self.range_min()), self.range_max())

    def random_values(self, count: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        values = rng.uniform(self.range_min(), self.range_max(), size=(count, len(self.parameters)))
        return self.snap_to_grid(values)

    def _as_parameter_matrix(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray,
        normalized: bool = False,
        snap_to_grid: bool = False,
    ) -> np.ndarray:
        if isinstance(values, Mapping):
            matrix = self.sizing_to_values(values)
        else:
            matrix = np.asarray(values, dtype=float)

        if matrix.ndim == 1:
            matrix = np.expand_dims(matrix, axis=0)
        if matrix.shape[1] != len(self.parameters):
            raise ValueError(
                f"Expected {len(self.parameters)} parameters, got {matrix.shape[1]}"
            )

        if normalized:
            matrix = self.denormalize_values(matrix)
        if snap_to_grid:
            matrix = self.snap_to_grid(matrix)
        return matrix

    def simulate(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray,
        normalized: bool = False,
        snap_to_grid: bool = False,
    ) -> list[dict[str, dict[str, Any]]]:
        parameter_values = self._as_parameter_matrix(values, normalized, snap_to_grid)
        return self.ngspice.simulate(parameter_values)

    def evaluate(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray,
        normalized: bool = False,
        snap_to_grid: bool = False,
    ) -> list[dict[str, Any]]:
        results = self.simulate(values, normalized=normalized, snap_to_grid=snap_to_grid)
        reports = []
        for result in results:
            objective, constraint, log = self.target.evaluate(result)
            reports.append(
                {
                    "measures": result,
                    "objective": objective,
                    "constraint": constraint,
                    "log": log,
                }
            )
        return reports



