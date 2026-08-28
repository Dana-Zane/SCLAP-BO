from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from Optimization_algorithm.TuRBO.turbo.gp import train_gp
from Optimization_algorithm.TuRBO.turbo.utils import latin_hypercube
from Optimization_algorithm.optimization_history import save_history as save_base_history
from ota_experiment import make_ota_evaluator
from ota_experiment import multi_seed_values
from ota_experiment import set_random_seed
from ota_experiment import write_multi_seed_csv

from recognized_binding.align_binding_pipeline import build_align_bindings


USE_PARAMETER_COMPRESSION = True
TURBO_PURE_NAME = "turbo_pure" if USE_PARAMETER_COMPRESSION else "turbo_pure_raw"


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        self.failure_tolerance = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))


def suggest_turbo_batch(
    *,
    x_unit: np.ndarray,
    y: np.ndarray,
    state: TurboState,
    batch_size: int,
    dim: int,
    n_candidates: int,
    n_training_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    f_min = -np.asarray(y, dtype=float).reshape(-1)
    mu, sigma = np.median(f_min), f_min.std()
    sigma = 1.0 if sigma < 1e-6 else sigma
    f_train = (f_min - mu) / sigma

    fit_device, fit_dtype = (torch.device("cpu"), torch.float64) if len(x_unit) < 1024 else (device, dtype)
    with gpytorch.settings.max_cholesky_size(2000):
        train_x = torch.tensor(x_unit).to(device=fit_device, dtype=fit_dtype)
        train_y = torch.tensor(f_train).to(device=fit_device, dtype=fit_dtype)
        gp = train_gp(train_x=train_x, train_y=train_y, use_ard=True, num_steps=n_training_steps, hypers={})

    x_center = x_unit[f_train.argmin().item(), :][None, :]
    weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
    weights = weights / weights.mean()
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
    tr_lb = np.clip(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = np.clip(x_center + weights * state.length / 2.0, 0.0, 1.0)

    sobol = SobolEngine(dim, scramble=True, seed=np.random.randint(int(1e6)))
    pert = sobol.draw(n_candidates).to(dtype=fit_dtype, device=fit_device).cpu().detach().numpy()
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = np.random.rand(n_candidates, dim) <= prob_perturb
    empty_rows = np.where(np.sum(mask, axis=1) == 0)[0]
    mask[empty_rows, np.random.randint(dim, size=len(empty_rows))] = 1

    x_cand = x_center.copy() * np.ones((n_candidates, dim))
    x_cand[mask] = pert[mask]

    sample_device, sample_dtype = (torch.device("cpu"), torch.float64) if len(x_cand) < 1024 else (device, dtype)
    gp = gp.to(dtype=sample_dtype, device=sample_device)
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(2000):
        x_cand_torch = torch.tensor(x_cand).to(device=sample_device, dtype=sample_dtype)
        y_cand = gp.likelihood(gp(x_cand_torch)).sample(torch.Size([batch_size])).t().cpu().detach().numpy()

    x_next = np.ones((batch_size, dim))
    for i in range(batch_size):
        best = np.argmin(y_cand[:, i])
        x_next[i, :] = x_cand[best, :]
        y_cand[best, :] = np.inf
    return x_next


def update_turbo_state(state: TurboState, y_next: np.ndarray) -> None:
    if len(y_next) and np.max(y_next) > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    if len(y_next):
        state.best_value = max(state.best_value, float(np.max(y_next)))
    state.restart_triggered = state.length < state.length_min


def parameter_compression_payload(parameter_space: Any) -> dict[str, Any]:
    return {
        "description": parameter_space.describe(),
        "original_names": parameter_space.original_names,
        "compressed_names": parameter_space.names,
        "groups": parameter_space.groups,
    }


def save_history(
    *,
    record_extras: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = save_base_history(**kwargs)
    if record_extras is None:
        return payload
    if len(record_extras) != len(payload["history"]):
        raise ValueError("record_extras must contain one item per evaluation.")
    for record, extra in zip(payload["history"], record_extras):
        record.update(extra)
    output_path = Path(kwargs["output_path"])
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


class PureTurboOptimizer:
    def __init__(
        self,
        evaluate: Callable[[np.ndarray], tuple[float, dict[str, Any]]],
        lb: np.ndarray,
        ub: np.ndarray,
        n_init: int,
        max_simulations: int,
        batch_size: int = 1,
        n_candidates: int | None = None,
        n_training_steps: int = 50,
        verbose: bool = True,
        device: str = "cpu",
        dtype: str = "float64",
    ) -> None:
        self.evaluate = evaluate
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.n_init = int(n_init)
        self.max_simulations = int(max_simulations)
        self.batch_size = int(batch_size)
        self.n_candidates = int(n_candidates) if n_candidates is not None else min(100 * self.dim, 5000)
        self.n_training_steps = int(n_training_steps)
        self.verbose = bool(verbose)
        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

        self.n_simulations = 0
        self.X = np.empty((0, self.dim))
        self.fX = np.empty((0, 1))
        self.records: list[dict[str, Any]] = []

    def optimize(self) -> None:
        start = time.time()
        while self.n_simulations < self.max_simulations:
            state = TurboState(dim=self.dim, batch_size=self.batch_size)
            x_unit, y = self._initial_data()
            if len(y) < 2:
                raise RuntimeError("Not enough initial points to fit TuRBO.")
            state.best_value = float(np.max(y))

            while self.n_simulations < self.max_simulations and not state.restart_triggered:
                batch_size = min(self.batch_size, self.max_simulations - self.n_simulations)
                x_next_unit = self._suggest_batch(x_unit, y, state, batch_size)
                x_next, y_next = self._evaluate_unit(x_next_unit, source="turbo")

                if len(y_next):
                    x_unit = np.vstack([x_unit, x_next])
                    y = np.vstack([y, y_next])
                update_turbo_state(state, y_next)

                if self.verbose:
                    print(
                        f"TuRBO sim={self.n_simulations}/{self.max_simulations} "
                        f"best={np.max(self.fX):.6g} length={state.length:.4g} "
                        f"time={time.time() - start:.1f}s",
                        flush=True,
                    )

    def _initial_data(self) -> tuple[np.ndarray, np.ndarray]:
        target = min(self.n_init, self.max_simulations - self.n_simulations)
        x_trial = latin_hypercube(target, self.dim)
        x_unit, y = self._evaluate_unit(x_trial, source="turbo_init")
        if self.verbose:
            print(f"TuRBO restart init={len(y)} sim={self.n_simulations}", flush=True)
        return x_unit, y

    def _evaluate_unit(self, x_unit: np.ndarray, source: str) -> tuple[np.ndarray, np.ndarray]:
        x_norm_array = np.asarray(x_unit, dtype=float)
        x = self._from_unit_cube(x_norm_array)
        x_rows = []
        y_rows = []

        for x_norm, row in zip(x_norm_array, x):
            if self.n_simulations >= self.max_simulations:
                break
            true_fom, extra = self.evaluate(row)
            self.n_simulations += 1
            record = {
                "sim": int(self.n_simulations),
                "source": source,
                "fom": float(true_fom),
                "x_norm": x_norm.tolist(),
            }
            record.update(extra)
            self.records.append(record)
            x_rows.append(x_norm)
            y_rows.append([float(true_fom)])

        if y_rows:
            x_array = np.asarray(x_rows, dtype=float)
            y_array = np.asarray(y_rows, dtype=float)
            self.X = np.vstack([self.X, x_array])
            self.fX = np.vstack([self.fX, y_array])
            return x_array, y_array
        return np.empty((0, self.dim)), np.empty((0, 1))

    def _suggest_batch(
        self,
        x_unit: np.ndarray,
        y: np.ndarray,
        state: TurboState,
        batch_size: int,
    ) -> np.ndarray:
        return suggest_turbo_batch(
            x_unit=x_unit,
            y=y,
            state=state,
            batch_size=batch_size,
            dim=self.dim,
            n_candidates=self.n_candidates,
            n_training_steps=self.n_training_steps,
            device=self.device,
            dtype=self.dtype,
        )

    def _from_unit_cube(self, x_unit: np.ndarray) -> np.ndarray:
        return self.lb + np.asarray(x_unit, dtype=float) * (self.ub - self.lb)


def run_turbo_pure(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results_{args.circuit}") / f"results_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    bindings_dir = Path(args.bindings_dir) if args.bindings_dir else None
    if USE_PARAMETER_COMPRESSION and args.binding_source == "align":
        align_output_dir = Path(args.align_output_dir) if args.align_output_dir else output_dir / "align"
        bindings_dir = build_align_bindings(args.circuit, args.ngspice_root, align_output_dir)

    parameter_space, evaluate = make_ota_evaluator(
        args.circuit,
        args.ngspice_root,
        use_parameter_compression=USE_PARAMETER_COMPRESSION,
        bindings_dir=bindings_dir,
    )
    compression = parameter_compression_payload(parameter_space)

    start = time.time()
    bounds = np.array([[0.0] * parameter_space.dim, [1.0] * parameter_space.dim])
    optimizer = PureTurboOptimizer(
        evaluate=evaluate,
        lb=bounds[0],
        ub=bounds[1],
        n_init=min(args.n_init, args.total_evals),
        max_simulations=args.total_evals,
        batch_size=args.batch_size,
        n_candidates=args.n_candidates,
        n_training_steps=args.n_training_steps,
        verbose=args.verbose,
        device=args.device,
        dtype=args.dtype,
    )
    optimizer.optimize()
    runtime_seconds = time.time() - start

    history_path = output_dir / f"{TURBO_PURE_NAME}_history.json"
    save_history(
        algorithm=TURBO_PURE_NAME,
        x_history=optimizer.X,
        fom_history=optimizer.fX.reshape(-1),
        output_path=history_path,
        variable_names=parameter_space.names,
        physical_bounds=parameter_space.physical_bounds,
        record_extras=optimizer.records,
        extra={
            "mode": TURBO_PURE_NAME,
            "simulation_budget": args.total_evals,
            "total_simulations": int(optimizer.n_simulations),
            "use_parameter_compression": USE_PARAMETER_COMPRESSION,
            "parameter_compression": compression,
            "binding_source": args.binding_source,
            "bindings_dir": str(bindings_dir) if bindings_dir else None,
            "runtime_seconds": runtime_seconds,
            "runtime_minutes": runtime_seconds / 60.0,
        },
    )

    history = json.loads(history_path.read_text(encoding="utf-8"))
    print(history_path)
    print(history["best"]["fom"])
    return {
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "history_file": str(history_path),
        "best_fom": history["best"]["fom"],
        "best_eval": history["best"]["eval"],
        "best_x": history["best"]["x"],
        "best_x_norm": history["best"]["x_norm"],
        "total_simulations": int(optimizer.n_simulations),
    }


def run_multi_seed(args: argparse.Namespace) -> None:
    results_root = Path(f"results_{args.circuit}")
    root_output_dir = Path(args.output_dir) if args.output_dir else results_root / f"{TURBO_PURE_NAME}_multi_seed"
    root_output_dir.mkdir(parents=True, exist_ok=True)
    seeds = multi_seed_values(args.seed_start, args.seed_stop, args.seed_count)
    summaries = []

    for seed in seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        seed_args.output_dir = str(results_root / f"results_{seed}")
        summaries.append(run_turbo_pure(seed_args))

    best_values = np.asarray([summary["best_fom"] for summary in summaries], dtype=float)
    payload = {
        "seeds": seeds,
        "n_runs": len(summaries),
        "best_fom_mean": float(np.mean(best_values)),
        "best_fom_std": float(np.std(best_values)),
        "best_fom_min": float(np.min(best_values)),
        "best_fom_max": float(np.max(best_values)),
        "runs": summaries,
    }
    summary_path = root_output_dir / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_multi_seed_csv(root_output_dir / "multi_seed_summary.csv", summaries)
    print(summary_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", default=os.getenv("OTA_CIRCUIT", "folded_cascode_ota"))
    parser.add_argument("--ngspice-root", default="Ngspice_Benchmark_unbound")
    parser.add_argument("--binding-source", default="align", choices=["align", "existing"])
    parser.add_argument("--bindings-dir")
    parser.add_argument("--align-output-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--total-evals", type=int, default=1000)
    parser.add_argument("--n-init", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-candidates", type=int)
    parser.add_argument("--n-training-steps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seed-start", type=int, default=4)
    parser.add_argument("--seed-stop", type=int, default=94)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    if parsed_args.multi_seed:
        run_multi_seed(parsed_args)
    else:
        run_turbo_pure(parsed_args)
