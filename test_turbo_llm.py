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

from SCLAP_BO.adaptive_group_perturb import ConstraintAdaptiveGroupSelector
from SCLAP_BO.adaptive_group_perturb import OtaConstraints
from SCLAP_BO.llm_group_binding_consensus import build_consensus_binding
from SCLAP_BO.llm_group_binding_consensus import run_llm_bindings
from SCLAP_BO.llm_group_binding_consensus import write_outputs
from Optimization_algorithm.TuRBO.turbo.gp import train_gp
from Optimization_algorithm.TuRBO.turbo.utils import latin_hypercube
from Optimization_algorithm.optimization_history import save_history as save_base_history
from ota_experiment import make_ota_evaluator
from ota_experiment import multi_seed_values
from ota_experiment import set_random_seed
from ota_experiment import write_multi_seed_csv
from recognized_binding.align_binding_pipeline import build_align_bindings

@dataclass
class TrustRegionBounds:
    lb: np.ndarray
    ub: np.ndarray


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
    perturb_groups: list[list[int]] | None = None,
    perturb_group_probs: list[float] | None = None,
) -> tuple[np.ndarray, TrustRegionBounds]:
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

    if perturb_groups:
        group_probs = np.asarray(perturb_group_probs or [0.8] * len(perturb_groups), dtype=float)
        #group_probs = np.clip(group_probs, 0.0, 1.0)
        group_mask = np.random.rand(n_candidates, len(perturb_groups)) <= group_probs[None, :]
        empty_rows = np.where(np.sum(group_mask, axis=1) == 0)[0]
        group_weights = group_probs / group_probs.sum()
        group_mask[empty_rows, np.random.choice(len(perturb_groups), size=len(empty_rows), p=group_weights)] = 1
        mask = np.zeros((n_candidates, dim), dtype=bool)
        for group_i, group in enumerate(perturb_groups):
            rows = np.where(group_mask[:, group_i])[0]
            mask[np.ix_(rows, group)] = True
    else:
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
    return x_next, TrustRegionBounds(lb=tr_lb.ravel(), ub=tr_ub.ravel())


def update_turbo_state(state: TurboState, y_next: np.ndarray, failure_count: int = 1) -> None:
    if len(y_next) and np.max(y_next) > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += failure_count

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
        raise ValueError("record_extras must contain one item per valid evaluation.")
    for record, extra in zip(payload["history"], record_extras):
        record.update(extra)
    output_path = Path(kwargs["output_path"])
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


class SourceTurboLLMValid:
    def __init__(
        self,
        evaluate: Callable[[np.ndarray], tuple[float, dict[str, Any]]],
        lb: np.ndarray,
        ub: np.ndarray,
        n_init: int,
        max_simulations: int,
        group_selector: Any | None,
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
        self.group_selector = group_selector
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
                x_next_unit, _ = self._suggest_batch(x_unit, y, state, batch_size)
                x_next, y_next = self._evaluate_unit(x_next_unit, sources="turbo")

                if len(y_next):
                    x_unit = np.vstack([x_unit, x_next])
                    y = np.vstack([y, y_next])
                self._update_state(state, y_next)

                if self.verbose:
                    print(
                        f"TuRBO-LLMValid sim={self.n_simulations}/{self.max_simulations} "
                        f"best={np.max(self.fX):.6g} length={state.length:.4g} "
                        f"time={time.time() - start:.1f}s",
                        flush=True,
                    )

    def _initial_data(self) -> tuple[np.ndarray, np.ndarray]:
        target = min(self.n_init, self.max_simulations - self.n_simulations)
        x_trial = latin_hypercube(target, self.dim)
        x_unit, y = self._evaluate_unit(x_trial, sources="turbo_init")
        if self.verbose:
            print(f"TuRBO-LLMValid restart init={len(y)} sim={self.n_simulations}", flush=True)
        return x_unit, y

    def _evaluate_unit(self, x_unit: np.ndarray, sources: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
        x_norm_array = np.asarray(x_unit, dtype=float)
        if isinstance(sources, str):
            source_list = [sources] * len(x_norm_array)
        else:
            source_list = list(sources)
        x = self._from_unit_cube(x_norm_array)
        x_rows = []
        y_rows = []

        for source, x_norm, row in zip(source_list, x_norm_array, x):
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
    ) -> tuple[np.ndarray, TrustRegionBounds]:
        perturb_groups = None
        perturb_group_probs = None
        if self.group_selector is not None:
            group_selection = self.group_selector.select_group_probabilities(center_record=self._center_record(x_unit, y))
            if group_selection is not None:
                perturb_groups, perturb_group_probs = group_selection
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
            perturb_groups=perturb_groups,
            perturb_group_probs=perturb_group_probs,
        )

    def _update_state(self, state: TurboState, y_next: np.ndarray) -> None:
        update_turbo_state(state, y_next, failure_count=1)

    def _from_unit_cube(self, x_unit: np.ndarray) -> np.ndarray:
        return self.lb + np.asarray(x_unit, dtype=float) * (self.ub - self.lb)

    def _center_record(self, x_unit: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        center_i = int(np.asarray(y, dtype=float).reshape(-1).argmax())
        x_norm = np.asarray(x_unit[center_i], dtype=float)
        fom = float(np.asarray(y, dtype=float).reshape(-1)[center_i])
        for record in reversed(self.records):
            if abs(float(record["fom"]) - fom) < 1e-12 and np.allclose(record["x_norm"], x_norm):
                return record
        return {"sim": None, "source": "turbo_center", "fom": fom, "x_norm": x_norm.tolist()}


def run_turbo_llm(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results_{args.circuit}") / f"results_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    bindings_dir = Path(args.bindings_dir) if args.bindings_dir else None
    if args.binding_source == "align":
        align_output_dir = Path(args.align_output_dir) if args.align_output_dir else output_dir / "align"
        bindings_dir = build_align_bindings(args.circuit, args.ngspice_root, align_output_dir)

    parameter_space, evaluate = make_ota_evaluator(args.circuit, args.ngspice_root, bindings_dir=bindings_dir)
    compression = parameter_compression_payload(parameter_space)
   
    bounds = np.array([[0.0] * parameter_space.dim, [1.0] * parameter_space.dim])

    start = time.time()
    group_selector = ConstraintAdaptiveGroupSelector(
        group_records=args.llm_group_records,
        parameter_names=parameter_space.names,
        constraints=ota_constraints(),
        top_k=args.adaptive_group_top_k,
        base_prob=args.adaptive_group_base_prob,
        max_prob=args.adaptive_group_max_prob,
    )
    optimizer = SourceTurboLLMValid(
        evaluate=evaluate,
        lb=bounds[0],
        ub=bounds[1],
        n_init=min(args.n_init, args.total_evals),
        max_simulations=args.total_evals,
        group_selector=group_selector,
        batch_size=args.batch_size,
        n_candidates=args.n_candidates,
        n_training_steps=args.n_training_steps,
        verbose=args.verbose,
        device=args.device,
        dtype=args.dtype,
    )
    optimizer.optimize()
    runtime_seconds = time.time() - start

    save_history(
        algorithm="turbo_llm_group_perturb",
        x_history=optimizer.X,
        fom_history=optimizer.fX.reshape(-1),
        output_path=output_dir / "turbo_llm_history.json",
        variable_names=parameter_space.names,
        physical_bounds=parameter_space.physical_bounds,
        record_extras=optimizer.records,
        extra={
            "mode": "single_call_circuit_context_llm_group_perturb",
            "simulation_budget": args.total_evals,
            "total_simulations": int(optimizer.n_simulations),
            "llm_temperature": args.llm_temperature,
            "llm_seed": args.llm_seed,
            "llm_model": args.llm_model,
            "llm_consensus_runs": args.llm_consensus_runs,
            "llm_consensus_threshold": args.llm_consensus_threshold,
            "llm_group_records": args.llm_group_records,
            "binding_source": args.binding_source,
            "bindings_dir": str(bindings_dir) if bindings_dir else None,
            "adaptive_group_top_k": args.adaptive_group_top_k,
            "adaptive_group_base_prob": args.adaptive_group_base_prob,
            "adaptive_group_max_prob": args.adaptive_group_max_prob,
            "adaptive_group_selection_records": group_selector.selection_records,
            "parameter_compression": compression,
            "runtime_seconds": runtime_seconds,
            "runtime_minutes": runtime_seconds / 60.0,
        },
    )
    history_path = output_dir / "turbo_llm_history.json"
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


def make_shared_group_records(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    bindings_dir = Path(args.bindings_dir) if args.bindings_dir else None
    if args.binding_source == "align":
        bindings_dir = build_align_bindings(args.circuit, args.ngspice_root, output_dir / "align")
        args.binding_source = "existing"
        args.bindings_dir = str(bindings_dir)

    binding_dir = output_dir / "shared_llm_group_binding"
    parameter_space, runs = run_llm_bindings(
        circuit=args.circuit,
        ngspice_root=args.ngspice_root,
        bindings_dir=bindings_dir,
        output_dir=binding_dir,
        model_id=args.llm_model,
        temperature=args.llm_temperature,
        llm_seed=args.llm_seed,
        n_runs=args.llm_consensus_runs,
    )
    binding, summary = build_consensus_binding(
        parameter_names=parameter_space.names,
        runs=runs,
        threshold=args.llm_consensus_threshold,
    )
    write_outputs(binding_dir, binding, summary)
    return binding["functional_groups"]


def ota_constraints() -> OtaConstraints:
    from OTA_score import constraints

    return OtaConstraints(
        gdc_min=float(constraints[0]),
        gbw_min=float(constraints[1]),
        pm_min=float(constraints[2]),
        pm_max=float(constraints[3]),
        idd_max=float(constraints[4]),
    )


def run_multi_seed(args: argparse.Namespace) -> None:
    results_root = Path(f"results_{args.circuit}")
    root_output_dir = Path(args.output_dir) if args.output_dir else results_root / "turbo_llm_multi_seed"
    root_output_dir.mkdir(parents=True, exist_ok=True)
    group_records = make_shared_group_records(args, root_output_dir)
    seeds = multi_seed_values(args.seed_start, args.seed_stop, args.seed_count)
    summaries = []

    for seed in seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        seed_args.llm_group_records = group_records
        seed_args.output_dir = str(results_root / f"results_{seed}")
        summaries.append(run_turbo_llm(seed_args))

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
    parser.add_argument("--circuit", default=os.getenv("OTA_CIRCUIT", "folded_vc_ota"))
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
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--adaptive-group-top-k", type=int, default=2)
    parser.add_argument("--adaptive-group-base-prob", type=float, default=0.6)
    parser.add_argument("--adaptive-group-max-prob", type=float, default=0.8)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-seed", type=int, default=42)
    parser.add_argument("--llm-consensus-runs", type=int, default=10)
    parser.add_argument("--llm-consensus-threshold", type=float, default=0.6)
    parser.set_defaults(llm_group_records=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seed-start", type=int, default=2)
    parser.add_argument("--seed-stop", type=int, default=92)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    parsed_args = build_arg_parser().parse_args()
    if parsed_args.multi_seed:
        run_multi_seed(parsed_args)
    else:
        output_dir = Path(parsed_args.output_dir) if parsed_args.output_dir else Path(f"results_{parsed_args.circuit}") / f"results_{parsed_args.seed}"
        parsed_args.llm_group_records = make_shared_group_records(parsed_args, output_dir)
        run_turbo_llm(parsed_args)
