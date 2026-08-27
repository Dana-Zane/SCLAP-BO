from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from Optimization_algorithm.optimization_history import save_history


torch.set_default_dtype(torch.double)


@dataclass
class BAxUSConfig:
    seed: int = 2
    n_trials: int = 2000
    n_init: int = 512
    batch_size: int = 4
    target_dim: int = 5
    raw_samples: int = 512
    num_restarts: int = 10
    maxiter: int = 100
    device: str = "cpu"
    output: Path = Path("results_ota/results_2/baxus_history.json")


def make_embedding(input_dim: int, target_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embedding = np.zeros((target_dim, input_dim), dtype=float)
    target_indices = rng.integers(0, target_dim, size=input_dim)
    signs = rng.choice(np.array([-1.0, 1.0]), size=input_dim)
    embedding[target_indices, np.arange(input_dim)] = signs
    return embedding


def embedded_to_input(z: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    x = embedding.T @ z
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def make_initial_data(
    n_init: int,
    input_dim: int,
    target_dim: int,
    embedding: np.ndarray,
    seed: int,
    device: torch.device,
    evaluate_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = torch.stack([-torch.ones(target_dim, device=device), torch.ones(target_dim, device=device)])
    z_init = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=seed).squeeze(1).cpu().numpy()
    x_init = np.array([embedded_to_input(z, embedding) for z in z_init], dtype=float).reshape(n_init, input_dim)
    y_init = np.array([evaluate_fn(x) for x in x_init], dtype=float)
    return z_init, x_init, y_init


def fit_model(z_history: list[list[float]], y_history: list[float], device: torch.device) -> SingleTaskGP:
    train_z = torch.as_tensor(np.asarray(z_history, dtype=float), device=device)
    train_y = torch.as_tensor(np.asarray(y_history, dtype=float), device=device).unsqueeze(-1)
    model = SingleTaskGP(train_z, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def suggest_next_batch(
    *,
    z_history: list[list[float]],
    y_history: list[float],
    batch_size: int,
    config: BAxUSConfig,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    target_dim = len(z_history[0])
    model = fit_model(z_history, y_history, device)
    best_f = float(np.max(y_history))
    acq = LogExpectedImprovement(model=model, best_f=best_f)

    bounds = torch.stack([-torch.ones(target_dim, device=device), torch.ones(target_dim, device=device)])
    candidates, _ = optimize_acqf(
        acq,
        bounds=bounds,
        q=1,
        num_restarts=config.num_restarts,
        raw_samples=config.raw_samples,
        options={"maxiter": config.maxiter, "batch_limit": 5, "seed": seed},
        return_best_only=False,
    )
    with torch.no_grad():
        acq_values = acq(candidates).detach().cpu().reshape(-1).numpy()

    z_pool = candidates.detach().cpu().numpy().reshape(-1, target_dim)
    selected = select_top_unique_indices(z_pool, acq_values, batch_size)
    infos = [{"acq_value": float(acq_values[index])} for index in selected]
    return z_pool[selected], infos


def select_top_unique_indices(x_pool: np.ndarray, scores: np.ndarray, count: int) -> list[int]:
    selected: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index in np.argsort(-scores):
        key = tuple(np.round(x_pool[index], 8))
        if key in seen:
            continue
        selected.append(int(index))
        seen.add(key)
        if len(selected) == count:
            break
    return selected


def run_baxus(
    config: BAxUSConfig,
    evaluate_fn: Callable[[np.ndarray], float],
    variable_names: Sequence[str],
    physical_bounds: np.ndarray,
) -> None:
    device = torch.device(config.device)
    input_dim = len(variable_names)
    target_dim = min(config.target_dim, input_dim)
    embedding = make_embedding(input_dim, target_dim, config.seed)
    start_time = time.time()
    z_init, x_init, y_init = make_initial_data(
        config.n_init,
        input_dim,
        target_dim,
        embedding,
        config.seed,
        device,
        evaluate_fn,
    )
    z_history = z_init.tolist()
    x_history = x_init.tolist()
    y_history = y_init.tolist()
    acquisition_records: list[dict[str, object]] = []

    while len(y_history) < config.n_trials:
        remaining = config.n_trials - len(y_history)
        batch_size = min(config.batch_size, remaining)
        best_before = float(np.max(y_history))
        z_batch, candidate_infos = suggest_next_batch(
            z_history=z_history,
            y_history=y_history,
            batch_size=batch_size,
            config=config,
            seed=config.seed + len(y_history),
            device=device,
        )
        x_batch = np.array([embedded_to_input(z, embedding) for z in z_batch], dtype=float)
        for batch_index, (z_next, x_next, candidate_info) in enumerate(
            zip(z_batch, x_batch, candidate_infos), start=1
        ):
            y_next = evaluate_fn(x_next)
            z_history.append(z_next.tolist())
            x_history.append(x_next.tolist())
            y_history.append(float(y_next))
            acquisition_records.append(
                {
                    "eval": len(y_history),
                    "batch_size": batch_size,
                    "batch_index": batch_index,
                    "fom": float(y_next),
                    "best_before": best_before,
                    "acq_value": candidate_info["acq_value"],
                    "z": z_next.tolist(),
                    "x_norm": x_next.tolist(),
                }
            )
            print(
                f"BAxUS seed={config.seed} eval={len(y_history)}/{config.n_trials} "
                f"batch={batch_index}/{batch_size} fom={y_next:.6g} best={max(y_history):.6g}",
                flush=True,
            )

    runtime_seconds = time.time() - start_time
    save_history(
        algorithm="BAxUS",
        x_history=x_history,
        fom_history=y_history,
        output_path=config.output,
        variable_names=list(variable_names),
        physical_bounds=physical_bounds,
        extra={
            "seed": config.seed,
            "device": str(device),
            "n_init": config.n_init,
            "n_trials": config.n_trials,
            "batch_size": config.batch_size,
            "input_dim": input_dim,
            "target_dim": target_dim,
            "raw_samples": config.raw_samples,
            "num_restarts": config.num_restarts,
            "maxiter": config.maxiter,
            "embedding": embedding.tolist(),
            "z_history": z_history,
            "acquisition_records": acquisition_records,
            "runtime_seconds": runtime_seconds,
            "runtime_minutes": runtime_seconds / 60.0,
        },
    )
