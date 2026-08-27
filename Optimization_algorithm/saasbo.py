from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples

from Optimization_algorithm.optimization_history import save_history


torch.set_default_dtype(torch.double)


@dataclass
class SAASBOConfig:
    seed: int = 2
    n_trials: int = 2000
    n_init: int = 512
    batch_size: int = 4
    raw_samples: int = 512
    num_restarts: int = 10
    maxiter: int = 100
    mc_samples: int = 128
    warmup_steps: int = 256
    num_samples: int = 128
    thinning: int = 16
    max_tree_depth: int = 6
    device: str = "cpu"
    output: Path = Path("results_ota/results_2/saasbo_history.json")


def make_initial_data(
    n_init: int,
    dim: int,
    seed: int,
    device: torch.device,
    evaluate_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
    bounds = torch.stack([torch.zeros(dim, device=device), torch.ones(dim, device=device)])
    xs = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=seed).squeeze(1).cpu().numpy()
    ys = np.array([evaluate_fn(x) for x in xs], dtype=float)
    return xs, ys


def fit_model(
    x_history: list[list[float]],
    y_history: list[float],
    config: SAASBOConfig,
    device: torch.device,
) -> SaasFullyBayesianSingleTaskGP:
    train_x = torch.as_tensor(np.asarray(x_history, dtype=float), device=device)
    train_y = torch.as_tensor(np.asarray(y_history, dtype=float), device=device).unsqueeze(-1)
    model = SaasFullyBayesianSingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=config.warmup_steps,
        num_samples=config.num_samples,
        thinning=config.thinning,
        max_tree_depth=config.max_tree_depth,
        disable_progbar=True,
    )
    return model


def suggest_next_batch(
    *,
    x_history: list[list[float]],
    y_history: list[float],
    batch_size: int,
    config: SAASBOConfig,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    dim = len(x_history[0])
    model = fit_model(x_history, y_history, config, device)
    best_f = float(np.max(y_history))
    sampler = SobolQMCNormalSampler(torch.Size([config.mc_samples]), seed=seed)
    acq = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)

    bounds = torch.stack([torch.zeros(dim, device=device), torch.ones(dim, device=device)])
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

    x_pool = candidates.detach().cpu().numpy().reshape(-1, dim)
    selected = select_top_unique_indices(x_pool, acq_values, batch_size)
    infos = [{"acq_value": float(acq_values[index])} for index in selected]
    return x_pool[selected], infos


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


def run_saasbo(
    config: SAASBOConfig,
    evaluate_fn: Callable[[np.ndarray], float],
    variable_names: Sequence[str],
    physical_bounds: np.ndarray,
) -> None:
    device = torch.device(config.device)
    dim = len(variable_names)
    start_time = time.time()
    x_init, y_init = make_initial_data(config.n_init, dim, config.seed, device, evaluate_fn)
    x_history = x_init.tolist()
    y_history = y_init.tolist()
    acquisition_records: list[dict[str, object]] = []

    while len(y_history) < config.n_trials:
        remaining = config.n_trials - len(y_history)
        batch_size = min(config.batch_size, remaining)
        best_before = float(np.max(y_history))
        x_batch, candidate_infos = suggest_next_batch(
            x_history=x_history,
            y_history=y_history,
            batch_size=batch_size,
            config=config,
            seed=config.seed + len(y_history),
            device=device,
        )
        for batch_index, (x_next, candidate_info) in enumerate(zip(x_batch, candidate_infos), start=1):
            y_next = evaluate_fn(x_next)
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
                    "x_norm": x_next.tolist(),
                }
            )
            print(
                f"SAASBO seed={config.seed} eval={len(y_history)}/{config.n_trials} "
                f"batch={batch_index}/{batch_size} fom={y_next:.6g} best={max(y_history):.6g}",
                flush=True,
            )

    runtime_seconds = time.time() - start_time
    save_history(
        algorithm="SAASBO",
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
            "raw_samples": config.raw_samples,
            "num_restarts": config.num_restarts,
            "maxiter": config.maxiter,
            "mc_samples": config.mc_samples,
            "warmup_steps": config.warmup_steps,
            "num_samples": config.num_samples,
            "thinning": config.thinning,
            "max_tree_depth": config.max_tree_depth,
            "acquisition_records": acquisition_records,
            "runtime_seconds": runtime_seconds,
            "runtime_minutes": runtime_seconds / 60.0,
        },
    )
