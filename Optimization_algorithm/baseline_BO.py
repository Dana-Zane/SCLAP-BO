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

FAILURE_FOM = -10.0


@dataclass
class BaselineBOConfig:
    seed: int = 2
    n_trials: int = 2000
    n_init: int = 512
    batch_size: int = 4
    raw_samples: int = 512
    num_restarts: int = 10
    maxiter: int = 100
    device: str = "cpu"
    output: Path = Path("results_ota/results_2/bo_history.json")
    failure_fom: float = FAILURE_FOM
    warm_start_hypers: bool = True


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


HYPERPARAMETER_KEYS = {
    "likelihood.noise_covar.raw_noise",
    "mean_module.raw_constant",
    "covar_module.raw_outputscale",
    "covar_module.base_kernel.raw_lengthscale",
}


def load_model_hypers(model: SingleTaskGP, hypers: dict[str, torch.Tensor] | None) -> None:
    if not hypers:
        return
    state_dict = model.state_dict()
    for key, value in hypers.items():
        if key in state_dict and state_dict[key].shape == value.shape:
            state_dict[key] = value.to(device=state_dict[key].device, dtype=state_dict[key].dtype)
    model.load_state_dict(state_dict)


def get_model_hypers(model: SingleTaskGP) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {key: state_dict[key].detach().clone() for key in HYPERPARAMETER_KEYS if key in state_dict}


def fit_model(
    x_history: list[list[float]],
    y_history: list[float],
    device: torch.device,
    hypers: dict[str, torch.Tensor] | None = None,
) -> SingleTaskGP:
    train_x = torch.as_tensor(np.asarray(x_history, dtype=float), device=device)
    train_y = torch.as_tensor(np.asarray(y_history, dtype=float), device=device).unsqueeze(-1)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    load_model_hypers(model, hypers)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def suggest_next_batch(
    *,
    dim: int,
    x_history: list[list[float]],
    y_history: list[float],
    batch_size: int,
    raw_samples: int,
    num_restarts: int,
    maxiter: int,
    seed: int,
    device: torch.device,
    hypers: dict[str, torch.Tensor] | None,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    model = fit_model(x_history, y_history, device, hypers)
    fitted_hypers = get_model_hypers(model)
    best_f = float(np.max(y_history))
    acq = LogExpectedImprovement(model=model, best_f=best_f)

    bounds = torch.stack([torch.zeros(dim, device=device), torch.ones(dim, device=device)])
    candidates, _ = optimize_acqf(
        acq,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"maxiter": maxiter, "batch_limit": 5, "seed": seed},
        return_best_only=False,
    )
    with torch.no_grad():
        acq_values = acq(candidates).detach().cpu().reshape(-1).numpy()

    x_pool = candidates.detach().cpu().numpy().reshape(-1, dim)
    selected = select_top_unique_indices(x_pool, acq_values, batch_size)
    return x_pool[selected], fitted_hypers


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


def run_bo(
    config: BaselineBOConfig,
    evaluate_fn: Callable[[np.ndarray], float],
    variable_names: Sequence[str],
    physical_bounds: np.ndarray,
) -> None:
    device = torch.device(config.device)
    dim = len(variable_names)
    algorithm = "BoTorch_LogEI"

    start_time = time.time()
    x_init, y_init = make_initial_data(config.n_init, dim, config.seed, device, evaluate_fn)
    x_history = x_init.tolist()
    y_history = y_init.tolist()
    model_hypers: dict[str, torch.Tensor] | None = None

    while len(y_history) < config.n_trials:
        remaining = config.n_trials - len(y_history)
        batch_size = min(config.batch_size, remaining)
        x_batch, fitted_hypers = suggest_next_batch(
            dim=dim,
            x_history=x_history,
            y_history=y_history,
            batch_size=batch_size,
            raw_samples=config.raw_samples,
            num_restarts=config.num_restarts,
            maxiter=config.maxiter,
            seed=config.seed + len(y_history),
            device=device,
            hypers=model_hypers if config.warm_start_hypers else None,
        )
        if config.warm_start_hypers:
            model_hypers = fitted_hypers
        for batch_index, x_next in enumerate(x_batch, start=1):
            y_next = evaluate_fn(x_next)
            failed = bool(y_next <= config.failure_fom)
            x_history.append(x_next.tolist())
            y_history.append(float(y_next))
            print(
                f"{algorithm} seed={config.seed} eval={len(y_history)}/{config.n_trials} "
                f"batch={batch_index}/{batch_size} fom={y_next:.6g} "
                f"failed={int(failed)} best={max(y_history):.6g}",
                flush=True,
            )

    runtime_seconds = time.time() - start_time
    save_history(
        algorithm=algorithm,
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
            "warm_start_hypers": config.warm_start_hypers,
            "failure_fom": config.failure_fom,
            "initial_failures": int(np.sum(np.asarray(y_init) <= config.failure_fom)),
            "total_failures": int(np.sum(np.asarray(y_history) <= config.failure_fom)),
            "runtime_seconds": runtime_seconds,
            "runtime_minutes": runtime_seconds / 60.0,
        },
    )
