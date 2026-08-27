from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import torch
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from morbo.benchmark_function import BenchmarkFunction
from morbo.gen import TS_select_batch_MORBO
from morbo.state import TRBOState
from morbo.trust_region import TurboHParams



def run_native_morbo(
    config: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    start_time = time.time()
   
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_evals = int(config["max_evals"])
    max_reference_point = config.get("max_reference_point")
    if max_reference_point is None:
        raise ValueError("config['max_reference_point'] is required for MORBO HV mode.")

    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.double
    tkwargs = {"dtype": dtype, "device": device}

    bounds = torch.tensor(config["bounds"], **tkwargs)
    if bounds.shape[0] != 2:
        raise ValueError("config['bounds'] must be shaped as [2, dim].")

    dim = int(bounds.shape[1])
    directions = config["directions"]
    num_outputs = len(directions)
    signs = torch.tensor(
        [1.0 if direction == "max" else -1.0 for direction in directions],
        **tkwargs,
    )

    objective_raw = _load_objective(config["objective"])

    def signed_objective(X: torch.Tensor) -> torch.Tensor:
        return _evaluate_objective(objective_raw, X).to(**tkwargs) * signs

    observation_noise_std = config.get("observation_noise_std",None)
    observation_noise_bias = config.get("observation_noise_bias",None)
    recompute_all_hvs = bool(config.get("recompute_all_hvs", True))
    save_during_opt = bool(config.get("save_during_opt", False))
    save_callback = _make_save_callback(config.get("save_back"),config_dir=config_dir)

    batch_size = int(config.get("batch_size", 1))
    n_initial_points = int(config.get("n_initial_points", max(2 * dim, 10)))
    min_tr_size = int(config.get("min_tr_size", min(n_initial_points, dim + 1)))
    # morbo_options = dict(config.get("morbo_options", {}))
    failure_streak = config.get("failure_streak",None)
    if failure_streak is None:
        failure_streak = max(dim // 3, 10)

    f = BenchmarkFunction(
        base_f=signed_objective,
        num_outputs=num_outputs,
        ref_point=torch.tensor(max_reference_point, **tkwargs),
        dim=dim,
        tkwargs=tkwargs,
        negate=False,
        observation_noise_std=observation_noise_std,
        observation_noise_bias=observation_noise_bias,
    )

    tr_hparams_kwargs = {
        "batch_size": batch_size,
        "n_initial_points": n_initial_points,
        "min_tr_size": min_tr_size,
        "max_reference_point": list(max_reference_point),
        "failure_streak": failure_streak,
        "verbose": bool(config.get("verbose", False)),
    }
    #tr_hparams_kwargs.update(morbo_options)
    tr_hparams = TurboHParams.from_dict(tr_hparams_kwargs)
    n_trust_regions = tr_hparams.n_trust_regions

    trbo_state = TRBOState(
        dim=dim,
        max_evals=max_evals,
        num_outputs=num_outputs,
        num_objectives=num_outputs,
        bounds=bounds,
        tr_hparams=tr_hparams,
        constraints=None,
        objective=IdentityMCMultiOutputObjective(),
    )

    n_evals = []
    true_hv = []
    pareto_X = []
    pareto_Y = []
    n_points_in_tr = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_other = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_sobol = [[] for _ in range(n_trust_regions)]
    tr_sizes = [[] for _ in range(n_trust_regions)]
    tr_centers = [[] for _ in range(n_trust_regions)]
    tr_restarts = [[] for _ in range(n_trust_regions)]
    fit_times = []
    gen_times = []
    true_ref_point = torch.tensor(max_reference_point, **tkwargs)

    n_points = min(n_initial_points, max_evals - int(trbo_state.n_evals.item()))
    X_init = draw_sobol_samples(bounds=bounds, n=n_points, q=1).squeeze(1)
    Y_init = f(X_init)
    trbo_state.update(
        X=X_init,
        Y=Y_init,
        new_ind=torch.full(
            (X_init.shape[0],),
            0,
            dtype=torch.long,
            device=X_init.device,
        ),
    )
    trbo_state.log_restart_points(X=X_init, Y=Y_init)

    for i in range(n_trust_regions):
        trbo_state.initialize_standard(
            tr_idx=i,
            restart=False,
            switch_strategy=False,
            X_init=X_init,
            Y_init=Y_init,
        )

    trbo_state.update_data_across_trs()
    trbo_state.TR_index_history.fill_(-2)

    all_tr_indices = [-1] * n_points
    while trbo_state.n_evals < max_evals:
        start_gen = time.time()
        selection_output = TS_select_batch_MORBO(trbo_state=trbo_state)
        gen_times.append(time.time() - start_gen)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on generating candidates: {gen_times[-1]:.1f} seconds")

        X_cand = selection_output.X_cand
        tr_indices = selection_output.tr_indices
        remaining = max_evals - int(trbo_state.n_evals.item())
        if X_cand.shape[0] > remaining:
            X_cand = X_cand[:remaining]
            tr_indices = tr_indices[:remaining]

        all_tr_indices.extend(tr_indices.tolist())
        trbo_state.tabu_set.log_iteration()
        Y_cand = f(X_cand)

        for i, tr in enumerate(trbo_state.trust_regions):
            inds = torch.cat(
                [torch.where((x == trbo_state.X_history).all(dim=-1))[0] for x in tr.X]
            )
            tr_inds = trbo_state.TR_index_history[inds]
            assert len(tr_inds) == len(tr.X)
            n_points_in_tr[i].append(len(tr_inds))
            n_points_in_tr_collected_by_sobol[i].append(sum(tr_inds == -2).cpu().item())
            n_points_in_tr_collected_by_other[i].append(
                sum((tr_inds != i) & (tr_inds != -2)).cpu().item()
            )
            tr_sizes[i].append(tr.length.item())
            tr_centers[i].append(tr.X_center.cpu().squeeze().tolist())

        start_fit = time.time()
        trbo_state.update(X=X_cand, Y=Y_cand, new_ind=tr_indices)
        should_restart_trs = trbo_state.update_trust_regions_and_log(
            X_cand=X_cand,
            Y_cand=Y_cand,
            tr_indices=tr_indices,
            batch_size=batch_size,
            verbose=bool(tr_hparams.verbose),
        )
        fit_times.append(time.time() - start_fit)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on model fitting: {fit_times[-1]:.1f} seconds")

        switch_strategy = trbo_state.check_switch_strategy()
        if switch_strategy:
            should_restart_trs = [True for _ in should_restart_trs]
        if any(should_restart_trs):
            for i in range(trbo_state.tr_hparams.n_trust_regions):
                if should_restart_trs[i]:
                    restart_points = min(
                        trbo_state.tr_hparams.n_restart_points,
                        max_evals - trbo_state.n_evals,
                    )
                    if restart_points <= 0:
                        break
                    if trbo_state.tr_hparams.verbose:
                        print(f"{trbo_state.n_evals}) Restarting trust region {i}")
                    trbo_state.TR_index_history[trbo_state.TR_index_history == i] = -1
                    init_kwargs = {}
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        X_center = trbo_state.gen_new_restart_design()
                        Y_center = f(X_center)
                        init_kwargs["X_init"] = X_center
                        init_kwargs["Y_init"] = Y_center
                        init_kwargs["X_center"] = X_center
                        trbo_state.update(
                            X=X_center,
                            Y=Y_center,
                            new_ind=torch.tensor(
                                [i],
                                dtype=torch.long,
                                device=X_center.device,
                            ),
                        )
                        trbo_state.log_restart_points(X=X_center, Y=Y_center)

                    trbo_state.initialize_standard(
                        tr_idx=i,
                        restart=True,
                        switch_strategy=switch_strategy,
                        **init_kwargs,
                    )
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        trbo_state.update_data_across_trs()
                    tr_restarts[i].append(trbo_state.n_evals.item())

        if trbo_state.tr_hparams.verbose:
            print(f"Total refill points: {trbo_state.total_refill_points}")

        n_evals.append(trbo_state.n_evals.item())
        obj = trbo_state.objective if trbo_state.objective else lambda x: x
        if trbo_state.hv is not None:
            partitioning = DominatedPartitioning(
                ref_point=true_ref_point,
                Y=obj(trbo_state.pareto_Y),
            )
            hv = partitioning.compute_hypervolume().item()
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume: {hv:.3f}")
            pareto_X.append(trbo_state.pareto_X.tolist())
            pareto_Y.append(trbo_state.pareto_Y.tolist())
            true_hv.append(hv)
            if observation_noise_std is not None:
                f.record_current_pf_and_hv(obj=obj, constraints=trbo_state.constraints)
        else:
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume is zero!")
            pareto_X.append([])
            pareto_Y.append([])
            true_hv.append(0.0)
        trbo_state.update_data_across_trs()

        output = _build_output(
            trbo_state=trbo_state,
            n_evals=n_evals,
            true_hv=true_hv,
            pareto_X=pareto_X,
            pareto_Y=pareto_Y,
            n_points_in_tr=n_points_in_tr,
            n_points_in_tr_collected_by_other=n_points_in_tr_collected_by_other,
            n_points_in_tr_collected_by_sobol=n_points_in_tr_collected_by_sobol,
            tr_sizes=tr_sizes,
            tr_centers=tr_centers,
            tr_restarts=tr_restarts,
            fit_times=fit_times,
            gen_times=gen_times,
            all_tr_indices=all_tr_indices,
        )
        if save_during_opt and save_callback is not None:
            save_callback(output)

    end_time = time.time()
    if trbo_state.tr_hparams.verbose:
        print(f"Total time: {end_time - start_time:.1f} seconds")

    if trbo_state.hv is not None and recompute_all_hvs:
        f.record_all_hvs(obj=obj, constraints=trbo_state.constraints)

    output = _build_output(
        trbo_state=trbo_state,
        n_evals=n_evals,
        true_hv=true_hv,
        pareto_X=pareto_X,
        pareto_Y=pareto_Y,
        n_points_in_tr=n_points_in_tr,
        n_points_in_tr_collected_by_other=n_points_in_tr_collected_by_other,
        n_points_in_tr_collected_by_sobol=n_points_in_tr_collected_by_sobol,
        tr_sizes=tr_sizes,
        tr_centers=tr_centers,
        tr_restarts=tr_restarts,
        fit_times=fit_times,
        gen_times=gen_times,
        all_tr_indices=all_tr_indices,
    )
    if trbo_state.hv is not None and recompute_all_hvs:
        output = {**output, **f.get_outputs()}
    if save_callback is not None:
        save_callback(output)

    raw_y_history = trbo_state.Y_history.cpu() / signs.cpu()
    return {
        "X_history": trbo_state.X_history.cpu().tolist(),
        "values_history": raw_y_history.tolist(),
        "pareto_X": trbo_state.pareto_X.cpu().tolist(),
        "pareto_values": (trbo_state.pareto_Y.cpu() / signs.cpu()).tolist(),
        "n_evals": int(trbo_state.n_evals.item()),
    }


def _build_output(
    trbo_state: TRBOState,
    n_evals,
    true_hv,
    pareto_X,
    pareto_Y,
    n_points_in_tr,
    n_points_in_tr_collected_by_other,
    n_points_in_tr_collected_by_sobol,
    tr_sizes,
    tr_centers,
    tr_restarts,
    fit_times,
    gen_times,
    all_tr_indices,
) -> Dict[str, Any]:
    return {
        "n_evals": n_evals,
        "X_history": trbo_state.X_history.cpu(),
        "metric_history": trbo_state.Y_history.cpu(),
        "true_pareto_X": pareto_X,
        "true_pareto_Y": pareto_Y,
        "true_hv": true_hv,
        "n_points_in_tr": n_points_in_tr,
        "n_points_in_tr_collected_by_other": n_points_in_tr_collected_by_other,
        "n_points_in_tr_collected_by_sobol": n_points_in_tr_collected_by_sobol,
        "tr_sizes": tr_sizes,
        "tr_centers": tr_centers,
        "tr_restarts": tr_restarts,
        "fit_times": fit_times,
        "gen_times": gen_times,
        "tr_indices": all_tr_indices,
    }


def _load_objective(spec: Any) -> Callable[[torch.Tensor], Any]:
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError("config['objective'] must be a callable or 'module:function'.")
    if ":" not in spec:
        raise ValueError("Objective must be specified as 'module:function'.")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _evaluate_objective(
    objective: Callable[[torch.Tensor], Any],
    X: torch.Tensor,
) -> torch.Tensor:
    try:
        Y = objective(X)
    except Exception:
        rows = [objective(row) for row in X]
        Y = torch.stack(
            [row if torch.is_tensor(row) else torch.tensor(row) for row in rows],
            dim=0,
        )
    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.double)
    Y = Y.to(dtype=torch.double, device=X.device)
    if Y.ndim == 1:
        Y = Y.unsqueeze(0)
    return Y


def _make_save_callback(save_back: Any, config_dir: Path):
    if save_back is False or save_back is None:
        return None

    if save_back is True:
        save_path = config_dir / "morbo_native_result.pt"
    else:
        save_path = Path(str(save_back)).expanduser()
        if not save_path.is_absolute():
            save_path = config_dir / save_path

    def save_callback(output: Dict[str, Any]) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)  
        torch.save(output, save_path)                       

    return save_callback







