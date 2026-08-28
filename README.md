# SCLAP-BO End-to-End OTA Sizing Workflow

This repository contains an end-to-end workflow for analog OTA sizing. The main
entry point is `test_turbo_llm.py`: it starts from an unbound Ngspice benchmark
netlist, recognizes circuit structures, binds sizing parameters, asks the LLM to
build functional perturbation groups by consensus, and runs TuRBO optimization.

## Repository Layout

`Ngspice_Benchmark/`

Open-source benchmark circuits with annotated sizing parameters. This directory
is the reference benchmark used to keep the recovered parameter binding order
consistent with the annotated benchmark.

`Ngspice_Benchmark_unbound/`

End-to-end test benchmark. The experiment starts from these unbound circuits and
recovers parameter bindings through structure recognition instead of relying on
pre-annotated parameter names.

`src/`

Circuit simulation platform. It provides the Ngspice-facing simulation
infrastructure used by `OTA_score.py` and the optimization scripts.

`Optimization_algorithm/`

General optimization algorithms. `test_turbo_llm.py` uses the TuRBO
implementation from `Optimization_algorithm/TuRBO`.

`recognized_binding/`

Structure recognition and deterministic parameter binding. It runs the ALIGN
structure recognizer, writes recognized netlists and structures, and converts
recognized structures into parameter binding JSON files.

`SCLAP_BO/`

LLM-based functional block analysis and adaptive group perturbation. It builds
LLM functional groups, runs the consensus mechanism, and supplies group
probabilities to the optimizer.

## Main Workflow

Run:

```bash
conda run -n rl python -u test_turbo_llm.py --multi-seed
```

The workflow is:

1. `test_turbo_llm.py` parses the experiment arguments.
2. With the default `--binding-source align`, it calls
   `recognized_binding.align_binding_pipeline.build_align_bindings`.
3. ALIGN structure recognition generates:
   - `results_<circuit>/turbo_llm_multi_seed/align/recognized_netlists/`
   - `results_<circuit>/turbo_llm_multi_seed/align/recognized_structures/`
   - `results_<circuit>/turbo_llm_multi_seed/align/structure_bindings/`
4. `ota_experiment.make_ota_evaluator` loads the circuit simulation function and
   compresses the unbound parameters using the freshly generated structure
   bindings.
5. For `Ngspice_Benchmark_unbound`, the compressed parameter order is
   canonicalized against `Ngspice_Benchmark` so the recovered end-to-end system
   is comparable with the annotated benchmark.
6. `SCLAP_BO.llm_group_binding_consensus.run_llm_bindings` calls the LLM
   multiple times. The default is 10 runs.
7. `SCLAP_BO.llm_group_binding_consensus.build_consensus_binding` builds one
   stable functional group binding from the repeated LLM outputs.
8. The consensus result is written to:
   `results_<circuit>/turbo_llm_multi_seed/shared_llm_group_binding/`
9. Each seed reuses the same consensus functional groups and runs TuRBO with
   adaptive group perturbation.
10. Each seed writes:
    `results_<circuit>/results_<seed>/turbo_llm_history.json`
11. The multi-seed summary is written to:
    `results_<circuit>/turbo_llm_multi_seed/multi_seed_summary.csv`

The current workflow does not load external LLM group binding JSON files. The
LLM group binding is generated inside the run.

## Common Commands

Run the default multi-seed experiment:

```bash
conda run -n rl python -u test_turbo_llm.py --multi-seed
```

Run one circuit explicitly:

```bash
conda run -n rl python -u test_turbo_llm.py \
  --multi-seed \
  --circuit folded_vc_ota
```

Available circuits:

```text
ss_vc_ota
folded_cascode_ota
folded_vc_ota
```

Run a small smoke test:

```bash
conda run -n rl python -u test_turbo_llm.py \
  --circuit folded_vc_ota \
  --total-evals 3 \
  --n-init 2 \
  --batch-size 1 \
  --n-candidates 4 \
  --n-training-steps 1 \
  --llm-consensus-runs 1 \
  --output-dir .verify_turbo_llm
```

Run the non-LLM TuRBO baseline with the same ALIGN binding path:

```bash
conda run -n rl python -u test_turbo_pure.py \
  --circuit folded_vc_ota \
  --multi-seed
```

## Key Arguments

`--circuit`

Circuit name. Default: `folded_vc_ota`.

`--ngspice-root`

Input benchmark root. Default: `Ngspice_Benchmark_unbound`.

`--binding-source`

Parameter binding source. Default: `align`, which means the workflow generates
fresh recognized binding files from the input netlist.

`--llm-consensus-runs`

Number of repeated LLM calls used for consensus. Default: `10`.

`--llm-consensus-threshold`

Minimum support ratio for stable parameter co-grouping. Default: `0.6`.

`--seed-start`, `--seed-stop`, `--seed-count`

Multi-seed configuration. The default generates 10 seeds from 2 to 92.

## LLM Configuration

The LLM client reads standard environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=gpt-5.5
```

`--llm-model` overrides `OPENAI_MODEL`.

## Output Files

For multi-seed runs, the important files are:

```text
results_<circuit>/turbo_llm_multi_seed/align/
results_<circuit>/turbo_llm_multi_seed/shared_llm_group_binding/
results_<circuit>/turbo_llm_multi_seed/multi_seed_summary.csv
results_<circuit>/results_<seed>/turbo_llm_history.json
results_<circuit>/results_<seed>/turbo_llm_history.csv
```

`turbo_llm_history.json` records the best result, full optimization history,
parameter compression result, LLM functional groups, binding source, and runtime
metadata.
