# AI-Assisted Symbolic Optimization for Strategic Facility Network Design
## Milestone 4 - Model Development and Offline Evaluation

### Table of Contents

- [1. Setup, Usage, and Workflow Guide](#1-setup-usage-and-workflow-guide)
  - [1.1 Repository Structure](#11-repository-structure)
  - [1.2 Installation and Setup](#12-installation-and-setup)
  - [1.3 Running the Single-Candidate Evaluation](#13-running-the-single-candidate-evaluation)
  - [1.4 Running the Model Comparison Workflow](#14-running-the-model-comparison-workflow)
  - [1.5 Running the ZenML Workflow](#15-running-the-zenml-workflow)
  - [1.6 Produced Runtime Outputs](#16-produced-runtime-outputs)
  - [1.7 Troubleshooting](#17-troubleshooting)
- [2. Milestone 4 - ML Pipeline Development - Model Training and Offline Evaluation](#2-milestone-4---ml-pipeline-development---model-training-and-offline-evaluation)
  - [2.1 Project Structure Definition and Modularity](#21-project-structure-definition-and-modularity)
  - [2.2 Code Versioning](#22-code-versioning)
  - [2.3 Experiment Tracking and Model Versioning](#23-experiment-tracking-and-model-versioning)
  - [2.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform](#24-integration-of-model-training-and-offline-evaluation-into-the-ml-pipeline--mlops-platform)
  - [2.5 Optional Energy Efficiency Measurement](#25-optional-energy-efficiency-measurement)
- [3. References](#3-references)
- [4. Presentation](#4-presentation)

## 1. Setup, Usage, and Workflow Guide

This milestone no longer trains a facility-level classifier. It now evaluates the actual Milestone 1 modeling component: an `LLM-as-modeler` system that generates solver-compatible UFLP code, executes it in a sandbox, and compares it against a deterministic CBC reference solver.

The milestone supports three execution modes:

- local single-candidate evaluation through MLflow
- multi-candidate offline comparison through MLflow
- end-to-end orchestration through ZenML

### 1.1 Repository Structure

- [`configs/`](configs/) - config files for single-candidate evaluation and multi-candidate comparison
- [`scripts/run_train.py`](scripts/run_train.py) - runs the single-candidate offline evaluation workflow
- [`scripts/run_compare.py`](scripts/run_compare.py) - runs the candidate comparison workflow
- [`scripts/run_zenml.py`](scripts/run_zenml.py) - runs the ZenML pipeline wrapper
- [`src/m4_model_dev/data/`](src/m4_model_dev/data/) - benchmark parsing, reference solutions, benchmark dataset, grouped splits, and SFT dataset creation
- [`src/m4_model_dev/models/`](src/m4_model_dev/models/) - candidate registry and LLM symbolic code generation
- [`src/m4_model_dev/evaluation/`](src/m4_model_dev/evaluation/) - sandbox execution and offline evaluation metrics
- [`src/m4_model_dev/pipelines/`](src/m4_model_dev/pipelines/) - local and ZenML-integrated training/evaluation workflows
- [`src/m4_model_dev/reporting/`](src/m4_model_dev/reporting/) - report tables, summaries, and figures
- [`src/m4_model_dev/tracking/`](src/m4_model_dev/tracking/) - MLflow and CodeCarbon integration
- [`tests/`](tests/) - config, registry, and comparison-selection tests
- [`data/reference/`](data/reference/) - deterministic reference solutions built from the shared OR-Library benchmark
- [`data/datasets/`](data/datasets/) - instance-level benchmark dataset
- [`data/splits/`](data/splits/) - grouped train/validation/test split definition
- [`data/sft/`](data/sft/) - prompt/response SFT dataset prepared for optional fine-tuning
- [`reports/`](reports/) - generated evaluation summaries, figures, comparison outputs, energy logs, and ZenML status artifacts

### 1.2 Installation and Setup

Create and activate a virtual environment, then install the milestone requirements:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
cd Milestone_4-Model_Dev
pip install -r requirements.txt
```

Optional environment variables for LLM-backed candidates:

```bash
$env:GROQ_API_KEY="YOUR_KEY"
$env:GROQ_MODEL="llama-3.1-8b-instant"
```

Notes:

- `zenml[local]` and its local-store dependencies are already included in [`requirements.txt`](requirements.txt).
- MLflow artifacts are written locally under `Milestone_4-Model_Dev/mlruns/`.
- ZenML local state is written under `Milestone_4-Model_Dev/.zen/`.
- Generated solver code is written under `reports/generated_code/` and is ignored as a local runtime artifact.

### 1.3 Running the Single-Candidate Evaluation

The default single-candidate configuration evaluates the `llm_robust_prompt_v1` candidate:

```bash
python scripts/run_train.py
```

Main outputs:

- [`reports/summary.txt`](reports/summary.txt)
- [`reports/run_manifest.json`](reports/run_manifest.json)
- [`reports/evaluation/single_candidate_metrics.csv`](reports/evaluation/single_candidate_metrics.csv)
- [`reports/evaluation/single_candidate_raw_results.csv`](reports/evaluation/single_candidate_raw_results.csv)
- [`reports/training_metrics.png`](reports/training_metrics.png)
- [`reports/training_status.png`](reports/training_status.png)
- [`reports/training_dashboard.png`](reports/training_dashboard.png)
- [`artifacts/llm_robust_prompt_v1_spec.json`](artifacts/llm_robust_prompt_v1_spec.json)

### 1.4 Running the Model Comparison Workflow

The comparison workflow evaluates the deterministic reference baseline, two off-the-shelf LLM candidates, and an optional fine-tuned placeholder:

```bash
python scripts/run_compare.py
```

Main outputs:

- [`reports/model_comparison.csv`](reports/model_comparison.csv)
- [`reports/model_comparison.json`](reports/model_comparison.json)
- [`reports/model_comparison_raw_results.csv`](reports/model_comparison_raw_results.csv)
- [`reports/model_comparison_summary.txt`](reports/model_comparison_summary.txt)
- [`reports/model_selection.json`](reports/model_selection.json)
- [`reports/comparison_validation_metrics.png`](reports/comparison_validation_metrics.png)
- [`reports/comparison_test_metrics.png`](reports/comparison_test_metrics.png)
- [`reports/comparison_dashboard.png`](reports/comparison_dashboard.png)

### 1.5 Running the ZenML Workflow

The same training/evaluation flow is exposed through ZenML:

```bash
python scripts/run_zenml.py
```

ZenML status outputs:

- [`reports/zenml_status.json`](reports/zenml_status.json)
- [`reports/zenml_status.txt`](reports/zenml_status.txt)

### 1.6 Produced Runtime Outputs

Key generated assets after the verified local runs:

- deterministic reference solutions: [`data/reference/reference_solutions.csv`](data/reference/reference_solutions.csv)
- benchmark instance dataset: [`data/datasets/benchmark_instances.csv`](data/datasets/benchmark_instances.csv)
- grouped split definition: [`data/splits/instance_splits.csv`](data/splits/instance_splits.csv)
- SFT dataset manifest: [`data/sft/sft_manifest.json`](data/sft/sft_manifest.json)
- MLflow-registered single-candidate artifact family: local registry name `m4-symbolic-generator-best`
- current ZenML integration status: [`reports/zenml_status.json`](reports/zenml_status.json)

### 1.7 Troubleshooting

- `Missing GROQ_API_KEY`
  - Set `GROQ_API_KEY` before running LLM-backed candidates.
  - The deterministic baseline still runs without any API key.

- `429 Too Many Requests`
  - The Groq backend is retried automatically.
  - If rate limits persist, wait and rerun the workflow.

- `Windows path too long`
  - This milestone avoids bundling the entire source tree into MLflow model artifacts.
  - Keep the repo in a reasonably short local path if you move it.

- ZenML local-store errors
  - Reinstall from [`requirements.txt`](requirements.txt) in a fresh virtual environment.
  - Then rerun `python scripts/run_zenml.py`.

## 2. Milestone 4 - ML Pipeline Development - Model Training and Offline Evaluation

This milestone realigns the project with Milestone 1 by evaluating symbolic optimization-model generation rather than a proxy supervised classifier. The deterministic CBC solver remains the trusted reference baseline. M4 develops and evaluates multiple candidate generator variants:

- `deterministic_baseline`
- `llm_token_prompt_v0`
- `llm_robust_prompt_v1`
- `llm_fine_tuned` placeholder for future fine-tuning

The current verified local runs show that the deterministic reference path remains reliable, while the off-the-shelf LLM candidates still fail to produce solver-valid code consistently under the stricter offline evaluation pipeline. That negative result is still useful and aligned with the milestone: the training/evaluation stack, tracking, model versioning, and MLOps integration are now measuring the correct project objective.

### 2.1 Project Structure Definition and Modularity

The milestone was restructured around clear module boundaries instead of the old classifier workflow:

- benchmark and reference-data construction in [`src/m4_model_dev/data/benchmark.py`](src/m4_model_dev/data/benchmark.py), [`src/m4_model_dev/data/build_reference_solutions.py`](src/m4_model_dev/data/build_reference_solutions.py), and [`src/m4_model_dev/data/build_benchmark_dataset.py`](src/m4_model_dev/data/build_benchmark_dataset.py)
- split definition and optional fine-tuning dataset creation in [`src/m4_model_dev/data/make_splits.py`](src/m4_model_dev/data/make_splits.py) and [`src/m4_model_dev/data/build_sft_dataset.py`](src/m4_model_dev/data/build_sft_dataset.py)
- candidate definitions in [`src/m4_model_dev/models/model_registry.py`](src/m4_model_dev/models/model_registry.py)
- LLM code generation and validation in [`src/m4_model_dev/models/symbolic_generator.py`](src/m4_model_dev/models/symbolic_generator.py)
- sandboxed execution in [`src/m4_model_dev/evaluation/generated_exec.py`](src/m4_model_dev/evaluation/generated_exec.py)
- split-level metric aggregation in [`src/m4_model_dev/evaluation/metrics.py`](src/m4_model_dev/evaluation/metrics.py)
- local and ZenML pipelines in [`src/m4_model_dev/pipelines/training_pipeline.py`](src/m4_model_dev/pipelines/training_pipeline.py), [`src/m4_model_dev/pipelines/comparison_pipeline.py`](src/m4_model_dev/pipelines/comparison_pipeline.py), and [`src/m4_model_dev/pipelines/zenml_pipeline.py`](src/m4_model_dev/pipelines/zenml_pipeline.py)
- reporting and figures in [`src/m4_model_dev/reporting/`](src/m4_model_dev/reporting/)

This modular layout supports both the current off-the-shelf candidates and a later fine-tuned candidate without changing the pipeline shape.

### 2.2 Code Versioning

Code versioning is handled through the Git repository and milestone-local modularization:

- Milestone 4 is isolated under [`Milestone_4-Model_Dev/`](.)
- runtime behavior is controlled through versioned config files in [`configs/`](configs/)
- reproducible entry points are exposed through [`scripts/`](scripts/)
- regression coverage is provided through [`tests/`](tests/)

The refactor also removed the old classifier-specific files and replaced them with the symbolic-evaluation implementation that matches the Milestone 1 project identity.

### 2.3 Experiment Tracking and Model Versioning

MLflow is used for experiment tracking and local model versioning through [`src/m4_model_dev/tracking/mlflow_utils.py`](src/m4_model_dev/tracking/mlflow_utils.py).

Tracked run contents include:

- candidate metadata: backend, model name, prompt template, token budget, enabled state
- split-level offline metrics from [`reports/evaluation/single_candidate_metrics.csv`](reports/evaluation/single_candidate_metrics.csv) and [`reports/model_comparison.csv`](reports/model_comparison.csv)
- raw evaluation tables from [`reports/evaluation/single_candidate_raw_results.csv`](reports/evaluation/single_candidate_raw_results.csv) and [`reports/model_comparison_raw_results.csv`](reports/model_comparison_raw_results.csv)
- run manifests and report figures
- a registered local model artifact family under `m4-symbolic-generator-best`

Model development is therefore tracked as versioned candidate specifications plus their offline solver-grounded evaluation outcomes, not as traditional classifier weights.

### 2.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform

The milestone integrates the full offline workflow into both a local pipeline and a ZenML pipeline.

Local pipeline flow:

1. Build deterministic reference solutions from the shared benchmark.
2. Build the benchmark instance dataset.
3. Build grouped train/validation/test splits.
4. Build an SFT prompt/response dataset for future fine-tuning.
5. Evaluate one candidate or multiple candidates offline.
6. Aggregate split-level metrics and write report artifacts.
7. Log the run to MLflow.

The same flow is wrapped by ZenML in [`src/m4_model_dev/pipelines/zenml_pipeline.py`](src/m4_model_dev/pipelines/zenml_pipeline.py), and the verified status is recorded in [`reports/zenml_status.json`](reports/zenml_status.json).

Offline evaluation metrics are solver-grounded and aligned with Milestone 1:

- generation success rate
- execution success rate
- feasibility rate
- exact match rate against the deterministic baseline
- objective gap versus the deterministic baseline
- objective gap versus the best known OR-Library optimum when available
- runtime per split

Current verified comparison evidence:

- the deterministic baseline remains exact and reliable
- the off-the-shelf LLM candidates still fail validation/execution on the current local runs
- the comparison workflow still selects the best non-baseline candidate according to validation-only rules and records that choice in [`reports/model_selection.json`](reports/model_selection.json)

### 2.5 Optional Energy Efficiency Measurement

Optional energy measurement is enabled for the single-candidate workflow through CodeCarbon in [`src/m4_model_dev/tracking/codecarbon_utils.py`](src/m4_model_dev/tracking/codecarbon_utils.py).

Current energy evidence is written to:

- [`reports/emissions.csv`](reports/emissions.csv)

This keeps the energy-efficiency bonus as a real runnable option instead of a documentation-only claim.

## 3. References

- OR-Library UFLP benchmark: [`../data/raw/`](../data/raw/)
- OR-Tools linear solver: https://developers.google.com/optimization
- MLflow documentation: https://mlflow.org/
- ZenML documentation: https://docs.zenml.io/
- CodeCarbon documentation: https://mlco2.github.io/codecarbon/
- Groq API documentation: https://console.groq.com/docs

## 4. Presentation

- Recorded presentation placeholder: add the final Milestone 4 demo link here.
