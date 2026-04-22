# Milestone 3 - Data Acquisition, Validation, and Preparation

This milestone implements the data layer for the UFLP project defined in Milestone 1 and used by the solver PoC in Milestone 2. The refactor keeps the project on one shared benchmark source, builds reproducible derived datasets, validates them against explicit contracts, versions the pipeline with DVC, and exposes engineered features through Feast.

### Table of Contents

- [1. Setup, Usage, and Pipeline Guide](#1-setup-usage-and-pipeline-guide)
  - [1.1 Repository Structure](#11-repository-structure)
  - [1.2 Installation and Setup](#12-installation-and-setup)
  - [1.3 Running the Pipeline](#13-running-the-pipeline)
  - [1.4 Troubleshooting](#14-troubleshooting)
- [2. Milestone 3 - ML Pipeline Development - Data Ingestion, Validation and Preparation](#2-milestone-3---ml-pipeline-development---data-ingestion-validation-and-preparation)
  - [2.1 Schema Definition](#21-schema-definition)
  - [2.2 Data Validation and Verification](#22-data-validation-and-verification)
  - [2.3 Data Versioning](#23-data-versioning)
  - [2.4 Setting up a Feature Store](#24-setting-up-a-feature-store)
  - [2.5 Setup of Data Pipeline within the Larger ML Pipeline / MLOps Platform](#25-setup-of-data-pipeline-within-the-larger-ml-pipeline--mlops-platform)
    - [2.5.1 Ingestion of Raw Data and Storage into a Repository](#251-ingestion-of-raw-data-and-storage-into-a-repository)
    - [2.5.2 Preprocessing and Feature Engineering](#252-preprocessing-and-feature-engineering)
- [3. References](#3-references)

## Quick Links

- Report PDF: [report/report.pdf](report/report.pdf)
- Shared raw dataset: [../data/raw/](../data/raw/)
- DVC pipeline: [dvc.yaml](dvc.yaml)
- Pipeline runner: [pipeline/run_data_pipeline.py](pipeline/run_data_pipeline.py)
- Full workflow runner: [run_full_workflow.py](run_full_workflow.py)
- ZenML workflow runner: [run_zenml_workflow.py](run_zenml_workflow.py)
- ZenML pipeline definition: [pipeline/zenml_pipeline.py](pipeline/zenml_pipeline.py)
- Validation summary: [reports/validation/validation_summary.json](reports/validation/validation_summary.json)
- ZenML status: [reports/zenml_status.json](reports/zenml_status.json)
- Feast demo: [feature_repo/run_feature_store_demo.py](feature_repo/run_feature_store_demo.py)
- Milestone 1 README: [../Milestone_1-Project_Inception/README.md](../Milestone_1-Project_Inception/README.md)
- Milestone 2 README: [../Milestone_2-PoC/README.md](../Milestone_2-PoC/README.md)

## 1. Setup, Usage, and Pipeline Guide

### 1.1 Repository Structure

Milestone 3 uses the shared root raw dataset and keeps all derived artifacts inside the milestone folder:

- [../data/raw/](../data/raw/): canonical OR-Library UFLP input and [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt)
- [pipeline/ingest_data.py](pipeline/ingest_data.py): manifest creation from the shared raw dataset
- [pipeline/preprocess_data.py](pipeline/preprocess_data.py): normalized relational tables
- [pipeline/engineer_features.py](pipeline/engineer_features.py): instance-, facility-, and customer-level features
- [pipeline/validate_data.py](pipeline/validate_data.py): schema checks, statistics, anomaly checks, and consistency validation
- [feature_repo/](feature_repo/): Feast entities, views, apply script, and retrieval demo
- [data/interim/](data/interim/), [data/processed/](data/processed/), [data/features/](data/features/): generated M3 outputs
- [reports/validation/](reports/validation/) and [reports/stats/](reports/stats/): validation and statistics outputs

### 1.2 Installation and Setup

From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
cd Milestone_3-Data_Prep
pip install -r requirements-m3.txt
```

[requirements-m3.txt](requirements-m3.txt) now includes the ZenML local-runtime dependencies used by [run_zenml_workflow.py](run_zenml_workflow.py), including `zenml[local]`, `sqlmodel`, `passlib[bcrypt]`, and the supporting SQL/runtime packages needed by the local ZenML store.

For deep Windows paths, DVC works more reliably with a short local cache path:

```powershell
..\.venv\Scripts\python.exe -m dvc config cache.dir D:/dvc-cache-csc5382-m3 --local
```

If you use a different short local path, substitute it in the command above.

### 1.3 Running the Pipeline

Run the data pipeline only:

```powershell
python pipeline/run_data_pipeline.py
```

Run the full workflow, including Feast apply and retrieval demo:

```powershell
python run_full_workflow.py
```

Run the ZenML-orchestrated workflow:

```powershell
python run_zenml_workflow.py
```

Run DVC reproduction:

```powershell
..\.venv\Scripts\python.exe -m dvc repro
```

### 1.4 Troubleshooting

- If raw data is not found, confirm that the shared dataset exists in [../data/raw/](../data/raw/).
- If DVC fails on Windows path length, use a short local cache path as shown above.
- If Feast retrieval fails, rerun the feature stage first, then apply the repo with [feature_repo/apply_repo.py](feature_repo/apply_repo.py).
- If ZenML reports missing modules such as `sqlmodel` or `passlib`, reinstall [requirements-m3.txt](requirements-m3.txt) in the active environment and rerun [run_zenml_workflow.py](run_zenml_workflow.py).
- If the environment has older conflicting ZenML dependencies, recreate the virtual environment and reinstall from [requirements-m3.txt](requirements-m3.txt).

## 2. Milestone 3 - ML Pipeline Development - Data Ingestion, Validation and Preparation

This milestone converts the project from the solver-centric PoC of Milestone 2 into a reproducible data pipeline built on the same UFLP benchmark defined in Milestone 1. The refactor keeps one shared raw benchmark source, produces structured derived datasets, validates them, versions the pipeline with DVC, and exposes features through Feast.

### 2.1 Schema Definition

Milestone 3 uses explicit schemas for each layer:

- [schema/raw_schema.json](schema/raw_schema.json)
- [schema/processed_schema.json](schema/processed_schema.json)
- [schema/feature_schema.json](schema/feature_schema.json)

These schemas define expected files, required columns, numeric and non-negative fields, and identifier keys. They serve as the contracts enforced during validation for the raw, processed, and feature layers.

### 2.2 Data Validation and Verification

[pipeline/validate_data.py](pipeline/validate_data.py) validates:

- raw dataset completeness against the expected OR-Library benchmark files
- required columns and numeric typing
- non-negativity constraints
- duplicate identifier detection
- processed consistency for `m`, `n`, and the complete `m x n` assignment matrix
- feature consistency against processed tables
- value range checks such as normalized fields remaining in `[0, 1]`

It writes:

- [reports/validation/validation_summary.json](reports/validation/validation_summary.json)
- [reports/validation/anomalies.json](reports/validation/anomalies.json)
- [reports/validation/anomalies.txt](reports/validation/anomalies.txt)
- [reports/stats/raw_statistics.json](reports/stats/raw_statistics.json)
- [reports/stats/processed_statistics.json](reports/stats/processed_statistics.json)
- [reports/stats/feature_statistics.json](reports/stats/feature_statistics.json)

Current result: `status = passed`, `anomaly_count = 0`.

### 2.3 Data Versioning

Milestone 3 uses DVC through [dvc.yaml](dvc.yaml) and [dvc.lock](dvc.lock). The refactor removed the duplicated milestone-local raw snapshot, switched DVC dependencies to the shared root dataset [../data/raw/](../data/raw/), and moved generated Parquet outputs out of normal Git tracking so DVC can manage the pipeline outputs correctly.

The current DVC stages are:

1. `ingest`
2. `preprocess`
3. `features`
4. `validate`

### 2.4 Setting up a Feature Store

Milestone 3 uses Feast through [feature_repo/](feature_repo/) with:

- entities: `instance_id`, `facility_key`, `customer_key`
- feature views over the generated Parquet files
- local registry and online store for demonstration

[feature_repo/views.py](feature_repo/views.py) now points directly at the generated M3 feature outputs through the shared path contract. Both [feature_repo/apply_repo.py](feature_repo/apply_repo.py) and [feature_repo/run_feature_store_demo.py](feature_repo/run_feature_store_demo.py) run successfully after the refactor, and the feature-store publish step is also included in the ZenML workflow.

### 2.5 Setup of Data Pipeline within the Larger ML Pipeline / MLOps Platform

Milestone 3 now exposes two orchestration layers:

- a lightweight local runner in [pipeline/run_data_pipeline.py](pipeline/run_data_pipeline.py) and [run_full_workflow.py](run_full_workflow.py)
- an MLOps-platform runner in [pipeline/zenml_pipeline.py](pipeline/zenml_pipeline.py), launched through [run_zenml_workflow.py](run_zenml_workflow.py)

The ZenML pipeline wraps the same refactored stage functions used by the local workflow and executes them in explicit order:

1. ingestion
2. preprocessing
3. feature engineering
4. validation
5. feature-store publication

This keeps the milestone reproducible for local development while also satisfying the requirement to integrate the data pipeline into a larger ML pipeline / MLOps platform. The most recent ZenML execution status is written to:

- [reports/zenml_status.json](reports/zenml_status.json)
- [reports/zenml_status.txt](reports/zenml_status.txt)

#### 2.5.1 Ingestion of Raw Data and Storage into a Repository

[pipeline/ingest_data.py](pipeline/ingest_data.py) scans the shared benchmark source in [../data/raw/](../data/raw/), excludes `uncapopt.txt`, reads each instance header, checks optimum availability, and writes a structured manifest to:

- [data/interim/dataset_manifest.csv](data/interim/dataset_manifest.csv)
- [data/interim/dataset_manifest.json](data/interim/dataset_manifest.json)

The manifest captures source path, file size, facility count, customer count, optimum availability, and ingestion status for all 15 benchmark instances.

#### 2.5.2 Preprocessing and Feature Engineering

[pipeline/preprocess_data.py](pipeline/preprocess_data.py) parses the OR-Library benchmark into normalized relational tables:

- [data/processed/instances.csv](data/processed/instances.csv)
- [data/processed/facilities.csv](data/processed/facilities.csv)
- [data/processed/customers.csv](data/processed/customers.csv)
- [data/processed/assignment_costs.csv](data/processed/assignment_costs.csv)

The refactor fixed the parser so it now handles both benchmark formats used in the dataset, including the larger `capa/capb/capc` files that contain literal `capacity` tokens in facility rows.

[pipeline/engineer_features.py](pipeline/engineer_features.py) then builds:

- instance features: counts, ratios, and cost summaries
- facility features: normalized fixed costs, z-scores, ranks, and assignment-cost summaries
- customer features: assignment-cost statistics and nearest-facility features

Current pipeline output counts are:

- 15 ingested benchmark instances
- 15 processed instance rows
- 664 processed facility rows
- 3600 processed customer rows
- 318200 assignment-cost rows
- 15 instance feature rows
- 664 facility feature rows
- 3600 customer feature rows

These counts are recorded in [reports/validation/validation_summary.json](reports/validation/validation_summary.json).

## 3. References

- OR-Library UFLP benchmark dataset, mirrored in [../data/raw/](../data/raw/)
- DVC documentation: https://dvc.org/doc
- Feast documentation: https://docs.feast.dev/
- Google OR-Tools documentation: https://developers.google.com/optimization
