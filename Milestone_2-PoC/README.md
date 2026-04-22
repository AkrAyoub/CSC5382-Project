# Milestone 2 - Development of Proof-of-Concepts

This milestone implements a proof of concept for the Milestone 1 project direction: using an off-the-shelf LLM as a symbolic modeler for the Uncapacitated Facility Location Problem (UFLP), while keeping a deterministic CBC solver as the trusted reference and verifier.

### Table of Contents

- [1. Setup, Usage, and Demo Guide](#1-setup-usage-and-demo-guide)
  - [1.1 Repository Structure](#11-repository-structure)
  - [1.2 Installation and Setup](#12-installation-and-setup)
  - [1.3 Running the Streamlit App](#13-running-the-streamlit-app)
  - [1.4 Demo Flow](#14-demo-flow)
  - [1.5 Screenshots](#15-screenshots)
  - [1.6 Troubleshooting](#16-troubleshooting)
- [2. Milestone 2 - Development of Proof-of-Concepts](#2-milestone-2---development-of-proof-of-concepts)
  - [2.1 Model Integration](#21-model-integration)
  - [2.2 App Development](#22-app-development)
  - [2.3 End-to-End Scenario Testing](#23-end-to-end-scenario-testing)
- [3. References](#3-references)

## Quick Links

- App: [app.py](app.py)
- Baseline solver: [src/baseline_solver.py](src/baseline_solver.py)
- Unified PoC pipeline: [src/poc_pipeline.py](src/poc_pipeline.py)
- LLM backend abstraction: [src/llm_backend.py](src/llm_backend.py)
- LLM generation pipeline: [src/llm_generate.py](src/llm_generate.py)
- Sandboxed execution: [src/exec_generated.py](src/exec_generated.py)
- Shared dataset paths: [src/paths.py](src/paths.py)
- End-to-end tests: [tests/e2e.py](tests/e2e.py)
- Shared dataset: [../data/raw/](../data/raw/)
- Best-known objectives: [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt)

## 1. Setup, Usage, and Demo Guide

### 1.1 Repository Structure

The milestone is organized around one baseline-first proof-of-concept flow:

- [app.py](app.py): Streamlit user interface
- [src/baseline_solver.py](src/baseline_solver.py): OR-Library parser and deterministic OR-Tools CBC baseline
- [src/poc_pipeline.py](src/poc_pipeline.py): unified PoC scenario runner
- [src/llm_backend.py](src/llm_backend.py): LLM backend abstraction
- [src/llm_generate.py](src/llm_generate.py): code generation and validation
- [src/exec_generated.py](src/exec_generated.py): sandboxed execution and strict output validation
- [src/pipeline_trace.py](src/pipeline_trace.py): execution trace for the UI
- [tests/e2e.py](tests/e2e.py): end-to-end milestone test
- [requirements.txt](requirements.txt): dependencies
- [assets/screenshots/](assets/screenshots/): demo screenshots
- [../data/raw/](../data/raw/): shared OR-Library UFLP instances and [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt)

### 1.2 Installation and Setup

Create and activate a virtual environment, install dependencies, and optionally configure the LLM environment variables from the `Milestone_2-PoC` folder.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The deterministic baseline path does not require an API key. The optional LLM verification path uses Groq:

```powershell
$env:GROQ_API_KEY="YOUR_KEY"
$env:GROQ_MODEL="llama-3.1-8b-instant"
```

A key can be obtained from: https://console.groq.com/keys

### 1.3 Running the Streamlit App

Launch the Streamlit app:

```powershell
python -m streamlit run app.py
```

The app allows the user to:

- select an OR-Library instance from the shared dataset
- run the deterministic CBC baseline
- optionally enable the LLM-generated solver verification path
- inspect the baseline result, optional generated result, and the pipeline trace

### 1.4 Demo Flow

The demo flow is:

- choose an OR-Library instance from the shared dataset
- run the deterministic CBC baseline
- optionally enable LLM-generated solver verification for the same instance
- inspect the baseline objective, best-known gap when available, generated objective when available, and the full pipeline trace

This replaces the older split between a baseline-only mode and a separate LLM mode with one clearer scenario centered on the trusted baseline.

### 1.5 Screenshots

Captured application screenshots are stored in [assets/screenshots/](assets/screenshots/). The folder includes baseline runs and LLM-verification runs that can be reused in the milestone report and demo package.

### 1.6 Troubleshooting

- If `streamlit` is not found, use `python -m streamlit run app.py`.
- If `No instances found` appears, verify that the dataset exists in [../data/raw/](../data/raw/).
- If the LLM path is disabled or no API key is set, the app still runs the deterministic baseline path.
- If the LLM path hits rate limits, the backend retries automatically; if the failure persists, wait briefly and rerun.
- The app shows a pipeline trace so generation, sandbox execution, and comparison failures can be inspected directly in the UI.

## 2. Milestone 2 - Development of Proof-of-Concepts

The proof of concept uses OR-Library UFLP instances from [../data/raw/](../data/raw/) and demonstrates one baseline-centered scenario: solve the selected instance with the deterministic CBC solver first, then optionally ask an off-the-shelf LLM to generate solver code for the same instance and verify the generated result against the baseline. This keeps the milestone focused on a working end-to-end PoC while staying aligned with Milestone 1, where the LLM assists symbolic modeling and the solver remains the trusted optimization engine.

### 2.1 Model Integration

The primary integrated model is the deterministic OR-Tools CBC baseline in [src/baseline_solver.py](src/baseline_solver.py). It parses the OR-Library UFLP format, builds the facility-open variables and customer-assignment variables, minimizes fixed plus assignment cost, and compares the objective against [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt) when best-known values are available. This satisfies the milestone requirement for a working baseline model on the project dataset.

The PoC also integrates an off-the-shelf LLM path through [src/llm_backend.py](src/llm_backend.py) and [src/llm_generate.py](src/llm_generate.py). The LLM does not replace the optimizer. Instead, it generates OR-Tools solver code, that code is statically checked, executed through [src/exec_generated.py](src/exec_generated.py), and compared against the deterministic baseline inside [src/poc_pipeline.py](src/poc_pipeline.py). This preserves the Milestone 1 architecture: the LLM acts as the symbolic modeler, while the solver provides the correctness reference.

The refactor also introduced [src/paths.py](src/paths.py) so the PoC uses the shared repository dataset root instead of milestone-local hardcoded paths. In addition, the LLM backend is abstracted so the current off-the-shelf model and any later fine-tuned model can use the same interface without changing the app flow or the evaluation logic.

### 2.2 App Development

The application is implemented in Streamlit in [app.py](app.py). The refactored app follows a single scenario instead of two disconnected modes:

- the user selects one OR-Library instance
- the deterministic CBC baseline always runs first
- the user may optionally enable LLM-generated solver verification for the same instance
- the UI reports the baseline objective, best-known gap when available, LLM objective when enabled, and the full execution trace

This app structure is intentionally simple because the milestone rubric asks for a working proof of concept rather than a full production interface. The refactor removed the older split between a baseline-only mode and a separate LLM mode, and replaced it with one clearer workflow that is easier to demonstrate, test, and extend in later milestones. The app also exposes the pipeline trace so generation, validation, sandbox execution, and comparison steps can be inspected during demos and debugging.

### 2.3 End-to-End Scenario Testing

The end-to-end milestone test is implemented in [tests/e2e.py](tests/e2e.py). It verifies the runnable scenario instead of UI-only behavior.

The baseline test runs on three repository instances and checks that the CBC result stays at or near the known optimum when [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt) contains a reference value. This gives direct evidence that the integrated baseline is functioning correctly on the project dataset and that the shared data path assumptions are valid.

The optional LLM verification test runs on one instance when `E2E_ENABLE_LLM=1` is set. It executes the same baseline-first pipeline used by the app, then checks that the generated solver objective is sufficiently close to the baseline objective. This preserves reproducibility for normal grading while still allowing the extended LLM path to be exercised when credentials are available. Together, the baseline and optional LLM checks provide milestone evidence for model integration, app integration, and an executable end-to-end scenario.

## 3. References

- OR-Library, UFLP benchmark instances and best-known solutions, stored in [../data/raw/](../data/raw/)
- Google OR-Tools CBC solver documentation: https://developers.google.com/optimization
- Groq API documentation: https://console.groq.com/docs
