## AI-Assisted Symbolic Optimization for Strategic Facility Network Design
## UFLP Baseline vs LLM-Generated Solver (Streamlit)


## Table of Contents

- [Overview](#overview)
- [Repository structure (Milestone_2-PoC)](#repository-structure-milestone_2-poc)
- [Dataset (OR-Library UFLP)](#dataset-or-library-uflp)
- [Highlights](#highlights)
- [Installation & Setup](#installation--setup)
- [Running the StreamlitApp](#running-the-streamlitapp)
  - [Mode A - Baseline Solver](#mode-a---baseline-solver)
  - [Mode B - LLM + Verification Pipeline](#mode-b---llm--verification-pipeline)
- [End-to-End Testing (Mode A and Mode B)](#end-to-end-testing-mode-a-and-mode-b)
- [Demo Screenshots](#demo-screenshots)
- [Troubleshooting](#troubleshooting)



## Overview

This milestone implements a **Proof of Concept (PoC)** for the **Uncapacitated Facility Location Problem (UFLP)** combining:

- **Mode A (Baseline):** Deterministic OR-Tools CBC MILP solver  
- **Mode B (LLM + Verify):** Groq LLM generates a solver module -> executed in a sandbox -> verified against the baseline  

The system demonstrates:

- Deterministic optimization baseline  
- LLM-based symbolic model generation  
- Automatic validation + sandbox execution  
- Streamlit user interface  
- Automated end-to-end testing covering both modes  



## Repository structure (Milestone_2-PoC)

- **App**
  - [`app.py`](app.py) — Streamlit UI (Mode A + Mode B)
- **Core modules**
  - [`src/baseline_solver.py`](src/baseline_solver.py) — OR-Library parser + OR-Tools CBC baseline
  - [`src/compare.py`](src/compare.py) — pipeline driver: baseline and LLM solver comparison
  - [`src/llm_generate.py`](src/llm_generate.py) — Groq code generation + static validation + retry handling
  - [`src/exec_generated.py`](src/exec_generated.py) — sandbox execution + strict output validation
  - [`src/pipeline_trace.py`](src/pipeline_trace.py) — structured trace for UI + debugging
- **Testing**
  - [`tests/e2e.py`](tests/e2e.py) — end-to-end smoke test (Mode A + Mode B)
- **Dependencies**
  - [`requirements.txt`](requirements.txt)
- **Data (in repo root, shared by all milestones)**
  - [`../data/raw/`](../data/raw/) — OR-Library instances + `uncapopt.txt`



## Dataset (OR-Library UFLP)

The PoC uses OR-Library UFLP instances stored in:

- **Instances directory:** [`../data/raw/`](../data/raw/)
- **Known optima file:** [`../data/raw/uncapopt.txt`](../data/raw/uncapopt.txt)

Supported instance files include: `cap71.txt … cap74.txt`, `cap101 …`, `cap131 …`, `capa/capb/capc`.



## Highlights

### Deterministic Baseline

- OR-Tools CBC MILP
- Verified against OR-Library optimal values

### LLM Safety Controls

- Static validation of generated code
- Sandboxed execution environment
- Import restrictions
- Strict output validation

### Verification Architecture

- LLM does not directly optimize.
- It generates symbolic solver code.
- The classical solver guarantees correctness.



## Installation & Setup

### 1. Create a Virtual Environment (Recommended)

**Windows (PowerShell)**

```bash
python -m venv .venv
```

### 2. Install Dependencies

```bash
pip install -r requiremments.txt
```

Dependencies:

- streamlit
- ortools
- groq

### 3. Groq API key setup

```bash
$env:GROQ_API_KEY="YOUR_KEY"
$env:GROQ_MODEL="llama-3.1-8b-instant"
```

A key can be obtained from: https://console.groq.com/keys



## Running the StreamlitApp

```bash
streamlit run app.py
```

The app allows:

- Selecting an OR-Library instance
- Running Mode A (Baseline)
- Running Mode B (LLM + Verify)
- Viewing full execution trace and generated code


### Mode A - Baseline Solver

- Parses OR-Library UFLP format
- Builds MILP with:
  - y[i] open facility (BoolVar)
  - x[j,i] customer assignment (BoolVar)
- Uses:
```bash
solver = pywraplp.Solver.CreateSolver("CBC")
```
- Computes objective:
  - sum fixed costs + sum assignment costs
- Validates against uncapopt.txt

Implementation: [`src/baseline_solver.py`](src/baseline_solver.py)


### Mode B - LLM + Verification Pipeline

- LLM generates OR-Tools CBC solver code

- Static validation ensures:
  - Correct solver constructor
  - Proper parsing structure
  - No unsafe imports
  - Correct objective construction
- Code executes in a sandbox with restricted builtins
- Output validated for:
  - Feasibility
  - Assignment correctness
  - Matching baseline objective
- Gap computed vs baseline


Core modules:
['src/llm_generate.py'](src/llm_generate.py)
['src/exec_generated.py'](src/exec_generated.py)
['src/compare.py'](src/compare.py)
['src/pipeline_trace.py'](src/pipeline_trace.py)



## End-to-End Testing (Mode A and Mode B)

```bash
python tests/e2e.py
```

### Mode A

- Runs baseline on 3 instances
- Validates near-zero optimality gap


### Mode B

- Runs full LLM pipeline
- Ensures objective matches baseline within tolerance
- Verifies feasibility

Expected output:
```bash
[E2E] Mode A — OK
[E2E] Mode B — OK
ALL E2E TESTS PASSED
```

You may customize behavior:
```bash
E2E_TOL=1e-6
E2E_MODE_B_INSTANCE=cap71.txt
```

## Demo Screenshots

Demonstration screenshots can be found here:  [`assets/screenshots/`](assets/screenshots/)


## Troubleshooting

- “No instances found”
  - Ensure dataset exists in: [`../data/raw/`](../data/raw/)

- Mode B fails due to 429 (rate limit)
  - The system automatically retries.
  - If it persists, wait briefly and rerun.

- Missing API Key
  - Set GROQ_API_KEY before running Mode B or the E2E test