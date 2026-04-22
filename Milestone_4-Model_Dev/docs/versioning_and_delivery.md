## Milestone 4 Versioning and Delivery

Milestone 4 uses the repository Git history and a branch-based workflow compatible with GitHub Flow:

1. Create a short-lived branch from the default branch (`main` or `master`)
2. Implement one scoped change set
3. Re-run the Milestone 4 scripts before opening a pull request
4. Commit source code, configs, benchmark metadata, and lightweight reports
5. Open a pull request into the default branch
6. Merge only after the milestone outputs are refreshed

Suggested branch names:

- `feature/m4-symbolic-eval-pipeline`
- `feature/m4-mlflow-model-versioning`
- `feature/m4-zenml-symbolic-workflow`
- `feature/m4-finetune-dataset`

Suggested verification commands before merge:

- `scripts\run_train.py`
- `scripts\run_compare.py`
- `scripts\run_tests.py`
- `scripts\run_zenml.py`

Artifacts that belong in Git:

- source code
- configs
- reference tables and split definitions
- report templates and lightweight metadata
- evaluation summaries needed as milestone evidence

Artifacts better kept out of Git unless explicitly required:

- local tracking databases
- generated model binaries
- ZenML local state
- generated solver scratch outputs
- very large run outputs
