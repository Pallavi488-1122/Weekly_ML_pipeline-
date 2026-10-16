# Weekly ML Model Retraining Pipeline

An end-to-end MLOps demo that automatically retrains a classifier weekly, evaluates the candidate model, and only promotes it when it passes a quality gate. It records success/failure, processed data volume, metrics, and errors.

## Architecture

1. **Schedule:** GitHub Actions starts the workflow every Sunday at 03:00 UTC (08:30 IST). `workflow_dispatch` also enables manual test runs.
2. **Load data:** the demo loads the Breast Cancer Wisconsin dataset bundled with scikit-learn. Replace `load_data()` in `src/pipeline.py` with your API, database, or new CSV source.
3. **Validate:** reject empty datasets, mismatched rows, duplicate columns, non-numeric features, and single-class targets.
4. **Preprocess and train:** median imputation and a Random Forest classifier are combined in a scikit-learn pipeline.
5. **Evaluate:** accuracy, precision, recall, and F1 are calculated on a stratified holdout set.
6. **Quality gate:** by default, candidate F1 must be at least `0.90` and must not be more than `0.01` below the saved model's F1.
7. **Promote safely:** if the candidate passes, the model and metrics are written to temporary files and moved into `artifacts/`. If it fails, the existing model is kept.
8. **Log:** `logs/pipeline.log` stores readable logs and `logs/run_history.jsonl` stores one JSON record per run, including row count, metrics, status, promotion decision, and errors. GitHub Actions also uploads logs/artifacts for 90 days.

## Project structure

```text
weekly_ml_retraining_pipeline/
├── .github/workflows/weekly-retrain.yml
├── artifacts/                 # promoted model + metrics (created on first pass)
├── logs/                      # logs + JSONL run history
├── src/pipeline.py            # end-to-end retraining workflow
├── tests/test_pipeline.py
├── requirements.txt
└── README.md
```

## Run locally

Recommended: Python 3.11 or 3.12.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/pipeline.py
```

### macOS/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/pipeline.py
```

After a successful run, inspect:
- `artifacts/model.joblib`
- `artifacts/current_metrics.json`
- `logs/pipeline.log`
- `logs/run_history.jsonl`

The first run creates a baseline only if it meets `MIN_F1`. Later runs must pass both quality gates.

## Adjust quality thresholds

Environment variables can tune the gate:

**PowerShell**
```powershell
$env:MIN_F1 = "0.90"
$env:MAX_ALLOWED_F1_DROP = "0.01"
python src/pipeline.py
```

**macOS/Linux**
```bash
MIN_F1=0.90 MAX_ALLOWED_F1_DROP=0.01 python src/pipeline.py
```

## Run tests

Install pytest, then run:

```bash
python -m pip install pytest
python -m pytest -q
```

## Configure GitHub Actions

1. Create a GitHub repository and push this project.
2. In repository **Settings → Actions → General**, ensure workflows have permission to write to the repository if you want the workflow to commit promoted artifacts.
3. Ensure the workflow file is at `.github/workflows/weekly-retrain.yml`.
4. GitHub Actions scheduled workflows use UTC. The included schedule is Sunday at 03:00 UTC / 08:30 IST. Scheduled runs can be delayed during high load, and GitHub's schedule runs from the default branch.
5. Open **Actions → Weekly ML Model Retraining → Run workflow** to trigger a manual run for testing/debugging.

The workflow uploads run logs even when the pipeline step fails. A successful pipeline run that does not promote a candidate is not treated as a runtime failure; inspect the log and run-history record for the gate decision.

## Important production notes

- This is a portfolio-ready demo, not a complete production deployment. The sample dataset is static and is not a live weekly feed. Replace `load_data()` with your real source to make weekly retraining meaningful.
- GitHub Actions artifacts are retained for 90 days. For long-term durable logs, send records to object storage, a database, or an experiment tracker such as MLflow.
- Committing binary model artifacts to Git is acceptable for a small demo, but use object storage/model registry for production.
- For a stricter atomic promotion design, store each model under a versioned filename and update a single manifest/pointer only after the model and metrics are validated together.
- Before deployment, use a time-aware or business-representative validation split, monitor data drift, add schema checks, and evaluate fairness/security where relevant.
- Never load untrusted `.joblib` files: deserialization can execute code.

## Troubleshooting

- **Candidate not promoted:** inspect `candidate_metrics` and `gate_reason` in `logs/run_history.jsonl`. The quality gate may have rejected it.
- **Workflow cannot push:** check repository Actions permissions and branch protection rules.
- **No weekly run:** verify the workflow is on the default branch and that scheduled workflows are enabled.
- **Bad data / training exception:** the run record includes the error; the old model is not deliberately overwritten by a failed training/evaluation step.

## License

Use and modify for learning and portfolio projects.
