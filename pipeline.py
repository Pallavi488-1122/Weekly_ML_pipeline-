"""
Weekly retraining pipeline demo.

Default data: sklearn Breast Cancer Wisconsin dataset (bundled with scikit-learn).
Replace load_data() with your own source when adapting this project.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
LOG_DIR = ROOT / "logs"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "current_metrics.json"
RUN_LOG_PATH = LOG_DIR / "pipeline.log"
RUN_HISTORY_PATH = LOG_DIR / "run_history.jsonl"

# Promotion gate: candidate F1 must not be worse than current F1 by more than this.
MAX_ALLOWED_F1_DROP = float(os.getenv("MAX_ALLOWED_F1_DROP", "0.01"))
MIN_F1 = float(os.getenv("MIN_F1", "0.90"))
RANDOM_STATE = 42


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(RUN_LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load fresh data. Replace this function with an API, database, or file loader."""
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.copy()
    return X, y


def validate_data(X: pd.DataFrame, y: pd.Series) -> None:
    if X.empty or len(y) == 0:
        raise ValueError("Dataset is empty.")
    if len(X) != len(y):
        raise ValueError(f"Feature/target row mismatch: {len(X)} vs {len(y)}.")
    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature column names detected.")
    if y.nunique() < 2:
        raise ValueError("Training data must contain at least two target classes.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("This demo expects numeric features.")
    if X.isna().all(axis=None):
        raise ValueError("All feature values are missing.")


def build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
    }


def read_current_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Could not read current metrics; refusing to overwrite baseline: %s", exc)
        raise RuntimeError("Current metrics file is unreadable; stopping safely.") from exc


def append_run_record(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def atomic_save(candidate: Pipeline, metrics: dict) -> None:
    """Write temp files first, then replace destinations to avoid partial artifacts."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ARTIFACT_DIR) as temp_dir:
        temp = Path(temp_dir)
        temp_model = temp / "model.joblib"
        temp_metrics = temp / "current_metrics.json"
        joblib.dump(candidate, temp_model)
        temp_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        # Each file is replaced atomically on the same filesystem.
        os.replace(temp_model, MODEL_PATH)
        os.replace(temp_metrics, METRICS_PATH)


def run_pipeline() -> dict:
    started = datetime.now(timezone.utc).isoformat()
    record = {
        "started_at": started,
        "status": "failure",
        "rows_processed": 0,
        "candidate_metrics": None,
        "previous_metrics": None,
        "promoted": False,
        "error": None,
    }
    try:
        logging.info("Pipeline started.")
        X, y = load_data()
        validate_data(X, y)
        record["rows_processed"] = int(len(X))
        logging.info("Loaded and validated %d rows and %d features.", len(X), X.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        candidate = build_model()
        candidate.fit(X_train, y_train)
        candidate_metrics = evaluate(candidate, X_test, y_test)
        record["candidate_metrics"] = candidate_metrics
        logging.info("Candidate metrics: %s", candidate_metrics)

        previous_metrics = read_current_metrics()
        record["previous_metrics"] = previous_metrics

        meets_absolute_gate = candidate_metrics["f1"] >= MIN_F1
        if previous_metrics is None:
            # First successful run bootstraps the baseline if it clears the minimum.
            promote = meets_absolute_gate
            reason = "No previous model; first model must meet MIN_F1."
        else:
            old_f1 = float(previous_metrics["f1"])
            meets_regression_gate = candidate_metrics["f1"] >= old_f1 - MAX_ALLOWED_F1_DROP
            promote = meets_absolute_gate and meets_regression_gate
            reason = (
                f"MIN_F1={MIN_F1}; previous F1={old_f1}; "
                f"maximum allowed drop={MAX_ALLOWED_F1_DROP}."
            )

        if promote:
            atomic_save(candidate, candidate_metrics)
            record["promoted"] = True
            logging.info("Candidate promoted. %s", reason)
        else:
            logging.warning("Candidate NOT promoted. Existing model remains unchanged. %s", reason)

        record["status"] = "success"
        record["gate_reason"] = reason
        return record

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        logging.exception("Pipeline failed; existing production model was not intentionally changed.")
        return record
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            append_run_record(record)
            logging.info("Run record saved: status=%s promoted=%s", record["status"], record["promoted"])
        except Exception:
            logging.exception("Could not persist run history.")


if __name__ == "__main__":
    setup_logging()
    result = run_pipeline()
    print(json.dumps(result, indent=2))
    if result["status"] != "success":
        sys.exit(1)
