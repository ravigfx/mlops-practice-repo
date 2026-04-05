"""
09 - End-to-End MLOps Pipeline: Iris Classification
Covers: data → preprocessing → training → evaluation → serving → monitoring
"""

import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts/iris_pipeline")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STEP 1 — Data Ingestion
# ══════════════════════════════════════════════════════════════
def ingest_data() -> tuple[pd.DataFrame, pd.Series]:
    log.info("Step 1: Ingesting data...")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    # Save raw data artifact
    raw = X.copy()
    raw["target"] = y
    raw.to_csv(ARTIFACT_DIR / "raw_data.csv", index=False)
    log.info(f"Data shape: {X.shape}")
    return X, y


# ══════════════════════════════════════════════════════════════
# STEP 2 — Preprocessing
# ══════════════════════════════════════════════════════════════
def preprocess(X: pd.DataFrame, y: pd.Series) -> tuple:
    log.info("Step 2: Preprocessing...")

    # Data quality checks
    assert X.isnull().sum().sum() == 0, "Missing values detected!"
    assert (X > 0).all().all(), "Non-positive values detected!"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save splits
    train_df = X_train.copy(); train_df["target"] = y_train
    test_df  = X_test.copy();  test_df["target"]  = y_test
    train_df.to_csv(ARTIFACT_DIR / "train.csv", index=False)
    test_df.to_csv(ARTIFACT_DIR  / "test.csv",  index=False)

    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════
# STEP 3 — Training
# ══════════════════════════════════════════════════════════════
def train(X_train: pd.DataFrame, y_train: pd.Series,
          params: dict = None) -> Pipeline:
    log.info("Step 3: Training model...")
    params = params or {"n_estimators": 100, "max_depth": 5, "random_state": 42}

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(**params)),
    ])
    pipeline.fit(X_train, y_train)

    # Save model
    model_path = ARTIFACT_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Save params
    with open(ARTIFACT_DIR / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    log.info(f"Model saved: {model_path}")
    return pipeline


# ══════════════════════════════════════════════════════════════
# STEP 4 — Evaluation
# ══════════════════════════════════════════════════════════════
def evaluate(model: Pipeline, X_test: pd.DataFrame,
             y_test: pd.Series) -> dict:
    log.info("Step 4: Evaluating model...")

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "timestamp": datetime.now().isoformat(),
        "n_test_samples": len(y_test),
    }

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Accuracy: {metrics['accuracy']}")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["setosa", "versicolor", "virginica"]))
    return metrics


# ══════════════════════════════════════════════════════════════
# STEP 5 — Quality Gate
# ══════════════════════════════════════════════════════════════
def quality_gate(metrics: dict, threshold: float = 0.90) -> bool:
    log.info("Step 5: Quality gate check...")
    acc = metrics["accuracy"]
    passed = acc >= threshold
    status = "✅ PASSED" if passed else "❌ FAILED"
    log.info(f"Quality Gate: accuracy={acc} threshold={threshold} → {status}")
    return passed


# ══════════════════════════════════════════════════════════════
# STEP 6 — Package for Serving
# ══════════════════════════════════════════════════════════════
def package_model(metrics: dict) -> dict:
    log.info("Step 6: Packaging model for serving...")
    manifest = {
        "name":      "iris-classifier",
        "version":   "1.0.0",
        "model_path": str(ARTIFACT_DIR / "model.pkl"),
        "metrics":   metrics,
        "framework": "sklearn",
        "classes":   ["setosa", "versicolor", "virginica"],
        "input_schema": {
            "sepal_length": "float",
            "sepal_width":  "float",
            "petal_length": "float",
            "petal_width":  "float",
        },
        "created_at": datetime.now().isoformat(),
    }
    with open(ARTIFACT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("✅ Model packaged")
    return manifest


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(params: dict = None, threshold: float = 0.90):
    log.info("=" * 60)
    log.info("🚀 Starting Iris MLOps Pipeline")
    log.info("=" * 60)

    X, y                          = ingest_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)
    model                         = train(X_train, y_train, params)
    metrics                       = evaluate(model, X_test, y_test)

    if not quality_gate(metrics, threshold):
        log.error("Pipeline halted: quality gate failed!")
        return None

    manifest = package_model(metrics)
    log.info("=" * 60)
    log.info("✅ Pipeline complete! Artifacts saved to:")
    for f in sorted(ARTIFACT_DIR.iterdir()):
        log.info(f"   {f}")
    log.info("=" * 60)
    return manifest


if __name__ == "__main__":
    manifest = run_pipeline(
        params={"n_estimators": 150, "max_depth": 6, "random_state": 42},
        threshold=0.90,
    )
    if manifest:
        print("\nManifest:", json.dumps(manifest, indent=2))
