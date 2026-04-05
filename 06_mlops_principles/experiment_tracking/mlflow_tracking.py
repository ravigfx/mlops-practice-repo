"""
06 - Experiment Tracking with MLflow
Practice: log params, metrics, artifacts, register models
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
import json
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://localhost:5000"   # or "sqlite:///mlflow.db" locally
EXPERIMENT_NAME = "iris-classification"
MODEL_NAME = "iris-classifier"


# ── Data Preparation ──────────────────────────────────────────
def load_and_split():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ── Model Training with MLflow ────────────────────────────────
def train_random_forest(params: dict):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split()

    with mlflow.start_run(run_name="random_forest") as run:
        # ── Log Parameters ──────────────────────────────────
        mlflow.log_params(params)
        mlflow.set_tags({
            "model_type": "random_forest",
            "framework": "sklearn",
            "dataset": "iris",
        })

        # ── Build Pipeline ──────────────────────────────────
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**params, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        # ── Evaluate ────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

        metrics = {
            "accuracy":        accuracy_score(y_test, y_pred),
            "f1_weighted":     f1_score(y_test, y_pred, average="weighted"),
            "precision":       precision_score(y_test, y_pred, average="weighted"),
            "recall":          recall_score(y_test, y_pred, average="weighted"),
            "cv_mean":         cv_scores.mean(),
            "cv_std":          cv_scores.std(),
        }

        # ── Log Metrics ─────────────────────────────────────
        mlflow.log_metrics(metrics)
        print("Metrics:", metrics)

        # ── Log Model ───────────────────────────────────────
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )

        # ── Log Artifacts ───────────────────────────────────
        report = classification_report(y_test, y_pred, output_dict=True)
        Path("metrics").mkdir(exist_ok=True)
        with open("metrics/classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("metrics/classification_report.json")

        print(f"\nRun ID: {run.info.run_id}")
        return run.info.run_id, metrics


# ── Hyperparameter Tuning with Optuna + MLflow ────────────────
def optuna_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth":    trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    _, metrics = train_random_forest(params)
    return metrics["accuracy"]


def hyperparameter_search(n_trials: int = 20):
    study = optuna.create_study(direction="maximize", study_name="iris-rf-tuning")
    study.optimize(optuna_objective, n_trials=n_trials)
    print("\n✅ Best params:", study.best_params)
    print("✅ Best accuracy:", study.best_value)
    return study.best_params


# ── Promote Best Model to Production ─────────────────────────
def promote_model(run_id: str, stage: str = "Staging"):
    """Transition a model version to Staging or Production."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Find model version from run
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        print("No model versions found for this run")
        return

    version = versions[0].version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    print(f"✅ Model v{version} transitioned to {stage}")


# ── Load & Predict from Registry ─────────────────────────────
def predict_from_registry(stage: str = "Production"):
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    iris = load_iris(as_frame=True)
    sample = iris.data.head(5)
    preds = model.predict(sample)
    print("Sample predictions:", preds)
    return preds


if __name__ == "__main__":
    # Basic training run
    params = {"n_estimators": 100, "max_depth": 5}
    run_id, metrics = train_random_forest(params)
    print("\nTraining complete!")
    print("To view results: mlflow ui --port 5000")
