"""
05 - Airflow DAG: ML Pipeline Orchestration
Practice: DAG structure, task dependencies, sensors, XComs
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule
import json


# ── Default args ──────────────────────────────────────────────
default_args = {
    "owner":            "mlops-team",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": True,
    "email":            ["mlops@yourcompany.com"],
}

ACCURACY_THRESHOLD = 0.90


# ── Task Functions ────────────────────────────────────────────
def ingest_data(**context):
    """Pull latest data from data warehouse."""
    import pandas as pd
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    data_path = "/tmp/iris_raw.csv"
    iris.data.assign(target=iris.target).to_csv(data_path, index=False)
    context["ti"].xcom_push(key="data_path", value=data_path)
    print(f"✅ Data ingested: {data_path}")


def validate_data(**context):
    """Run data quality checks."""
    import pandas as pd
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="ingest_data")
    df = pd.read_csv(data_path)

    assert df.isnull().sum().sum() == 0, "Nulls detected!"
    assert len(df) > 0, "Empty dataset!"
    print(f"✅ Validation passed: {len(df)} rows")


def preprocess(**context):
    """Feature engineering and train/test split."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="ingest_data")
    df = pd.read_csv(data_path)

    X, y = df.drop("target", axis=1), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv("/tmp/X_train.csv", index=False)
    X_test.to_csv("/tmp/X_test.csv",   index=False)
    y_train.to_csv("/tmp/y_train.csv", index=False, header=True)
    y_test.to_csv("/tmp/y_test.csv",   index=False, header=True)
    print("✅ Preprocessing done")


def train_model(**context):
    """Train model and push run_id to XCom."""
    import pandas as pd, pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = pd.read_csv("/tmp/X_train.csv")
    y_train = pd.read_csv("/tmp/y_train.csv").squeeze()

    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipeline.fit(X_train, y_train)

    with open("/tmp/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    context["ti"].xcom_push(key="model_path", value="/tmp/model.pkl")
    print("✅ Model trained")


def evaluate_model(**context):
    """Evaluate and push accuracy to XCom."""
    import pandas as pd, pickle, json
    from sklearn.metrics import accuracy_score

    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv("/tmp/X_test.csv")
    y_test = pd.read_csv("/tmp/y_test.csv").squeeze()
    acc = accuracy_score(y_test, model.predict(X_test))

    metrics = {"accuracy": round(float(acc), 4)}
    with open("/tmp/metrics.json", "w") as f:
        json.dump(metrics, f)

    context["ti"].xcom_push(key="accuracy", value=acc)
    print(f"✅ Accuracy: {acc}")


def quality_gate(**context):
    """Branch: deploy if accuracy passes threshold."""
    acc = context["ti"].xcom_pull(key="accuracy", task_ids="evaluate_model")
    if acc >= ACCURACY_THRESHOLD:
        return "deploy_model"
    return "retrain_alert"


def deploy_model(**context):
    print("🚀 Deploying model to production API...")
    # In real life: kubectl rollout / MLflow Registry transition


def send_retrain_alert(**context):
    acc = context["ti"].xcom_pull(key="accuracy", task_ids="evaluate_model")
    print(f"🚨 Model accuracy {acc} below threshold {ACCURACY_THRESHOLD}. Alert sent!")


# ── DAG Definition ────────────────────────────────────────────
with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="End-to-end ML training DAG",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "training", "iris"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ONE_SUCCESS)

    ingest    = PythonOperator(task_id="ingest_data",    python_callable=ingest_data)
    validate  = PythonOperator(task_id="validate_data",  python_callable=validate_data)
    preproc   = PythonOperator(task_id="preprocess",     python_callable=preprocess)
    train     = PythonOperator(task_id="train_model",    python_callable=train_model)
    evaluate  = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    gate      = BranchPythonOperator(task_id="quality_gate", python_callable=quality_gate)
    deploy    = PythonOperator(task_id="deploy_model",   python_callable=deploy_model)
    alert     = PythonOperator(task_id="retrain_alert",  python_callable=send_retrain_alert)

    # ── Task Dependencies ─────────────────────────────────────
    start >> ingest >> validate >> preproc >> train >> evaluate >> gate
    gate  >> [deploy, alert]
    [deploy, alert] >> end
