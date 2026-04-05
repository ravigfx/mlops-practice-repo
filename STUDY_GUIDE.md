# 🗺️ MLOps Roadmap — Study Guide & Progress Tracker

> Based on [roadmap.sh/mlops](https://roadmap.sh/mlops)

---

## How to Use This Repo

1. Work through each section **in order** — earlier sections are prerequisites for later ones
2. Run every script, read every comment, understand every line
3. Check off topics as you complete them
4. Build the end-to-end projects to consolidate learning

---

## 📅 Suggested Study Plan (12 Weeks)

| Week | Focus | Files to Study |
|------|-------|---------------|
| 1    | Python Fundamentals | `01_programming_fundamentals/python_basics/` |
| 2    | Bash + Git Workflows | `01_programming_fundamentals/bash_scripts/`, `02_version_control/git_workflows/` |
| 3    | DVC + Data Versioning | `02_version_control/dvc_data_versioning/` |
| 4    | Docker + Kubernetes | `03_cloud_computing/` |
| 5    | ML Fundamentals | `04_ml_fundamentals/supervised/`, `04_ml_fundamentals/model_evaluation/` |
| 6    | Unsupervised + Anomaly Detection | `04_ml_fundamentals/unsupervised/` |
| 7    | Airflow + Spark | `05_data_engineering/airflow_pipelines/`, `05_data_engineering/spark_basics/` |
| 8    | Kafka + Feature Stores | `05_data_engineering/kafka_basics/`, `06_mlops_principles/feature_stores/` |
| 9    | MLflow + CI/CD | `06_mlops_principles/experiment_tracking/`, `06_mlops_principles/ci_cd/` |
| 10   | Model Serving (FastAPI + Go) | `07_model_serving/`, `01_programming_fundamentals/go_basics/` |
| 11   | Monitoring + Drift Detection | `08_monitoring/` |
| 12   | End-to-End Projects | `09_end_to_end_projects/` |

---

## ✅ Topic Checklist

### Phase 1 — Foundations
- [ ] Python: OOP, dataclasses, type hints, generators
- [ ] Python: virtual environments, packaging (`pip`, `pyproject.toml`)
- [ ] Python: `pytest` unit testing and fixtures
- [ ] Bash: scripting, piping, `set -euo pipefail`
- [ ] Go: HTTP server, JSON, goroutines (bonus)
- [ ] Git: branching strategies (Gitflow, trunk-based)
- [ ] Git: conventional commits
- [ ] Git: pre-commit hooks

### Phase 2 — Data & Versioning
- [ ] DVC: `dvc init`, `dvc add`, `dvc push`, `dvc pull`
- [ ] DVC: `dvc.yaml` pipelines with stages and params
- [ ] DVC: experiment comparison with `dvc exp`
- [ ] Remote storage: S3/GCS with DVC

### Phase 3 — Cloud & Containers
- [ ] Docker: write a multi-stage `Dockerfile`
- [ ] Docker: `docker build`, `docker run`, `docker-compose`
- [ ] Kubernetes: Deployments, Services, ConfigMaps, Secrets
- [ ] Kubernetes: HorizontalPodAutoscaler (HPA)
- [ ] Cloud: understand AWS SageMaker / GCP Vertex AI / Azure ML concepts

### Phase 4 — Machine Learning
- [ ] Supervised: classification (RF, LR, SVM, GBT)
- [ ] Supervised: regression (Ridge, GBR)
- [ ] Model evaluation: CV, ROC-AUC, Precision-Recall
- [ ] Bias-variance tradeoff and learning curves
- [ ] Model calibration (Platt scaling, Isotonic)
- [ ] Feature importance and SHAP values
- [ ] Unsupervised: K-Means, DBSCAN, PCA
- [ ] Anomaly detection: Isolation Forest

### Phase 5 — Data Engineering
- [ ] Airflow: write a DAG with dependencies and branching
- [ ] Airflow: XComs, sensors, operators
- [ ] Spark: DataFrames, transformations, actions
- [ ] Spark: Spark ML pipelines (VectorAssembler, StandardScaler)
- [ ] Kafka: producers, consumers, topics, partitions
- [ ] Kafka: streaming ML inference events

### Phase 6 — MLOps Core
- [ ] MLflow: `mlflow.start_run()`, log params, metrics, artifacts
- [ ] MLflow: Model Registry — register, stage, promote
- [ ] MLflow: load model from registry for inference
- [ ] Hyperparameter tuning: Optuna + MLflow integration
- [ ] Feature stores: Feast — define, materialize, fetch
- [ ] CI/CD: GitHub Actions ML pipeline with quality gate
- [ ] CI/CD: automated Docker build & push in CI
- [ ] CI/CD: rolling deployment with `kubectl`

### Phase 7 — Serving
- [ ] FastAPI: POST `/predict`, GET `/health`, batch endpoint
- [ ] FastAPI: Pydantic request/response validation
- [ ] FastAPI: lifespan events (model loading)
- [ ] Go: basic HTTP server for high-performance serving
- [ ] gRPC: understand when to use gRPC vs REST

### Phase 8 — Monitoring
- [ ] Drift detection: KS test, PSI
- [ ] Performance monitoring: accuracy windows, alerting
- [ ] Evidently AI: generate HTML drift reports
- [ ] Prometheus: instrument API with counters, histograms, gauges
- [ ] Grafana: build a monitoring dashboard
- [ ] Alerting: configure thresholds and notifications

### Phase 9 — End-to-End
- [ ] Complete Project 1: Iris classification pipeline
- [ ] Complete Project 2: NLP sentiment pipeline
- [ ] Deploy a model locally with `docker-compose`
- [ ] Start the full local stack: MLflow + Airflow + Kafka + Grafana

---

## 🛠️ Local Stack Quick Start

```bash
# 1. Clone and install
git clone <your-fork>
cd mlops-practice-repo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start infrastructure
docker-compose up -d postgres minio mlflow

# 3. Open MLflow UI
open http://localhost:5000

# 4. Run the Iris end-to-end pipeline
python 09_end_to_end_projects/project1_iris_pipeline/pipeline.py

# 5. Start the model API
uvicorn 07_model_serving.fastapi.serve:app --port 8000 --reload
open http://localhost:8000/docs

# 6. Run tests
pytest tests/ -v
```

---

## 📚 Key Tools Summary

| Area | Tool | Purpose |
|------|------|---------|
| Data versioning | DVC | Track datasets + models like Git |
| Experiment tracking | MLflow | Log params, metrics, artifacts |
| Orchestration | Airflow | Schedule and monitor ML pipelines |
| Batch processing | Spark | Process large datasets |
| Streaming | Kafka | Real-time ML event pipeline |
| Feature store | Feast | Reusable feature management |
| Containerization | Docker | Package model + dependencies |
| Orchestration | Kubernetes | Scale model serving |
| Serving | FastAPI | REST API for predictions |
| Drift detection | Evidently | Monitor data/model drift |
| Metrics | Prometheus | Scrape and store metrics |
| Dashboards | Grafana | Visualize model health |
| CI/CD | GitHub Actions | Automate training → deployment |

---

## 🎯 Portfolio Projects to Build

After completing the exercises, build these for your portfolio:

1. **End-to-end tabular pipeline** — DVC + MLflow + FastAPI + Docker
2. **NLP pipeline** — text preprocessing + TF-IDF/BERT + serving
3. **Real-time drift monitor** — Kafka + Evidently + Grafana alerts
4. **AutoML system** — Optuna HPO + MLflow + automated promotion
5. **Multi-model serving** — model registry + A/B testing + shadow mode
