# 📋 Section READMEs Content

# This script generates README.md files for each section
sections = {
    "01_programming_fundamentals": {
        "title": "Programming Fundamentals",
        "topics": ["Python OOP, dataclasses, type hints", "Bash scripting for automation", "Virtual environments & packaging", "pytest & unit testing"],
        "practice": "Run `python python_basics/python_for_mlops.py` to see all examples",
    },
    "02_version_control": {
        "title": "Version Control",
        "topics": ["Git branching strategies (Gitflow, trunk-based)", "DVC for data & model versioning", "DVC pipelines & experiments", "Remote storage (S3, GCS)"],
        "practice": "Read `dvc_data_versioning/dvc_workflow.sh` and follow the steps",
    },
    "03_cloud_computing": {
        "title": "Cloud & Containerization",
        "topics": ["Docker multi-stage builds", "Kubernetes Deployments, Services, HPA", "Cloud ML services (SageMaker, Vertex AI, AzureML)", "Infrastructure as Code (Terraform)"],
        "practice": "Build the Docker image: `docker build -f docker/Dockerfile -t iris:dev .`",
    },
    "04_ml_fundamentals": {
        "title": "Machine Learning Fundamentals",
        "topics": ["Supervised learning (classification, regression)", "Unsupervised learning (clustering, PCA)", "Model evaluation metrics", "Feature engineering & selection"],
        "practice": "Explore notebooks in each subfolder",
    },
    "05_data_engineering": {
        "title": "Data Engineering",
        "topics": ["Apache Airflow DAGs", "Apache Spark for batch processing", "Apache Kafka for streaming", "Data Lakes & Warehouses (Delta Lake, Snowflake)"],
        "practice": "Set up Airflow locally: `pip install apache-airflow && airflow standalone`",
    },
    "06_mlops_principles": {
        "title": "MLOps Principles & Tools",
        "topics": ["CI/CD with GitHub Actions", "MLflow experiment tracking & model registry", "Feature stores (Feast)", "Data lineage"],
        "practice": "Start MLflow UI: `mlflow ui --port 5000` then run `experiment_tracking/mlflow_tracking.py`",
    },
    "07_model_serving": {
        "title": "Model Serving",
        "topics": ["FastAPI REST endpoints", "Flask serving", "Batch vs real-time inference", "Input validation with Pydantic"],
        "practice": "Start server: `uvicorn fastapi/serve:app --reload` → http://localhost:8000/docs",
    },
    "08_monitoring": {
        "title": "Monitoring & Observability",
        "topics": ["Data drift detection (KS test, PSI)", "Model performance monitoring", "Evidently AI reports", "Alerting & dashboards"],
        "practice": "Run `python model_monitoring/drift_detection.py` to see drift simulation",
    },
    "09_end_to_end_projects": {
        "title": "End-to-End Projects",
        "topics": ["Iris classification pipeline (beginner)", "NLP sentiment pipeline (intermediate)", "Computer vision pipeline (advanced)"],
        "practice": "Run `python project1_iris_pipeline/pipeline.py` for the full pipeline",
    },
}

import os, json
from pathlib import Path

for folder, info in sections.items():
    content = f"""# {info['title']}

## 🎯 What You'll Learn
{chr(10).join(f"- {t}" for t in info['topics'])}

## 🛠️ Practice
{info['practice']}

## 📚 Resources
See the main [README](../README.md) for resource links.
"""
    readme_path = Path(f"/home/claude/mlops-practice-repo/{folder}/README.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(content)
    print(f"Created: {readme_path}")
