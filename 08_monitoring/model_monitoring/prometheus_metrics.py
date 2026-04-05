"""
08 - Monitoring: Instrument FastAPI with Prometheus metrics
Exposes /metrics endpoint for Prometheus scraping
"""

import time
import random
from pathlib import Path

# pip install prometheus-client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from fastapi import FastAPI, Response
import uvicorn


# ── Metric Definitions ────────────────────────────────────────
PREDICTION_COUNTER = None
PREDICTION_LATENCY = None
MODEL_ACCURACY     = None
ACTIVE_REQUESTS    = None
PREDICTION_CONFIDENCE = None

def setup_metrics():
    global PREDICTION_COUNTER, PREDICTION_LATENCY, MODEL_ACCURACY
    global ACTIVE_REQUESTS, PREDICTION_CONFIDENCE

    if not PROMETHEUS_AVAILABLE:
        print("prometheus-client not installed: pip install prometheus-client")
        return

    PREDICTION_COUNTER = Counter(
        "model_predictions_total",
        "Total number of predictions served",
        ["model_version", "predicted_class", "status"],
    )

    PREDICTION_LATENCY = Histogram(
        "model_prediction_latency_seconds",
        "Prediction latency in seconds",
        ["model_version"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    MODEL_ACCURACY = Gauge(
        "model_accuracy",
        "Current model accuracy on validation window",
        ["model_version", "window"],
    )

    ACTIVE_REQUESTS = Gauge(
        "model_active_requests",
        "Number of currently active prediction requests",
    )

    PREDICTION_CONFIDENCE = Histogram(
        "model_prediction_confidence",
        "Distribution of prediction confidence scores",
        ["model_version"],
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    )

    print("✅ Prometheus metrics initialized")


# ── Instrumented FastAPI App ──────────────────────────────────
def create_monitored_app() -> FastAPI:
    setup_metrics()
    app = FastAPI(title="Monitored ML API", version="1.0.0")

    @app.middleware("http")
    async def track_requests(request, call_next):
        if ACTIVE_REQUESTS:
            ACTIVE_REQUESTS.inc()
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        if ACTIVE_REQUESTS:
            ACTIVE_REQUESTS.dec()
        return response

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        if not PROMETHEUS_AVAILABLE:
            return Response("prometheus-client not installed", status_code=503)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/predict")
    async def predict(features: dict):
        """Simulated prediction endpoint with full instrumentation."""
        model_version = "1.0.0"
        start = time.time()

        # Simulate prediction
        time.sleep(random.uniform(0.001, 0.05))
        pred_class   = str(random.randint(0, 2))
        confidence   = random.uniform(0.6, 0.99)
        latency      = time.time() - start

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNTER.labels(
                model_version=model_version,
                predicted_class=pred_class,
                status="success",
            ).inc()

            PREDICTION_LATENCY.labels(model_version=model_version).observe(latency)
            PREDICTION_CONFIDENCE.labels(model_version=model_version).observe(confidence)

            # Simulate periodic accuracy updates
            if random.random() < 0.05:
                MODEL_ACCURACY.labels(
                    model_version=model_version, window="1h"
                ).set(random.uniform(0.88, 0.97))

        return {
            "prediction":  int(pred_class),
            "confidence":  round(confidence, 4),
            "latency_ms":  round(latency * 1000, 2),
        }

    return app


# ── Sample Grafana Dashboard JSON Template ───────────────────
GRAFANA_DASHBOARD = {
    "title": "MLOps Model Monitoring",
    "panels": [
        {
            "title": "Prediction Rate (req/s)",
            "type": "graph",
            "targets": [{"expr": "rate(model_predictions_total[5m])"}],
        },
        {
            "title": "P95 Prediction Latency",
            "type": "graph",
            "targets": [{"expr": "histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))"}],
        },
        {
            "title": "Model Accuracy",
            "type": "stat",
            "targets": [{"expr": "model_accuracy"}],
        },
        {
            "title": "Confidence Distribution",
            "type": "heatmap",
            "targets": [{"expr": "rate(model_prediction_confidence_bucket[10m])"}],
        },
        {
            "title": "Active Requests",
            "type": "gauge",
            "targets": [{"expr": "model_active_requests"}],
        },
    ],
}

import json
Path("infra/grafana/dashboards").mkdir(parents=True, exist_ok=True)
with open("infra/grafana/dashboards/ml-monitoring.json", "w") as f:
    json.dump(GRAFANA_DASHBOARD, f, indent=2)


if __name__ == "__main__":
    app = create_monitored_app()
    print("Start: uvicorn prometheus_metrics:app --port 8001")
    print("Then visit: http://localhost:8001/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8001)
