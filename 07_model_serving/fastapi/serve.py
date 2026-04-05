"""
07 - Model Serving with FastAPI
Practice: REST API, input validation, async prediction, health checks
"""

import pickle
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn


# ── Schemas ───────────────────────────────────────────────────
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="Sepal length in cm")
    sepal_width:  float = Field(..., gt=0, lt=20, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=20, description="Petal length in cm")
    petal_width:  float = Field(..., gt=0, lt=20, description="Petal width in cm")

    model_config = {"json_schema_extra": {
        "example": {"sepal_length": 5.1, "sepal_width": 3.5,
                    "petal_length": 1.4, "petal_width": 0.2}
    }}


class BatchPredictRequest(BaseModel):
    samples: list[IrisFeatures]

    @field_validator("samples")
    @classmethod
    def check_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 samples")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict[str, float]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


# ── App State ─────────────────────────────────────────────────
class ModelState:
    model = None
    version = "unknown"
    start_time = time.time()
    CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


# ── Lifespan (load model at startup) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    model_path = Path("models/model.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            ModelState.model = pickle.load(f)
        ModelState.version = "1.0.0"
        print(f"✅ Model loaded: {model_path}")
    else:
        # Fallback: train a quick model for demo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        iris = load_iris()
        ModelState.model = RandomForestClassifier(n_estimators=10, random_state=42)
        ModelState.model.fit(iris.data, iris.target)
        ModelState.version = "demo-1.0.0"
        print("⚠️  No saved model found — using demo model")
    yield
    # Shutdown
    print("Shutting down model server...")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Model Server",
    description="Iris classifier REST API — Practice project",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health():
    return HealthResponse(
        status="healthy" if ModelState.model else "degraded",
        model_loaded=ModelState.model is not None,
        model_version=ModelState.version,
        uptime_seconds=round(time.time() - ModelState.start_time, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures):
    if ModelState.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    X = np.array([[
        features.sepal_length, features.sepal_width,
        features.petal_length, features.petal_width,
    ]])

    pred = int(ModelState.model.predict(X)[0])
    proba = ModelState.model.predict_proba(X)[0]
    latency = (time.time() - start) * 1000

    return PredictionResponse(
        prediction=pred,
        class_name=ModelState.CLASS_NAMES[pred],
        probabilities={ModelState.CLASS_NAMES[i]: round(float(p), 4)
                       for i, p in enumerate(proba)},
        model_version=ModelState.version,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(request: BatchPredictRequest):
    if ModelState.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([[
        s.sepal_length, s.sepal_width, s.petal_length, s.petal_width
    ] for s in request.samples])

    preds = ModelState.model.predict(X)
    probas = ModelState.model.predict_proba(X)

    return {
        "predictions": [
            {
                "prediction": int(p),
                "class_name": ModelState.CLASS_NAMES[int(p)],
                "probabilities": {ModelState.CLASS_NAMES[i]: round(float(prob), 4)
                                  for i, prob in enumerate(proba)},
            }
            for p, proba in zip(preds, probas)
        ],
        "count": len(preds),
        "model_version": ModelState.version,
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    return {
        "name": "iris-classifier",
        "version": ModelState.version,
        "classes": ModelState.CLASS_NAMES,
        "framework": "scikit-learn",
        "input_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    }


if __name__ == "__main__":
    # Start: uvicorn serve:app --reload --port 8000
    # Docs:  http://localhost:8000/docs
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
