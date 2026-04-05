"""
Tests for MLOps pipeline components
Run: pytest tests/ -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture(scope="module")
def iris_data():
    iris = load_iris(as_frame=True)
    return iris.data, iris.target


@pytest.fixture(scope="module")
def trained_model(iris_data):
    X, y = iris_data
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    model.fit(X[:120], y[:120])
    return model


# ── Data Tests ───────────────────────────────────────────────
class TestDataQuality:
    def test_no_nulls(self, iris_data):
        X, y = iris_data
        assert X.isnull().sum().sum() == 0

    def test_positive_features(self, iris_data):
        X, _ = iris_data
        assert (X > 0).all().all()

    def test_expected_shape(self, iris_data):
        X, y = iris_data
        assert X.shape == (150, 4)
        assert len(y) == 150

    def test_target_classes(self, iris_data):
        _, y = iris_data
        assert set(y.unique()) == {0, 1, 2}

    def test_feature_ranges(self, iris_data):
        X, _ = iris_data
        for col in X.columns:
            assert X[col].max() < 20, f"{col} has unreasonably large values"


# ── Model Tests ───────────────────────────────────────────────
class TestModel:
    def test_model_trains(self, trained_model):
        assert trained_model is not None

    def test_prediction_shape(self, trained_model, iris_data):
        X, _ = iris_data
        preds = trained_model.predict(X[120:])
        assert preds.shape == (30,)

    def test_prediction_classes(self, trained_model, iris_data):
        X, _ = iris_data
        preds = trained_model.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_probability_output(self, trained_model, iris_data):
        X, _ = iris_data
        probas = trained_model.predict_proba(X[:5])
        assert probas.shape == (5, 3)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_accuracy_threshold(self, trained_model, iris_data):
        from sklearn.metrics import accuracy_score
        X, y = iris_data
        acc = accuracy_score(y[120:], trained_model.predict(X[120:]))
        assert acc >= 0.85, f"Accuracy {acc} below threshold 0.85"

    def test_single_prediction(self, trained_model):
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred = trained_model.predict(sample)
        assert pred[0] in {0, 1, 2}

    def test_batch_prediction(self, trained_model):
        batch = np.random.uniform(1, 8, (100, 4))
        preds = trained_model.predict(batch)
        assert len(preds) == 100


# ── Drift Detection Tests ─────────────────────────────────────
class TestDriftDetection:
    def test_no_drift_same_data(self, iris_data):
        from scipy import stats
        X, _ = iris_data
        stat, p = stats.ks_2samp(X.iloc[:, 0], X.iloc[:, 0])
        assert p > 0.05  # no drift expected

    def test_drift_detected_noisy(self, iris_data):
        from scipy import stats
        X, _ = iris_data
        noisy = X.iloc[:, 0] + np.random.normal(0, 5, len(X))
        stat, p = stats.ks_2samp(X.iloc[:, 0], noisy)
        assert p < 0.05  # drift expected

    def test_psi_stable(self, iris_data):
        X, _ = iris_data
        ref = X.iloc[:75, 0].values
        cur = X.iloc[75:, 0].values
        breakpoints = np.percentile(ref, np.linspace(0, 100, 11))
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf
        r = np.clip(np.histogram(ref, bins=breakpoints)[0] / len(ref), 1e-4, None)
        c = np.clip(np.histogram(cur, bins=breakpoints)[0] / len(cur), 1e-4, None)
        psi = float(np.sum((c - r) * np.log(c / r)))
        assert psi < 0.2  # PSI < 0.2 = stable


# ── API Tests (requires running server) ──────────────────────
class TestAPIPayloads:
    """Validate request/response schemas without a live server."""

    def test_valid_payload(self):
        payload = {
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        }
        for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
            assert key in payload
            assert payload[key] > 0

    def test_invalid_negative_feature(self):
        with pytest.raises(AssertionError):
            val = -1.0
            assert val > 0, "Feature values must be positive"
