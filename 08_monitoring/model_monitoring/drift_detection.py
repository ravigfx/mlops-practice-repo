"""
08 - Model Monitoring & Data Drift Detection
Practice: statistical drift tests, Evidently reports, alerting
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from pathlib import Path
import json
from datetime import datetime


# ══════════════════════════════════════════════════════════════
# 1. Statistical Drift Detection (no extra libraries)
# ══════════════════════════════════════════════════════════════

class DriftDetector:
    """Detect distribution drift using KS test and PSI."""

    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray,
                threshold: float = 0.05) -> dict:
        """Kolmogorov–Smirnov test for distribution shift."""
        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            "test": "KS",
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(p_value < threshold),
            "threshold": threshold,
        }

    @staticmethod
    def psi(reference: np.ndarray, current: np.ndarray,
            buckets: int = 10) -> dict:
        """Population Stability Index — PSI < 0.1 = stable."""
        breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current,   bins=breakpoints)[0]

        ref_pct = np.clip(ref_counts / len(reference), 1e-4, None)
        cur_pct = np.clip(cur_counts / len(current),   1e-4, None)

        psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        if psi_value < 0.1:
            severity = "none"
        elif psi_value < 0.2:
            severity = "slight"
        else:
            severity = "significant"

        return {
            "test": "PSI",
            "psi": round(psi_value, 4),
            "severity": severity,
            "drift_detected": psi_value >= 0.2,
        }

    def check_all_features(self, ref_df: pd.DataFrame,
                           cur_df: pd.DataFrame) -> dict:
        """Run drift checks on all numeric features."""
        results = {}
        for col in ref_df.select_dtypes(include=np.number).columns:
            results[col] = {
                "ks":  self.ks_test(ref_df[col].values, cur_df[col].values),
                "psi": self.psi(ref_df[col].values, cur_df[col].values),
            }
        return results


# ══════════════════════════════════════════════════════════════
# 2. Model Performance Monitoring
# ══════════════════════════════════════════════════════════════

class ModelMonitor:
    """Track prediction quality over time windows."""

    def __init__(self, alert_threshold: float = 0.85):
        self.alert_threshold = alert_threshold
        self.history: list[dict] = []

    def log_window(self, y_true: np.ndarray, y_pred: np.ndarray,
                   window_id: str = None) -> dict:
        from sklearn.metrics import accuracy_score, f1_score

        window_id = window_id or datetime.now().strftime("%Y%m%d_%H%M")
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="weighted")

        entry = {
            "window_id": window_id,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(y_true),
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "alert": acc < self.alert_threshold,
        }
        self.history.append(entry)
        if entry["alert"]:
            self._trigger_alert(entry)
        return entry

    def _trigger_alert(self, entry: dict):
        print(f"🚨 ALERT [{entry['window_id']}]: "
              f"Accuracy dropped to {entry['accuracy']} "
              f"(threshold={self.alert_threshold})")
        # In production: send to PagerDuty / Slack / email

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save_report(self, path: str = "metrics/monitoring_report.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Report saved to {path}")


# ══════════════════════════════════════════════════════════════
# 3. Evidently AI Report (requires: pip install evidently)
# ══════════════════════════════════════════════════════════════

def generate_evidently_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame,
                               target_col: str = "target"):
    """Generate HTML drift report using Evidently."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, ClassificationPreset

        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)

        out_path = "metrics/evidently_report.html"
        Path("metrics").mkdir(exist_ok=True)
        report.save_html(out_path)
        print(f"✅ Evidently report saved: {out_path}")
        return out_path
    except ImportError:
        print("Install evidently: pip install evidently")


# ══════════════════════════════════════════════════════════════
# 4. Demo Run
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Simulate reference data (training distribution)
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # ── Drift Detection ──────────────────────────────────────
    # Simulate drifted data by adding noise
    rng = np.random.RandomState(99)
    X_drifted = X_test.copy()
    X_drifted["sepal length (cm)"] += rng.normal(0, 1.5, size=len(X_test))

    detector = DriftDetector()
    drift_results = detector.check_all_features(X_train, X_drifted)
    for feature, result in drift_results.items():
        ks = result["ks"]
        psi = result["psi"]
        flag = "🚨" if ks["drift_detected"] else "✅"
        print(f"{flag} {feature}: KS p={ks['p_value']} | PSI={psi['psi']} ({psi['severity']})")

    # ── Performance Monitoring ───────────────────────────────
    monitor = ModelMonitor(alert_threshold=0.90)

    # Simulate multiple windows
    for i, noise in enumerate([0, 0.5, 1.5, 3.0]):
        X_noisy = X_test.copy()
        X_noisy.iloc[:, 0] += rng.normal(0, noise, size=len(X_test))
        y_pred = model.predict(X_noisy)
        entry = monitor.log_window(y_test.values, y_pred, window_id=f"window_{i+1}")
        print(f"Window {i+1}: accuracy={entry['accuracy']}")

    monitor.save_report()
    print("\n", monitor.summary().to_string())
