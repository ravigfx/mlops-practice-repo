"""
04 - ML Fundamentals: Model Evaluation
Practice: cross-validation, ROC-AUC, calibration, learning curves
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve,
    StratifiedKFold, validation_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. Cross-Validation Strategies
# ══════════════════════════════════════════════════════════════

def cross_validation_demo():
    print("\n── Cross-Validation Strategies ─────────────────────────")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    clf = Pipeline([("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])

    strategies = {
        "5-Fold CV":           cross_val_score(clf, X, y, cv=5),
        "10-Fold CV":          cross_val_score(clf, X, y, cv=10),
        "Stratified 5-Fold":   cross_val_score(clf, X, y,
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)),
    }

    for name, scores in strategies.items():
        print(f"  {name:<22} {scores.mean():.4f} ± {scores.std():.4f}")


# ══════════════════════════════════════════════════════════════
# 2. ROC & Precision-Recall Curves (multi-class)
# ══════════════════════════════════════════════════════════════

def plot_roc_curves():
    print("\n── ROC Curves (One-vs-Rest) ────────────────────────────")
    iris = load_iris(as_frame=True)
    X, y = iris.data.values, iris.target.values
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = 3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42
    )

    clf = Pipeline([("scaler", StandardScaler()),
                    ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))])
    clf.fit(X_train, y_train.argmax(axis=1))
    y_score = clf.predict_proba(X_test)

    colors = ["#e6194b", "#3cb44b", "#4363d8"]
    class_names = iris.target_names

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC
    for i, (color, name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves (One-vs-Rest)")
    axes[0].legend(loc="lower right")

    # Precision-Recall
    for i, (color, name) in enumerate(zip(colors, class_names)):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        ap = average_precision_score(y_test[:, i], y_score[:, i])
        axes[1].plot(recall, precision, color=color, label=f"{name} (AP={ap:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_pr_curves.png", dpi=120)
    plt.close()
    print(f"ROC/PR plot saved: {OUTPUT_DIR}/roc_pr_curves.png")


# ══════════════════════════════════════════════════════════════
# 3. Learning Curves — Bias-Variance
# ══════════════════════════════════════════════════════════════

def plot_learning_curves():
    print("\n── Learning Curves (Bias-Variance Tradeoff) ────────────")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    models = {
        "High Bias (Depth=1)":     RandomForestClassifier(max_depth=1, n_estimators=50),
        "Good Fit (Depth=5)":      RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42),
        "High Variance (Depth=∞)": RandomForestClassifier(max_depth=None, n_estimators=100),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, model) in zip(axes, models.items()):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X, y, cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color="#e6194b")
        ax.fill_between(train_sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color="#3cb44b")
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#e6194b", label="Train")
        ax.plot(train_sizes, val_scores.mean(axis=1),   "o-", color="#3cb44b", label="Val")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Training samples"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0.5, 1.05); ax.legend()

    fig.suptitle("Learning Curves — Bias vs Variance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=120)
    plt.close()
    print(f"Learning curves saved: {OUTPUT_DIR}/learning_curves.png")


# ══════════════════════════════════════════════════════════════
# 4. Model Calibration
# ══════════════════════════════════════════════════════════════

def plot_calibration():
    print("\n── Model Calibration ───────────────────────────────────")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest (uncalibrated)": RandomForestClassifier(n_estimators=100, random_state=42),
        "RF + Platt Scaling":           CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42), cv=5, method="sigmoid"
        ),
        "RF + Isotonic":                CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42), cv=5, method="isotonic"
        ),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, model in models.items():
        model.fit(X_train, y_train)
        prob_pos = model.predict_proba(X_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10
        )
        brier = brier_score_loss(y_test, prob_pos)
        ax.plot(mean_predicted_value, fraction_of_positives, "o-",
                label=f"{name} (Brier={brier:.3f})")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration.png", dpi=120)
    plt.close()
    print(f"Calibration plot saved: {OUTPUT_DIR}/calibration.png")


if __name__ == "__main__":
    cross_validation_demo()
    plot_roc_curves()
    plot_learning_curves()
    plot_calibration()
    print(f"\n✅ All evaluation exercises complete! → {OUTPUT_DIR.resolve()}")
