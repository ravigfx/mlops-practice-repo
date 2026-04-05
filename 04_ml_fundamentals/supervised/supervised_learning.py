"""
04 - ML Fundamentals: Supervised Learning
Practice: classification, regression, cross-validation, pipelines
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import load_iris, load_boston, fetch_california_housing
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, r2_score, mean_absolute_error
)

# Classifiers
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

OUTPUT_DIR = Path("outputs/supervised")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. Classification Benchmark
# ══════════════════════════════════════════════════════════════

def run_classification_benchmark():
    """Compare multiple classifiers on Iris dataset."""
    print("\n── Classification Benchmark ────────────────────────────")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "Logistic Regression":  LogisticRegression(max_iter=1000),
        "Decision Tree":        DecisionTreeClassifier(max_depth=5),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100),
        "SVM":                  SVC(probability=True),
        "KNN":                  KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    for name, clf in classifiers.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append({
            "Model":        name,
            "CV Mean":      round(cv_scores.mean(), 4),
            "CV Std":       round(cv_scores.std(), 4),
            "Test Acc":     round(accuracy_score(y_test, y_pred), 4),
            "F1 Weighted":  round(f1_score(y_test, y_pred, average="weighted"), 4),
        })
        print(f"  {name:<25} CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
              f"Test={accuracy_score(y_test, y_pred):.3f}")

    df = pd.DataFrame(results).sort_values("Test Acc", ascending=False)
    df.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/benchmark_results.csv")
    return df


# ══════════════════════════════════════════════════════════════
# 2. Hyperparameter Tuning with GridSearchCV
# ══════════════════════════════════════════════════════════════

def tune_random_forest():
    print("\n── Hyperparameter Tuning (GridSearchCV) ────────────────")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth":    [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(OUTPUT_DIR / "gridsearch_results.csv", index=False)
    return grid_search.best_estimator_


# ══════════════════════════════════════════════════════════════
# 3. Regression
# ══════════════════════════════════════════════════════════════

def run_regression():
    print("\n── Regression: California Housing ─────────────────────")
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from sklearn.ensemble import GradientBoostingRegressor
    models = {
        "Ridge":    Ridge(alpha=1.0),
        "GBR":      GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        print(f"  {name:<10} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")


# ══════════════════════════════════════════════════════════════
# 4. Confusion Matrix Plot
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = Pipeline([("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Random Forest — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close()
    print(f"Confusion matrix saved: {OUTPUT_DIR}/confusion_matrix.png")


# ══════════════════════════════════════════════════════════════
# 5. Feature Importance
# ══════════════════════════════════════════════════════════════

def plot_feature_importance():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=120)
    plt.close()
    print(f"Feature importance saved: {OUTPUT_DIR}/feature_importance.png")


if __name__ == "__main__":
    df_results = run_classification_benchmark()
    print("\n", df_results.to_string(index=False))

    best_model  = tune_random_forest()
    run_regression()
    plot_confusion_matrix()
    plot_feature_importance()

    print("\n✅ All supervised learning exercises complete!")
    print(f"   Outputs in: {OUTPUT_DIR.resolve()}")
