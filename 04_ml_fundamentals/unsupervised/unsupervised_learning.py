"""
04 - ML Fundamentals: Unsupervised Learning
Practice: clustering, PCA, anomaly detection
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest

OUTPUT_DIR = Path("outputs/unsupervised")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. PCA — Dimensionality Reduction
# ══════════════════════════════════════════════════════════════

def run_pca():
    print("\n── PCA: Dimensionality Reduction ───────────────────────")
    iris = load_iris(as_frame=True)
    X, y = iris.data.values, iris.target.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#e6194b", "#3cb44b", "#4363d8"]

    # Scree plot
    pca_full = PCA().fit(X_scaled)
    axes[0].plot(np.cumsum(pca_full.explained_variance_ratio_), "o-", color="#4C72B0")
    axes[0].axhline(0.95, linestyle="--", color="gray", label="95% threshold")
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("Scree Plot")
    axes[0].legend()

    # 2D projection
    for label, color, name in zip([0, 1, 2], colors, iris.target_names):
        mask = y == label
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, label=name, alpha=0.8, edgecolors="k", linewidths=0.3)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Iris — PCA 2D Projection")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_analysis.png", dpi=120)
    plt.close()
    print(f"PCA plot saved: {OUTPUT_DIR}/pca_analysis.png")
    return X_pca, y


# ══════════════════════════════════════════════════════════════
# 2. K-Means Clustering + Elbow Method
# ══════════════════════════════════════════════════════════════

def run_kmeans():
    print("\n── K-Means Clustering ──────────────────────────────────")
    iris = load_iris(as_frame=True)
    X = StandardScaler().fit_transform(iris.data.values)
    y_true = iris.target.values

    # Elbow method
    inertias, silhouettes = [], []
    K_range = range(2, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(K_range, inertias, "o-", color="#e6194b")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method"); axes[0].axvline(3, linestyle="--", alpha=0.5)
    axes[1].plot(K_range, silhouettes, "o-", color="#3cb44b")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Scores")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kmeans_elbow.png", dpi=120)
    plt.close()

    # Final model (k=3)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X, labels)
    print(f"K=3 → Silhouette={sil:.3f}  ARI={ari:.3f}")
    return labels


# ══════════════════════════════════════════════════════════════
# 3. DBSCAN — Density-Based Clustering
# ══════════════════════════════════════════════════════════════

def run_dbscan():
    print("\n── DBSCAN Clustering ───────────────────────────────────")
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)

    print(f"Clusters found: {n_clusters}  |  Noise points: {n_noise}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", alpha=0.7)
    axes[0].set_title("True Labels")
    axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.7)
    axes[1].set_title(f"DBSCAN (eps=0.5, min_samples=5)\n{n_clusters} clusters, {n_noise} noise pts")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dbscan_clusters.png", dpi=120)
    plt.close()
    print(f"DBSCAN plot saved: {OUTPUT_DIR}/dbscan_clusters.png")


# ══════════════════════════════════════════════════════════════
# 4. Anomaly Detection — Isolation Forest
# ══════════════════════════════════════════════════════════════

def run_anomaly_detection():
    print("\n── Anomaly Detection: Isolation Forest ─────────────────")
    rng = np.random.RandomState(42)
    X_normal = rng.randn(300, 2)
    X_anomaly = rng.uniform(-5, 5, (20, 2))
    X = np.vstack([X_normal, X_anomaly])

    iso = IsolationForest(contamination=0.06, random_state=42)
    preds = iso.fit_predict(X)   # -1 = anomaly, 1 = normal

    n_anomalies = (preds == -1).sum()
    print(f"Anomalies detected: {n_anomalies}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[preds == 1, 0],  X[preds == 1, 1],
               c="#3cb44b", label="Normal", alpha=0.6, s=30)
    ax.scatter(X[preds == -1, 0], X[preds == -1, 1],
               c="#e6194b", label="Anomaly", s=80, edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_title("Isolation Forest — Anomaly Detection")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "anomaly_detection.png", dpi=120)
    plt.close()
    print(f"Anomaly plot saved: {OUTPUT_DIR}/anomaly_detection.png")


if __name__ == "__main__":
    run_pca()
    run_kmeans()
    run_dbscan()
    run_anomaly_detection()
    print(f"\n✅ All unsupervised exercises complete! → {OUTPUT_DIR.resolve()}")
