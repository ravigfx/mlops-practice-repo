"""
06 - MLOps Principles: Feature Stores
Practice: Feast feature store — define, materialize, fetch features
Requirements: pip install feast
"""

# ══════════════════════════════════════════════════════════════
# PART A: feature_repo/feature_store.yaml
# ══════════════════════════════════════════════════════════════

FEATURE_STORE_YAML = """
project: iris_feature_store
provider: local
registry: data/registry.db
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
entity_key_serialization_version: 2
"""

# ══════════════════════════════════════════════════════════════
# PART B: Feature Definitions (features.py)
# ══════════════════════════════════════════════════════════════

FEATURES_PY = '''
from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float32, Int32

# ── Entity ──────────────────────────────────────────────────
sample_entity = Entity(
    name="sample_id",
    description="Unique identifier for each Iris sample",
)

# ── Data Source ─────────────────────────────────────────────
iris_source = FileSource(
    path="data/iris_features.parquet",
    timestamp_field="event_timestamp",
)

# ── Feature View ────────────────────────────────────────────
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[sample_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width",  dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width",  dtype=Float32),
    ],
    online=True,
    source=iris_source,
    tags={"team": "ml", "version": "1"},
)
'''

# ══════════════════════════════════════════════════════════════
# PART C: Python utilities for working with the feature store
# ══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def create_feature_dataset(output_path: str = "data/iris_features.parquet"):
    """Generate a Feast-compatible parquet file with timestamps."""
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df["sample_id"]        = range(len(df))
    df["target"]           = iris.target
    df["event_timestamp"]  = pd.Timestamp.now() - pd.to_timedelta(
        np.random.randint(0, 30, size=len(df)), unit="D"
    )
    df["created"]          = pd.Timestamp.now()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Feature dataset saved: {output_path}  shape={df.shape}")
    return df


def feast_workflow_demo():
    """
    Demonstrates the Feast workflow end-to-end.
    Requires Feast installed + feature store initialized.
    """
    print("""
    ── Feast Feature Store Workflow ──────────────────────────

    1. Initialize repository:
       $ feast init iris_feature_store
       $ cd iris_feature_store

    2. Apply feature definitions:
       $ feast apply

    3. Materialize features (offline → online store):
       $ feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

    4. Fetch historical features (for training):
    """)

    # Example Python code for historical fetch
    HISTORICAL_FETCH = '''
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")

# Training: fetch historical features
entity_df = pd.DataFrame({
    "sample_id": list(range(150)),
    "event_timestamp": [pd.Timestamp.now()] * 150,
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
).to_df()

print("Training features shape:", training_df.shape)
    '''
    print(HISTORICAL_FETCH)

    print("    5. Fetch online features (for inference):\n")
    ONLINE_FETCH = '''
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Inference: fetch online (low latency) features
feature_vector = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=[{"sample_id": 42}],
).to_dict()

print("Online features:", feature_vector)
    '''
    print(ONLINE_FETCH)


def manual_feature_store():
    """
    Lightweight in-memory feature store (no Feast required).
    Demonstrates the concept of a feature store.
    """
    print("\n── Manual Feature Store (in-memory demo) ───────────────")
    from sklearn.datasets import load_iris

    class SimpleFeatureStore:
        def __init__(self):
            self._store: dict[int, dict] = {}
            self._metadata: dict = {}

        def materialize(self, df: pd.DataFrame, entity_col: str):
            """Load features into the online store."""
            for _, row in df.iterrows():
                entity_id = int(row[entity_col])
                self._store[entity_id] = row.drop(entity_col).to_dict()
            self._metadata["last_materialized"] = datetime.utcnow().isoformat()
            print(f"Materialized {len(df)} entities into online store")

        def get_online_features(self, entity_ids: list[int],
                                 feature_names: list[str]) -> pd.DataFrame:
            rows = []
            for eid in entity_ids:
                if eid in self._store:
                    row = {k: self._store[eid].get(k) for k in feature_names}
                    row["entity_id"] = eid
                    rows.append(row)
                else:
                    print(f"⚠️  Entity {eid} not found in store")
            return pd.DataFrame(rows)

    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df["sample_id"] = range(len(df))

    store = SimpleFeatureStore()
    store.materialize(df, entity_col="sample_id")

    features = store.get_online_features(
        entity_ids=[0, 5, 42, 99],
        feature_names=["sepal_length", "petal_length"],
    )
    print("Retrieved features:")
    print(features.to_string(index=False))


if __name__ == "__main__":
    # Write config files
    Path("feature_repo").mkdir(exist_ok=True)
    Path("feature_repo/feature_store.yaml").write_text(FEATURE_STORE_YAML)
    Path("feature_repo/features.py").write_text(FEATURES_PY)
    print("Feature repo files written to feature_repo/")

    create_feature_dataset()
    feast_workflow_demo()
    manual_feature_store()
