"""
01 - Python Fundamentals for MLOps
Practice: virtual envs, data structures, OOP, file I/O, testing
"""

# ── 1. Data Structures ──────────────────────────────────────────────────────
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Typed config object — common pattern in MLOps projects."""
    name: str
    version: str
    hyperparams: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "hyperparams": self.hyperparams,
            "tags": self.tags,
            "description": self.description,
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)


# ── 2. File I/O & Path Operations ───────────────────────────────────────────
def setup_project_dirs(base: str = "outputs") -> dict:
    """Create standard MLOps output directories."""
    dirs = {
        "models": Path(base) / "models",
        "data":   Path(base) / "data",
        "logs":   Path(base) / "logs",
        "plots":  Path(base) / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


# ── 3. List / Dict Comprehensions ───────────────────────────────────────────
def filter_high_accuracy_runs(runs: list[dict], threshold: float = 0.90) -> list[dict]:
    """Filter experiment runs by accuracy — common data wrangling task."""
    return [r for r in runs if r.get("accuracy", 0) >= threshold]


def extract_metric_trend(runs: list[dict], metric: str) -> list[float]:
    return [r[metric] for r in runs if metric in r]


# ── 4. Generators for large datasets ────────────────────────────────────────
def batch_generator(data: list, batch_size: int = 32):
    """Yield batches — memory-efficient processing for large datasets."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


# ── 5. Error Handling ────────────────────────────────────────────────────────
class ModelLoadError(Exception):
    pass


def safe_load_model(path: str):
    """Gracefully handle model loading failures."""
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ModelLoadError(f"Model file not found: {path}")
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}") from e


# ── PRACTICE EXERCISES ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Exercise 1: Create and save a config
    cfg = ModelConfig(
        name="iris_classifier",
        version="1.0.0",
        hyperparams={"n_estimators": 100, "max_depth": 5},
        tags=["classification", "sklearn"],
    )
    print(cfg.to_dict())

    # Exercise 2: Filter runs
    runs = [
        {"run_id": "a1", "accuracy": 0.91, "f1": 0.90},
        {"run_id": "b2", "accuracy": 0.85, "f1": 0.83},
        {"run_id": "c3", "accuracy": 0.95, "f1": 0.94},
    ]
    good_runs = filter_high_accuracy_runs(runs, threshold=0.90)
    print("High accuracy runs:", good_runs)

    # Exercise 3: Batch generator
    data = list(range(100))
    for batch in batch_generator(data, batch_size=10):
        pass  # process batch
    print("Batching complete")

    # Exercise 4: Create project dirs
    dirs = setup_project_dirs("practice_output")
    print("Dirs created:", dirs)
