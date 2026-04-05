"""
09 - End-to-End Project 2: NLP Sentiment Classification Pipeline
Covers: text preprocessing → TF-IDF/embeddings → training → serving → monitoring
"""

import re
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts/nlp_pipeline")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {0: "negative", 1: "neutral", 2: "positive"}


# ══════════════════════════════════════════════════════════════
# STEP 1 — Synthetic Dataset
# ══════════════════════════════════════════════════════════════

def create_sentiment_dataset(n: int = 500) -> pd.DataFrame:
    """Create a synthetic labelled sentiment dataset."""
    positive = [
        "This model is absolutely fantastic and works perfectly",
        "Amazing performance, exceeded all expectations",
        "Love how easy it is to use, great experience",
        "The predictions are incredibly accurate and fast",
        "Outstanding results, highly recommend this approach",
        "Brilliant solution that solved all our problems",
        "Excellent accuracy with very low latency",
        "Great model, deployment was smooth and efficient",
    ]
    negative = [
        "Terrible accuracy, model keeps failing in production",
        "Very disappointed, the model drifted badly",
        "Awful performance, way too slow for real time use",
        "Complete disaster, the pipeline broke everything",
        "Horrible results, cannot trust these predictions",
        "The deployment was a nightmare from start to finish",
        "Extremely slow inference, completely unacceptable",
        "Poor model quality, needs to be retrained urgently",
    ]
    neutral = [
        "The model ran without errors today",
        "Predictions were processed within expected time",
        "Standard training run completed as scheduled",
        "Model evaluation metrics are within normal range",
        "Deployment finished with no major incidents",
        "Data pipeline executed without any warnings",
        "Model performance is consistent with baseline",
        "Weekly retraining job completed successfully",
    ]

    rng = np.random.RandomState(42)
    texts, labels = [], []
    sources = [(positive, 2), (negative, 0), (neutral, 1)]

    for _ in range(n):
        src, label = sources[rng.randint(0, 3)]
        text = src[rng.randint(0, len(src))]
        # Add small variations
        text = text + " " + rng.choice([".", "!", "...", ""])
        texts.append(text.strip())
        labels.append(label)

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(ARTIFACT_DIR / "sentiment_raw.csv", index=False)
    log.info(f"Dataset created: {df.shape}  label_dist={df.label.value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════
# STEP 2 — Text Preprocessing
# ══════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 2: Text preprocessing...")
    df = df.copy()
    df["text_clean"] = df["text"].apply(preprocess_text)
    df["text_len"]   = df["text_clean"].apply(len)
    df["word_count"] = df["text_clean"].apply(lambda x: len(x.split()))

    # Filter out very short texts
    df = df[df["word_count"] >= 3].reset_index(drop=True)
    log.info(f"After preprocessing: {len(df)} samples")
    df.to_csv(ARTIFACT_DIR / "sentiment_processed.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════
# STEP 3 — Training
# ══════════════════════════════════════════════════════════════

def train_nlp_model(df: pd.DataFrame) -> tuple:
    log.info("Step 3: Training NLP model...")
    X = df["text_clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),         # unigrams + bigrams
            sublinear_tf=True,           # apply log normalization
            min_df=2,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, multi_class="multinomial"
        )),
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    log.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    model_path = ARTIFACT_DIR / "nlp_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    log.info(f"Model saved: {model_path}")

    return pipeline, X_test, y_test


# ══════════════════════════════════════════════════════════════
# STEP 4 — Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_nlp_model(model, X_test, y_test) -> dict:
    log.info("Step 4: Evaluating model...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba, multi_class="ovr"), 4),
        "timestamp": datetime.now().isoformat(),
        "n_test":    len(y_test),
    }

    report = classification_report(
        y_test, y_pred,
        target_names=[LABELS[i] for i in sorted(LABELS)],
        output_dict=True
    )
    metrics["classification_report"] = report

    with open(ARTIFACT_DIR / "nlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + classification_report(
        y_test, y_pred,
        target_names=[LABELS[i] for i in sorted(LABELS)]
    ))
    log.info(f"Accuracy: {metrics['accuracy']}  ROC-AUC: {metrics['roc_auc']}")
    return metrics


# ══════════════════════════════════════════════════════════════
# STEP 5 — Inference Helper
# ══════════════════════════════════════════════════════════════

def predict_sentiment(texts: list[str], model_path: str = None) -> list[dict]:
    """Load model and predict sentiment for new texts."""
    model_path = model_path or str(ARTIFACT_DIR / "nlp_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    clean_texts = [preprocess_text(t) for t in texts]
    preds  = model.predict(clean_texts)
    probas = model.predict_proba(clean_texts)

    results = []
    for text, pred, proba in zip(texts, preds, probas):
        results.append({
            "text":       text[:60] + "..." if len(text) > 60 else text,
            "sentiment":  LABELS[pred],
            "confidence": round(float(max(proba)), 4),
            "scores":     {LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
        })
    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_nlp_pipeline():
    log.info("=" * 60)
    log.info("🚀 Starting NLP Sentiment Pipeline")
    log.info("=" * 60)

    df        = create_sentiment_dataset(n=600)
    df        = preprocess_dataset(df)
    model, X_test, y_test = train_nlp_model(df)
    metrics   = evaluate_nlp_model(model, X_test, y_test)

    # Sample inference
    test_texts = [
        "The model accuracy is amazing, great deployment!",
        "Horrible results, everything keeps crashing",
        "Pipeline ran without major issues today",
    ]
    results = predict_sentiment(test_texts)
    print("\n── Sample Predictions ────────────────────────────────")
    for r in results:
        print(f"  [{r['sentiment'].upper():<10} {r['confidence']:.0%}] {r['text']}")

    log.info("=" * 60)
    log.info(f"✅ NLP Pipeline complete! Artifacts: {ARTIFACT_DIR.resolve()}")
    log.info("=" * 60)
    return metrics


if __name__ == "__main__":
    run_nlp_pipeline()
