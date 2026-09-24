"""Train TF-IDF + LogisticRegression pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocess import preprocess_dataframe, train_test_split_data

# Best params from MLflow LogReg Imbalanced run + class_weight=balanced
# Training uses balanced data (no make_disbalance).
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True
TFIDF_NGRAM_RANGE = (1, 1)
TFIDF_MAX_FEATURES = 40000

LOGREG_C = 0.5
LOGREG_MAX_ITER = 1000
LOGREG_CLASS_WEIGHT = "balanced"
LOGREG_RANDOM_STATE = 42

DEFAULT_MODEL_DIR = Path("model")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "model.pkl"
DEFAULT_VECTORIZER_PATH = DEFAULT_MODEL_DIR / "vectorizer.pkl"


def build_pipeline() -> Pipeline:
    """Build TF-IDF + LogReg pipeline with fixed best params."""
    vectorizer = TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
    )
    model = LogisticRegression(
        C=LOGREG_C,
        max_iter=LOGREG_MAX_ITER,
        class_weight=LOGREG_CLASS_WEIGHT,
        random_state=LOGREG_RANDOM_STATE,
    )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("logreg", model),
        ]
    )


def evaluate(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Compute classification metrics on the test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }


def save_artifacts(
    pipeline: Pipeline,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
) -> None:
    """Save fitted model and vectorizer as separate pkl files."""
    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline.named_steps["logreg"], model_path)
    joblib.dump(pipeline.named_steps["tfidf"], vectorizer_path)


def train(
    df: pd.DataFrame | None = None,
    path: str | None = None,
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path | None = DEFAULT_VECTORIZER_PATH,
) -> tuple[Pipeline, dict[str, float]]:
    """Preprocess, split, train LogReg on balanced data, optionally save."""
    if df is None:
        df = preprocess_dataframe(path=path)

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)

    if model_path is not None and vectorizer_path is not None:
        save_artifacts(pipeline, model_path=model_path, vectorizer_path=vectorizer_path)

    return pipeline, metrics


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a saved LogReg model from disk."""
    return joblib.load(model_path)


def load_vectorizer(vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH) -> Any:
    """Load a saved TF-IDF vectorizer from disk."""
    return joblib.load(vectorizer_path)
