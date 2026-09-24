"""EDA and preprocessing for the IMDB dataset."""

from __future__ import annotations

from typing import Tuple

import html

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_download import get_csv_path


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the IMDB CSV. Downloads the dataset if path is not given."""
    csv_path = get_csv_path(path)
    return pd.read_csv(csv_path)


def encode_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Encode sentiment: positive -> 1, otherwise -> 0."""
    df = df.copy()
    if df["sentiment"].dtype == object:
        df["sentiment"] = np.where(df["sentiment"] == "positive", 1, 0)
    return df


def add_review_len(df: pd.DataFrame) -> pd.DataFrame:
    """Add review_len column (word count)."""
    df = df.copy()
    df["review_len"] = df["review"].str.split().str.len()
    return df


def max_review_len(df: pd.DataFrame) -> int:
    """Return max review length in words."""
    if "review_len" not in df.columns:
        df = add_review_len(df)
    return int(df["review_len"].max())


def count_reviews_with_digits(df: pd.DataFrame) -> int:
    """Count reviews that still contain digits."""
    return int(df["review"].str.contains(r"\d+").sum())


def drop_review_len(df: pd.DataFrame) -> pd.DataFrame:
    """Drop temporary review_len column if present."""
    df = df.copy()
    if "review_len" in df.columns:
        df = df.drop("review_len", axis=1)
    return df


def run_eda(df: pd.DataFrame) -> dict:
    """Run the same basic EDA checks as in the notebook."""
    df = add_review_len(df)
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "max_review_len": max_review_len(df),
        "reviews_with_digits": count_reviews_with_digits(df),
    }
    return info


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean review text: HTML, tags, punctuation/digits, whitespace, lower."""
    df = df.copy()
    df["review"] = df["review"].map(html.unescape)
    df["review"] = df["review"].str.replace(r"<[^>]+>", "", regex=True)
    df["review"] = df["review"].str.replace(r"[^a-z\s]", " ", regex=True)
    df["review"] = df["review"].str.replace(r"[\s+]", " ", regex=True).str.strip()
    df["review"] = df["review"].str.lower()
    return df


def preprocess_dataframe(df: pd.DataFrame | None = None, path: str | None = None) -> pd.DataFrame:
    """Full preprocessing: load if needed, encode labels, clean text."""
    if df is None:
        df = load_raw_data(path)
    df = encode_sentiment(df)
    df = clean_reviews(df)
    df = drop_review_len(df)
    return df[["review", "sentiment"]]


def train_test_split_data(
    df: pd.DataFrame | None = None,
    path: str | None = None,
    test_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split. Preprocesses if df is not given."""
    if df is None:
        df = preprocess_dataframe(path=path)
    X = df["review"]
    y = df["sentiment"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def make_disbalance(
    X: pd.Series,
    y: pd.Series,
    pos_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """Make a train set imbalanced by positive-class ratio."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_neg = len(neg_idx)
    n_pos_keep = int(n_neg * pos_ratio / (1 - pos_ratio))

    rng = np.random.RandomState(random_state)
    pos_keep = rng.choice(pos_idx, size=n_pos_keep, replace=False)

    keep_idx = np.concatenate([pos_keep, neg_idx.values])
    rng.shuffle(keep_idx)

    return X.loc[keep_idx].reset_index(drop=True), y.loc[keep_idx].reset_index(drop=True)
