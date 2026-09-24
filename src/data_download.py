"""Download IMDB dataset from Kaggle."""

from __future__ import annotations

import kagglehub

DATASET_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
CSV_NAME = "IMDB Dataset.csv"


def download_dataset() -> str:
    """Download dataset via kagglehub and return the folder path."""
    return kagglehub.dataset_download(DATASET_SLUG)


def get_csv_path(path: str | None = None) -> str:
    """Return full CSV path. Downloads the dataset if path is not given."""
    if path is None:
        path = download_dataset()
    return f"{path}/{CSV_NAME}"
