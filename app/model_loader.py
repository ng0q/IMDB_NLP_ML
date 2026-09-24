import joblib
from pathlib import Path

MODEL_DIR = Path('model')

model = None
vectorizer = None

def load_artifacts() -> None:
    global model, vectorizer

    model_path = MODEL_DIR / 'model.pkl'
    vectorizer_path = MODEL_DIR / 'vectorizer.pkl'

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path.resolve()}. "
            f"Start `python -m src.train` before."
        )
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {vectorizer_path.resolve()}. "
            f"Start `python -m src.train` before."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    print(f"Model loaded from {model_path.resolve()}")
    print(f"Vectorizer loaded from{vectorizer_path.resolve()}")