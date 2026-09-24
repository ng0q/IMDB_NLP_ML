from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app import model_loader
from app.schemas import PredictionRequest, PredictionResponse
from src.preprocess import clean_reviews


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_artifacts()
    yield


app = FastAPI(
    title="IMDB Sentiment Prediction API",
    description="Predict sentiment of a movie review",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Service"])
def health():
    if model_loader.model is None or model_loader.vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if model_loader.model is None or model_loader.vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame({"review": [request.review]})
    input_df = clean_reviews(input_df)
    features = model_loader.vectorizer.transform(input_df["review"])

    y_pred = model_loader.model.predict(features)[0]
    y_proba = model_loader.model.predict_proba(features)[0]
    confidence = float(y_proba[int(y_pred)])

    return PredictionResponse(
        predicted_sentiment='Positive' if y_pred==1 else 'Negative',
        predicted_proba_sentiment=f"{confidence:.2f}",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
