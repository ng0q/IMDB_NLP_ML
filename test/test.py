from app.schemas import PredictionRequest, PredictionResponse
from src.preprocess import clean_reviews
import pandas as pd


def test_prediction_request_schema():
    request = PredictionRequest(review="This movie was great")
    assert request.review == "This movie was great"


def test_prediction_response_schema():
    response = PredictionResponse(
        predicted_sentiment="Positive",
        predicted_proba_sentiment="0.93",
    )
    assert response.predicted_sentiment == "Positive"
    assert response.predicted_proba_sentiment == "0.93"


def test_clean_reviews_strips_html_tags():
    df = pd.DataFrame({"review": ["hello <b>world</b>!"]})
    cleaned = clean_reviews(df)
    assert cleaned["review"].iloc[0] == "hello world"
