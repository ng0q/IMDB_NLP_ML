from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    review: str = Field(
        ...,
        description='Text review',
        examples=['This movie so good']
    )
    

class PredictionResponse(BaseModel):
    predicted_sentiement: str = Field(
        ...,
        description='Positive or Negative',
        examples=['Positive']
    )
    predicted_proba_sentiement: str = Field(
        ...,
        description='Model`s Confidence',
        examples=['0.93']
    )