from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import uvicorn
from src.inference import InferencePipeline
from src.logger import setup_logger

logger = setup_logger(__name__)
app = FastAPI(title="Fraud Detection API")

# Initialize pipeline
pipeline = InferencePipeline()

class PredictionRequest(BaseModel):
    features: List[float]

    @validator('features')
    def check_feature_length(cls, v):
        if len(v) != 30:
            raise ValueError('Features list must contain exactly 30 values.')
        return v

class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts fraud for a single transaction.
    """
    logger.info("Received prediction request.")
    try:
        result = pipeline.predict(request.features)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
