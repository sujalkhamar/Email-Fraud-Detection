import pytest
import joblib
import os
from src.inference import InferencePipeline
from src.config import MODEL_SAVE_PATH, SCALER_SAVE_PATH

def test_inference_pipeline_load():
    # This test might fail if models aren't trained yet, 
    # but we are testing the logic.
    pipeline = InferencePipeline()
    # If assets exist, they should be loaded
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
        assert pipeline.model is not None
        assert pipeline.scaler is not None

def test_prediction_logic():
    # Mock prediction if assets don't exist, or skip
    if not os.path.exists(MODEL_SAVE_PATH):
        pytest.skip("Model not trained yet.")
    
    pipeline = InferencePipeline()
    # 30 features
    dummy_input = [0.0] * 30
    dummy_input[28] = 100.0 # Amount
    
    result = pipeline.predict(dummy_input)
    assert "fraud_probability" in result
    assert "prediction" in result
    assert result["prediction"] in [0, 1]
