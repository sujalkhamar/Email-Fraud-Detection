import joblib
from src.config import MODEL_SAVE_PATH, SCALER_SAVE_PATH
from src.logger import setup_logger

logger = setup_logger(__name__)

class InferencePipeline:
    def __init__(self):
        try:
            logger.info("Loading model and vectorizer for inference.")
            self.model = joblib.load(MODEL_SAVE_PATH)
            # SCALER_SAVE_PATH now holds our TF-IDF vectorizer
            self.vectorizer = joblib.load(SCALER_SAVE_PATH)
            
            logger.info("Inference assets loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading inference assets: {str(e)}")
            self.model = None
            self.vectorizer = None

    def predict(self, subject: str, body: str):
        """
        Takes email subject and body, applies TF-IDF, and returns prediction.
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model or Vectorizer not loaded.")

        # Combine text
        combined_text = subject + " " + body
        
        # Vectorize
        X_vec = self.vectorizer.transform([combined_text])
        
        # Predict
        prediction = int(self.model.predict(X_vec)[0])
        
        # Probability if available
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X_vec)[0][1]
        else:
            # Approximate probability for SVMs using decision function
            # Not a true probability, but gives a score > 0 for class 1
            dec = self.model.decision_function(X_vec)[0]
            prob = 1 / (1 + (2.71828 ** -dec)) # Sigmoid
        
        logger.info(f"Prediction made: {prediction} (Prob: {prob:.4f})")
        return {
            "fraud_probability": float(prob),
            "prediction": prediction,
            "explanation": {"status": "SHAP not supported for TF-IDF in this version"}
        }
