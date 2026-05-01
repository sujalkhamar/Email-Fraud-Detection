import joblib
import pandas as pd
import numpy as np
import shap
from src.config import MODEL_SAVE_PATH, SCALER_SAVE_PATH, PROCESSED_DATA_PATH
from src.logger import setup_logger

logger = setup_logger(__name__)

class InferencePipeline:
    def __init__(self):
        try:
            logger.info("Loading model and scaler for inference.")
            self.model = joblib.load(MODEL_SAVE_PATH)
            self.scaler = joblib.load(SCALER_SAVE_PATH)
            
            # Load background data for SHAP (sample of training data)
            logger.info("Loading background data for SHAP explainer.")
            data = joblib.load(PROCESSED_DATA_PATH)
            self.background_data = data["X_train"].iloc[:100] # Use 100 samples for speed
            
            # Initialize explainer
            # If XGBoost/RandomForest, TreeExplainer is best. Otherwise KernelExplainer.
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP TreeExplainer initialized.")
            except:
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.background_data)
                logger.info("SHAP KernelExplainer initialized.")
                
            logger.info("Inference assets and SHAP explainer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading inference assets: {str(e)}")
            self.model = None
            self.scaler = None
            self.explainer = None

    def predict(self, raw_features: list):
        """
        Takes raw features, applies scaling, and returns prediction with SHAP explanation.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or Scaler not loaded.")

        # Convert to DataFrame
        columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        df = pd.DataFrame([raw_features], columns=columns)
        
        # Scale Amount (assuming index -2 corresponds to Amount in raw_features)
        # However, the df construction above uses the list directly.
        # Let's ensure the scaler matches the expected input.
        df["Amount"] = self.scaler.transform(df[["Amount"]])
        
        # Ensure column order matches training (X_train columns)
        # self.background_data.columns contains the correct order
        df = df[self.background_data.columns]
        
        # Predict
        prob = self.model.predict_proba(df)[0][1]
        prediction = int(self.model.predict(df)[0])
        
        # SHAP Explanation
        explanation = {}
        if self.explainer:
            shap_values = self.explainer.shap_values(df)
            # TreeExplainer returns a list for multiclass, index 1 is positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Map features to their SHAP values
            explanation = dict(zip(df.columns, shap_values[0].tolist()))
        
        logger.info(f"Prediction made: {prediction} (Prob: {prob:.4f})")
        return {
            "fraud_probability": float(prob),
            "prediction": prediction,
            "explanation": explanation
        }
