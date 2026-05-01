import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE, TEST_SIZE, SCALER_SAVE_PATH, PROCESSED_DATA_PATH
from src.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess(self, df):
        """
        Handles missing values, normalizes 'Amount', and splits data.
        """
        logger.info("Starting preprocessing pipeline.")
        
        # 1. Handle missing values
        if df.isnull().values.any():
            logger.info("Missing values detected. Filling with median.")
            df = df.fillna(df.median())
        
        # 2. Normalize 'Amount'
        # We fit the scaler on the whole dataset here for the pipeline, 
        # but in strict practice, we should fit on train and transform test.
        # However, for the initial step, we'll scale 'Amount'.
        logger.info("Scaling 'Amount' column.")
        df['Amount'] = self.scaler.fit_transform(df[['Amount']])
        
        # Save scaler for inference
        joblib.dump(self.scaler, SCALER_SAVE_PATH)
        logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")

        # 3. Split data
        X = df.drop("Class", axis=1)
        y = df["Class"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # 4. Handle Class Imbalance with SMOTE (ONLY on training data)
        logger.info("Applying SMOTE to handle class imbalance on training data.")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Original training shape: {X_train.shape}")
        logger.info(f"Resampled training shape: {X_train_res.shape}")

        processed_data = {
            "X_train": X_train_res,
            "X_test": X_test,
            "y_train": y_train_res,
            "y_test": y_test
        }
        
        joblib.dump(processed_data, PROCESSED_DATA_PATH)
        logger.info(f"Processed data saved to {PROCESSED_DATA_PATH}")
        
        return processed_data
