import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE, TEST_SIZE, SCALER_SAVE_PATH, PROCESSED_DATA_PATH
from src.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        # We will repurpose SCALER_SAVE_PATH to save the TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def preprocess(self, df):
        """
        Combines Subject and Body, applies TF-IDF, and splits data.
        """
        logger.info("Starting NLP preprocessing pipeline.")
        
        # 1. Handle missing values
        df = df.fillna("")
        
        # 2. Combine text
        logger.info("Combining Subject and Body into single text feature.")
        df['text'] = df['Subject'] + " " + df['Body']
        
        # 3. Split data before vectorization to avoid data leakage
        X_raw = df['text']
        y = df['Class']
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # 4. Apply TF-IDF Vectorization
        logger.info("Applying TF-IDF Vectorizer.")
        X_train_vec = self.vectorizer.fit_transform(X_train_raw)
        X_test_vec = self.vectorizer.transform(X_test_raw)
        
        # Save vectorizer for inference
        joblib.dump(self.vectorizer, SCALER_SAVE_PATH)
        logger.info(f"TF-IDF Vectorizer saved to {SCALER_SAVE_PATH}")

        # 5. Handle Class Imbalance with SMOTE (ONLY on training data)
        logger.info("Applying SMOTE to handle class imbalance on training data.")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)
        
        logger.info(f"Original training shape: {X_train_vec.shape}")
        logger.info(f"Resampled training shape: {X_train_res.shape}")

        processed_data = {
            "X_train": X_train_res,
            "X_test": X_test_vec,
            "y_train": y_train_res,
            "y_test": y_test
        }
        
        joblib.dump(processed_data, PROCESSED_DATA_PATH)
        logger.info(f"Processed data saved to {PROCESSED_DATA_PATH}")
        
        return processed_data
