import pandas as pd
import os
from src.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads data from CSV and validates schema."""
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path)
            self._validate_schema(df)
            logger.info("Data loaded and validated successfully.")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_schema(self, df):
        """Validates that the required columns are present."""
        # Standard columns for the Email Spam dataset
        required_columns = ["Subject", "Body", "Class"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if 'Class' is binary
        if not set(df["Class"].unique()).issubset({0, 1}):
            raise ValueError("Target column 'Class' must contain only 0 and 1.")

        logger.info("Schema validation passed.")
