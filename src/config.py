import os
from pathlib import Path

# Project Roots
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data Paths
RAW_DATA_PATH = DATA_DIR / "emails_sample.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.joblib"

# Model Paths
MODEL_SAVE_PATH = MODELS_DIR / "best_model.joblib"
SCALER_SAVE_PATH = MODELS_DIR / "scaler.joblib"

# Training Config
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Logging Config
LOG_FILE = LOGS_DIR / "app.log"
