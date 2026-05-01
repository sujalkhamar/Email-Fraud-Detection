from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.config import RAW_DATA_PATH
from src.generate_data import generate_synthetic_data
from src.logger import setup_logger
import os

logger = setup_logger("TrainingPipeline")

def main():
    # 1. Generate data if not exists
    if not os.path.exists(RAW_DATA_PATH):
        logger.info("Raw data not found. Generating synthetic dataset...")
        generate_synthetic_data(n_samples=20000)

    # 2. Load Data
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()

    # 3. Preprocess
    preprocessor = DataPreprocessor()
    preprocessor.preprocess(df)

    # 4. Train
    trainer = ModelTrainer()
    trainer.train()

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
