import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.config import RAW_DATA_PATH

def generate_synthetic_data(n_samples=10000):
    """
    Generates a synthetic fraud dataset similar to the Kaggle Credit Card Fraud dataset.
    """
    print(f"Generating synthetic dataset with {n_samples} samples...")
    
    # 28 PCA components + Amount + Time
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=25,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.98, 0.02], # Imbalanced classes
        flip_y=0.01,
        random_state=42
    )
    
    columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    df = pd.DataFrame(X, columns=columns)
    df["Class"] = y
    
    # Adjust Amount to be positive and realistic
    df["Amount"] = np.abs(df["Amount"] * 100)
    
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset saved to {RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()
