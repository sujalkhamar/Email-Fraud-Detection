import pytest
import pandas as pd
import numpy as np
import os
from src.data_loader import DataLoader
from src.config import RAW_DATA_PATH

def test_data_loader_file_not_found():
    loader = DataLoader("invalid_path.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_data()

def test_data_loader_schema_validation():
    # Create a dummy df with missing columns
    df = pd.DataFrame({"V1": [1, 2], "Amount": [10.0, 20.0]})
    df.to_csv("test_invalid.csv", index=False)
    
    loader = DataLoader("test_invalid.csv")
    with pytest.raises(ValueError):
        loader.load_data()
    
    os.remove("test_invalid.csv")

def test_synthetic_data_generation():
    # This ensures our generator works
    from src.generate_data import generate_synthetic_data
    generate_synthetic_data(n_samples=100)
    assert os.path.exists(RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) == 100
    assert "Class" in df.columns
    assert "Amount" in df.columns
