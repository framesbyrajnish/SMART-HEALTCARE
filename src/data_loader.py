import pandas as pd
from pathlib import Path

def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

def load_pima(path='data/pima.csv'):
    return load_csv(path)

def load_heart(path='data/heart.csv'):
    return load_csv(path)
