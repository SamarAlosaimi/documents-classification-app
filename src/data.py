import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_data(filename: str):
    file_path = DATA_DIR / filename
    return pd.read_csv(file_path)
