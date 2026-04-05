import pandas as pd
from pathlib import Path


def load_data(input_path: str) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(path)
    print(f"Loaded data from {input_path}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = load_data("data/raw/us_accident.csv")
    print(df.head())