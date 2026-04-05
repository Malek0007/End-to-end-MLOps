import pandas as pd
from pathlib import Path


FEATURE_COLUMNS = [
    'Start_Lat',
    'Start_Lng',
    'Distance(mi)',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Severity'
]


def process_data(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    df = df[FEATURE_COLUMNS]

    df = df.head(30000)

    # df = df.dropna()
    df = df.drop_duplicates()

    df.to_csv(output_file, index=False)

    print(f"Processed data saved to {output_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    process_data(
        input_path="data/raw/us_accident.csv",
        output_path="data/processed/us_accident_clean.csv"
    )