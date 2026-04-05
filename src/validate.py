import pandas as pd
from pathlib import Path

DATA = Path("data/processed/us_accident_clean.csv")
OUT = Path("reports/validation.txt")

OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

lines = []
lines.append(f"rows={len(df)}")
lines.append(f"cols={len(df.columns)}")
lines.append(f"severity_min={df['Severity'].min()}")
lines.append(f"severity_max={df['Severity'].max()}")
lines.append(f"distance_min={df['Distance(mi)'].min()}")
lines.append(f"distance_max={df['Distance(mi)'].max()}")
lines.append(f"temperature_min={df['Temperature(F)'].min()}")
lines.append(f"temperature_max={df['Temperature(F)'].max()}")
lines.append(f"humidity_min={df['Humidity(%)'].min()}")
lines.append(f"humidity_max={df['Humidity(%)'].max()}")

OUT.write_text("\n".join(lines) + "\n")
print("Wrote:", OUT)