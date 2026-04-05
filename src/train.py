import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_model(input_path: str, model_path: str) -> None:
    df = pd.read_csv(input_path)

    # Features only
    X = df[
    [
        'Start_Lat',
        'Start_Lng',
        'Distance(mi)',
        'Temperature(F)',
        'Humidity(%)',
        'Pressure(in)',
        'Visibility(mi)',
        'Wind_Speed(mph)'
    ]
    ]

    # Target
    y = df['Severity']
    y = (y > 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # create output folder
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    # save model
    joblib.dump(pipeline, model_file)
    print(f"Model saved to {model_file}")


if __name__ == "__main__":
    train_model(
        input_path="data/processed/us_accident_clean.csv",
        model_path="models/us_accident_model.pkl"
    )