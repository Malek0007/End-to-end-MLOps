import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def train_models(input_path: str, model_path: str) -> None:
    mlflow.set_experiment("us_accident_model_comparison")

    df = pd.read_csv(input_path)

    X = df[
        [
            "Start_Lat",
            "Start_Lng",
            "Distance(mi)",
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)"
        ]
    ]

    y = (df["Severity"] > 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_configs = {
        "logistic_regression": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(random_state=42, max_iter=1000))
            ]),
            "params": {
                "model__C": [0.1, 1.0, 10.0]
            }
        },
        "decision_tree": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(random_state=42))
            ]),
            "params": {
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5]
            }
        },
        "random_forest": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(random_state=42))
            ]),
            "params": {
                "model__n_estimators": [50, 100],
                "model__max_depth": [10, None]
            }
        },
        "adaboost": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", AdaBoostClassifier(random_state=42))
            ]),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.5, 1.0]
            }
        }
    }

    best_model_name = None
    best_estimator = None
    best_f1 = -1.0
    best_run_id = None

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "model_results.txt"

    report_lines = []

    for model_name, config in model_configs.items():
        print(f"\n===== Training {model_name} =====")

        with mlflow.start_run(run_name=model_name) as run:
            grid = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_pipeline = grid.best_estimator_

            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            roc_auc = roc_auc_score(y_test, y_proba)

            print("Best Parameters:", grid.best_params_)
            print(classification_report(y_test, y_pred))
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("input_path", input_path)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            for param_name, param_value in grid.best_params_.items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log the model under a constant artifact folder name
            mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

            report_lines.append(f"Model: {model_name}")
            report_lines.append(f"Best Parameters: {grid.best_params_}")
            report_lines.append(f"Accuracy: {accuracy:.4f}")
            report_lines.append(f"Precision: {precision:.4f}")
            report_lines.append(f"Recall: {recall:.4f}")
            report_lines.append(f"F1-Score: {f1:.4f}")
            report_lines.append(f"ROC-AUC: {roc_auc:.4f}")
            report_lines.append("-" * 60)

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_estimator = best_pipeline
                best_run_id = run.info.run_id

    report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nModel comparison report saved to {report_file}")

    print(f"\nChampion model: {best_model_name}")
    print(f"Best F1-Score: {best_f1:.4f}")

    if best_run_id is not None:
        client = MlflowClient()
        registered_model_name = "us_accident_model"
        model_uri = f"runs:/{best_run_id}/model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

        client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=result.version
        )

        print(f"Model registered as '{registered_model_name}' version {result.version}")
        print("Alias 'champion' assigned")

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, model_file)

    print(f"Champion model saved to {model_file}")


if __name__ == "__main__":
    train_models(
        input_path="data/processed/us_accident_clean.csv",
        model_path="models/us_accident_model.pkl"
    )