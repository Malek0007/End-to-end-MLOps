import mlflow.sklearn
import joblib

MODEL_NAME = "us_accident_model"
MODEL_ALIAS = "champion"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
print(f"Loading model from MLflow URI: {model_uri}")

# Load model from MLflow
sk_model = mlflow.sklearn.load_model(model_uri)
print("Loaded model:", type(sk_model))

# Save as .pkl
joblib.dump(sk_model, "models/us_accident_model.pkl")

print("Saved model to models/us_accident_model.pkl")