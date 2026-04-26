# End-to-end MLOps — US Accidents

A complete MLOps project for training and serving a US Accidents prediction model.
Includes a DVC pipeline (process → validate → train), a FastAPI backend, a web
frontend, Docker Compose, Kubernetes manifests, and a GitHub Actions workflow.

<img width="545" height="210" alt="image" src="https://github.com/user-attachments/assets/8bc7f659-59f0-4457-b1f5-8f4541525735" />


## Stack

- **Backend**: FastAPI + Uvicorn
- **ML**: scikit-learn 1.6.1, pandas, numpy, joblib
- **Pipeline & versioning**: DVC
- **Experiment tracking**: MLflow
- **Frontend**: HTML (served via Docker on port 3000)
- **Deploy**: Docker Compose + Kubernetes (`k8s/`)

## Prerequisites

- Docker + Docker Compose **OR** Python 3.11
- (Optional) DVC, if you want to re-run the training pipeline

---

## Quick start — Docker

The fastest way to run everything:

```bash
git clone https://github.com/Malek0007/End-to-end-MLOps.git
cd End-to-end-MLOps

docker compose up --build
```

Then open:
- Frontend: http://localhost:3000
- Health check: http://localhost:8000/health

To stop:
```bash
docker compose down
```

> The backend image expects a trained model in `models/`. If the folder is
> empty, run the training pipeline first (see below) or place a `.pkl` file in
> `models/` before building.

---

## Run locally without Docker

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

### 2. (Optional) Re-run the ML pipeline with DVC

```bash
pip install dvc
dvc repro
```

This runs the three stages defined in `dvc.yaml`:
1. `process_data` — cleans `data/raw/us_accident.csv` → `data/processed/us_accident_clean.csv`
2. `validate` — writes `reports/validation.txt`
3. `train_model` — trains and saves `models/us_accident_model.pkl`

You can also run stages individually:
```bash
python src/process_data.py
python src/validate.py
python src/train.py
```

### 3. Start the API

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit http://localhost:8000/docs.

---

## Kubernetes deployment

Manifests are in `k8s/`. Apply them with:

```bash
kubectl apply -f k8s/
```

(Make sure your image is built and pushed to a registry your cluster can reach,
or load it into your local cluster, e.g. `minikube image load`.)

---

## Useful commands

| Action                       | Command                          |
| ---------------------------- | -------------------------------- |
| Build & run all services     | `docker compose up --build`      |
| Stop all services            | `docker compose down`            |
| Re-run the full ML pipeline  | `dvc repro`                      |
| Run only the API locally     | `uvicorn app.app:app --reload`   |

---

## Troubleshooting

- **`Cannot find model`** — make sure `models/us_accident_model.pkl` exists
  before starting the API. Run `dvc repro` or `python src/train.py`.
- **Port already in use** — change the host port in `docker-compose.yml`
  (e.g., `"8001:8000"`).
- **Frontend can't reach the API** — verify the `api` service is healthy:
  `docker compose ps`.
