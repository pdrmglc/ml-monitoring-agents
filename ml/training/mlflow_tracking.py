import os
import shutil
import subprocess
import time
from dotenv import load_dotenv
import mlflow

# ----------------------------------------------------------------------------------------

load_dotenv()

MLFLOW_TRACKING_URI = "http://localhost:5000"

MLFLOW_DB_PATH = "./mlflow_server/mlflow.db"
SNAPSHOT_PATH = "./mlflow_server/mlflow_snapshot.db"
GCS_PATH = os.getenv("MLFLOW_DB_GCS_PATH")

# ----------------------------------------------------------------------------------------

def setup_mlflow(experiment_name: str | None = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

# ----------------------------------------------------------------------------------------

def sync_mlflow_db():
    print("Sincronizando MLflow DB...")

    # pequena espera pra garantir flush do SQLite
    time.sleep(2)

    shutil.copy(MLFLOW_DB_PATH, SNAPSHOT_PATH)

    subprocess.run(
        ["gcloud", "storage", "cp", SNAPSHOT_PATH, GCS_PATH],
        check=False
    )