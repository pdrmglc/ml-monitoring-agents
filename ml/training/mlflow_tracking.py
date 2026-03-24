import os
from dotenv import load_dotenv
import mlflow

# ----------------------------------------------------------------------------------------

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# ----------------------------------------------------------------------------------------

def setup_mlflow(experiment_name: str | None = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

# ----------------------------------------------------------------------------------------
