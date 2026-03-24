import os
from datetime import datetime, timezone

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from google.cloud import storage


# ========================
# Config
# ========================

ML_GCS_PREDICTIONS_ROOT = os.getenv("ML_GCS_PREDICTIONS_ROOT")
BUCKET_NAME = ML_GCS_PREDICTIONS_ROOT.replace("gs://", "")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = "production"


# ========================
# MLflow setup
# ========================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ========================
# Load model + metadata
# ========================

def load_model_and_metadata():
    client = MlflowClient()

    # usa alias (mais moderno que stage)
    mv = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    return model, mv.run_id, mv.version


model, RUN_ID, MODEL_VERSION = load_model_and_metadata()


# ========================
# Prediction
# ========================

def process_predict(data):

    df = pd.DataFrame(data)

    prediction = model.predict_proba(df)[:, 1].tolist()

    return prediction


# ========================
# Save to GCS
# ========================

def save_prediction(df: pd.DataFrame):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    filename = f"predictions_{timestamp_str}.parquet"

    path = (
        f"{MODEL_NAME}/"
        f"v{MODEL_VERSION}/"
        f"{date_str}/"
        f"{filename}"
    )

    blob = bucket.blob(path)

    blob.upload_from_string(
        df.to_parquet(index=False),
        content_type="application/octet-stream"
    )