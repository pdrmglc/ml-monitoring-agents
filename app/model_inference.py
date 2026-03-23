import pandas as pd
import mlflow.sklearn
import mlflow
import os
from datetime import datetime, timezone
from google.cloud import storage

ML_GCS_PREDICTIONS_ROOT = os.getenv("ML_GCS_PREDICTIONS_ROOT")
BUCKET_NAME = ML_GCS_PREDICTIONS_ROOT.replace("gs://", "")

MODEL_URI = os.getenv("MODEL_URI")
RUN_ID = MODEL_URI.split("/")[-3]  # Extrai o RUN_ID do MODEL_URI

model = mlflow.sklearn.load_model(MODEL_URI)


def process_predict(data):

    df = pd.DataFrame(data)

    prediction = model.predict_proba(df)[:, 1].tolist()

    return prediction


def save_prediction(df: pd.DataFrame):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    filename = f"predictions_{timestamp_str}_{RUN_ID}.parquet"
    path = f"{RUN_ID}/{date_str}/{filename}"

    blob = bucket.blob(path)

    blob.upload_from_string(
        df.to_parquet(index=False),
        content_type="application/octet-stream"
    )