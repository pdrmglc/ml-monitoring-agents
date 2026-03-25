import os
from datetime import datetime, timezone

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from google.cloud import storage

from sqlalchemy import create_engine, text
from ml.schema.data_definition import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, BIN_COLUMNS, ID_COLUMN, FEATURE_COLUMNS


from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.presets import DataDriftPreset
from ml.schema.validate_schema import enforce_schema

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
# Carrega snapshot do dataset de treino
# ========================

def load_reference_dataset(run_id):
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="dataset_snapshot"
    )

    path = os.path.join(local_dir, "data.parquet")
    return pd.read_parquet(path)

def load_data_from_db(
    table_name="raw_data",
    data_start=None,
    data_end=None,
    inclusive_start=True
):
    DB_URL = os.getenv("CONN_STRING")
    engine = create_engine(DB_URL)

    start_op = ">=" if inclusive_start else ">"

    query = text(f"""
    SELECT *
    FROM {table_name}
    WHERE created_at {start_op} :start
      AND created_at <= :end
    """)

    return pd.read_sql(
        query,
        engine,
        params={
            "start": data_start,
            "end": data_end
        }
    )

def get_evidently_schema():
    return DataDefinition(
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS+BIN_COLUMNS,
        id_column=ID_COLUMN,
    )

def compute_drift(df_ref, df_cur, model):

    if df_cur.empty:
        return {"error": "No current data available"}

    preprocessor = model.named_steps["preprocessor"]

    # aplica schema antes (importante)
    df_ref = enforce_schema(df_ref)
    df_cur = enforce_schema(df_cur)

    # transforma
    X_ref = pd.DataFrame(
        preprocessor.transform(df_ref[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS
    )

    X_cur = pd.DataFrame(
        preprocessor.transform(df_cur[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS
    )

    # schema do evidently
    schema = get_evidently_schema()

    dataset_ref = Dataset.from_pandas(
        X_ref,
        data_definition=schema
    )

    dataset_cur = Dataset.from_pandas(
        X_cur,
        data_definition=schema
    )

    report = Report([
        DataDriftPreset()
    ])

    result = report.run(dataset_ref, dataset_cur)

    return result.dict()

def run_drift():

    model, run_id, _ = load_model_and_metadata()

    df_ref = load_reference_dataset(run_id)

    data_start = df_ref["created_at"].max() + pd.Timedelta(microseconds=1)
    data_end = datetime.now(timezone.utc)

    df_cur = load_data_from_db(
        table_name="raw_data",
        data_start=data_start,
        data_end=data_end
    )

    return compute_drift(df_ref, df_cur, model)

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