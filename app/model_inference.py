import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.dialects.postgresql import insert
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

def create_connected_engine():
    DB_URL = os.getenv("CONN_STRING")
    engine = create_engine(DB_URL)

    return engine

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

    prediction = model.predict_proba(df)[:, 1]

    return prediction, MODEL_NAME, MODEL_VERSION, RUN_ID

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

def get_training_ids(run_id):
    df_ref = load_reference_dataset(run_id)
    return set(df_ref[ID_COLUMN].values)

def load_data_from_db(
    table_name="raw_data",
    data_start=None,
    data_end=None,
    inclusive_start=True
):
    engine = create_connected_engine()

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

def prepare_datasets(df_ref, df_cur, model):
    if df_cur.empty:
        return None, None

    preprocessor = model.named_steps["preprocessor"]

    # enforce schema
    df_ref = enforce_schema(df_ref)
    df_cur = enforce_schema(df_cur)

    X_ref = pd.DataFrame(
        preprocessor.transform(df_ref[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS
    )

    X_cur = pd.DataFrame(
        preprocessor.transform(df_cur[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS
    )

    return X_ref, X_cur

def build_drift_report(X_ref, X_cur):
    schema = get_evidently_schema()

    dataset_ref = Dataset.from_pandas(X_ref, data_definition=schema)
    dataset_cur = Dataset.from_pandas(X_cur, data_definition=schema)

    report = Report([DataDriftPreset()])
    return report.run(dataset_ref, dataset_cur)

def load_ref_and_current():
    model, run_id, _ = load_model_and_metadata()

    df_ref = load_reference_dataset(run_id)

    data_start = df_ref["created_at"].max()
    data_end = datetime.now(timezone.utc)

    df_cur = load_data_from_db(
        table_name="raw_data",
        data_start=data_start,
        data_end=data_end,
        inclusive_start=False
    )

    return model, df_ref, df_cur

def run_drift():
    model, df_ref, df_cur = load_ref_and_current()

    if df_cur.empty:
        return {"error": "No current data available"}

    X_ref, X_cur = prepare_datasets(df_ref, df_cur, model)

    report = build_drift_report(X_ref, X_cur)

    return report.dict()

def run_drift_html():
    model, df_ref, df_cur = load_ref_and_current()

    if df_cur.empty:
        return "<h1>No data</h1>"

    X_ref, X_cur = prepare_datasets(df_ref, df_cur, model)

    report = build_drift_report(X_ref, X_cur)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        report.save_html(tmp.name)

        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()

    return html

# ========================
# Save to PostgreSQL
# ========================

def save_prediction(df: pd.DataFrame):
    if df.empty:
        return

    engine = create_connected_engine()

    metadata = MetaData()
    predictions_log_table = Table(
        "predictions_log",
        metadata,
        autoload_with=engine
    )

    df["updated_at"] = datetime.now(timezone.utc)
    records = df.to_dict(orient="records")

    with engine.begin() as conn:
        stmt = insert(predictions_log_table).values(records)

        stmt = stmt.on_conflict_do_update(
            index_elements=["id", "model", "model_version"],
            set_={
                "predict_proba": stmt.excluded.predict_proba,
                "prediction": stmt.excluded.prediction,
                "threshold": stmt.excluded.threshold,
                "used_in_training": stmt.excluded.used_in_training,
                "updated_at": stmt.excluded.updated_at,
            }
        )

        conn.execute(stmt)