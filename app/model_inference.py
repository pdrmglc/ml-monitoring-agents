import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.dialects.postgresql import insert
from ml.schema.data_definition import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, BIN_COLUMNS, ID_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN


from evidently import DataDefinition, BinaryClassification, Dataset, Report
from evidently.presets import DataDriftPreset, ClassificationPreset
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
        classification=[BinaryClassification(
            target=TARGET_COLUMN,
            prediction_labels="prediction",
            # prediction_probas="predict_proba"
        )],
        numerical_columns=NUMERICAL_COLUMNS, # + ["predict_proba"],
        categorical_columns=CATEGORICAL_COLUMNS + BIN_COLUMNS + [TARGET_COLUMN,"prediction"]
    )

def build_drift_report(X_ref, X_cur):
    schema = get_evidently_schema()

    dataset_ref = Dataset.from_pandas(X_ref, data_definition=schema)
    dataset_cur = Dataset.from_pandas(X_cur, data_definition=schema)

    report = Report(
    metrics=[
        DataDriftPreset(),
        ClassificationPreset()])

    return report.run(dataset_cur, dataset_ref)

def load_ref_and_current():
    model, run_id, model_version = load_model_and_metadata()

    df = load_full_dataset(MODEL_NAME, model_version)

    df_ref = df[df["used_in_training"] == 1].copy()
    df_cur = df[df["used_in_training"] == 0].copy()

    return model, df_ref, df_cur

def run_drift():
    model, df_ref, df_cur = load_ref_and_current()

    # remove linhas sem predição (join incompleto)
    df_ref = df_ref.dropna(subset=["prediction", "predict_proba"])
    df_cur = df_cur.dropna(subset=["prediction", "predict_proba"])

    # só onde tem churn para métricas de modelo
    df_cur_valid = df_cur[df_cur[TARGET_COLUMN].notna()]

    if df_cur_valid.empty:
        return {"error": f"No current data with {TARGET_COLUMN} available"}

    X_ref = build_full_evidently_dataset(df_ref, model)
    X_cur = build_full_evidently_dataset(df_cur_valid, model)

    report = build_drift_report(X_ref, X_cur)

    return report.dict()


def run_drift_html():
    model, df_ref, df_cur = load_ref_and_current()

    df_ref = df_ref.dropna(subset=["prediction", "predict_proba"])
    df_cur = df_cur.dropna(subset=["prediction", "predict_proba"])

    df_cur_valid = df_cur[df_cur[TARGET_COLUMN].notna()]

    if df_cur_valid.empty:
        return f"<h1>No data with {TARGET_COLUMN} available</h1>"

    X_ref = build_full_evidently_dataset(df_ref, model)
    X_cur = build_full_evidently_dataset(df_cur_valid, model)

    report = build_drift_report(X_ref, X_cur)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        report.save_html(tmp.name)

        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()

    return html

def build_full_evidently_dataset(df, model):
    preprocessor = model.named_steps["preprocessor"]

    extra_cols = df[[TARGET_COLUMN, "prediction", "predict_proba"]].copy()

    df_features = enforce_schema(df)

    X = pd.DataFrame(
        preprocessor.transform(df_features[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS
    )

    # adiciona outputs
    X[TARGET_COLUMN] = extra_cols[TARGET_COLUMN].values
    X["prediction"] = extra_cols["prediction"].values
    X["predict_proba"] = extra_cols["predict_proba"].values

    return X

def load_full_dataset(model_name, model_version):
    engine = create_connected_engine()

    query = f"""
    SELECT 
        r.*,
        p.predict_proba,
        p.prediction,
        p.used_in_training
    FROM raw_data r
    JOIN predictions_log p
        ON r.{ID_COLUMN} = p.id
    WHERE p.model = '{model_name}'
    AND p.model_version = '{model_version}'
    """

    return pd.read_sql(
        query,
        engine
    )

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