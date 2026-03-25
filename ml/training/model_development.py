# %% Imports
from ml.training.mlflow_tracking import setup_mlflow

from datetime import datetime, timezone
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from lightgbm import LGBMClassifier

from ml.preprocessing.preprocessor import Preprocessor
from ml.schema.data_definition import TARGET_COLUMN, FEATURE_COLUMNS

from sqlalchemy import create_engine
import os
import tempfile

# %% MLflow setup
setup_mlflow("churn_model")
run_name = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

def load_data_from_db(table_name="raw_data", data_start=None, data_end=None):
    DB_URL = os.getenv("CONN_STRING")
    engine = create_engine(DB_URL)

    query = f"""
    SELECT *
    FROM {table_name}
    WHERE created_at BETWEEN '{data_start}' AND '{data_end}'
    """

    return pd.read_sql(query, engine)

# %% Load data
def main():
    table_name = "raw_data"
    data_start = datetime.fromisoformat("2026-03-24 20:18:59.244813")
    data_end = datetime.fromisoformat("2026-03-24 20:19:03.898733")
    test_size = 0.2
    random_state = 42

    df = load_data_from_db(table_name, data_start, data_end)

    target = TARGET_COLUMN

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[target].astype(int)


    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[target].astype(int)

    # ----------------------------------------------------------------------------------------

    with mlflow.start_run(run_name=run_name):

        # %% Build pipeline
        preprocessor = Preprocessor()

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            random_state=random_state,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # %% Train
        pipeline.fit(X_train, y_train)

        # ------------------------------------------------------------------------------------

        # %% Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_auc": roc_auc_score(y_test, y_proba),
            "test_logloss": log_loss(y_test, y_proba),
        }

        mlflow.log_metrics(metrics)

        # ------------------------------------------------------------------------------------

        # %% Log model parameters
        mlflow.log_params(model.get_params())

        # ------------------------------------------------------------------------------------
        # Log dataset used for train
        mlflow.log_params({
        "table_name": table_name,
        "data_start": data_start,
        "data_end": data_end,
        "test_size": test_size,
        "random_state": random_state
            })
        

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "data.parquet")
            df.to_parquet(path)
            mlflow.log_artifact(path, artifact_path="dataset_snapshot")

        # ------------------------------------------------------------------------------------

        # %% Log full pipeline
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )

        # ------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")