# %% Imports
from ml.training.mlflow_tracking import setup_mlflow, sync_mlflow_db

from datetime import datetime
import json
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from lightgbm import LGBMClassifier

from ml.preprocessing.preprocessor import Preprocessor

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER_PATH = PROJECT_ROOT / "data"
DATA_PATH = DATA_FOLDER_PATH / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = DATA_FOLDER_PATH / "output"

OUTPUT_PATH.mkdir(exist_ok=True)

# %% MLflow setup
setup_mlflow("churn_model")
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# %% Load data
def main():
    df = pd.read_csv(DATA_PATH)

    target = "Churn"
    id_col = "customerID"

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42
    )

    X_train = df_train.drop(columns=[target, id_col])
    y_train = df_train[target].map({"Yes": 1, "No": 0})

    X_test = df_test.drop(columns=[target, id_col])
    y_test = df_test[target].map({"Yes": 1, "No": 0})

    # %% Identify categorical features
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # ----------------------------------------------------------------------------------------

    with mlflow.start_run(run_name=run_name):

        # %% Build pipeline
        preprocessor = Preprocessor(categorical_features)

        model = LGBMClassifier(
            n_estimators=5,
            learning_rate=0.01,
            random_state=42,
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

        # %% Log feature distributions (baseline for drift)

        distributions = {}

        for col in X_train.columns:

            if X_train[col].dtype == "object":

                distributions[col] = (
                    X_train[col]
                    .value_counts(normalize=True)
                    .to_dict()
                )

            else:

                distributions[col] = {
                    "mean": float(X_train[col].mean()),
                    "std": float(X_train[col].std()),
                    "min": float(X_train[col].min()),
                    "max": float(X_train[col].max()),
                }

        with open(f"{OUTPUT_PATH}/feature_distributions.json", "w") as f:
            json.dump(distributions, f, indent=2)

        mlflow.log_artifact(f"{OUTPUT_PATH}/feature_distributions.json")

        # ------------------------------------------------------------------------------------

        # %% Log full pipeline
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )

        # ------------------------------------------------------------------------------------

        # %% Save test set for later evaluation / monitoring

        df_test.to_csv(f"{OUTPUT_PATH}/df_test.csv", index=False)
        mlflow.log_artifact(f"{OUTPUT_PATH}/df_test.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
    try:
        sync_mlflow_db()
    except Exception as e:
        print(f"Erro durante a sincronização: {e}")