import os

import mlflow
from mlflow.tracking import MlflowClient
from ml.training.mlflow_tracking import setup_mlflow

# ----------------------------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME")
setup_mlflow()
# ----------------------------------------------------------------------------------------

def register_model(run_id: str):

    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"

    print(f"Registrando modelo do run {run_id}...")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Versão criada: {result.version}")

    # Promove para Production
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=result.version
    )

    print("Modelo promovido para Production.")

# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    run_id = input("Digite o RUN_ID: ").strip()
    try:
        register_model(run_id)
    except Exception as e:
        print(f"Erro durante o registro do modelo: {e}")