echo "Iniciando MLflow..."

mlflow server \
  --backend-store-uri $CONN_STRING_MLFLOW \
  --default-artifact-root $MLFLOW_GCS_ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000