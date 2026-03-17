#!/bin/bash
set -e

MLFLOW_DIR=./mlflow_server
DB_PATH=$MLFLOW_DIR/mlflow.db

mkdir -p $MLFLOW_DIR

# Mata instâncias anteriores do MLflow
pkill -f "mlflow server" || true

echo "Verificando mlflow.db no GCS..."

if gcloud storage ls $MLFLOW_DB_GCS_PATH > /dev/null 2>&1; then
  echo "Baixando banco existente..."
  gcloud storage cp $MLFLOW_DB_GCS_PATH $DB_PATH
else
  echo "Banco ainda não existe, criando novo..."
  touch $DB_PATH
fi

echo "Iniciando MLflow..."

mlflow server \
  --backend-store-uri sqlite:///$DB_PATH \
  --default-artifact-root $MLFLOW_GCS_ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000