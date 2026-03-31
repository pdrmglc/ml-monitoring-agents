#!/bin/bash

set -e

PROJECT_ID=$GOOGLE_CLOUD_PROJECT
REGION=$GOOGLE_CLOUD_REGION
SERVICE_NAME="mlflow"

IMAGE="gcr.io/$PROJECT_ID/mlflow"

echo "==> Buildando imagem"
gcloud builds submit mlflow_server --tag $IMAGE

echo "==> Deployando no Cloud Run"
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory=2Gi \
  --min-instances=0 \
  --max-instances=1
  --set-env-vars BACKEND_STORE_URI=$BACKEND_STORE_URI,ARTIFACT_ROOT=$ARTIFACT_ROOT

echo "==> Deploy finalizado"