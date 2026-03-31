#!/bin/bash

echo "Iniciando MLflow..."

#!/bin/bash

echo "Iniciando MLflow..."

mlflow server \
  --backend-store-uri $BACKEND_STORE_URI \
  --default-artifact-root $ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port ${PORT:-8080} \
  --workers 1 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"