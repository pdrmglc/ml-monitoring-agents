# ml-monitoring-agents

## Desenvolvimento local

### MLflow Server

Inicia o servidor do MLflow com download automático do `mlflow.db` (caso exista no GCS):

```bash
chmod +x mlflow_server/start.sh
./mlflow_server/start.sh
```

O servidor ficará disponível em:
http://localhost:5000

### API (FastAPI)
Inicia o serviço da API:

```bash
uvicorn app.api:main --host 0.0.0.0 --port 8080
```

A API ficará disponível em:
http://localhost:8080