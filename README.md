# ml-monitoring-agents

## Desenvolvimento local

### MLflow Server

O MLflow é utilizado **apenas localmente** para rastreamento de experimentos.  
Os artefatos (modelos) são armazenados no GCS, evitando a necessidade de manter o MLflow em produção.

O script `start.sh`:

- baixa o `mlflow.db` do GCS (se existir)
- inicia o servidor local
- permite recuperar o histórico de experimentos

**Trade-off:**
- ✔ Sem custo e menor complexidade (sem deploy do MLflow)
- ✘ Necessidade de usar explicitamente o caminho do modelo no bucket (GCS)

#### Executar:
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