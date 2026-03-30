# ml-monitoring-agents

Projeto que propõe uma arquitetura completa para:

* desenvolvimento de modelos de machine learning
* versionamento de modelos e pipelines
* deploy e serving de modelos
* monitoramento de modelo e das variáveis de entrada (data drift)

O objetivo não é apenas treinar modelos, mas estruturar um fluxo consistente, reproduzível e escalável de ponta a ponta.

---

## Desenvolvimento local

### MLflow Server

O MLflow é executado em um **container dedicado**, sendo utilizado para:

* rastreamento de experimentos
* versionamento de modelos
* gerenciamento de artefatos

Essa abordagem garante isolamento, reprodutibilidade e facilita a migração para ambientes produtivos.

O script `start.sh`:

* inicializa o servidor MLflow (a ser implementado futuramente em container)
* configura o backend de tracking
* permite acesso centralizado aos experimentos

#### Execução

```bash
chmod +x mlflow_server/start.sh
./mlflow_server/start.sh
```

A interface estará disponível em:

```
http://localhost:5000
```

---

### API (FastAPI)

Responsável por:

* servir modelos em produção
* executar inferência
* calcular métricas de drift
* retornar resultados em JSON ou HTML

#### Execução

```bash
uvicorn app.api:main --host 0.0.0.0 --port 8080
```

A API estará disponível em:

```
http://localhost:8080
```

---

## Arquitetura do projeto

A estrutura foi organizada para separar claramente responsabilidades entre:

* treinamento
* inferência
* validação de dados
* monitoramento
* processamento batch

---

### `.devcontainer/`

Ambiente de desenvolvimento isolado.

* **devcontainer.json**
  Define o ambiente de desenvolvimento (extensões, configurações e runtime)

* **Dockerfile.dev**
  Build do container de desenvolvimento, garantindo consistência entre máquinas

---

### `app/`

Camada de aplicação (API e inferência online)

* **api.py**
  Ponto de entrada da API. Define endpoints para:

  * inferência
  * cálculo de drift
  * geração de relatórios

* **model_inference.py**
  Orquestra o fluxo de inferência:

  * validação
  * pré-processamento
  * carregamento do modelo
  * predição

---

### `jobs/`

Processamento batch

* **batch_predict.py**
  Executa inferência em larga escala fora da API (modo offline)

---

### `ml/`

Núcleo de machine learning

#### `encoders/`

* Implementações customizadas de encoding
* Exemplo: target encoding com estratégia out-of-fold para evitar leakage

#### `preprocessing/`

* **preprocessor.py**
  Pipeline de transformação de dados:

  * limpeza
  * engenharia de features
  * padronização

Projetado para ser integrado a pipelines (ex: sklearn Pipeline), garantindo:

* reprodutibilidade
* versionamento consistente

#### `schema/`

* **data_definition.py** → definição das variáveis
* **pydantic_schema.py** → validação de entrada da API
* **validate_schema.py** → regras adicionais de consistência

Responsável por garantir integridade antes da inferência

#### `training/`

Pipeline de treinamento e versionamento

* **model_development.py** → experimentação e treinamento
* **mlflow_tracking.py** → logging de métricas e artefatos
* **model_registry.py** → versionamento e promoção de modelos

---

### `mlflow_server/`

Infraestrutura do MLflow

* **start.sh**
  Script de inicialização do servidor em container

---

### Docker

* **Dockerfile**
  Build da aplicação para ambiente de produção

---

### Configuração

* **.env.example**
  Template de variáveis de ambiente

---

### Dependências

* **requirements.dev.txt**
  Ambiente de desenvolvimento (inclui produção)

* **requirements.prod.txt**
  Ambiente mínimo para execução em produção

