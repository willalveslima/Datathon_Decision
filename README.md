# Sistema de Recomendação de Candidatos para Vagas

Este projeto implementa um sistema de recomendação de candidatos para vagas de emprego, utilizando um modelo de Machine Learning (ML) e uma arquitetura baseada em microsserviços com FastAPI, PostgreSQL, Prometheus e Grafana. O pipeline de MLOps inclui pré-processamento de dados, treinamento de modelo, API de predição, serviço de monitoramento de drift e um pipeline de CI/CD automatizado com GitHub Actions e Google Cloud Platform (GCP).

## 🚀 Funcionalidades

* **Recomendação de Candidatos**: Previsão da probabilidade de um candidato ser contratado para uma vaga específica.
* **Engenharia de Features Avançada**: Utiliza TF-IDF e extração de habilidades/tecnologias de dados textuais, além de features categóricas e de localização.
* **API RESTful de Predição**: Endpoint para receber dados de vagas e candidatos e retornar um ranking de probabilidades.
* **Serviço de Coleta de Feedback**: Endpoint para registrar o resultado real das contratações, essencial para o monitoramento de drift.
* **Monitoramento de Drift do Modelo**: Cálculo agendado de métricas de Data Drift (mudança na distribuição dos dados de entrada) e Concept Drift (queda de desempenho do modelo).
* **Monitoramento de Aplicação**: Coleta de métricas HTTP (latência, requisições, erros) de todos os serviços usando Prometheus e Grafana.
* **Persistência de Dados**: Armazenamento de logs de predição e métricas de drift em um banco de dados PostgreSQL.
* **CI/CD Automatizado**: Pipeline de Integração Contínua e Implantação Contínua com GitHub Actions para automatizar testes, builds e deploys na GCP.


## 🏛️ Arquitetura

O projeto é dividido em vários componentes, orquestrados via Docker Compose para desenvolvimento local e implantados na Google Cloud Platform para produção.

* **API de Predição (FastAPI)**:
    * Recebe requisições de predição.
    * Aplica pré-processamento e inferência do modelo.
    * Salva logs de predição no PostgreSQL.
    * Expõe métricas Prometheus.
    * Implantada no **Google Cloud Run**.
* **Serviço de Monitoramento de Drift (FastAPI)**:
    * Recebe feedback sobre o resultado real das contratações.
    * Atualiza logs de predição no PostgreSQL.
    * Executa um job agendado para calcular métricas de drift.
    * Salva métricas de drift no PostgreSQL.
    * Expõe métricas Prometheus.
    * Implantada no **Google Cloud Run**.
* **Banco de Dados (PostgreSQL)**:
    * Armazena logs de predição e métricas de drift.
    * Utiliza **Google Cloud SQL (PostgreSQL)**.
* **Prometheus**:
    * Coleta métricas dos serviços da API e de Drift Monitor.
    * Executado em uma **Imagem Docker** (para deploy local simplificado).
* **Grafana**:
    * Visualiza as métricas coletadas pelo Prometheus.
    * Painéis para monitorar a saúde da aplicação e o drift do modelo.
    * Executado em uma **Imagem Docker** (para deploy local simplificado).
* **Google Cloud Storage**:
    * Armazena dados brutos e artefatos de modelo (modelos `.pkl`, vetorizadores).
* **Google Artifact Registry**:
    * Registro privado para imagens Docker construídas.

## ⚙️ Começando (Setup Local)

Siga estes passos para configurar e rodar o projeto em sua máquina local.

### Pré-requisitos

* [Python 3.11](https://www.python.org/downloads/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (inclui Docker Compose)
* [Google Cloud CLI (gcloud)](https://cloud.google.com/sdk/docs/install)

### 1. Clonar o Repositório

```bash
git clone [https://github.com/willalveslima/Datathon_Decision.git](https://github.com/willalveslima/Datathon_Decision.git)
cd Datathon_Decision
```

### 2. Configurar Ambiente Virtual e Ferramentas

Este script irá criar um ambiente virtual Python (`.venv`), instalar as dependências, configurar os hooks de pré-commit e criar os arquivos de configuração para `isort`, `black` e `pylint`.

```bash
./.conf/setup_env.sh
```

### 3. Preparar Dados Brutos

Os dados brutos (`applicants.json`, `vagas.json`, `prospects.json`) devem ser colocados na pasta `data/raw/`. Se eles não estiverem no seu repositório Git, você precisará baixá-los manualmente para `data/raw/`.

### 4. Executar Pré-processamento de Dados

Este passo gera o arquivo `data/processed/merged_data_processed.csv`.

```bash
python src/training/data_preprocessing_merge.py
```

### 5. Treinar o Modelo

Este passo treina o modelo de ML e salva os artefatos (modelo, vetorizadores, etc.) na pasta `models/`.

```bash
python src/training/model_training.py
```

### 6. Configurar Variáveis de Ambiente do Banco de Dados

Crie um arquivo `.env` na **raiz do seu projeto** com as credenciais do banco de dados (mesmas usadas no Cloud SQL para compatibilidade).

```
# .env
DB_NAME=ml_drift_db
DB_USER=ml_user
DB_PASSWORD=ml_password_secure
```

### 7. Iniciar a Stack Local

Este comando irá construir as imagens Docker (se necessário) e iniciar todos os serviços (API, Drift Monitor, PostgreSQL, Prometheus, Grafana) via Docker Compose.

```bash
./run_local.sh
```

### 8. Acessar os Serviços Localmente

* **API de Predição**: `http://localhost:8000` (documentação Swagger em `http://localhost:8000/docs`)
* **Serviço de Monitoramento de Drift**: `http://localhost:8001` (documentação Swagger em `http://localhost:8001/docs`)
* **Prometheus**: `http://localhost:9090`
* **Grafana**: `http://localhost:3000` (usuário: `admin`, senha: `grafana-api`)

### 9. Testar o Pipeline Funcional Completo

Execute este script para simular requisições de predição e feedback, alimentando o banco de dados e as métricas de drift. Ele rodará em loop até ser interrompido (`Ctrl+C`).

```bash
python tests/functional/test_full_pipeline.py
```

### 10. Parar a Stack Local

```bash
./stop_local.sh
```

## 🌳 Estrutura do Projeto

```
.
├── docker-compose.yml
├── .env
├── run_local.sh
├── stop_local.sh
├── README.md
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
│
├── data/
│   ├── raw/
│   │   ├── applicants.json
│   │   ├── vagas.json
│   │   └── prospects.json
│   └── processed/
│       └── merged_data_processed.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── tfidf_vectorizer_applicant.pkl
│   ├── tfidf_vectorizer_job.pkl
│   ├── categorical_cols.pkl
│   ├── encoded_feature_names.pkl
│   ├── common_skills.pkl
│   └── model_version.txt
│
├── src/
│   ├── training/
│   │   ├── data_preprocessing_merge.py
│   │   └── model_training.py
│   │
│   ├── prediction_service/
│   │   └── prediction_service.py
│   │   └── __init__.py # Para importação do app
│   │
│   ├── drift_monitor_service/
│   │   └── main.py
│   │   └── __init__.py # Para importação do app
│   │
│   └── utils/
│       └── common_utils.py
│
├── docker/
│   ├── api/
│   │   ├── Dockerfile
│   │   └── Dockerfile 
│   │
│   ├── drift_monitor/
│   │   ├── Dockerfile
│   │   └── Dockerfile
│   │
│   ├── prometheus/
│   │   └── prometheus.yml
│   │
│   └── grafana/
│       ├── grafana.ini
│       ├── datasources/
│       │   └── prometheus.yml
│       └── dashboards/
│
│
└── tests/
    ├── unit/
    │   ├── test_api.py
    │   ├── test_data_preprocessing.py    
    │   ├── test_model_training.py    
    │   └── test_drift_monitor.py
    ├── functional/
    │   ├── test_full_pipilene_GCP.py
    │   └── test_full_pipeline.py
    └── .gitkeep
```
##  🧐 Testes Automatizados e Verificação da Qualidade do Código

O projeto utiliza o pre-commit para verificar qualidade do código e Teste automatizado:

```
repos:
- repo: local
  hooks:
  - id: isort
    name: Run isort
    types: [python]
    exclude: ^tests/
    entry: isort
    language: system
  - id: black
    name: Run black
    types: [python]
    exclude: ^tests/
    entry: black
    language: system
    args: ["--line-length=100"]
  - id: pydocstyle
    name: Roda pydocstyle
    types: [python]
    exclude: ^tests/
    entry: pydocstyle
    language: system
  - id: pylint
    name: Roda pylint
    types: [python]
    exclude: ^tests/
    entry: pylint
    language: system
  - id: pytest
    name: Roda pytest
    entry: bash -c "PYTHONPATH=. python -m pytest"
    language: system
    pass_filenames: false
    always_run: true
```


## 🚀 CI/CD com GitHub Actions e GCP

O projeto utiliza GitHub Actions para automatizar o pipeline de CI/CD, desde a validação do código até o deploy dos serviços no Google Cloud Platform.

### Visão Geral do Pipeline (`.github/workflows/main.yml`)

1.  **Gatilhos**: `push` para `main`, `develop`, `feature/*` e `pull_request`.
2.  **Autenticação GCP**: Autentica o GitHub Actions na GCP usando uma Service Account Key (`secrets.GCP_SA_KEY`).
3.  **Setup Python e Dependências**: Configura o ambiente Python e instala todas as dependências.
4.  **Download de Dados Brutos**: Baixa os arquivos JSON brutos do **Cloud Storage** (`gs://datathon-decision/raw/`) para o ambiente da Action.
5.  **Pré-processamento de Dados**: Executa `src/training/data_preprocessing_merge.py` para gerar o CSV processado.
6.  **Treinamento do Modelo**: Executa `src/training/model_training.py` para treinar o modelo e salvar os artefatos (`.pkl`, `.txt`) na pasta `models/` do runner.
7.  **Upload de Artefatos do Modelo**: Faz o upload dos modelos treinados da pasta `models/` do runner para o **Cloud Storage** (`gs://datathon-decision/models/`).
8.  **Testes com Cobertura**: Roda `pytest` com `pytest-cov` e impõe um limite mínimo de cobertura.
9. **Build de Imagens Docker**: Constrói as imagens Docker para a API e o Serviço de Drift usando `Dockerfile.ci` (que baixam os modelos do GCS).
10. **Push de Imagens Docker**: Envia as imagens construídas para o **Artifact Registry** da GCP.
11. **Deploy para Cloud Run**: Implanta as novas versões da API de Predição e do Serviço de Monitoramento de Drift no **Google Cloud Run**, passando as variáveis de ambiente do DB via GitHub Secrets.

### Configuração de Segredos (GitHub Secrets)

Para que o pipeline funcione, você deve configurar os seguintes segredos no seu repositório GitHub (`Settings > Secrets and variables > Actions`):

* `GCP_SA_KEY`: Conteúdo JSON completo da chave da Service Account da GCP (com as permissões necessárias).
* `GCP_PROJECT_ID`: O ID do seu projeto GCP.
* `DB_NAME_SECRET`: Nome do banco de dados PostgreSQL.
* `DB_USER_SECRET`: Usuário do banco de dados PostgreSQL.
* `DB_PASSWORD_SECRET`: Senha do banco de dados PostgreSQL.

### Permissões da Service Account da GCP

A Service Account usada pelo GitHub Actions (`GCP_SA_KEY`) e a Service Account usada pelos serviços Cloud Run em tempo de execução (`cloud-run-sa-...`) precisam das seguintes permissões no seu projeto GCP:

* **Service Account do GitHub Actions (para o pipeline de CI/CD):**
    * `Cloud Run Admin`
    * `Artifact Registry Writer`
    * `Storage Object Viewer` (para baixar dados brutos)
    * `Storage Object Admin` (para fazer upload de modelos)
    * `Service Account User` (para atuar como a SA do Cloud Run)
    * `Cloud SQL Client` (se a Action precisar descrever instâncias SQL)
    * `Service Usage Consumer`
* **Service Account do Cloud Run (para os serviços em execução):**
    * `Cloud SQL Client` (para conectar ao Cloud SQL)
    * `Storage Object Viewer` (para ler artefatos de modelo do GCS)

## 📊 Monitoramento e Drift Detection

A stack de monitoramento é composta por Prometheus e Grafana, que coletam métricas dos serviços da API e de Drift Monitor.

* **endpoint /feedback (serviço de drift)**: Recebe o feedback com os dados reais de contratação para serem usados como insumos para os calculos de drift.

* **endpoint /metrics (serviço de drift)**: Expõe as metricas de Drift Calculadas para o modelo

* **Prometheus**: Coleta métricas de:
    * Requisições HTTP (`http_requests_total`, `http_request_duration_seconds`)
    * Latência de predição (`prediction_duration_seconds`)
    * Erros de predição (`prediction_errors_total`)
    * Número de candidatos por requisição (`applicants_per_prediction_request`)
    * Versão do modelo (`model_version`)
    * Métricas de Drift (`data_drift_score`, `concept_drift_score`, `model_f1_score_production`, etc.)
* **Grafana**: Permite criar dashboards personalizados para visualizar essas métricas, identificar tendências, anomalias e receber alertas sobre o desempenho da aplicação e a saúde do modelo.

## 🤝 Contribuição

Contribuições são bem-vindas! Siga estas diretrizes:

1.  Faça um fork do repositório.
2.  Crie uma nova branch para sua feature (`git checkout -b feature/minha-nova-feature`).
3.  Faça suas alterações e certifique-se de que os testes passem.
4.  Execute `pre-commit run --all-files` antes de commitar.
5.  Envie suas alterações (`git push origin feature/minha-nova-feature`).
6.  Abra um Pull Request.

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
