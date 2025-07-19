"""Serviço de Predição de Contratação de Candidatos.

Este script implementa uma API FastAPI que serve um modelo de Machine Learning
para prever a probabilidade de um candidato ser contratado para uma vaga específica.

Funcionalidades Principais:
1.  **Carregamento de Artefatos**: Na inicialização, carrega o modelo de Regressão
    Logística treinado e todos os objetos de pré-processamento necessários
    (vetorizadores TF-IDF, listas de colunas, etc.).
2.  **Endpoint de Predição (`/predict`)**:
    - Recebe os dados de uma vaga e uma lista de candidatos.
    - Para cada candidato, aplica o mesmo pipeline de pré-processamento usado
      no treinamento (engenharia de features de texto, extração de habilidades,
      codificação de features categóricas).
    - Usa o modelo carregado para prever a probabilidade de contratação.
    - Retorna uma lista de candidatos ranqueados pela probabilidade, da maior
      para a menor.
3.  **Logging de Predições**: Salva cada predição realizada em um banco de dados
    PostgreSQL, incluindo as features de entrada e a probabilidade prevista.
    Esses logs são essenciais para o monitoramento de drift.
4.  **Monitoramento com Prometheus**: Expõe um endpoint `/metrics` com métricas
    sobre o desempenho da API, como latência de requisição, latência de predição,
    contagem de erros e número de candidatos por requisição.
"""

import json  # Para serializar/desserializar JSON
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from scipy.sparse import csr_matrix, hstack
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from starlette.responses import PlainTextResponse

# Importar utilitários comuns


# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. Carregar o Modelo e Objetos de Pré-processamento ---
# Estes objetos são carregados uma única vez quando a API é iniciada.
try:
    loaded_model: Any = joblib.load("models/logistic_regression_model.pkl")
    loaded_tfidf_applicant: Any = joblib.load("models/tfidf_vectorizer_applicant.pkl")
    loaded_tfidf_job: Any = joblib.load("models/tfidf_vectorizer_job.pkl")
    loaded_categorical_cols: List[str] = joblib.load("models/categorical_cols.pkl")
    loaded_encoded_feature_names: List[str] = joblib.load("models/encoded_feature_names.pkl")
    loaded_common_skills: List[str] = joblib.load("models/common_skills.pkl")
    logger.info("Modelo e objetos de pré-processamento carregados com sucesso.")
    with open("models/model_version.txt", "r", encoding="utf-8") as f:
        model_version_str = f.read().strip()
    logger.info(
        "Modelo e objetos de pré-processamento carregados com sucesso. Versão do modelo: %s",
        model_version_str,
    )

except FileNotFoundError as e:
    logger.error(
        "Erro ao carregar arquivos do modelo. "
        "Certifique-se de que os arquivos .pkl estão na pasta 'models/'. "
        "Erro: %s",
        e,
    )
    raise RuntimeError(
        "Erro ao carregar arquivos do modelo. Certifique-se de que os arquivos"
        f".pkl estão na pasta 'models/'. Erro: {e}"
    ) from e
except Exception as e:
    logger.error(
        "Erro inesperado ao carregar o modelo ou objetos de pré-processamento: %s",
        e,
    )
    raise RuntimeError(
        f"Erro inesperado ao carregar o modelo ou objetos de pré-processamento: {e}"
    ) from e

# Lista de stop words comuns em português (deve ser a mesma usada no treinamento)

# --- 2. Configuração do Banco de Dados (SQLAlchemy) ---
Base: Any = declarative_base()


class DatabaseConnection:
    """Classe para gerenciar a conexão com o banco de dados."""

    def __init__(self):
        """Incializa a conexão com o banco de dados."""
        self.engine: Optional[Engine] = None
        self.session_local: Optional[sessionmaker] = None

    def initialize(self, database_url: str) -> None:
        """Inicializa a conexão com o banco de dados."""
        self.engine = create_engine(database_url)
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def get_session(self):
        """Retorna uma nova sessão do banco de dados."""
        if self.session_local is None:
            raise RuntimeError("Database connection not initialized")
        return self.session_local()


# Instância global da conexão do banco
db_connection = DatabaseConnection()


# Definição do modelo da tabela para logs de predição
class PredictionLog(Base):
    """Modelo SQLAlchemy para a tabela 'prediction_logs'."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    codigo_vaga = Column(String, index=True)
    codigo_profissional = Column(String, index=True)
    predicted_probability = Column(Float)
    actual_outcome = Column(Boolean, nullable=True)
    input_features_applicant = Column(Text)
    input_features_job = Column(Text)

    def __repr__(self):
        """Representação do objeto PredictionLog."""
        return (
            f"<PredictionLog(id={self.id}, codigo_vaga='{self.codigo_vaga}', "
            f"codigo_profissional='{self.codigo_profissional}',"
            f"predicted_probability={self.predicted_probability})>"
        )

    def to_dict(self):
        """Convert o objeto PredictionLog em um dicionário."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "codigo_vaga": self.codigo_vaga,
            "codigo_profissional": self.codigo_profissional,
            "predicted_probability": self.predicted_probability,
            "actual_outcome": self.actual_outcome,
            "input_features_applicant": self.input_features_applicant,
            "input_features_job": self.input_features_job,
        }


# --- 3. Definir Métricas Prometheus ---
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP Request Latency", ["method", "endpoint", "status_code"]
)
PREDICTION_LATENCY = Histogram("prediction_duration_seconds", "Prediction Latency")
PREDICTION_ERRORS_TOTAL = Counter("prediction_errors_total", "Total Prediction Errors")
APPLICANTS_PER_PREDICTION = Histogram(
    "applicants_per_prediction_request", "Number of applicants per prediction request"
)
MODEL_VERSION = Gauge("model_version", "Version of the deployed model")
SUCCESSFUL_PREDICTIONS_TOTAL = Counter(
    "successful_predictions_total", "Total Successful Predictions"
)

MODEL_VERSION.set(float(model_version_str))


# --- 4. Definir Modelos Pydantic para Validação de Entrada e Saída ---
class ApplicantData(BaseModel):
    """Modelo Pydantic para os dados de entrada de um único candidato."""

    codigo_profissional: str  # Campo obrigatório
    objetivo_profissional: str  # Campo obrigatório
    historico_profissional_texto: str  # Campo obrigatório
    cv_completo: str  # Campo obrigatório
    local: str  # Campo obrigatório


class JobData(BaseModel):
    """Modelo Pydantic para os dados de entrada de uma única vaga."""

    codigo_vaga: str  # Campo obrigatório
    titulo_vaga_prospect: str  # Campo obrigatório
    titulo_vaga: str  # Campo obrigatório
    principais_atividades: str  # Campo obrigatório
    competencia_tecnicas_e_comportamentais: str  # Campo obrigatório
    demais_observacoes: str  # Campo obrigatório
    areas_atuacao: str  # Campo obrigatório
    nivel_profissional: str  # Campo obrigatório
    nivel_academico: str  # Campo obrigatório
    nivel_ingles: str  # Campo obrigatório
    nivel_espanhol: str  # Campo obrigatório
    pais: str  # Campo obrigatório
    estado: str  # Campo obrigatório
    cidade: str  # Campo obrigatório
    regiao: str  # Campo obrigatório


class PredictionRequest(BaseModel):
    """Modelo Pydantic para o corpo da requisição de predição."""

    job: JobData
    applicants: List[ApplicantData]


class RankedApplicant(BaseModel):
    """Modelo Pydantic para um candidato ranqueado na resposta da predição."""

    applicant_index: int
    probability: float
    codigo_profissional: str


class PredictionResponse(BaseModel):
    """Modelo Pydantic para a resposta da API de predição."""

    ranked_applicants: List[RankedApplicant]


# --- 5. Inicializar a Aplicação FastAPI ---
app = FastAPI(
    title="API de Recomendação de Candidatos",
    description=(
        "API para prever a probabilidade de contratação de candidatos " "para vagas e ranqueá-los."
    ),
    version="1.0.0",
)


# --- Evento de Startup para Inicializar o Banco de Dados ---
@app.on_event("startup")
def startup_db_connection() -> None:
    """
    Inicializa a conexão com o banco de dados na inicialização da API.

    Lê as credenciais do banco de dados a partir de variáveis de ambiente,
    cria o motor (engine) do SQLAlchemy e a tabela de logs, se não existir.
    """
    db_name = os.getenv("DB_NAME", "ml_drift_db")
    db_user = os.getenv("DB_USER", "ml_user")
    db_password = os.getenv("DB_PASSWORD", "ml_password_secure")
    db_host = os.getenv("DB_HOST", "db")
    db_port = os.getenv("DB_PORT", "5432")

    database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    db_connection.initialize(database_url)
    logger.info(
        "Conexão com o banco de dados inicializada e tabela 'prediction_logs' verificada/criada."
    )


# --- Middleware para Métricas de Requisição ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable) -> PlainTextResponse:
    """
    Middleware para capturar métricas de requisição HTTP.

    Calcula a latência de cada requisição e incrementa contadores Prometheus
    para monitoramento de volume, latência e status das requisições.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    method = request.method
    endpoint = request.url.path
    status_code = str(response.status_code)

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status_code=status_code).observe(
        process_time
    )
    logger.info(
        "Requisição %s %s com status %s processada em %.4f segundos.",
        method,
        endpoint,
        status_code,
        process_time,
    )
    return response


# pylint: disable=duplicate-code
# --- Função para extrair habilidades (replicada do treinamento) ---
def extract_skills_from_text(text: str, skills_list: List[str]) -> List[str]:
    """
    Extrai habilidades de uma lista predefinida que estão presentes em um texto.

    Args:
        text: O texto de entrada para ser analisado.
        skills_list: Uma lista de strings, onde cada string é uma habilidade
                     a ser procurada (deve estar em minúsculas).

    Returns:
        Uma lista de habilidades encontradas no texto.
    """
    found_skills: List[str] = []
    text_lower = str(text).lower()
    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills


# pylint: enable=duplicate-code
# pylint: disable=duplicate-code
# --- 6. Funções auxiliares de pré-processamento ---
def _prepare_text_features(applicant_data: Dict[str, Any], job_data: Dict[str, Any]) -> tuple:
    """Prepara e combina features de texto para candidato e vaga."""
    text_cols_applicants = ["objetivo_profissional", "historico_profissional_texto", "cv_completo"]
    text_cols_vagas = [
        "titulo_vaga_prospect",
        "titulo_vaga",
        "principais_atividades",
        "competencia_tecnicas_e_comportamentais",
        "demais_observacoes",
        "areas_atuacao",
    ]

    temp_applicant_df = pd.DataFrame([applicant_data])
    temp_job_df = pd.DataFrame([job_data])

    # Preenche campos ausentes com string vazia
    for col in text_cols_applicants:
        temp_applicant_df[col] = (
            temp_applicant_df[col].fillna("") if col in temp_applicant_df.columns else ""
        )

    for col in text_cols_vagas:
        temp_job_df[col] = temp_job_df[col].fillna("") if col in temp_job_df.columns else ""

    # Combina features de texto
    applicant_text = " ".join([temp_applicant_df[col].iloc[0] for col in text_cols_applicants])
    job_text = " ".join([temp_job_df[col].iloc[0] for col in text_cols_vagas])

    return applicant_text, job_text


# pylint: enable=duplicate-code
# pylint: disable=duplicate-code
def _prepare_categorical_features(
    applicant_data: Dict[str, Any], job_data: Dict[str, Any]
) -> csr_matrix:
    """Prepara features categóricas usando one-hot encoding."""
    categorical_cols = [
        "nivel profissional",
        "nivel_academico",
        "nivel_ingles",
        "nivel_espanhol",
        "local",
        "pais",
        "estado",
        "cidade",
        "regiao",
    ]

    categorical_data = {}
    for col in categorical_cols:
        if col in applicant_data:
            categorical_data[col] = [applicant_data[col]]
        elif col in job_data:
            if col == "nivel profissional" and "nivel_profissional" in job_data:
                categorical_data[col] = [job_data["nivel_profissional"]]
            else:
                categorical_data[col] = [job_data[col]]
        else:
            categorical_data[col] = [""]

    temp_categorical_df = pd.DataFrame(categorical_data)
    for col in temp_categorical_df.columns:
        temp_categorical_df[col] = temp_categorical_df[col].astype(str).fillna("")

    encoded_features = pd.get_dummies(temp_categorical_df[loaded_categorical_cols], dummy_na=False)
    encoded_features = encoded_features.reindex(columns=loaded_encoded_feature_names, fill_value=0)

    return csr_matrix(encoded_features.astype(float))


# pylint: enable=duplicate-code
def _prepare_skill_features(applicant_text: str, job_text: str) -> csr_matrix:
    """Prepara features de habilidades baseadas no texto."""
    applicant_skills = extract_skills_from_text(applicant_text, loaded_common_skills)
    job_skills = extract_skills_from_text(job_text, loaded_common_skills)

    skill_features_data = {}
    for skill in loaded_common_skills:
        skill_features_data[f"applicant_has_{skill}"] = [1 if skill in applicant_skills else 0]
        skill_features_data[f"job_requires_{skill}"] = [1 if skill in job_skills else 0]

    skill_features_data["common_skills_count"] = [
        len(set(applicant_skills).intersection(set(job_skills)))
    ]

    skill_features_df = pd.DataFrame(skill_features_data)
    return csr_matrix(skill_features_df.values)


def preprocess_and_predict(applicant_data: Dict[str, Any], job_data: Dict[str, Any]) -> float:
    """
    Pré-processa os dados de um candidato e uma vaga e retorna a probabilidade de contratação.

    Esta função replica o pipeline de pré-processamento do script de treinamento.

    Args:
        applicant_data: Dicionário com os dados de um candidato.
        job_data: Dicionário com os dados de uma vaga.

    Returns:
        A probabilidade (float) de o candidato ser contratado para a vaga.
    """
    prediction_start_time = time.time()
    try:
        # Prepara features de texto
        applicant_text, job_text = _prepare_text_features(applicant_data, job_data)

        # Aplica vetorização TF-IDF
        applicant_tfidf = loaded_tfidf_applicant.transform([applicant_text])
        job_tfidf = loaded_tfidf_job.transform([job_text])

        # Prepara features categóricas
        encoded_features_sparse = _prepare_categorical_features(applicant_data, job_data)

        # Prepara features de habilidades
        skill_features_sparse = _prepare_skill_features(applicant_text, job_text)

        # Combina todas as features
        x_inference = hstack(
            [applicant_tfidf, job_tfidf, encoded_features_sparse, skill_features_sparse]
        )

        # Faz a predição
        probability = loaded_model.predict_proba(x_inference)[:, 1][0]

        prediction_end_time = time.time()
        PREDICTION_LATENCY.observe(prediction_end_time - prediction_start_time)
        logger.info(
            "Predição concluída em %.4f segundos.", prediction_end_time - prediction_start_time
        )

        return probability

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.error("Erro durante o pré-processamento ou predição: %s", e, exc_info=True)
        raise


# --- 7. Definir Endpoints da API ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    /predict.

    Recebe dados de uma vaga e uma lista de candidatos,
    calcula a probabilidade de contratação para cada candidato,
    salva os logs no banco de dados e retorna os candidatos ranqueados por probabilidade.

    Args:
        request: Objeto PredictionRequest contendo os dados da vaga e dos candidatos.

    Returns:
        PredictionResponse: Lista de candidatos ranqueados por probabilidade de contratação.
    """
    num_applicants = len(request.applicants) if isinstance(request.applicants, list) else 0
    logger.info(
        "Requisição de predição recebida para %d candidatos e vaga %s.",
        num_applicants,
        request.job.codigo_vaga,
    )
    APPLICANTS_PER_PREDICTION.observe(num_applicants)

    results = []
    db_session = db_connection.get_session()

    try:
        for i, applicant_data in enumerate(request.applicants):
            try:
                prob = preprocess_and_predict(applicant_data.dict(), request.job.dict())

                results.append(
                    RankedApplicant(
                        applicant_index=i,
                        probability=prob,
                        codigo_profissional=applicant_data.codigo_profissional,
                    )
                )

                log_entry = PredictionLog(
                    codigo_vaga=request.job.codigo_vaga,
                    codigo_profissional=applicant_data.codigo_profissional,
                    predicted_probability=float(prob),
                    actual_outcome=None,
                    input_features_applicant=json.dumps(applicant_data.dict()),
                    input_features_job=json.dumps(request.job.dict()),
                )
                db_session.add(log_entry)
                logger.info(
                    "Log de predição salvo para candidato %s na vaga %s.",
                    applicant_data.codigo_profissional,
                    request.job.codigo_vaga,
                )

            except (ValueError, TypeError, KeyError) as e:
                logger.warning(
                    "Não foi possível processar ou logar o candidato índice %d (código: %s)"
                    "devido a um erro: %s. O candidato será ignorado.",
                    i,
                    applicant_data.codigo_profissional,
                    e,
                )

        db_session.commit()

    except Exception as e:
        db_session.rollback()
        logger.error("Erro ao salvar logs de predição no banco de dados: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Erro interno ao processar e salvar predições."
        ) from e
    finally:
        db_session.close()

    results.sort(key=lambda x: x.probability, reverse=True)

    SUCCESSFUL_PREDICTIONS_TOTAL.inc()
    logger.info("Predição concluída com sucesso para %d candidatos ranqueados.", len(results))
    return PredictionResponse(ranked_applicants=results)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Endpoint para o Prometheus coletar as métricas."""
    logger.info("Requisição para o endpoint /metrics recebida.")
    return PlainTextResponse(generate_latest().decode("utf-8"))


# --- Como Rodar a API ---
# Para rodar esta API, salve o código acima como um arquivo Python (ex: src/prediction_service.py).
#
# Certifique-se de que seus arquivos .pkl (modelo e transformadores) estejam na pasta 'models/'
# no mesmo nível da pasta 'src'.
#
# Instale as bibliotecas necessárias:
# pip install fastapi uvicorn pandas scikit-learn
# joblib pydantic prometheus_client psycopg2-binary SQLAlchemy
#
# Abra seu terminal no diretório raiz do projeto (onde está a pasta 'src' e 'models') e execute:
# uvicorn src.prediction_service:app --host 0.0.0.0 --port 8000 --reload
#
# A API estará disponível em http://127.0.0.1:8000
# Você pode acessar a documentação interativa em http://127.0.0.1:8000/docs
# As métricas estarão disponíveis em http://127.0.0.1:8000/metrics
