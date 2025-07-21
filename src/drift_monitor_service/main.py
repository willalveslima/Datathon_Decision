"""Serviço de Monitoramento de Drift e Coleta de Feedback.

Este serviço FastAPI tem duas responsabilidades principais:

1.  **Coleta de Feedback**: Expõe um endpoint `/feedback` que recebe o resultado
    real (ground truth) de uma previsão. Por exemplo, se um candidato foi
    efetivamente contratado ou não. Essa informação é usada para atualizar
    os logs de predição no banco de dados.

2.  **Monitoramento de Drift**: Executa uma tarefa agendada (a cada 5 minutos)
    para calcular métricas de Data Drift e Concept Drift.
    - **Data Drift**: Compara a distribuição estatística das features de entrada
      recentes com uma janela de tempo de referência (baseline) para detectar
      mudanças nos dados de entrada.
    - **Concept Drift**: Mede a degradação do desempenho do modelo ao longo do
      tempo, comparando métricas como F1-score, precisão e recall com valores
      de baseline, utilizando os feedbacks recebidos.

As métricas calculadas são salvas na tabela `drift_metrics` e também expostas
através do endpoint `/metrics` para monitoramento contínuo pelo Prometheus.
"""

import json  # Para serializar/desserializar JSON
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import joblib  # Para carregar objetos de pré-processamento se necessário para drift
import numpy as np  # Importar numpy para lidar com tipos numpy
import pandas as pd
import sqlalchemy  # Importação necessária para exceções SQLAlchemy
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # Para agendamento
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
from sqlalchemy.orm import Session, sessionmaker
from starlette.responses import PlainTextResponse

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. Configuração do Banco de Dados (SQLAlchemy) ---
# Variáveis globais inicializadas como None. Serão preenchidas no evento de startup.
engine: Optional[Engine] = None
SessionLocal: Optional[sessionmaker] = None
Base: Any = declarative_base()


# pylint: disable=duplicate-code
# Definição do modelo da tabela para logs de predição (compartilhada com a API de predição)
class PredictionLog(Base):
    """Modelo SQLAlchemy para a tabela 'prediction_logs'."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    codigo_vaga = Column(String, index=True)
    codigo_profissional = Column(String, index=True)
    predicted_probability = Column(Float)
    actual_outcome = Column(Boolean, nullable=True)  # Pode ser NULL inicialmente
    input_features_applicant = Column(Text)
    input_features_job = Column(Text)

    # pylint: disable=duplicate-code
    def to_dict(self) -> dict:
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

    # pylint: enable=duplicate-code
    def update_outcome(self, outcome: bool) -> None:
        """Atualiza o resultado real (actual_outcome) do log."""
        self.actual_outcome = outcome


# pylint: enable=duplicate-code
# Definição do modelo da tabela para métricas de drift (NOVA TABELA)
class DriftMetric(Base):
    """Modelo SQLAlchemy para a tabela 'drift_metrics'."""

    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String, index=True)
    metric_value = Column(Float)
    feature_name = Column(String, nullable=True)  # Para data drift por feature
    drift_score = Column(Float, nullable=True)  # Ex: p-value de um teste KS ou diferença

    def to_dict(self) -> dict:
        """Convert o objeto DriftMetric em um dicionário."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "feature_name": self.feature_name,
            "drift_score": self.drift_score,
        }

    def update_metric(self, metric_value: float, drift_score: float = None) -> None:
        """Atualiza os valores de metric_value e drift_score."""
        self.metric_value = metric_value
        if drift_score is not None:
            self.drift_score = drift_score


# --- 2. Definir Modelos Pydantic para Endpoints ---
class FeedbackRequest(BaseModel):
    """Modelo Pydantic para o corpo da requisição de feedback."""

    codigo_vaga: str
    codigo_profissional: str
    was_hired: bool  # True se contratado, False caso contrário


class DriftMetricResponse(BaseModel):
    """Modelo Pydantic para a resposta de métricas de drift (se houvesse um endpoint para isso)."""

    timestamp: datetime
    metric_name: str
    metric_value: float
    feature_name: Optional[str] = None
    drift_score: Optional[float] = None


# --- 3. Definir Métricas Prometheus para Drift ---

REQUEST_COUNT = Counter(
    "drift_http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "drift_http_request_duration_seconds",
    "HTTP Request Latency",
    ["method", "endpoint", "status_code"],
)

DATA_DRIFT_SCORE = Gauge("data_drift_score", "Data Drift Score for a feature", ["feature_name"])
CONCEPT_DRIFT_SCORE = Gauge("concept_drift_score", "Concept Drift Score (Model Performance Drop)")
TOTAL_DRIFT_CHECKS = Counter("total_drift_checks", "Total number of drift calculation runs")
DRIFT_CALCULATION_LATENCY = Histogram(
    "drift_calculation_duration_seconds", "Duration of drift calculation job"
)
MODEL_F1_SCORE = Gauge("model_f1_score_production", "F1-Score of the model in production")
MODEL_ACCURACY_SCORE = Gauge(
    "model_accuracy_score_production", "Accuracy Score of the model in production"
)
MODEL_PRECISION_SCORE = Gauge(
    "model_precision_score_production", "Precision Score of the model in production"
)  # Nova métrica
MODEL_RECALL_SCORE = Gauge(
    "model_recall_score_production", "Recall Score of the model in production"
)  # Nova métrica

# --- 4. Inicializar a Aplicação FastAPI ---
app = FastAPI(
    title="Serviço de Feedback e Monitoramento de Drift",
    description="Coleta feedback de contratação e calcula métricas de drift do modelo.",
    version="1.0.0",
)


# --- Evento de Startup para Inicializar o Banco de Dados ---
@app.on_event("startup")
def startup_db_connection() -> None:
    """
    Inicializa a conexão com o banco de dados na inicialização da API.

    Lê as credenciais do banco de dados a partir de variáveis de ambiente,
    cria o motor (engine) do SQLAlchemy e as tabelas, se não existirem.
    """
    # pylint: disable=global-statement
    global engine, SessionLocal  # Declara que estamos usando as variáveis globais
    # pylint: enable=global-statement
    # pylint: disable=duplicate-code
    db_name = os.getenv("DB_NAME", "ml_drift_db")
    db_user = os.getenv("DB_USER", "ml_user")
    db_password = os.getenv("DB_PASSWORD", "ml_password_secure")
    db_host = os.getenv("DB_HOST", "db")
    # db_port = os.getenv("DB_PORT", "5432")

    # database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    database_url = f"postgresql+psycopg2://{db_user}:{db_password}@/{db_name}?host={db_host}"

    # pylint: disable=duplicate-code
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)  # Cria as tabelas no DB
    logger.info("Conexão com o banco de dados inicializada e tabelas verificadas/criadas.")


# --- Middleware para Métricas de Requisição ---
# pylint: disable=duplicate-code
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


# --- 5. Funções Auxiliares para Cálculo de Drift ---
def _get_logs_for_period(
    db_session: Session, start_time: datetime, end_time: datetime
) -> List[PredictionLog]:
    """Busca logs de predição para um período de tempo específico."""
    return (
        db_session.query(PredictionLog)
        .filter(
            PredictionLog.timestamp >= start_time,
            PredictionLog.timestamp < end_time,
        )
        .all()
    )


def _process_logs_to_dataframe(logs: List[PredictionLog]) -> pd.DataFrame:
    """Convert uma lista de PredictionLog em um DataFrame."""
    return pd.DataFrame(
        [
            {
                "codigo_vaga": log.codigo_vaga,
                "codigo_profissional": log.codigo_profissional,
                "predicted_probability": log.predicted_probability,
                "actual_outcome": log.actual_outcome,
                "input_features_applicant": json.loads(log.input_features_applicant),
                "input_features_job": json.loads(log.input_features_job),
            }
            for log in logs
        ]
    )


def _get_nested_feature(data_dict: Dict[str, Any], feature_name: str) -> str:
    """Extrai uma feature de um dicionário aninhado (applicant/job features)."""
    if feature_name in data_dict:
        return data_dict[feature_name]
    if feature_name == "nivel profissional" and "nivel_profissional" in data_dict:
        return data_dict["nivel_profissional"]
    return ""


# pylint: disable=duplicate-code
def _calculate_data_drift(  # pylint: disable=R0914,too-many-branches
    recent_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    db_session: Session,
    loaded_common_skills: List[str],
) -> None:  # pylint: disable=R0914,too-many-branches
    """Calcula e registra métricas de Data Drift."""
    categorical_features_to_monitor: List[str] = [
        "nivel_academico",
        "nivel_ingles",
        "nivel_espanhol",
        "local",
        "pais",
        "estado",
        "cidade",
        "regiao",
        "nivel profissional",
    ]

    recent_features_for_drift = pd.DataFrame()
    baseline_features_for_drift = pd.DataFrame()
    # pylint: disable=duplicate-code
    for feature in categorical_features_to_monitor:
        recent_features_for_drift[feature] = recent_df.apply(
            lambda row, f=feature: (  # Captura a variável 'feature' no escopo da lambda
                _get_nested_feature(row["input_features_applicant"], f)
                if f in row["input_features_applicant"]
                else _get_nested_feature(row["input_features_job"], f)
            ),
            axis=1,
        )
        baseline_features_for_drift[feature] = baseline_df.apply(
            lambda row, f=feature: (  # Captura a variável 'feature' no escopo da lambda
                _get_nested_feature(row["input_features_applicant"], f)
                if f in row["input_features_applicant"]
                else _get_nested_feature(row["input_features_job"], f)
            ),
            axis=1,
        )
        recent_features_for_drift[feature] = (
            recent_features_for_drift[feature].astype(str).fillna("")
        )
        baseline_features_for_drift[feature] = (
            baseline_features_for_drift[feature].astype(str).fillna("")
        )
    # pylint: disable=duplicate-code
    for feature in categorical_features_to_monitor:
        if (
            feature in recent_features_for_drift.columns
            and feature in baseline_features_for_drift.columns
        ):
            if (
                not recent_features_for_drift[feature].empty
                and not baseline_features_for_drift[feature].empty
            ):
                recent_counts: pd.Series = recent_features_for_drift[feature].value_counts(
                    normalize=True
                )
                baseline_counts: pd.Series = baseline_features_for_drift[feature].value_counts(
                    normalize=True
                )

                if not baseline_counts.empty:
                    most_common_baseline_cat = baseline_counts.index[0]
                    prop_recent: float = recent_counts.get(most_common_baseline_cat, 0)
                    prop_baseline: float = baseline_counts.get(most_common_baseline_cat, 0)
                    drift_value = abs(prop_recent - prop_baseline)

                    DATA_DRIFT_SCORE.labels(feature_name=feature).set(float(drift_value))
                    db_session.add(
                        DriftMetric(
                            metric_name=f"data_drift_{feature}",
                            metric_value=float(drift_value),
                            feature_name=feature,
                        )
                    )
                    logger.info("Data Drift para '%s': %.4f", feature, drift_value)
                else:
                    logger.warning(
                        "Baseline vazio para feature '%s'. Pulando cálculo de drift "
                        "para esta feature.",
                        feature,
                    )
            else:
                logger.warning(
                    "Dados recentes ou baseline vazios para feature '%s'. Pulando cálculo "
                    "de drift para esta feature.",
                    feature,
                )

    if loaded_common_skills:
        recent_applicant_text: pd.Series = recent_df["input_features_applicant"].apply(
            lambda x: x.get("objetivo_profissional", "")
            + " "
            + x.get("historico_profissional_texto", "")
            + " "
            + x.get("cv_completo", "")
        )
        baseline_applicant_text: pd.Series = baseline_df["input_features_applicant"].apply(
            lambda x: x.get("objetivo_profissional", "")
            + " "
            + x.get("historico_profissional_texto", "")
            + " "
            + x.get("cv_completo", "")
        )

        key_skill: str = "python"
        recent_has_skill: float = recent_applicant_text.apply(
            lambda x: 1 if key_skill in x.lower() else 0
        ).mean()
        baseline_has_skill: float = baseline_applicant_text.apply(
            lambda x: 1 if key_skill in x.lower() else 0
        ).mean()
        drift_skill_freq = abs(recent_has_skill - baseline_has_skill)
        DATA_DRIFT_SCORE.labels(feature_name=f"skill_{key_skill}").set(float(drift_skill_freq))
        db_session.add(
            DriftMetric(
                metric_name=f"data_drift_skill_{key_skill}",
                metric_value=float(drift_skill_freq),
                feature_name=f"skill_{key_skill}",
            )
        )
        logger.info("Data Drift para habilidade '%s': %.4f", key_skill, drift_skill_freq)


# pylint: enable=duplicate-code


def _calculate_concept_drift(recent_df: pd.DataFrame, db_session: Session) -> None:
    """Calcula e registra métricas de Concept Drift (performance do modelo)."""
    recent_evaluated_df: pd.DataFrame = recent_df.dropna(subset=["actual_outcome"])
    if not recent_evaluated_df.empty:
        y_true: pd.Series = recent_evaluated_df["actual_outcome"].astype(int)
        y_pred_proba: pd.Series = recent_evaluated_df["predicted_probability"]
        y_pred_binary: pd.Series = (y_pred_proba >= 0.7).astype(int)

        if len(np.unique(y_true)) > 1 and np.sum(y_true) > 0 and np.sum(y_pred_binary) > 0:
            current_f1: float = f1_score(y_true, y_pred_binary, zero_division=0)
            current_accuracy: float = accuracy_score(y_true, y_pred_binary)
            current_precision: float = precision_score(y_true, y_pred_binary, zero_division=0)
            current_recall: float = recall_score(y_true, y_pred_binary, zero_division=0)

            baseline_f1_score: float = 0.47
            concept_drift_value: float = baseline_f1_score - current_f1

            CONCEPT_DRIFT_SCORE.set(float(concept_drift_value))
            MODEL_F1_SCORE.set(float(current_f1))
            MODEL_ACCURACY_SCORE.set(float(current_accuracy))
            MODEL_PRECISION_SCORE.set(float(current_precision))
            MODEL_RECALL_SCORE.set(float(current_recall))
            logger.info(
                "Concept Drift (Queda F1-Score): %.4f "
                "(F1 atual: %.4f, "
                "Acc atual: %.4f, "
                "Precision atual: %.4f, "
                "Recall atual: %.4f)",
                concept_drift_value,
                current_f1,
                current_accuracy,
                current_precision,
                current_recall,
            )

            db_session.add(
                DriftMetric(
                    metric_name="concept_drift_f1_drop",
                    metric_value=float(concept_drift_value),
                )
            )
            db_session.add(
                DriftMetric(metric_name="model_f1_score", metric_value=float(current_f1))
            )
            db_session.add(
                DriftMetric(
                    metric_name="model_accuracy_score",
                    metric_value=float(current_accuracy),
                )
            )
            db_session.add(
                DriftMetric(
                    metric_name="model_precision_score",
                    metric_value=float(current_precision),
                )
            )
            db_session.add(
                DriftMetric(
                    metric_name="model_recall_score",
                    metric_value=float(current_recall),
                )
            )
        else:
            logger.warning(
                "Dados insuficientes com classes diversas em 'actual_outcome' para calcular"
                "métricas de desempenho significativas. (Pode ser apenas uma classe presente "
                "ou nenhum positivo/predito positivo)."
            )
    else:
        logger.warning("Dados insuficientes com 'actual_outcome' para calcular Concept Drift.")


async def calculate_drift_metrics() -> None:
    """Função principal agendada para calcular e armazenar métricas de drift."""
    logger.info("Iniciando cálculo de métricas de drift...")
    start_time = datetime.now()
    if not SessionLocal:
        logger.error("SessionLocal não inicializado. Não é possível calcular métricas de drift.")
        return
    db_session: Session = SessionLocal()
    try:
        end_date_recent = datetime.utcnow()
        start_date_recent = end_date_recent - timedelta(minutes=5)
        end_date_baseline = start_date_recent
        start_date_baseline = end_date_baseline - timedelta(minutes=10)

        recent_logs = _get_logs_for_period(db_session, start_date_recent, end_date_recent)
        baseline_logs = _get_logs_for_period(db_session, start_date_baseline, end_date_baseline)

        if not recent_logs or not baseline_logs:
            logger.warning(
                "Dados insuficientes para calcular drift. Pelo menos 5 minutos de logs recentes "
                "e 10 minutos de logs de baseline são necessários."
            )
            return

        recent_df = _process_logs_to_dataframe(recent_logs)
        baseline_df = _process_logs_to_dataframe(baseline_logs)

        loaded_common_skills: List[str] = []
        try:
            loaded_common_skills = joblib.load("models/common_skills.pkl")
        except FileNotFoundError:
            logger.warning(
                "common_skills.pkl não encontrado. Não será possível monitorar "
                "drift em habilidades."
            )
        except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
            logger.error("Erro de worker ao carregar common_skills.pkl: %s", e, exc_info=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Erro inesperado ao carregar common_skills.pkl: %s", e, exc_info=True)

        _calculate_data_drift(recent_df, baseline_df, db_session, loaded_common_skills)
        _calculate_concept_drift(recent_df, db_session)

        db_session.commit()
        TOTAL_DRIFT_CHECKS.inc()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        DRIFT_CALCULATION_LATENCY.observe(duration)
        logger.info("Cálculo de métricas de drift concluído em %.4f segundos.", duration)

    except Exception as e:  # pylint: disable=broad-exception-caught
        db_session.rollback()
        logger.error("Erro durante o cálculo de métricas de drift: %s", e, exc_info=True)
    finally:
        db_session.close()


# --- 6. Agendador de Tarefas (APScheduler) ---
scheduler = AsyncIOScheduler()


@app.on_event("startup")
async def startup_event() -> None:
    """Inicia o agendador de tarefas na inicialização da API."""
    scheduler.add_job(calculate_drift_metrics, "interval", minutes=5)
    scheduler.start()
    logger.info("Agendador de cálculo de drift iniciado.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Encerra o agendador de tarefas de forma limpa ao desligar a API."""
    scheduler.shutdown()
    logger.info("Agendador de cálculo de drift desligado.")


# --- 7. Endpoints do Serviço de Drift ---
@app.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest) -> Dict[str, str]:
    """
    Recebe o feedback de contratação para um par candidato-vaga.

    Este endpoint atualiza o registro de log de predição correspondente
    com o resultado real (contratado ou não), que será usado posteriormente
    para calcular o Concept Drift.

    Args:
        feedback: Um objeto FeedbackRequest com os códigos da vaga, do profissional
                  e o status de contratação.
    """
    if not SessionLocal:
        raise HTTPException(status_code=503, detail="Database not configured")
    db_session: Session = SessionLocal()
    try:
        log_entry: Optional[PredictionLog] = (
            db_session.query(PredictionLog)
            .filter(
                PredictionLog.codigo_vaga == feedback.codigo_vaga,
                PredictionLog.codigo_profissional == feedback.codigo_profissional,
            )
            .order_by(PredictionLog.timestamp.desc())
            .first()
        )

        if not log_entry:
            raise HTTPException(
                status_code=404,
                detail="Log de predição não encontrado para o candidato/vaga.",
            )

        log_entry.actual_outcome = feedback.was_hired
        db_session.commit()
        logger.info(
            "Feedback recebido e atualizado para candidato %s na vaga %s. Contratado: %s",
            feedback.codigo_profissional,
            feedback.codigo_vaga,
            feedback.was_hired,
        )
        return {"message": "Feedback recebido e log de predição atualizado com sucesso."}
    except HTTPException:
        raise
    except sqlalchemy.exc.SQLAlchemyError as e:
        db_session.rollback()
        logger.error("Erro ao receber ou atualizar feedback (SQLAlchemy): %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Erro de banco de dados ao processar feedback."
        ) from e
    except Exception as e:
        db_session.rollback()
        logger.error("Erro inesperado ao receber ou atualizar feedback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Erro interno inesperado ao processar feedback."
        ) from e
    finally:
        db_session.close()


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Endpoint para o Prometheus coletar as métricas de drift e de requisição."""
    logger.info("Requisição para o endpoint /metrics do serviço de drift recebida.")
    return PlainTextResponse(generate_latest().decode("utf-8"))


# --- Como Rodar o Serviço de Drift ---
# Para rodar este serviço, salve o código acima como um arquivo Python
# (ex: src/drift_monitor_service/main.py).
#
# Certifique-se de que seus arquivos .pkl (modelos/transformadores) estejam na pasta 'models/'
# no mesmo nível da pasta 'src'.
#
# Instale as bibliotecas necessárias:
# pip install fastapi uvicorn pydantic SQLAlchemy psycopg2-binary pandas scikit-learn
# prometheus_client APScheduler evidently
#
# Abra seu terminal no diretório raiz do projeto e execute:
# uvicorn src.drift_monitor_service.main:app --host 0.0.0.0 --port 8001 --reload
#
# O serviço estará disponível em http://127.0.0.1:8001
# Você pode acessar a documentação interativa em http://127.0.0.1:8001/docs
# As métricas estarão disponíveis em http://127.0.0.1:8001/metrics
