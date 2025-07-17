from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json # Para serializar/desserializar JSON
from prometheus_client import generate_latest, Counter, Histogram, Gauge
from starlette.responses import PlainTextResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler # Para agendamento
import pandas as pd
import joblib # Para carregar objetos de pré-processamento se necessário para drift
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score # Importar mais métricas
import numpy as np # Importar numpy para lidar com tipos numpy

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Configuração do Banco de Dados (SQLAlchemy) ---
# Variáveis globais inicializadas como None. Serão preenchidas no evento de startup.
engine = None
SessionLocal = None
Base = declarative_base()

# Definição do modelo da tabela para logs de predição (compartilhada com a API de predição)
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    codigo_vaga = Column(String, index=True)
    codigo_profissional = Column(String, index=True)
    predicted_probability = Column(Float)
    actual_outcome = Column(Boolean, nullable=True) # Pode ser NULL inicialmente
    input_features_applicant = Column(Text)
    input_features_job = Column(Text)

# Definição do modelo da tabela para métricas de drift (NOVA TABELA)
class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String, index=True)
    metric_value = Column(Float)
    feature_name = Column(String, nullable=True) # Para data drift por feature
    drift_score = Column(Float, nullable=True) # Ex: p-value de um teste KS ou diferença

# --- 2. Definir Modelos Pydantic para Endpoints ---
class FeedbackRequest(BaseModel):
    codigo_vaga: str
    codigo_profissional: str
    was_hired: bool # True se contratado, False caso contrário

class DriftMetricResponse(BaseModel):
    timestamp: datetime
    metric_name: str
    metric_value: float
    feature_name: Optional[str] = None
    drift_score: Optional[float] = None

# --- 3. Definir Métricas Prometheus para Drift ---

REQUEST_COUNT = Counter('drift_http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('drift_http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint', 'status_code'])

DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Data Drift Score for a feature', ['feature_name'])
CONCEPT_DRIFT_SCORE = Gauge('concept_drift_score', 'Concept Drift Score (Model Performance Drop)')
TOTAL_DRIFT_CHECKS = Counter('total_drift_checks', 'Total number of drift calculation runs')
DRIFT_CALCULATION_LATENCY = Histogram('drift_calculation_duration_seconds', 'Duration of drift calculation job')
MODEL_F1_SCORE = Gauge('model_f1_score_production', 'F1-Score of the model in production')
MODEL_ACCURACY_SCORE = Gauge('model_accuracy_score_production', 'Accuracy Score of the model in production')
MODEL_PRECISION_SCORE = Gauge('model_precision_score_production', 'Precision Score of the model in production') # Nova métrica
MODEL_RECALL_SCORE = Gauge('model_recall_score_production', 'Recall Score of the model in production') # Nova métrica

# --- 4. Inicializar a Aplicação FastAPI ---
app = FastAPI(
    title="Serviço de Feedback e Monitoramento de Drift",
    description="Coleta feedback de contratação e calcula métricas de drift do modelo.",
    version="1.0.0",
)

# --- Evento de Startup para Inicializar o Banco de Dados ---
@app.on_event("startup")
def startup_db_connection():
    global engine, SessionLocal # Declara que estamos usando as variáveis globais
    DB_NAME = os.getenv("DB_NAME", "ml_drift_db")
    DB_USER = os.getenv("DB_USER", "ml_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "ml_password_secure")
    DB_HOST = os.getenv("DB_HOST", "db")
    DB_PORT = os.getenv("DB_PORT", "5432")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine) # Cria as tabelas no DB
    logger.info("Conexão com o banco de dados inicializada e tabelas verificadas/criadas.")


# --- Middleware para Métricas de Requisição ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    method = request.method
    endpoint = request.url.path
    status_code = str(response.status_code)

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status_code=status_code).observe(process_time)
    logger.info(f"Requisição {method} {endpoint} com status {status_code} processada em {process_time:.4f} segundos.")
    return response

# --- 5. Lógica de Cálculo de Drift (Função Agendada) ---
async def calculate_drift_metrics():
    logger.info("Iniciando cálculo de métricas de drift...")
    start_time = datetime.now()
    db_session = SessionLocal()
    try:
        # Definir períodos de tempo para dados recentes e baseline
        end_date_recent = datetime.utcnow()
        start_date_recent = end_date_recent - timedelta(minutes=5) # Últimos 5 minutos como "recentes"
        end_date_baseline = start_date_recent
        start_date_baseline = end_date_baseline - timedelta(minutes=10) # 10 minutos antes do período recente como "baseline"

        # Buscar logs de predição
        recent_logs = db_session.query(PredictionLog).filter(
            PredictionLog.timestamp >= start_date_recent,
            PredictionLog.timestamp < end_date_recent
        ).all()

        baseline_logs = db_session.query(PredictionLog).filter(
            PredictionLog.timestamp >= start_date_baseline,
            PredictionLog.timestamp < end_date_baseline
        ).all()

        if not recent_logs or not baseline_logs:
            logger.warning("Dados insuficientes para calcular drift. Pelo menos 5 minutos de logs recentes e 10 minutos de logs de baseline são necessários.")
            return

        # Converter logs para DataFrames para facilitar o processamento
        recent_df = pd.DataFrame([
            {
                'codigo_vaga': log.codigo_vaga,
                'codigo_profissional': log.codigo_profissional,
                'predicted_probability': log.predicted_probability,
                'actual_outcome': log.actual_outcome,
                'input_features_applicant': json.loads(log.input_features_applicant),
                'input_features_job': json.loads(log.input_features_job)
            } for log in recent_logs
        ])

        baseline_df = pd.DataFrame([
            {
                'codigo_vaga': log.codigo_vaga,
                'codigo_profissional': log.codigo_profissional,
                'predicted_probability': log.predicted_probability,
                'actual_outcome': log.actual_outcome,
                'input_features_applicant': json.loads(log.input_features_applicant),
                'input_features_job': json.loads(log.input_features_job)
            } for log in baseline_logs
        ])

        # --- Cálculo de Data Drift (Exemplo Expandido) ---
        categorical_features_to_monitor = [
            'nivel_academico', 'nivel_ingles', 'nivel_espanhol',
            'local', 'pais', 'estado', 'cidade', 'regiao',
            'nivel profissional'
        ]

        try:
            loaded_common_skills = joblib.load('models/common_skills.pkl')
        except FileNotFoundError:
            logger.warning("common_skills.pkl não encontrado. Não será possível monitorar drift em habilidades.")
            loaded_common_skills = []

        def get_nested_feature(data_dict, feature_name):
            if feature_name in data_dict:
                return data_dict[feature_name]
            if feature_name == 'nivel profissional' and 'nivel_profissional' in data_dict:
                return data_dict['nivel_profissional']
            return ''

        recent_features_for_drift = pd.DataFrame()
        baseline_features_for_drift = pd.DataFrame()

        for feature in categorical_features_to_monitor:
            recent_features_for_drift[feature] = recent_df.apply(
                lambda row: get_nested_feature(row['input_features_applicant'], feature) if feature in row['input_features_applicant'] else get_nested_feature(row['input_features_job'], feature), axis=1
            )
            baseline_features_for_drift[feature] = baseline_df.apply(
                lambda row: get_nested_feature(row['input_features_applicant'], feature) if feature in row['input_features_applicant'] else get_nested_feature(row['input_features_job'], feature), axis=1
            )
            recent_features_for_drift[feature] = recent_features_for_drift[feature].astype(str).fillna('')
            baseline_features_for_drift[feature] = baseline_features_for_drift[feature].astype(str).fillna('')

        for feature in categorical_features_to_monitor:
            if feature in recent_features_for_drift.columns and feature in baseline_features_for_drift.columns:
                if not recent_features_for_drift[feature].empty and not baseline_features_for_drift[feature].empty:
                    recent_counts = recent_features_for_drift[feature].value_counts(normalize=True)
                    baseline_counts = baseline_features_for_drift[feature].value_counts(normalize=True)

                    if not baseline_counts.empty:
                        most_common_baseline_cat = baseline_counts.index[0]
                        prop_recent = recent_counts.get(most_common_baseline_cat, 0)
                        prop_baseline = baseline_counts.get(most_common_baseline_cat, 0)
                        drift_value = abs(prop_recent - prop_baseline)

                        DATA_DRIFT_SCORE.labels(feature_name=feature).set(float(drift_value))
                        db_session.add(DriftMetric(metric_name=f'data_drift_{feature}', metric_value=float(drift_value), feature_name=feature))
                        logger.info(f"Data Drift para '{feature}': {drift_value:.4f}")
                    else:
                        logger.warning(f"Baseline vazio para feature '{feature}'. Pulando cálculo de drift para esta feature.")
                else:
                    logger.warning(f"Dados recentes ou baseline vazios para feature '{feature}'. Pulando cálculo de drift para esta feature.")

        if loaded_common_skills:
            recent_applicant_text = recent_df['input_features_applicant'].apply(lambda x: x.get('objetivo_profissional', '') + ' ' + x.get('historico_profissional_texto', '') + ' ' + x.get('cv_completo', ''))
            baseline_applicant_text = baseline_df['input_features_applicant'].apply(lambda x: x.get('objetivo_profissional', '') + ' ' + x.get('historico_profissional_texto', '') + ' ' + x.get('cv_completo', ''))

            key_skill = 'python'
            recent_has_skill = recent_applicant_text.apply(lambda x: 1 if key_skill in x.lower() else 0).mean()
            baseline_has_skill = baseline_applicant_text.apply(lambda x: 1 if key_skill in x.lower() else 0).mean()
            drift_skill_freq = abs(recent_has_skill - baseline_has_skill)
            DATA_DRIFT_SCORE.labels(feature_name=f'skill_{key_skill}').set(float(drift_skill_freq))
            db_session.add(DriftMetric(metric_name=f'data_drift_skill_{key_skill}', metric_value=float(drift_skill_freq), feature_name=f'skill_{key_skill}'))
            logger.info(f"Data Drift para habilidade '{key_skill}': {drift_skill_freq:.4f}")


        # --- Cálculo de Concept Drift (Performance do Modelo) ---
        recent_evaluated_df = recent_df.dropna(subset=['actual_outcome'])
        if not recent_evaluated_df.empty:
            y_true = recent_evaluated_df['actual_outcome'].astype(int)
            y_pred_proba = recent_evaluated_df['predicted_probability']
            y_pred_binary = (y_pred_proba >= 0.7).astype(int)

            # CORREÇÃO: Verificar se há amostras da classe positiva para evitar zero_division
            if len(np.unique(y_true)) > 1 and np.sum(y_true) > 0 and np.sum(y_pred_binary) > 0:
                current_f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                current_accuracy = accuracy_score(y_true, y_pred_binary)
                current_precision = precision_score(y_true, y_pred_binary, zero_division=0) # Nova métrica
                current_recall = recall_score(y_true, y_pred_binary, zero_division=0) # Nova métrica

                BASELINE_F1_SCORE = 0.47
                concept_drift_value = BASELINE_F1_SCORE - current_f1

                CONCEPT_DRIFT_SCORE.set(float(concept_drift_value))
                MODEL_F1_SCORE.set(float(current_f1))
                MODEL_ACCURACY_SCORE.set(float(current_accuracy))
                MODEL_PRECISION_SCORE.set(float(current_precision)) # Definir nova métrica
                MODEL_RECALL_SCORE.set(float(current_recall)) # Definir nova métrica
                logger.info(f"Concept Drift (Queda F1-Score): {concept_drift_value:.4f} (F1 atual: {current_f1:.4f}, Acc atual: {current_accuracy:.4f}, Precision atual: {current_precision:.4f}, Recall atual: {current_recall:.4f})")

                db_session.add(DriftMetric(metric_name='concept_drift_f1_drop', metric_value=float(concept_drift_value)))
                db_session.add(DriftMetric(metric_name='model_f1_score', metric_value=float(current_f1)))
                db_session.add(DriftMetric(metric_name='model_accuracy_score', metric_value=float(current_accuracy)))
                db_session.add(DriftMetric(metric_name='model_precision_score', metric_value=float(current_precision))) # Salvar nova métrica
                db_session.add(DriftMetric(metric_name='model_recall_score', metric_value=float(current_recall))) # Salvar nova métrica
            else:
                logger.warning("Dados insuficientes com classes diversas em 'actual_outcome' para calcular métricas de desempenho significativas. (Pode ser apenas uma classe presente ou nenhum positivo/predito positivo).")
        else:
            logger.warning("Dados insuficientes com 'actual_outcome' para calcular Concept Drift.")


        db_session.commit()
        TOTAL_DRIFT_CHECKS.inc()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        DRIFT_CALCULATION_LATENCY.observe(duration)
        logger.info(f"Cálculo de métricas de drift concluído em {duration:.4f} segundos.")

    except Exception as e:
        db_session.rollback()
        logger.error(f"Erro durante o cálculo de métricas de drift: {e}", exc_info=True)
    finally:
        db_session.close()

# --- 6. Agendador de Tarefas (APScheduler) ---
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    scheduler.add_job(calculate_drift_metrics, 'interval', minutes=5)
    scheduler.start()
    logger.info("Agendador de cálculo de drift iniciado.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    logger.info("Agendador de cálculo de drift desligado.")

# --- 7. Endpoints do Serviço de Drift ---
@app.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest):
    db_session = SessionLocal()
    try:
        log_entry = db_session.query(PredictionLog).filter(
            PredictionLog.codigo_vaga == feedback.codigo_vaga,
            PredictionLog.codigo_profissional == feedback.codigo_profissional
        ).order_by(PredictionLog.timestamp.desc()).first()

        if not log_entry:
            raise HTTPException(status_code=404, detail="Log de predição não encontrado para o candidato/vaga.")

        log_entry.actual_outcome = feedback.was_hired
        db_session.commit()
        logger.info(f"Feedback recebido e atualizado para candidato {feedback.codigo_profissional} na vaga {feedback.codigo_vaga}. Contratado: {feedback.was_hired}")
        return {"message": "Feedback recebido e log de predição atualizado com sucesso."}
    except HTTPException:
        raise
    except Exception as e:
        db_session.rollback()
        logger.error(f"Erro ao receber ou atualizar feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao processar feedback.")
    finally:
        db_session.close()

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    logger.info("Requisição para o endpoint /metrics do serviço de drift recebida.")
    return PlainTextResponse(generate_latest().decode('utf-8'))

# --- Como Rodar o Serviço de Drift ---
# Para rodar este serviço, salve o código acima como um arquivo Python (ex: src/drift_monitor_service/main.py).
#
# Certifique-se de que seus arquivos .pkl (modelos/transformadores) estejam na pasta 'models/'
# no mesmo nível da pasta 'src'.
#
# Instale as bibliotecas necessárias:
# pip install fastapi uvicorn pydantic SQLAlchemy psycopg2-binary pandas scikit-learn prometheus_client APScheduler evidently
#
# Abra seu terminal no diretório raiz do projeto e execute:
# uvicorn src.drift_monitor_service.main:app --host 0.0.0.0 --port 8001 --reload
#
# O serviço estará disponível em http://127.0.0.1:8001
# Você pode acessar a documentação interativa em http://127.0.0.1:8001/docs
# As métricas estarão disponíveis em http://127.0.0.1:8001/metrics
