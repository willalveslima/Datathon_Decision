from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from typing import List, Dict, Any, Optional
from prometheus_client import generate_latest, Counter, Histogram, Gauge
from starlette.responses import PlainTextResponse
import time
import logging
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json # Para serializar/desserializar JSON
import numpy as np # Importar numpy para lidar com tipos numpy

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Carregar o Modelo e Objetos de Pré-processamento ---
# Estes objetos são carregados uma única vez quando a API é iniciada.
try:
    loaded_model = joblib.load('models/logistic_regression_model.pkl')
    loaded_tfidf_applicant = joblib.load('models/tfidf_vectorizer_applicant.pkl')
    loaded_tfidf_job = joblib.load('models/tfidf_vectorizer_job.pkl')
    loaded_categorical_cols = joblib.load('models/categorical_cols.pkl')
    loaded_encoded_feature_names = joblib.load('models/encoded_feature_names.pkl')
    loaded_common_skills = joblib.load('models/common_skills.pkl')
    logger.info("Modelo e objetos de pré-processamento carregados com sucesso.")
except FileNotFoundError as e:
    logger.error(f"Erro ao carregar arquivos do modelo. Certifique-se de que os arquivos .pkl estão na pasta 'models/'. Erro: {e}")
    raise RuntimeError(f"Erro ao carregar arquivos do modelo. Certifique-se de que os arquivos .pkl estão na pasta 'models/'. Erro: {e}")
except Exception as e:
    logger.error(f"Erro inesperado ao carregar o modelo ou objetos de pré-processamento: {e}")
    raise RuntimeError(f"Erro inesperado ao carregar o modelo ou objetos de pré-processamento: {e}")

# Lista de stop words comuns em português (deve ser a mesma usada no treinamento)
portuguese_stop_words = [
    'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'às', 'até', 'com', 'como', 'da', 'das',
    'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'entre',
    'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'estas', 'este', 'estes', 'estou', 'está', 'estão', 'eu',
    'foi', 'fomos', 'for', 'fora', 'foram', 'fosse', 'fossem', 'fui', 'há', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mais',
    'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'nas', 'nem', 'no', 'nos', 'nossa', 'nossas',
    'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'porque', 'qual',
    'quando', 'que', 'quem', 'se', 'sem', 'ser', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam', 'seu', 'seus',
    'só', 'somos', 'sou', 'sua', 'suas', 'também', 'te', 'tem', 'tém', 'temos', 'tenho', 'terá', 'terão', 'terei',
    'teremos', 'teria', 'teriam', 'teu', 'teus', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tivesse', 'tivessem',
    'todas', 'todo', 'todos', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês', 'vos'
]

# --- 2. Configuração do Banco de Dados (SQLAlchemy) ---
# Variáveis globais inicializadas como None. Serão preenchidas no evento de startup.
engine = None
SessionLocal = None
Base = declarative_base()

# Definição do modelo da tabela para logs de predição
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    codigo_vaga = Column(String, index=True)
    codigo_profissional = Column(String, index=True)
    predicted_probability = Column(Float)
    actual_outcome = Column(Boolean, nullable=True)
    input_features_applicant = Column(Text)
    input_features_job = Column(Text)

# --- 3. Definir Métricas Prometheus ---
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint', 'status_code'])
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction Latency')
PREDICTION_ERRORS_TOTAL = Counter('prediction_errors_total', 'Total Prediction Errors')
APPLICANTS_PER_PREDICTION = Histogram('applicants_per_prediction_request', 'Number of applicants per prediction request')
MODEL_VERSION = Gauge('model_version', 'Version of the deployed model')
SUCCESSFUL_PREDICTIONS_TOTAL = Counter('successful_predictions_total', 'Total Successful Predictions')

MODEL_VERSION.set(1.0)

# --- 4. Definir Modelos Pydantic para Validação de Entrada e Saída ---
class ApplicantData(BaseModel):
    codigo_profissional: str # Campo obrigatório
    objetivo_profissional: str # Campo obrigatório
    historico_profissional_texto: str # Campo obrigatório
    cv_completo: str # Campo obrigatório
    local: str # Campo obrigatório

class JobData(BaseModel):
    codigo_vaga: str # Campo obrigatório
    titulo_vaga_prospect: str # Campo obrigatório
    titulo_vaga: str # Campo obrigatório
    principais_atividades: str # Campo obrigatório
    competencia_tecnicas_e_comportamentais: str # Campo obrigatório
    demais_observacoes: str # Campo obrigatório
    areas_atuacao: str # Campo obrigatório
    nivel_profissional: str # Campo obrigatório
    nivel_academico: str # Campo obrigatório
    nivel_ingles: str # Campo obrigatório
    nivel_espanhol: str # Campo obrigatório
    pais: str # Campo obrigatório
    estado: str # Campo obrigatório
    cidade: str # Campo obrigatório
    regiao: str # Campo obrigatório

class PredictionRequest(BaseModel):
    job: JobData
    applicants: List[ApplicantData]

class RankedApplicant(BaseModel):
    applicant_index: int
    probability: float
    codigo_profissional: str

class PredictionResponse(BaseModel):
    ranked_applicants: List[RankedApplicant]

# --- 5. Inicializar a Aplicação FastAPI ---
app = FastAPI(
    title="API de Recomendação de Candidatos",
    description="API para prever a probabilidade de contratação de candidatos para vagas e ranqueá-los.",
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
    Base.metadata.create_all(bind=engine) # Cria a tabela no DB
    logger.info("Conexão com o banco de dados inicializada e tabela 'prediction_logs' verificada/criada.")


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

# --- Função para extrair habilidades (replicada do treinamento) ---
def extract_skills_from_text(text, skills_list):
    found_skills = []
    text_lower = str(text).lower()
    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

# --- 6. Função de Pré-processamento e Previsão para um Único Par Candidato-Vaga ---
def preprocess_and_predict(applicant_data: Dict[str, Any], job_data: Dict[str, Any]) -> float:
    prediction_start_time = time.time()
    try:
        temp_applicant_df = pd.DataFrame([applicant_data])
        temp_job_df = pd.DataFrame([job_data])

        text_cols_applicants_inference = ['objetivo_profissional', 'historico_profissional_texto', 'cv_completo']
        text_cols_vagas_inference = ['titulo_vaga_prospect', 'titulo_vaga', 'principais_atividades', 'competencia_tecnicas_e_comportamentais', 'demais_observacoes', 'areas_atuacao']

        for col in text_cols_applicants_inference:
            if col in temp_applicant_df.columns:
                temp_applicant_df[col] = temp_applicant_df[col].fillna('')
            else:
                temp_applicant_df[col] = ''

        for col in text_cols_vagas_inference:
            if col in temp_job_df.columns:
                temp_job_df[col] = temp_job_df[col].fillna('')
            else:
                temp_job_df[col] = ''

        temp_applicant_df['applicant_text_features'] = temp_applicant_df['objetivo_profissional'] + ' ' + \
                                                       temp_applicant_df['historico_profissional_texto'] + ' ' + \
                                                       temp_applicant_df['cv_completo']

        temp_job_df['job_text_features'] = temp_job_df['titulo_vaga_prospect'] + ' ' + \
                                            temp_job_df['titulo_vaga'] + ' ' + \
                                            temp_job_df['principais_atividades'] + ' ' + \
                                            temp_job_df['competencia_tecnicas_e_comportamentais'] + ' ' + \
                                            temp_job_df['demais_observacoes'] + ' ' + \
                                            temp_job_df['areas_atuacao']

        applicant_tfidf_inference = loaded_tfidf_applicant.transform(temp_applicant_df['applicant_text_features'])
        job_tfidf_inference = loaded_tfidf_job.transform(temp_job_df['job_text_features'])

        # --- Processar Colunas Categóricas e de Localização ---
        # ATUALIZADO: Lista de colunas categóricas que o modelo espera (mesma do model_training.py)
        categorical_cols_for_inference = [
            'nivel profissional',
            'nivel_academico',
            'nivel_ingles',
            'nivel_espanhol',
            'local',
            'pais',
            'estado',
            'cidade',
            'regiao'
        ]

        categorical_data_inference = {}
        for col in categorical_cols_for_inference:
            # CORREÇÃO: Acessar diretamente o valor do dicionário applicant_data ou job_data
            # e garantir que a chave exista antes de tentar acessá-la.
            # Se o campo for obrigatório no Pydantic, ele estará presente, mas se for opcional
            # e não enviado, pode não estar.
            if col in applicant_data: # Verifica se a chave existe no dicionário do candidato
                categorical_data_inference[col] = [applicant_data[col]]
            elif col in job_data: # Verifica se a chave existe no dicionário da vaga
                if col == 'nivel profissional' and 'nivel_profissional' in job_data:
                    categorical_data_inference[col] = [job_data['nivel_profissional']]
                else:
                    categorical_data_inference[col] = [job_data[col]]
            else:
                # Se um campo obrigatório não foi fornecido (o que Pydantic já pegaria),
                # ou um campo opcional não foi fornecido, preencher com string vazia
                # para evitar erros na transformação.
                categorical_data_inference[col] = ['']

        temp_categorical_df = pd.DataFrame(categorical_data_inference)

        for col in temp_categorical_df.columns:
            temp_categorical_df[col] = temp_categorical_df[col].astype(str).fillna('')

        encoded_features_inference = pd.get_dummies(temp_categorical_df[loaded_categorical_cols], dummy_na=False)
        encoded_features_inference = encoded_features_inference.reindex(columns=loaded_encoded_feature_names, fill_value=0)
        encoded_features_inference = encoded_features_inference.astype(float)
        encoded_features_inference_sparse = csr_matrix(encoded_features_inference)

        # --- Processar Features de Habilidades ---
        # Certifique-se de que os dicionários de dados de entrada tenham as chaves esperadas para 'applicant_text_features'/'job_text_features'
        # para que extract_skills_from_text não falhe.
        applicant_text_for_skills = applicant_data.get('objetivo_profissional', '') + ' ' + \
                                    applicant_data.get('historico_profissional_texto', '') + ' ' + \
                                    applicant_data.get('cv_completo', '')

        job_text_for_skills = job_data.get('titulo_vaga_prospect', '') + ' ' + \
                              job_data.get('titulo_vaga', '') + ' ' + \
                              job_data.get('principais_atividades', '') + ' ' + \
                              job_data.get('competencia_tecnicas_e_comportamentais', '') + ' ' + \
                              job_data.get('demais_observacoes', '') + ' ' + \
                              job_data.get('areas_atuacao', '')

        applicant_skills = extract_skills_from_text(applicant_text_for_skills, loaded_common_skills)
        job_skills = extract_skills_from_text(job_text_for_skills, loaded_common_skills)

        skill_features_data = {}
        for skill in loaded_common_skills:
            skill_features_data[f'applicant_has_{skill}'] = [1 if skill in applicant_skills else 0]
            skill_features_data[f'job_requires_{skill}'] = [1 if skill in job_skills else 0]
        
        skill_features_data['common_skills_count'] = [len(set(applicant_skills).intersection(set(job_skills)))]

        skill_features_df_inference = pd.DataFrame(skill_features_data)
        skill_features_sparse_inference = csr_matrix(skill_features_df_inference.values)

        # Combinar todas as features para a inferência
        X_inference = hstack([applicant_tfidf_inference, job_tfidf_inference, encoded_features_inference_sparse, skill_features_sparse_inference])

        probability = loaded_model.predict_proba(X_inference)[:, 1][0]

        prediction_end_time = time.time()
        PREDICTION_LATENCY.observe(prediction_end_time - prediction_start_time)
        logger.info(f"Predição concluída em {prediction_end_time - prediction_start_time:.4f} segundos.")
        return probability

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.error(f"Erro durante o pré-processamento ou predição: {e}", exc_info=True)
        raise

# --- 7. Definir Endpoints da API ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Requisição de predição recebida para {len(request.applicants)} candidatos e vaga {request.job.codigo_vaga}.")
    APPLICANTS_PER_PREDICTION.observe(len(request.applicants))

    results = []
    # Acessa SessionLocal via global scope
    global SessionLocal
    if SessionLocal is None:
        logger.error("SessionLocal não inicializado. O banco de dados pode não ter sido configurado corretamente no startup.")
        raise HTTPException(status_code=500, detail="Serviço não inicializado corretamente (DB).")

    db_session = SessionLocal()
    try:
        for i, applicant_data in enumerate(request.applicants):
            try:
                # O Pydantic já validou a entrada. Se houver um erro aqui, é lógico/de processamento.
                prob = preprocess_and_predict(applicant_data.dict(), request.job.dict())
                
                results.append(RankedApplicant(
                    applicant_index=i,
                    probability=prob,
                    codigo_profissional=applicant_data.codigo_profissional
                ))

                log_entry = PredictionLog(
                    codigo_vaga=request.job.codigo_vaga,
                    codigo_profissional=applicant_data.codigo_profissional,
                    predicted_probability=float(prob), # Garante que é um float nativo do Python
                    actual_outcome=None,
                    input_features_applicant=json.dumps(applicant_data.dict()),
                    input_features_job=json.dumps(request.job.dict())
                )
                db_session.add(log_entry)
                logger.info(f"Log de predição salvo para candidato {applicant_data.codigo_profissional} na vaga {request.job.codigo_vaga}.")

            except Exception as e:
                # Se preprocess_and_predict levantar um erro, ele será capturado aqui.
                # Não é um erro 4xx de validação Pydantic, mas um erro de processamento.
                logger.warning(f"Não foi possível processar ou logar o candidato índice {i} (código: {applicant_data.codigo_profissional}) devido a um erro: {e}. O candidato será ignorado.")
                pass # O candidato é ignorado, e a requisição continua para os outros.
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        logger.error(f"Erro ao salvar logs de predição no banco de dados: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao processar e salvar predições.")
    finally:
        db_session.close()

    results.sort(key=lambda x: x.probability, reverse=True)

    SUCCESSFUL_PREDICTIONS_TOTAL.inc()
    logger.info(f"Predição concluída com sucesso para {len(results)} candidatos ranqueados.")
    return PredictionResponse(ranked_applicants=results)

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Endpoint para o Prometheus coletar as métricas.
    """
    logger.info("Requisição para o endpoint /metrics recebida.")
    return PlainTextResponse(generate_latest().decode('utf-8'))

# --- Como Rodar a API ---
# Para rodar esta API, salve o código acima como um arquivo Python (ex: src/prediction_service.py).
#
# Certifique-se de que seus arquivos .pkl (modelo e transformadores) estejam na pasta 'models/'
# no mesmo nível da pasta 'src'.
#
# Instale as bibliotecas necessárias:
# pip install fastapi uvicorn pandas scikit-learn joblib pydantic prometheus_client psycopg2-binary SQLAlchemy
#
# Abra seu terminal no diretório raiz do projeto (onde está a pasta 'src' e 'models') e execute:
# uvicorn src.prediction_service:app --host 0.0.0.0 --port 8000 --reload
#
# A API estará disponível em http://127.0.0.1:8000
# Você pode acessar a documentação interativa em http://127.0.0.1:8000/docs
# As métricas estarão disponíveis em http://127.0.0.1:8000/metrics
