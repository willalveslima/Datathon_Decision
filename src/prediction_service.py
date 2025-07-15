from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from typing import List, Dict, Any
from prometheus_client import generate_latest, Counter, Histogram, Gauge
from starlette.responses import PlainTextResponse
import time
import logging

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Carregar o Modelo e Objetos de Pré-processamento ---
# Estes objetos são carregados uma única vez quando a API é iniciada.
try:
    # Ajuste o caminho para carregar os arquivos .pkl da pasta 'models/'
    loaded_model = joblib.load('models/logistic_regression_model.pkl')
    loaded_tfidf_applicant = joblib.load('models/tfidf_vectorizer_applicant.pkl')
    loaded_tfidf_job = joblib.load('models/tfidf_vectorizer_job.pkl')
    loaded_categorical_cols = joblib.load('models/categorical_cols.pkl')
    loaded_encoded_feature_names = joblib.load('models/encoded_feature_names.pkl')
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
    'tivemos', 'tiveram', 'todas', 'todo', 'todos', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês', 'vos'
]

# --- 2. Definir Métricas Prometheus ---
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint', 'status_code']) # Também pode adicionar aqui
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction Latency')
PREDICTION_ERRORS_TOTAL = Counter('prediction_errors_total', 'Total Prediction Errors')
APPLICANTS_PER_PREDICTION = Histogram('applicants_per_prediction_request', 'Number of applicants per prediction request')
MODEL_VERSION = Gauge('model_version', 'Version of the deployed model')
SUCCESSFUL_PREDICTIONS_TOTAL = Counter('successful_predictions_total', 'Total Successful Predictions')


# Definir a versão do modelo (exemplo, você pode carregar de um arquivo de configuração)
MODEL_VERSION.set(1.0) # Exemplo: Versão 1.0 do seu modelo

# --- 3. Definir Modelos Pydantic para Validação de Entrada e Saída ---
# Estes modelos garantem que os dados de entrada e saída da API estejam no formato correto.

class ApplicantData(BaseModel):
    # Campos de texto do candidato
    objetivo_profissional: str = ""
    historico_profissional_texto: str = ""
    cv_completo: str = ""
    # Campos categóricos do candidato (apenas 'pcd' agora)
    pcd: str = ""

class JobData(BaseModel):
    # Campos de texto da vaga
    titulo_vaga_prospect: str = ""
    titulo_vaga: str = ""
    principais_atividades: str = ""
    competencia_tecnicas_e_comportamentais: str = ""
    demais_observacoes: str = ""
    areas_atuacao: str = ""
    # Campos categóricos da vaga (atualizado)
    modalidade_vaga_prospect: str = ""
    nivel_profissional: str = "" # Renomeado de 'nivel profissional' para consistência
    nivel_academico: str = ""
    nivel_ingles: str = ""
    nivel_espanhol: str = ""
    faixa_etaria: str = ""
    horario_trabalho: str = ""
    vaga_especifica_para_pcd: str = ""

class PredictionRequest(BaseModel):
    job: JobData
    applicants: List[ApplicantData]

class RankedApplicant(BaseModel):
    applicant_index: int # Ou um ID único do candidato se disponível
    probability: float
    # Opcional: incluir outros dados do candidato para facilitar a visualização
    # Ex: nome: str

class PredictionResponse(BaseModel):
    ranked_applicants: List[RankedApplicant]

# --- 4. Inicializar a Aplicação FastAPI ---
app = FastAPI(
    title="API de Recomendação de Candidatos",
    description="API para prever a probabilidade de contratação de candidatos para vagas e ranqueá-los.",
    version="1.0.0",
)

# --- Middleware para Métricas de Requisição ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    method = request.method
    endpoint = request.url.path
    status_code = str(response.status_code) # Captura o código de status

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status_code=status_code).observe(process_time)
    logger.info(f"Requisição {method} {endpoint} com status {status_code} processada em {process_time:.4f} segundos.")
    return response

# --- 5. Função de Pré-processamento e Previsão para um Único Par Candidato-Vaga ---
def preprocess_and_predict(applicant_data: Dict[str, Any], job_data: Dict[str, Any]) -> float:
    """
    Realiza o pré-processamento e a previsão para um único par candidato-vaga.
    """
    prediction_start_time = time.time() # Início da medição do tempo de predição
    try:
        # Criar DataFrames temporários para aplicar os transformadores
        temp_applicant_df = pd.DataFrame([applicant_data])
        temp_job_df = pd.DataFrame([job_data])

        # Preencher NaNs para as colunas de texto e categóricas, assim como no treinamento
        text_cols_applicants_inference = ['objetivo_profissional', 'historico_profissional_texto', 'cv_completo']
        text_cols_vagas_inference = ['titulo_vaga_prospect', 'titulo_vaga', 'principais_atividades', 'competencia_tecnicas_e_comportamentais', 'demais_observacoes', 'areas_atuacao']

        for col in text_cols_applicants_inference:
            if col in temp_applicant_df.columns:
                temp_applicant_df[col] = temp_applicant_df[col].fillna('')
            else:
                temp_applicant_df[col] = '' # Adiciona a coluna se não existir, preenchendo com vazio

        for col in text_cols_vagas_inference:
            if col in temp_job_df.columns:
                temp_job_df[col] = temp_job_df[col].fillna('')
            else:
                temp_job_df[col] = '' # Adiciona a coluna se não existir, preenchendo com vazio

        # Criar as features de texto combinadas
        temp_applicant_df['applicant_text_features'] = temp_applicant_df['objetivo_profissional'] + ' ' + \
                                                       temp_applicant_df['historico_profissional_texto'] + ' ' + \
                                                       temp_applicant_df['cv_completo']

        temp_job_df['job_text_features'] = temp_job_df['titulo_vaga_prospect'] + ' ' + \
                                            temp_job_df['titulo_vaga'] + ' ' + \
                                            temp_job_df['principais_atividades'] + ' ' + \
                                            temp_job_df['competencia_tecnicas_e_comportamentais'] + ' ' + \
                                            temp_job_df['demais_observacoes'] + ' ' + \
                                            temp_job_df['areas_atuacao']

        # Pré-processar a entrada usando os *mesmos* transformadores TF-IDF
        applicant_tfidf_inference = loaded_tfidf_applicant.transform(temp_applicant_df['applicant_text_features'])
        job_tfidf_inference = loaded_tfidf_job.transform(temp_job_df['job_text_features'])

        # Processar colunas categóricas
        categorical_data_inference = {}
        for col in loaded_categorical_cols:
            if col in temp_applicant_df.columns:
                categorical_data_inference[col] = [temp_applicant_df[col].iloc[0]]
            elif col in temp_job_df.columns:
                categorical_data_inference[col] = [temp_job_df[col].iloc[0]]
            else:
                categorical_data_inference[col] = ['']

        temp_categorical_df = pd.DataFrame(categorical_data_inference)

        for col in temp_categorical_df.columns:
            temp_categorical_df[col] = temp_categorical_df[col].astype(str).fillna('')

        encoded_features_inference = pd.get_dummies(temp_categorical_df[loaded_categorical_cols], dummy_na=False)

        encoded_features_inference = encoded_features_inference.reindex(columns=loaded_encoded_feature_names, fill_value=0)

        # Garante que o DataFrame seja numérico antes da conversão para sparse matrix
        encoded_features_inference = encoded_features_inference.astype(float)
        encoded_features_inference_sparse = csr_matrix(encoded_features_inference)

        # Combinar todas as features para a inferência
        X_inference = hstack([applicant_tfidf_inference, job_tfidf_inference, encoded_features_inference_sparse])

        # Usar o modelo para obter a probabilidade de contratação
        probability = loaded_model.predict_proba(X_inference)[:, 1][0]

        prediction_end_time = time.time()
        PREDICTION_LATENCY.observe(prediction_end_time - prediction_start_time) # Medir tempo de predição
        logger.info(f"Predição concluída em {prediction_end_time - prediction_start_time:.4f} segundos.")
        return probability

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc() # Incrementar contador de erros de predição
        logger.error(f"Erro durante o pré-processamento ou predição: {e}", exc_info=True)
        raise # Re-lança a exceção para ser tratada pelo endpoint

# --- 6. Definir Endpoints da API ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Recebe dados de uma vaga e uma lista de candidatos, e retorna uma lista
    ranqueada de candidatos pela probabilidade de serem contratados para a vaga.
    """
    logger.info(f"Requisição de predição recebida para {len(request.applicants)} candidatos.")
    APPLICANTS_PER_PREDICTION.observe(len(request.applicants)) # Medir número de candidatos na requisição

    results = []
    for i, applicant_data in enumerate(request.applicants):
        try:
            prob = preprocess_and_predict(applicant_data.dict(), request.job.dict())
            results.append(RankedApplicant(applicant_index=i, probability=prob))
        except Exception as e:
            logger.warning(f"Não foi possível processar o candidato índice {i} devido a um erro: {e}. O candidato será ignorado.")
            # O erro já foi incrementado em PREDICTION_ERRORS_TOTAL dentro de preprocess_and_predict
            pass

    # Ordenar os candidatos pela probabilidade em ordem decrescente
    results.sort(key=lambda x: x.probability, reverse=True)

    SUCCESSFUL_PREDICTIONS_TOTAL.inc() # Incrementa o contador de predições bem-sucedidas
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
# pip install fastapi uvicorn pandas scikit-learn joblib pydantic prometheus_client
#
# Abra seu terminal no diretório raiz do projeto (onde está a pasta 'src' e 'models') e execute:
# uvicorn src.prediction_service:app --host 0.0.0.0 --port 8000 --reload
#
# A API estará disponível em http://127.0.0.1:8000
# Você pode acessar a documentação interativa em http://127.0.0.1:8000/docs
# As métricas estarão disponíveis em http://127.0.0.1:8000/metrics
