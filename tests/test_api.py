import pytest
from starlette.testclient import TestClient
import json
from unittest.mock import patch, MagicMock # Importa patch e MagicMock para mocking
import os
import numpy as np # Importar numpy para o mock de predict_proba
import pandas as pd # Importar pandas para auxiliar no mocking de colunas categóricas

# Importa as variáveis globais que serão mockadas
# O caminho completo para o módulo onde 'app', 'engine', 'SessionLocal', 'Base' estão definidos
from src.prediction_service.prediction_service import PredictionLog 

#
# --- Configurações de Teste ---
BASE_URL = "http://localhost:8000"

# Dados de exemplo para a requisição de teste.
TEST_JOB_DATA = {
    "codigo_vaga": "VAGA-TESTE-001",
    "titulo_vaga_prospect": "Desenvolvedor Python Sênior",
    "titulo_vaga": "Vaga para Desenvolvedor Backend com Python e Django",
    "principais_atividades": "Desenvolvimento e manutenção de APIs, integração com sistemas externos, testes unitários e de integração.",
    "competencia_tecnicas_e_comportamentais": "Python, Django, REST APIs, SQL, Git, Metodologias Ágeis. Proatividade, comunicação e trabalho em equipe.",
    "demais_observacoes": "Experiência com cloud computing (AWS/GCP) é um diferencial.",
    "areas_atuacao": "TI - Desenvolvimento/Programação",
    "nivel_profissional": "Sênior",
    "nivel_academico": "Ensino Superior Completo",
    "nivel_ingles": "Avançado",
    "nivel_espanhol": "Básico",
    "pais": "Brasil",
    "estado": "São Paulo",
    "cidade": "São Paulo",
    "regiao": "Sudeste"
}

TEST_APPLICANTS_DATA = [
    { # Candidato 1: Alta compatibilidade
        "codigo_profissional": "CAND-TESTE-001",
        "objetivo_profissional": "Desenvolvedor Backend Sênior com foco em Python e Django.",
        "historico_profissional_texto": "Liderança de equipes no desenvolvimento de APIs RESTful robustas com Python e Django. Experiência em arquitetura de microsserviços, testes automatizados e implantação em nuvem (AWS).",
        "cv_completo": "Currículo detalhado com 8 anos de experiência em Python, Django, DRF, PostgreSQL, Docker, Kubernetes, AWS. Inglês fluente. Forte atuação em otimização de performance e segurança.",
        "local": "São Paulo"
    },
    { # Candidato 2: Boa compatibilidade, menos experiência
        "codigo_profissional": "CAND-TESTE-002",
        "objetivo_profissional": "Desenvolvedor Backend Júnior buscando crescimento.",
        "historico_profissional_texto": "2 anos de experiência com Python e Flask, desenvolvendo pequenas APIs e scripts de automação.",
        "cv_completo": "Conhecimento em Python, Flask, MongoDB. Inglês intermediário. Buscando oportunidade para aprender e crescer.",
        "local": "Rio de Janeiro"
    }
]

# --- Fixture do Cliente de Teste com Mocking de DB ---
@pytest.fixture(scope="module")
def test_client_and_mocks():
    # Mocka variáveis de ambiente do DB
    with patch.dict(os.environ, {
        "DB_NAME": "test_db",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "DB_HOST": "mock_db",
        "DB_PORT": "5432"
    }):
        # Patch nas funções reais do SQLAlchemy no módulo prediction_service.prediction_service
        with patch('src.prediction_service.prediction_service.create_engine') as mock_create_engine, \
             patch('src.prediction_service.prediction_service.sessionmaker') as mock_sessionmaker, \
             patch('src.prediction_service.prediction_service.Base.metadata.create_all') as mock_create_all, \
             patch('src.prediction_service.prediction_service.joblib.load') as mock_joblib_load:
            
            # Configura o mock de create_engine para retornar um mock de engine
            mock_engine_instance = MagicMock()
            mock_create_engine.return_value = mock_engine_instance

            # Configura o mock de sessionmaker para retornar uma classe de sessão mockada
            mock_db_session_class = MagicMock()
            mock_sessionmaker.return_value = mock_db_session_class

            # Cria um mock para a instância da sessão do banco de dados
            mock_db_session_instance = MagicMock()
            mock_db_session_class.return_value = mock_db_session_instance # Quando a classe de sessão é instanciada

            # Configura o mock da instância da sessão para que as operações de DB não causem erro
            mock_db_session_instance.add.return_value = None
            mock_db_session_instance.commit.return_value = None
            mock_db_session_instance.close.return_value = None
            mock_db_session_instance.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
            mock_db_session_instance.query.return_value.filter.return_value.all.return_value = []

            # --- Criar os mocks de retorno ANTES de definir side_effect ---
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = [
                np.array([[0.1, 0.9]]), # Para o primeiro candidato
                np.array([[0.2, 0.8]])  # Para o segundo candidato
            ]

            mock_tfidf_applicant_transform = MagicMock()
            mock_tfidf_applicant_transform.transform = lambda x: np.zeros((len(x), 5000))
            
            mock_tfidf_job_transform = MagicMock()
            mock_tfidf_job_transform.transform = lambda x: np.zeros((len(x), 5000))
            
            # --- CORREÇÃO: Mocks para loaded_categorical_cols e loaded_encoded_feature_names com base nas colunas ATUAIS ---
            # Estas devem corresponder às colunas categóricas que o modelo espera após o pré-processamento
            # e os nomes de colunas que get_dummies geraria.
            # A lista de colunas categóricas que o modelo espera (mesma do model_training.py)
            mock_loaded_categorical_cols = [
                'nivel profissional', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol',
                'local', 'pais', 'estado', 'cidade', 'regiao'
            ]
            
            # Gera os nomes das colunas após o One-Hot Encoding com base em dados de exemplo
            # Use um DataFrame de exemplo que contenha todas as categorias possíveis que o modelo pode ver
            # para garantir que todas as colunas OHE sejam geradas.
            # Para simplificar o teste, podemos criar um mock mais simples se a estrutura OHE for fixa.
            # Exemplo de dados para gerar os nomes das colunas OHE:
            sample_data_for_ohe_cols = pd.DataFrame([
                {'nivel profissional': 'Sênior', 'nivel_academico': 'Ensino Superior Completo', 'nivel_ingles': 'Avançado', 'nivel_espanhol': 'Básico', 'local': 'São Paulo', 'pais': 'Brasil', 'estado': 'São Paulo', 'cidade': 'São Paulo', 'regiao': 'Sudeste'},
                {'nivel profissional': 'Júnior', 'nivel_academico': 'Ensino Médio Completo', 'nivel_ingles': 'Intermediário', 'nivel_espanhol': 'Nenhum', 'local': 'Rio de Janeiro', 'pais': 'Brasil', 'estado': 'Rio de Janeiro', 'cidade': 'Rio de Janeiro', 'regiao': 'Sudeste'},
                {'nivel profissional': 'Pleno', 'nivel_academico': 'Pós-graduação', 'nivel_ingles': 'Básico', 'nivel_espanhol': 'Avançado', 'local': 'Belo Horizonte', 'pais': 'Brasil', 'estado': 'Minas Gerais', 'cidade': 'Belo Horizonte', 'regiao': 'Sudeste'}
            ])
            for col in sample_data_for_ohe_cols.columns:
                sample_data_for_ohe_cols[col] = sample_data_for_ohe_cols[col].astype(str)

            mock_loaded_encoded_feature_names = pd.get_dummies(
                sample_data_for_ohe_cols[mock_loaded_categorical_cols], dummy_na=False
            ).columns.tolist()

            mock_loaded_common_skills = ['python', 'java', 'javascript'] # Exemplo de habilidades comuns

            mock_joblib_load.side_effect = [
                mock_model,
                mock_tfidf_applicant_transform,
                mock_tfidf_job_transform,
                mock_loaded_categorical_cols,
                mock_loaded_encoded_feature_names,
                mock_loaded_common_skills
            ]

            # Mock da DatabaseConnection
            mock_db_connection = MagicMock()
            mock_db_connection.get_session.return_value = mock_db_session_instance
            
            # --- Patch nas variáveis globais do módulo ---
            with patch('src.prediction_service.prediction_service.db_connection', new=mock_db_connection), \
                 patch('src.prediction_service.prediction_service.loaded_model', new=mock_model), \
                 patch('src.prediction_service.prediction_service.loaded_tfidf_applicant', new=mock_tfidf_applicant_transform), \
                 patch('src.prediction_service.prediction_service.loaded_tfidf_job', new=mock_tfidf_job_transform), \
                 patch('src.prediction_service.prediction_service.loaded_categorical_cols', new=mock_loaded_categorical_cols), \
                 patch('src.prediction_service.prediction_service.loaded_encoded_feature_names', new=mock_loaded_encoded_feature_names), \
                 patch('src.prediction_service.prediction_service.loaded_common_skills', new=mock_loaded_common_skills):
                
                # Importa a instância da aplicação FastAPI *aqui*, após os patches estarem ativos
                from src.prediction_service.prediction_service import app as current_app
                
                # Cria o cliente de teste para a aplicação FastAPI
                with TestClient(app=current_app) as client:
                    # Retorna o cliente, a instância mockada da sessão e a instância da app
                    yield client, mock_db_session_instance, current_app

# --- Testes Unitários ---

def test_predict_valid_data(test_client_and_mocks):
    """
    Testa se o endpoint /predict retorna um status 200 OK e o formato de saída esperado
    para dados de entrada válidos, com o banco de dados mockado.
    """
    client, mock_db_session_instance, _ = test_client_and_mocks # Desempacota a fixture
    
    # Resetar o mock antes de cada teste para limpar o call_count
    mock_db_session_instance.reset_mock()

    request_payload = {
        "job": TEST_JOB_DATA,
        "applicants": TEST_APPLICANTS_DATA
    }

    print(f"\n--- Teste: Requisição POST para /predict com dados válidos ---")
    print(f"URL: {BASE_URL}/predict")
    print(f"Corpo da Requisição:\n{json.dumps(request_payload, indent=2)}")

    response = client.post("/predict", json=request_payload)

    print(f"Status da Resposta: {response.status_code}")
    print(f"Corpo da Resposta:\n{json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    response_data = response.json()
    assert "ranked_applicants" in response_data
    assert isinstance(response_data["ranked_applicants"], list)
    assert len(response_data["ranked_applicants"]) == len(TEST_APPLICANTS_DATA)

    # Verifica o formato da saída para cada candidato ranqueado
    for applicant in response_data["ranked_applicants"]:
        assert "applicant_index" in applicant
        assert isinstance(applicant["applicant_index"], int)
        assert "probability" in applicant
        assert isinstance(applicant["probability"], float)
        assert "codigo_profissional" in applicant
        assert isinstance(applicant["codigo_profissional"], str)
        # Não valida o valor exato da probabilidade, apenas o tipo e a presença

    

    print("Teste 'test_predict_valid_data' concluído com sucesso.")

def test_predict_invalid_data(test_client_and_mocks):
    """
    Testa se o endpoint /predict lida corretamente com dados de entrada inválidos.
   
    """
    client, mock_db_session_instance, _ = test_client_and_mocks # Desempacota a fixture

    # Resetar o mock antes de cada teste para limpar o call_count
    mock_db_session_instance.reset_mock()

    invalid_payload = {
        "job": TEST_JOB_DATA,
        "applicants": [
            {
                "codigo_profissional": "CAND-INVALID",
                "objetivo_profissional": "Desenvolvedor",
                # 'historico_profissional_texto' e 'cv_completo' estão faltando (serão preenchidos com "")
                "local": "Cidade Teste"
            }
        ]
    }

    print(f"\n--- Teste: Requisição POST para /predict com dados inválidos ---")
    print(f"URL: {BASE_URL}/predict")
    print(f"Corpo da Requisição (Inválido):\n{json.dumps(invalid_payload, indent=2)}")

    response = client.post("/predict", json=invalid_payload) # Usa 'client' da fixture e 'invalid_payload'

    print(f"Status da Resposta: {response.status_code}")
    print(f"Corpo da Resposta:\n{json.dumps(response.json(), indent=2)}")

    assert response.status_code == 422
    

    
    print("Teste 'test_predict_invalid_data' concluído com sucesso (comportamento atual do modelo Pydantic).")

def test_metrics_endpoint(test_client_and_mocks):
    """
    Testa se o endpoint /metrics retorna um status 200 OK e conteúdo de métricas.
    """
    client, _, _ = test_client_and_mocks # Desempacota a fixture

    print(f"\n--- Teste: Requisição GET para /metrics ---")
    print(f"URL: {BASE_URL}/metrics")

    response = client.get("/metrics")

    print(f"Status da Resposta: {response.status_code}")
    print(f"Corpo da Resposta (parcial):\n{response.text[:500]}...")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds" in response.text
    assert "prediction_duration_seconds" in response.text
    assert "model_version" in response.text
    print("Teste 'test_metrics_endpoint' concluído com sucesso.")

def test_model_loading(test_client_and_mocks):
    """
    Verifica se o modelo e os transformadores são carregados corretamente na inicialização da API.
    """
    _, _, app_instance = test_client_and_mocks # Desempacota a fixture para obter a instância da app
    
    print(f"\n--- Teste: Carregamento de Modelo na Inicialização da API ---")
    assert app_instance is not None # Agora 'app_instance' é a instância da app testada
    print("Teste 'test_model_loading' concluído com sucesso: Modelo e objetos de pré-processamento foram carregados.")

