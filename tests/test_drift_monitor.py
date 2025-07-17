import pytest
from starlette.testclient import TestClient
import json
from unittest.mock import patch, MagicMock, AsyncMock
import os
from datetime import datetime, timedelta

# Importa a instância da aplicação FastAPI e os componentes de DB do serviço de drift
# Importa as variáveis globais que serão mockadas e a função real
from src.drift_monitor_service.main import app, PredictionLog, DriftMetric, calculate_drift_metrics as actual_calculate_drift_metrics

# Importa o módulo core do prometheus_client para acessar o registro padrão
import prometheus_client.core


# --- Dados de Teste ---
# ATUALIZADO: Removidos os campos 'pcd', 'modalidade_vaga_prospect', 'faixa_etaria', 'horario_trabalho', 'vaga_especifica_para_pcd'
TEST_PREDICTION_LOG_ENTRY = {
    "id": 1,
    "timestamp": datetime.utcnow().isoformat(),
    "codigo_vaga": "VAGA-TESTE-DRIFT-001",
    "codigo_profissional": "CAND-TESTE-DRIFT-001",
    "predicted_probability": 0.85,
    "actual_outcome": None,
    "input_features_applicant": json.dumps({"objetivo_profissional": "Dev Python", "local": "São Paulo"}),
    "input_features_job": json.dumps({
        "nivel_ingles": "Avançado", "areas_atuacao": "TI", "nivel_profissional": "Sênior",
        "nivel_academico": "Ensino Superior Completo", "nivel_espanhol": "Básico",
        "pais": "Brasil", "estado": "São Paulo", "cidade": "São Paulo", "regiao": "Sudeste"
    })
}

TEST_PREDICTION_LOG_ENTRY_HIRED = {
    "id": 2,
    "timestamp": (datetime.utcnow() - timedelta(days=5)).isoformat(),
    "codigo_vaga": "VAGA-TESTE-DRIFT-002",
    "codigo_profissional": "CAND-TESTE-DRIFT-002",
    "predicted_probability": 0.90,
    "actual_outcome": True,
    "input_features_applicant": json.dumps({"objetivo_profissional": "Dev Java", "local": "Rio de Janeiro"}),
    "input_features_job": json.dumps({
        "nivel_ingles": "Intermediário", "areas_atuacao": "TI", "nivel_profissional": "Júnior",
        "nivel_academico": "Ensino Superior Incompleto", "nivel_espanhol": "Intermediário",
        "pais": "Brasil", "estado": "Rio de Janeiro", "cidade": "Rio de Janeiro", "regiao": "Sudeste"
    })
}

TEST_PREDICTION_LOG_ENTRY_NOT_HIRED = {
    "id": 3,
    "timestamp": (datetime.utcnow() - timedelta(days=10)).isoformat(),
    "codigo_vaga": "VAGA-TESTE-DRIFT-003",
    "codigo_profissional": "CAND-TESTE-DRIFT-003",
    "predicted_probability": 0.10,
    "actual_outcome": False,
    "input_features_applicant": json.dumps({"objetivo_profissional": "Analista", "local": "Belo Horizonte"}),
    "input_features_job": json.dumps({
        "nivel_ingles": "Básico", "areas_atuacao": "RH", "nivel_profissional": "Pleno",
        "nivel_academico": "Ensino Médio Completo", "nivel_espanhol": "Nenhum",
        "pais": "Brasil", "estado": "Minas Gerais", "cidade": "Belo Horizonte", "regiao": "Sudeste"
    })
}


# --- Fixture do Cliente de Teste com Mocking de DB e Agendador ---
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
        # Patch nas funções reais do SQLAlchemy no módulo drift_monitor_service.main
        # REMOVIDO: patch('src.drift_monitor_service.main.os.getenv', ...)
        with patch('src.drift_monitor_service.main.create_engine') as mock_create_engine, \
             patch('src.drift_monitor_service.main.sessionmaker') as mock_sessionmaker, \
             patch('src.drift_monitor_service.main.Base.metadata.create_all') as mock_create_all, \
             patch('src.drift_monitor_service.main.joblib.load') as mock_joblib_load:
            
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

            # Mocka o agendador para que as tarefas não rodem automaticamente durante o teste
            with patch('src.drift_monitor_service.main.AsyncIOScheduler') as MockScheduler:
                mock_scheduler_instance = MockScheduler.return_value
                mock_scheduler_instance.add_job.return_value = None
                mock_scheduler_instance.start.return_value = None
                mock_scheduler_instance.shutdown.return_value = None

                # Mocka joblib.load para common_skills
                mock_joblib_load.return_value = ['python', 'java'] # Lista de habilidades para teste

                # --- Patch nas variáveis globais 'engine' e 'SessionLocal' no módulo ---
                # Isso garante que 'engine' e 'SessionLocal' no módulo sejam mocks
                with patch('src.drift_monitor_service.main.engine', new=mock_engine_instance), \
                     patch('src.drift_monitor_service.main.SessionLocal', new=mock_db_session_class):
                    
                    # Cria o cliente de teste para a aplicação FastAPI
                    with TestClient(app=app) as client:
                        yield client, mock_db_session_instance, app # Retorna o cliente, a instância mockada da sessão e a app

# --- Testes Unitários para o Endpoint /feedback ---

def test_receive_feedback_success(test_client_and_mocks):
    client, mock_db_session_instance, _ = test_client_and_mocks
    
    # Resetar o mock antes de cada teste para limpar o call_count
    mock_db_session_instance.reset_mock()

    mock_log_entry = MagicMock(spec=PredictionLog)
    mock_log_entry.codigo_vaga = TEST_PREDICTION_LOG_ENTRY["codigo_vaga"]
    mock_log_entry.codigo_profissional = TEST_PREDICTION_LOG_ENTRY["codigo_profissional"]
    mock_log_entry.actual_outcome = None

    mock_db_session_instance.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_log_entry

    feedback_payload = {
        "codigo_vaga": TEST_PREDICTION_LOG_ENTRY["codigo_vaga"],
        "codigo_profissional": TEST_PREDICTION_LOG_ENTRY["codigo_profissional"],
        "was_hired": True
    }

    print(f"\n--- Teste: Requisição POST para /feedback (Sucesso) ---")
    print(f"Corpo da Requisição:\n{json.dumps(feedback_payload, indent=2)}")

    response = client.post("/feedback", json=feedback_payload)

    print(f"Status da Resposta: {response.status_code}")
    print(f"Corpo da Resposta:\n{json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["message"] == "Feedback recebido e log de predição atualizado com sucesso."
    assert mock_log_entry.actual_outcome is True
    mock_db_session_instance.commit.assert_called_once()
    mock_db_session_instance.close.assert_called_once()
    print("Teste 'test_receive_feedback_success' concluído com sucesso.")


def test_receive_feedback_not_found(test_client_and_mocks):
    client, mock_db_session_instance, _ = test_client_and_mocks
    
    # Resetar o mock antes de cada teste para limpar o call_count
    mock_db_session_instance.reset_mock()

    mock_db_session_instance.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    feedback_payload = {
        "codigo_vaga": "VAGA-NAO-EXISTE",
        "codigo_profissional": "CAND-NAO-EXISTE",
        "was_hired": False
    }

    print(f"\n--- Teste: Requisição POST para /feedback (Não Encontrado) ---")
    response = client.post("/feedback", json=feedback_payload)

    print(f"Status da Resposta: {response.status_code}")
    print(f"Corpo da Resposta:\n{json.dumps(response.json(), indent=2)}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Log de predição não encontrado para o candidato/vaga."
    # Remover a asserção de rollback, pois não é esperado para 404
    # mock_db_session_instance.rollback.assert_called_once() # Removido
    mock_db_session_instance.close.assert_called_once()
    print("Teste 'test_receive_feedback_not_found' concluído com sucesso.")

def test_receive_feedback_db_error(test_client_and_mocks):
    client, mock_db_session_instance, _ = test_client_and_mocks
    
    # Resetar o mock antes de cada teste para limpar o call_count
    mock_db_session_instance.reset_mock()

    mock_log_entry = MagicMock(spec=PredictionLog)
    mock_log_entry.codigo_vaga = TEST_PREDICTION_LOG_ENTRY["codigo_vaga"]
    mock_log_entry.codigo_profissional = TEST_PREDICTION_LOG_ENTRY["codigo_profissional"]
    mock_log_entry.actual_outcome = None

    mock_db_session_instance.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_log_entry
    mock_db_session_instance.commit.side_effect = Exception("Erro simulado no DB")

    feedback_payload = {
        "codigo_vaga": TEST_PREDICTION_LOG_ENTRY["codigo_vaga"],
        "codigo_profissional": TEST_PREDICTION_LOG_ENTRY["codigo_profissional"],
        "was_hired": True
    }

    print(f"\n--- Teste: Requisição POST para /feedback (Erro DB) ---")
    response = client.post("/feedback", json=feedback_payload)

    print(f"Status da Resposta: {response.status_code}")
    print(f"Corpo da Resposta:\n{json.dumps(response.json(), indent=2)}")

    assert response.status_code == 500
    assert response.json()["detail"] == "Erro interno ao processar feedback."
    mock_db_session_instance.rollback.assert_called_once()
    mock_db_session_instance.close.assert_called_once()
    print("Teste 'test_receive_feedback_db_error' concluído com sucesso.")

# --- Testes Unitários para a Lógica de Cálculo de Drift ---
@patch('src.drift_monitor_service.main.SessionLocal')
@patch('src.drift_monitor_service.main.joblib.load')
@patch('src.drift_monitor_service.main.datetime')
def test_calculate_drift_metrics(mock_datetime, mock_joblib_load, mock_session_local):
    mock_now = datetime(2023, 1, 30, 12, 0, 0)
    mock_datetime.utcnow.return_value = mock_now
    mock_datetime.now.return_value = mock_now

    mock_db_session_instance = MagicMock()
    mock_session_local.return_value = mock_db_session_instance

    mock_baseline_log1 = MagicMock(spec=PredictionLog)
    mock_baseline_log1.timestamp = datetime(2023, 1, 1, 10, 0, 0)
    mock_baseline_log1.predicted_probability = 0.8
    mock_baseline_log1.actual_outcome = True
    mock_baseline_log1.input_features_applicant = json.dumps({"objetivo_profissional": "Python Dev", "local": "São Paulo"})
    mock_baseline_log1.input_features_job = json.dumps({"nivel_ingles": "Avançado", "nivel_profissional": "Sênior", "pais": "Brasil"})

    mock_baseline_log2 = MagicMock(spec=PredictionLog)
    mock_baseline_log2.timestamp = datetime(2023, 1, 5, 10, 0, 0)
    mock_baseline_log2.predicted_probability = 0.2
    mock_baseline_log2.actual_outcome = False
    mock_baseline_log2.input_features_applicant = json.dumps({"objetivo_profissional": "Java Dev", "local": "Rio de Janeiro"})
    mock_baseline_log2.input_features_job = json.dumps({"nivel_ingles": "Básico", "nivel_profissional": "Júnior", "pais": "Brasil"})

    mock_recent_log1 = MagicMock(spec=PredictionLog)
    mock_recent_log1.timestamp = datetime(2023, 1, 25, 10, 0, 0)
    mock_recent_log1.predicted_probability = 0.7
    mock_recent_log1.actual_outcome = True
    mock_recent_log1.input_features_applicant = json.dumps({"objetivo_profissional": "Python Dev", "local": "São Paulo"})
    mock_recent_log1.input_features_job = json.dumps({"nivel_ingles": "Avançado", "nivel_profissional": "Sênior", "pais": "Brasil"})

    mock_recent_log2 = MagicMock(spec=PredictionLog)
    mock_recent_log2.timestamp = datetime(2023, 1, 26, 10, 0, 0)
    mock_recent_log2.predicted_probability = 0.3
    mock_recent_log2.actual_outcome = False
    mock_recent_log2.input_features_applicant = json.dumps({"objetivo_profissional": "Node.js Dev", "local": "Rio de Janeiro"})
    mock_recent_log2.input_features_job = json.dumps({"nivel_ingles": "Intermediário", "nivel_profissional": "Pleno", "pais": "Brasil"})

    mock_db_session_instance.query.return_value.filter.side_effect = [
        MagicMock(all=MagicMock(return_value=[mock_recent_log1, mock_recent_log2])),
        MagicMock(all=MagicMock(return_value=[mock_baseline_log1, mock_baseline_log2]))
    ]

    mock_joblib_load.return_value = ['python', 'java']

    print(f"\n--- Teste: Cálculo de Métricas de Drift ---")

    import asyncio
    from src.drift_monitor_service.main import calculate_drift_metrics as actual_calculate_drift_metrics
    asyncio.run(actual_calculate_drift_metrics())

    assert mock_db_session_instance.add.call_count >= 3
    mock_db_session_instance.commit.assert_called_once()
    mock_db_session_instance.close.assert_called_once()

    print("Teste 'test_calculate_drift_metrics' concluído com sucesso.")

# Teste para o endpoint /metrics do serviço de drift
def test_drift_metrics_endpoint(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    
    print(f"\n--- Teste: Requisição GET para /metrics do serviço de drift ---")
    response = client.get("/metrics")

    print(f"Status da Resposta: {response.status_code}")
    print(f"Corpo da Resposta (parcial):\n{response.text[:500]}...")

    assert response.status_code == 200
    assert "data_drift_score" in response.text
    assert "concept_drift_score" in response.text
    assert "total_drift_checks" in response.text
    assert "drift_calculation_duration_seconds" in response.text
    assert "model_f1_score_production" in response.text
    print("Teste 'test_drift_metrics_endpoint' concluído com sucesso.")
