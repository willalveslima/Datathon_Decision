import requests
import json
import random
import os
from datetime import datetime
import time # Importar a biblioteca time para usar sleep

# --- Configurações da API ---
PREDICTION_API_URL = "https://prediction-api-service-160589948885.southamerica-east1.run.app/predict"
FEEDBACK_API_URL = "https://drift-monitor-service-160589948885.southamerica-east1.run.app/feedback"
HEADERS = {"Content-Type": "application/json"}

# --- Caminhos dos arquivos de dados brutos (relativos à raiz do projeto) ---
# Assumindo que este script está em tests/functional/
RAW_DATA_PATH = '../../data/raw/'
APPLICANTS_FILE = RAW_DATA_PATH + 'applicants.json'
VAGAS_FILE = RAW_DATA_PATH + 'vagas.json'
PROSPECTS_FILE = RAW_DATA_PATH + 'prospects.json'

# --- Carregar dados brutos (uma vez) ---
try:
    with open(APPLICANTS_FILE, 'r', encoding='utf-8') as f:
        applicants_raw_data = json.load(f)
    with open(VAGAS_FILE, 'r', encoding='utf-8') as f:
        vagas_raw_data = json.load(f)
    with open(PROSPECTS_FILE, 'r', encoding='utf-8') as f:
        prospects_raw_data = json.load(f)
    print("Dados brutos carregados com sucesso para o teste funcional.")
except FileNotFoundError as e:
    print(f"Erro: Arquivo de dados brutos não encontrado. Certifique-se de que os caminhos estão corretos e os arquivos existem. Erro: {e}")
    exit()
except json.JSONDecodeError as e:
    print(f"Erro ao decodificar JSON nos arquivos brutos. Erro: {e}")
    exit()

# --- Funções Auxiliares para Extrair Dados e Mapear para Modelos Pydantic ---

def get_applicant_details(codigo_profissional: str) -> dict:
    """Extrai os detalhes de um candidato pelo seu código."""
    applicant_data = applicants_raw_data.get(str(codigo_profissional))
    if not applicant_data:
        return {} # Retorna dicionário vazio se não encontrar

    # Mapeia para o formato esperado pelo ApplicantData da API
    # ATUALIZADO: Removido 'pcd'
    details = {
        "codigo_profissional": str(codigo_profissional),
        "objetivo_profissional": applicant_data.get("infos_basicas", {}).get("objetivo_profissional", ""),
        "historico_profissional_texto": " ".join([exp.get("descricao_atividades", "") for exp in applicant_data.get("historico_profissional", [])]),
        "cv_completo": applicant_data.get("cv", ""),
        "local": applicant_data.get("infos_basicas", {}).get("local", "")
    }
    return details

def get_job_details(codigo_vaga: str) -> dict:
    """Extrai os detalhes de uma vaga pelo seu código."""
    job_data = vagas_raw_data.get(str(codigo_vaga))
    if not job_data:
        return {} # Retorna dicionário vazio se não encontrar

    # Mapeia para o formato esperado pelo JobData da API
    # ATUALIZADO: Removidos 'modalidade_vaga_prospect', 'faixa_etaria', 'horario_trabalho', 'vaga_especifica_para_pcd'
    details = {
        "codigo_vaga": str(codigo_vaga),
        "titulo_vaga_prospect": job_data.get("informacoes_basicas", {}).get("titulo_vaga", ""), # Pode ser ajustado se o prospects tiver um título diferente
        "titulo_vaga": job_data.get("informacoes_basicas", {}).get("titulo_vaga", ""),
        "principais_atividades": job_data.get("perfil_vaga", {}).get("principais_atividades", ""),
        "competencia_tecnicas_e_comportamentais": job_data.get("perfil_vaga", {}).get("competencia_tecnicas_e_comportamentais", ""),
        "demais_observacoes": job_data.get("perfil_vaga", {}).get("demais_observacoes", ""),
        "areas_atuacao": job_data.get("perfil_vaga", {}).get("areas_atuacao", ""),
        "nivel_profissional": job_data.get("perfil_vaga", {}).get("nivel profissional", ""), # Atenção ao espaço
        "nivel_academico": job_data.get("perfil_vaga", {}).get("nivel_academico", ""),
        "nivel_ingles": job_data.get("perfil_vaga", {}).get("nivel_ingles", ""),
        "nivel_espanhol": job_data.get("perfil_vaga", {}).get("nivel_espanhol", ""),
        "pais": job_data.get("perfil_vaga", {}).get("pais", ""),
        "estado": job_data.get("perfil_vaga", {}).get("estado", ""),
        "cidade": job_data.get("perfil_vaga", {}).get("cidade", ""),
        "regiao": job_data.get("perfil_vaga", {}).get("regiao", "")
    }
    return details

# --- Variáveis para Estatísticas ---
total_predictions_sent = 0
total_hired_feedback = 0
total_not_hired_feedback = 0
total_prediction_api_response_time = 0.0 # Novo: Tempo total de resposta da API de Predição
prediction_api_requests_count = 0 # Novo: Contador de requisições da API de Predição
total_feedback_api_response_time = 0.0 # Novo: Tempo total de resposta da API de Feedback
feedback_api_requests_count = 0 # Novo: Contador de requisições da API de Feedback


# --- Função Principal do Teste Funcional ---
def run_full_functional_test_iteration():
    global total_predictions_sent, total_hired_feedback, total_not_hired_feedback
    global total_prediction_api_response_time, prediction_api_requests_count
    global total_feedback_api_response_time, feedback_api_requests_count
    
    print("\n--- INICIANDO ITERAÇÃO DO TESTE FUNCIONAL ---")

    # 1. Selecionar uma vaga aleatória que tenha candidatos prospectados
    all_prospect_job_codes = list(prospects_raw_data.keys())
    if not all_prospect_job_codes:
        print("Erro: Nenhuma vaga com prospects encontrada nos dados brutos.")
        return False # Indica falha para o loop principal

    selected_job_code = None
    selected_job_prospect_data = None
    for _ in range(100): # Tenta até 100 vezes para encontrar uma vaga com prospects
        random_job_code = random.choice(all_prospect_job_codes)
        job_prospect_data = prospects_raw_data.get(random_job_code)
        if job_prospect_data and job_prospect_data.get("prospects"):
            selected_job_code = random_job_code
            selected_job_prospect_data = job_prospect_data
            break
    
    if not selected_job_code:
        print("Erro: Não foi possível encontrar uma vaga com candidatos prospectados após várias tentativas.")
        return False # Indica falha

    print(f"\nVaga Selecionada Aleatoriamente (Código): {selected_job_code}")
    job_details = get_job_details(selected_job_code)

    if not job_details:
        print(f"Erro: Detalhes da vaga {selected_job_code} não encontrados em vagas.json. Pulando este teste.")
        return True # Permite continuar o loop, mas este teste de iteração falhou

    all_applicants_for_job = selected_job_prospect_data["prospects"]
    print(f"Total de candidatos prospectados para esta vaga: {len(all_applicants_for_job)}")

    # ATUALIZADO: Lista de status que o modelo considera como "contratado" para simulação de feedback
    hired_statuses_for_feedback = [
        "Proposta Aceita",
        "Aprovado",
        "Contratado como Hunting",
        "Contratado pela Decision",
        "Entrevista Técnica",
        "Entrevista com Cliente",
        "Documentação CLT",
        "Documentação Cooperado",
        "Documentação PJ"
    ]

    processed_applicants_count = 0
    for i, prospect_entry in enumerate(all_applicants_for_job):
        codigo_profissional_selected = prospect_entry["codigo"]
        
        print(f"\n--- Processando Candidato {i+1}/{len(all_applicants_for_job)} (Código: {codigo_profissional_selected}) ---")
        print(f"  Situação Original: {prospect_entry['situacao_candidado']}")

        applicant_details = get_applicant_details(codigo_profissional_selected)

        if not applicant_details:
            print(f"Erro: Detalhes do candidato {codigo_profissional_selected} não encontrados em applicants.json. Pulando este candidato.")
            continue
        
        # 3. Preparar payload para a API de Predição (para um único candidato por vez)
        prediction_payload = {
            "job": job_details,
            "applicants": [applicant_details]
        }

        print(f"\n  Chamando a API de Predição ({PREDICTION_API_URL})")
        # print(f"  Payload da Predição:\n{json.dumps(prediction_payload, indent=2)}") # Descomente para ver o payload completo

        try:
            prediction_response = requests.post(PREDICTION_API_URL, headers=HEADERS, json=prediction_payload)
            prediction_response.raise_for_status() # Levanta HTTPError para 4xx/5xx

            # Novo: Registrar tempo de resposta da API de Predição
            total_prediction_api_response_time += prediction_response.elapsed.total_seconds()
            prediction_api_requests_count += 1

            prediction_result = prediction_response.json()
            print(f"  Status da Resposta da Predição: {prediction_response.status_code}")
            print(f"  Tempo de Resposta Predição: {prediction_response.elapsed.total_seconds():.4f}s")
            # print(f"  Resultado da Predição:\n{json.dumps(prediction_result, indent=2)}") # Descomente para ver o resultado completo

            predicted_prob = None
            if prediction_result and prediction_result.get("ranked_applicants"):
                predicted_prob = prediction_result["ranked_applicants"][0]["probability"]
                print(f"  Probabilidade Predita: {predicted_prob:.4f}")
                total_predictions_sent += 1 # Incrementa contador de previsões enviadas
            else:
                print("  Nenhuma probabilidade predita retornada para este candidato.")
                continue # Pula para o próximo candidato se a predição falhar

            # 4. Simular feedback para a API de Drift Monitor
            # ATUALIZADO: Usa a situação real do candidato para simular o feedback
            simulated_was_hired = prospect_entry['situacao_candidado'] in hired_statuses_for_feedback
            
            feedback_payload = {
                "codigo_vaga": selected_job_code,
                "codigo_profissional": codigo_profissional_selected,
                "was_hired": simulated_was_hired
            }

            print(f"\n  Chamando a API de Feedback ({FEEDBACK_API_URL})")
            # print(f"  Payload do Feedback:\n{json.dumps(feedback_payload, indent=2)}") # Descomente para ver o payload completo

            feedback_response = requests.post(FEEDBACK_API_URL, headers=HEADERS, json=feedback_payload)
            feedback_response.raise_for_status() # Levanta HTTPError para 4xx/5xx

            # Novo: Registrar tempo de resposta da API de Feedback
            total_feedback_api_response_time += feedback_response.elapsed.total_seconds()
            feedback_api_requests_count += 1

            print(f"  Status da Resposta do Feedback: {feedback_response.status_code}")
            print(f"  Tempo de Resposta Feedback: {feedback_response.elapsed.total_seconds():.4f}s")
            # print(f"  Resultado do Feedback:\n{json.dumps(feedback_response.json(), indent=2)}") # Descomente para ver o resultado completo
            print(f"  Feedback Enviado: Candidato {codigo_profissional_selected} (Vaga {selected_job_code}) foi Contratado: {simulated_was_hired}")
            
            # Incrementa contadores de feedback
            if simulated_was_hired:
                total_hired_feedback += 1
            else:
                total_not_hired_feedback += 1
            
            processed_applicants_count += 1

        except requests.exceptions.ConnectionError:
            print(f"\n  Erro de Conexão para o candidato {codigo_profissional_selected}: Uma das APIs não está acessível. Certifique-se de que ambas as APIs estão rodando.")
            return False # Interrompe o loop principal se houver problema de conexão
        except requests.exceptions.HTTPError as e:
            print(f"\n  Erro HTTP para o candidato {codigo_profissional_selected}: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 422:
                print(f"    Detalhes do erro 422: {e.response.json()}")
            continue # Continua para o próximo candidato mesmo com erro HTTP
        except json.JSONDecodeError:
            print(f"\n  Erro ao decodificar JSON da resposta para o candidato {codigo_profissional_selected}. Resposta: {prediction_response.text if 'prediction_response' in locals() else feedback_response.text}")
            continue # Continua para o próximo candidato
        except Exception as e:
            print(f"\n  Ocorreu um erro inesperado durante o processamento do candidato {codigo_profissional_selected}: {e}")
            continue # Continua para o próximo candidato

    print(f"\n--- {processed_applicants_count} de {len(all_applicants_for_job)} candidatos processados para a vaga {selected_job_code} ---")
    return True # Indica que a iteração foi concluída (com ou sem erros em candidatos individuais)

# --- Função para exibir estatísticas finais ---
def display_final_statistics():
    print("\n--- ESTATÍSTICAS FINAIS DO TESTE FUNCIONAL ---")
    print(f"Total de Previsões Enviadas para a API: {total_predictions_sent}")
    print(f"Total de Feedbacks 'Contratado' Enviados: {total_hired_feedback}")
    print(f"Total de Feedbacks 'Não Contratado' Enviados: {total_not_hired_feedback}")
    
    # Novo: Estatísticas de tempo médio de resposta
    if prediction_api_requests_count > 0:
        avg_pred_time = total_prediction_api_response_time / prediction_api_requests_count
        print(f"Tempo Médio de Resposta da API de Predição: {avg_pred_time:.4f}s")
    else:
        print("Nenhuma requisição enviada para a API de Predição.")
    
    if feedback_api_requests_count > 0:
        avg_feedback_time = total_feedback_api_response_time / feedback_api_requests_count
        print(f"Tempo Médio de Resposta da API de Feedback: {avg_feedback_time:.4f}s")
    else:
        print("Nenhuma requisição enviada para a API de Feedback.")

    print("---------------------------------------------")

# --- Loop Principal do Teste Funcional ---
if __name__ == "__main__":
    test_interval_minutes = 0.5
    print(f"Iniciando loop de teste funcional. Repetindo a cada {test_interval_minutes} minutos. Pressione Ctrl+C para interromper.")
    try:
        while True:
            success = run_full_functional_test_iteration()
            if not success: # Se houver um erro de conexão grave, interrompe o loop
                print("Erro grave detectado. Interrompendo o loop de teste.")
                break
            print(f"\n--- Aguardando {test_interval_minutes} minutos para a próxima iteração... ---")
            time.sleep(test_interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nTeste funcional interrompido pelo usuário (Ctrl+C).")
        display_final_statistics() # Exibe estatísticas ao ser interrompido
    except Exception as e:
        print(f"\nOcorreu um erro inesperado no loop principal: {e}")
        display_final_statistics() # Exibe estatísticas em caso de erro inesperado
