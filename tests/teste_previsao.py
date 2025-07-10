import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib # Importar joblib para carregar modelos

#arquivos de modelo
pasta_modelo = '../models/'
logistic_regression_model_pkl = pasta_modelo + 'logistic_regression_model.pkl'
tfidf_vectorizer_applicant_pkl = pasta_modelo + 'tfidf_vectorizer_applicant.pkl'
tfidf_vectorizer_job_pkl = pasta_modelo + 'tfidf_vectorizer_job.pkl'
categorical_cols_pkl = pasta_modelo + 'categorical_cols.pkl'
encoded_feature_names_pkl = pasta_modelo + 'encoded_feature_names.pkl'


# --- Função para Testar o Modelo com Novos Dados ---
# Esta função simula o processo de inferência em um ambiente de produção (e.g., uma API REST).
# Ela carrega o modelo e os transformadores salvos e os usa para fazer previsões em novos dados.

def predict_hiring_probability(applicant_data: dict, job_data: dict) -> float:
    """
    Prevê a probabilidade de um candidato ser contratado para uma vaga específica.

    Args:
        applicant_data (dict): Um dicionário contendo as informações do candidato,
                               com chaves como 'objetivo_profissional', 'historico_profissional_texto',
                               'cv_completo', e as colunas categóricas relevantes.
        job_data (dict): Um dicionário contendo as informações da vaga,
                         com chaves como 'titulo_vaga_prospect', 'titulo_vaga',
                         'principais_atividades', 'competencia_tecnicas_e_comportamentais',
                         'demais_observacoes', 'areas_atuacao', e as colunas categóricas relevantes.

    Returns:
        float: A probabilidade prevista de contratação (valor entre 0 e 1).
               Retorna -1.0 em caso de erro no pré-processamento ou carregamento do modelo.
    """
    try:
        # 1. Carregar o modelo e os objetos de pré-processamento salvos
        # Certifique-se de que os arquivos .pkl estejam no diretório correto (e.g., na pasta 'models/')
        loaded_model = joblib.load(logistic_regression_model_pkl)
        loaded_tfidf_applicant = joblib.load(tfidf_vectorizer_applicant_pkl)
        loaded_tfidf_job = joblib.load(tfidf_vectorizer_job_pkl)
        loaded_categorical_cols = joblib.load(categorical_cols_pkl)
        loaded_encoded_feature_names = joblib.load(encoded_feature_names_pkl)

        print("\nModelo e objetos de pré-processamento carregados para teste.")

        # 2. Preparar os dados de entrada (candidato e vaga)
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

        # 3. Pré-processar a entrada usando os *mesmos* transformadores
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

        # Adiciona esta linha para garantir que o DataFrame seja numérico antes da conversão para sparse matrix
        encoded_features_inference = encoded_features_inference.astype(float)

        encoded_features_inference_sparse = csr_matrix(encoded_features_inference)


        # Combinar todas as features para a inferência
        X_inference = hstack([applicant_tfidf_inference, job_tfidf_inference, encoded_features_inference_sparse])

        # 4. Usar o modelo para obter a probabilidade de contratação
        probability = loaded_model.predict_proba(X_inference)[:, 1][0]
        return probability

    except Exception as e:
        print(f"Erro durante a previsão: {e}")
        return -1.0 # Retorna um valor indicando erro

# --- Exemplo de Uso da Função de Previsão ---
print("\n--- Testando a função de previsão com dados de exemplo ---")

# Exemplo de dados de uma nova vaga (mantida constante para os 10 candidatos)
new_job_data = {
    'titulo_vaga_prospect': 'Desenvolvedor Python Sênior',
    'titulo_vaga': 'Vaga para Desenvolvedor Backend com Python e Django',
    'principais_atividades': 'Desenvolvimento e manutenção de APIs, integração com sistemas externos, testes unitários e de integração.',
    'competencia_tecnicas_e_comportamentais': 'Python, Django, REST APIs, SQL, Git, Metodologias Ágeis. Proatividade, comunicação e trabalho em equipe.',
    'demais_observacoes': 'Experiência com cloud computing (AWS/GCP) é um diferencial.',
    'areas_atuacao': 'TI - Desenvolvimento/Programação',
    'modalidade_vaga_prospect': 'CLT Full',
    'nivel profissional': 'Sênior',
    'nivel_academico': 'Ensino Superior Completo',
    'nivel_ingles': 'Avançado',
    'nivel_espanhol': 'Básico',
    'faixa_etaria': 'De: 25 Até: 40',
    'horario_trabalho': 'Comercial',
    'vaga_especifica_para_pcd': 'Não'
}

# Lista de 10 exemplos de dados de novos candidatos
new_applicant_data_list = [
    { # Candidato 1: Alta compatibilidade (ajustado para ser MAIS compatível)
        'objetivo_profissional': 'Desenvolvedor Backend Sênior com foco em Python, Django e Cloud.',
        'historico_profissional_texto': 'Liderança de equipes no desenvolvimento de APIs RESTful robustas com Python e Django. Experiência em arquitetura de microsserviços, testes automatizados e implantação em nuvem (AWS, GCP).',
        'cv_completo': 'Currículo detalhado com 8 anos de experiência em Python, Django, DRF, PostgreSQL, Docker, Kubernetes, AWS, GCP. Inglês fluente. Forte atuação em otimização de performance e segurança.',
        'pcd': 'Não'
    },
    { # Candidato 2: Boa compatibilidade, menos experiência
        'objetivo_profissional': 'Desenvolvedor Backend Júnior buscando crescimento.',
        'historico_profissional_texto': '2 anos de experiência com Python e Flask, desenvolvendo pequenas APIs e scripts de automação.',
        'cv_completo': 'Conhecimento em Python, Flask, MongoDB. Inglês intermediário. Buscando oportunidade para aprender e crescer.',
        'pcd': 'Não'
    },
    { # Candidato 3: Experiência em Java, não Python
        'objetivo_profissional': 'Arquiteto de Software com foco em sistemas distribuídos.',
        'historico_profissional_texto': 'Mais de 10 anos de experiência em arquitetura de sistemas Java, Spring Boot e microserviços.',
        'cv_completo': 'Experiência sólida em Java, Spring, Kafka, Docker. Inglês fluente.',
        'pcd': 'Não'
    },
    { # Candidato 4: Pouca experiência e habilidades genéricas
        'objetivo_profissional': 'Estagiário em TI.',
        'historico_profissional_texto': 'Cursando Análise e Desenvolvimento de Sistemas. Conhecimento básico em lógica de programação e pacote Office.',
        'cv_completo': 'Interesse em aprender novas tecnologias. Inglês básico.',
        'pcd': 'Não'
    },
    { # Candidato 5: Experiência relevante, mas em outra área (Data Science)
        'objetivo_profissional': 'Cientista de Dados com Python.',
        'historico_profissional_texto': '3 anos de experiência em análise de dados, machine learning com Python (Pandas, Scikit-learn).',
        'cv_completo': 'Projetos em análise preditiva, visualização de dados. Inglês avançado.',
        'pcd': 'Não'
    },
    { # Candidato 6: Boa experiência em Python, mas sem Django explícito (ajustado para ser mais compatível)
        'objetivo_profissional': 'Engenheiro de Software Python Sênior.',
        'historico_profissional_texto': 'Desenvolvimento de sistemas escaláveis em Python, com foco em performance e segurança. Experiência com APIs RESTful e bancos de dados relacionais (PostgreSQL).',
        'cv_completo': 'Proficiência em Python, SQL, AWS. Experiência com testes automatizados e integração contínua. Inglês avançado.',
        'pcd': 'Não'
    },
    { # Candidato 7: Nível júnior, mas com bom potencial
        'objetivo_profissional': 'Desenvolvedor Python Júnior.',
        'historico_profissional_texto': 'Recém-formado em Ciência da Computação. Projeto de TCC em Python para web scraping.',
        'cv_completo': 'Conhecimento em Python, HTML, CSS. Inglês intermediário.',
        'pcd': 'Não'
    },
    { # Candidato 8: Experiência em Python, mas com foco em infraestrutura (DevOps)
        'objetivo_profissional': 'Engenheiro DevOps com Python.',
        'historico_profissional_texto': 'Automação de infraestrutura com Python, Ansible e Terraform. Experiência com CI/CD e Docker.',
        'cv_completo': 'Foco em automação e otimização de processos. Inglês fluente.',
        'pcd': 'Não'
    },
    { # Candidato 9: Experiência em .NET, não relevante
        'objetivo_profissional': 'Desenvolvedor .NET Sênior.',
        'historico_profissional_texto': 'Mais de 8 anos de experiência em desenvolvimento de aplicações web com C# e .NET Core.',
        'cv_completo': 'Conhecimento em ASP.NET, SQL Server, Azure. Inglês avançado.',
        'pcd': 'Não'
    },
    { # Candidato 10: Perfil generalista, sem foco claro
        'objetivo_profissional': 'Analista de Sistemas.',
        'historico_profissional_texto': 'Suporte a sistemas, levantamento de requisitos e documentação. Conhecimento em diversas ferramentas de gestão.',
        'cv_completo': 'Habilidades de comunicação e resolução de problemas. Inglês básico.',
        'pcd': 'Não'
    },
    { # Candidato 11: Novo candidato com altíssima compatibilidade
        'objetivo_profissional': 'Especialista em Backend Python/Django, buscando desafios em nuvem.',
        'historico_profissional_texto': 'Comprovada experiência de 7 anos no desenvolvimento e otimização de sistemas críticos em Python/Django, incluindo design de banco de dados (PostgreSQL) e integração contínua. Proficiência em AWS e GCP para escalabilidade e resiliência.',
        'cv_completo': 'Portfólio com projetos complexos em Python, Django REST Framework, microsserviços, Docker, Kubernetes, AWS (EC2, Lambda, S3, RDS), GCP (Compute Engine, Cloud SQL). Inglês fluente. Forte capacidade analítica e de resolução de problemas.',
        'pcd': 'Não'
    }
]

# Loop através da lista de candidatos e faça a previsão para cada um
for i, applicant_data in enumerate(new_applicant_data_list):
    print(f"\n--- Previsão para o Candidato {i+1} ---")
    predicted_probability = predict_hiring_probability(applicant_data, new_job_data)

    if predicted_probability != -1.0:
        print(f"Probabilidade prevista de contratação: {predicted_probability:.4f}")
        if predicted_probability >= 0.5: # Exemplo de threshold
            print("Recomendação: Candidato com alta probabilidade de ser contratado.")
        else:
            print("Recomendação: Candidato com baixa probabilidade de ser contratado.")
    else:
        print("Não foi possível fazer a previsão para este candidato. Verifique os logs de erro.")
