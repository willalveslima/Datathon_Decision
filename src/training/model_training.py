import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib # Importar joblib para salvar e carregar modelos

# Importações para Regressão Logística
from sklearn.linear_model import LogisticRegression # Importar LogisticRegression

# Load the processed data
# Carrega o DataFrame processado que foi salvo na etapa anterior.
# Este DataFrame já deve ter as colunas indesejadas removidas conforme sua última solicitação.
try:
    # Ajustado para que o script possa ser executado a partir de src/training/
    merged_df = pd.read_csv('../../data/processed/merged_data_processed.csv')
    print("DataFrame processado 'data/processed/merged_data_processed.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: 'data/processed/merged_data_processed.csv' não encontrado. Por favor, execute as etapas anteriores para criá-lo.")
    exit()

# --- Text Vectorization (TF-IDF) ---
# Vetorização de features de texto usando TF-IDF.
print("\nIniciando vetorização TF-IDF para features de texto com stop words em português...")

# Lista de stop words comuns em português.
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


# Inicializa o TfidfVectorizer para as features de texto do candidato e da vaga.
# max_features limita o número de palavras (features) para evitar alta dimensionalidade.
tfidf_vectorizer_applicant = TfidfVectorizer(stop_words=portuguese_stop_words, max_features=5000, lowercase=True)
tfidf_vectorizer_job = TfidfVectorizer(stop_words=portuguese_stop_words, max_features=5000, lowercase=True)


# Aplica o TF-IDF nas features de texto dos candidatos.
applicant_tfidf = tfidf_vectorizer_applicant.fit_transform(merged_df['applicant_text_features'])
print(f"Shape do TF-IDF para candidatos: {applicant_tfidf.shape}")

# Aplica o TF-IDF nas features de texto das vagas.
job_tfidf = tfidf_vectorizer_job.fit_transform(merged_df['job_text_features'])
print(f"Shape do TF-IDF para vagas: {job_tfidf.shape}")

# --- Feature Engineering: Extração de Habilidades/Tecnologias ---
print("\nIniciando extração de features de habilidades/tecnologias...")

# Lista de habilidades/tecnologias comuns (exemplo, expanda conforme seu domínio)
# Use termos em minúsculas para facilitar a correspondência.
common_skills = [
    'python', 'java', 'javascript', 'c#', '.net', 'sql', 'nosql', 'django', 'flask',
    'react', 'angular', 'vue', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
    'machine learning', 'deep learning', 'ci/cd', 'agile', 'scrum', 'git',
    'postgresql', 'mongodb', 'mysql', 'linux', 'cloud', 'devops', 'api', 'rest'
]

# Função para verificar a presença de habilidades em um texto
def extract_skills_from_text(text, skills_list):
    found_skills = []
    text_lower = str(text).lower() # Garante que o texto seja string e em minúsculas
    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

# Aplica a função para extrair habilidades para candidatos e vagas
merged_df['applicant_skills'] = merged_df['applicant_text_features'].apply(lambda x: extract_skills_from_text(x, common_skills))
merged_df['job_skills'] = merged_df['job_text_features'].apply(lambda x: extract_skills_from_text(x, common_skills))

# Cria features binárias para cada habilidade (One-Hot Encoding manual para habilidades)
for skill in common_skills:
    merged_df[f'applicant_has_{skill}'] = merged_df['applicant_skills'].apply(lambda x: 1 if skill in x else 0)
    merged_df[f'job_requires_{skill}'] = merged_df['job_skills'].apply(lambda x: 1 if skill in x else 0)

# Cria uma feature de contagem de habilidades em comum
merged_df['common_skills_count'] = merged_df.apply(lambda row: len(set(row['applicant_skills']).intersection(set(row['job_skills']))), axis=1)

print(f"Features de habilidades criadas. Total de {len(common_skills) * 2 + 1} novas features.")

# --- Categorical Feature Encoding (One-Hot Encoding) ---
# Codificação de features categóricas usando One-Hot Encoding.
print("\nIniciando One-Hot Encoding para features categóricas...")

# Lista de colunas categóricas a serem codificadas.
# ATUALIZADO para remover as colunas especificadas.
categorical_cols = [
    'nivel profissional', # Atenção ao espaço no nome da coluna
    'nivel_academico',
    'nivel_ingles',
    'nivel_espanhol',
    'local', # Do candidato
    'pais',  # Da vaga
    'estado', # Da vaga
    'cidade', # Da vaga
    'regiao'  # Da vaga
]

# Filtra as colunas que realmente existem no DataFrame mesclado.
existing_categorical_cols = [col for col in categorical_cols if col in merged_df.columns]
missing_categorical_cols = [col for col in categorical_cols if col not in merged_df.columns]

if missing_categorical_cols:
    print(f"Aviso: As seguintes colunas categóricas não foram encontradas no DataFrame mesclado e serão ignoradas: {missing_categorical_cols}")

# Converte todos os valores das colunas categóricas existentes para string e preenche NaNs com string vazia.
for col in existing_categorical_cols:
    merged_df[col] = merged_df[col].astype(str).fillna('')

# Realiza o One-Hot Encoding. `dummy_na=False` evita a criação de uma coluna para valores NaN.
encoded_features = pd.get_dummies(merged_df[existing_categorical_cols], dummy_na=False)
print(f"Shape das features categóricas codificadas: {encoded_features.shape}")

# --- Combining all features ---
# Combina todas as features (TF-IDF, categóricas codificadas e novas features de habilidade) em uma única matriz.
# Converte o DataFrame de features codificadas para uma matriz esparsa.
encoded_features_sparse = csr_matrix(encoded_features)

# Adiciona as novas features de habilidade ao X
# Cria um DataFrame com as features de habilidade para converter em sparse matrix
skill_features_df = merged_df[[f'applicant_has_{skill}' for skill in common_skills] +
                              [f'job_requires_{skill}' for skill in common_skills] +
                              ['common_skills_count']]
skill_features_sparse = csr_matrix(skill_features_df.values)


# Empilha horizontalmente todas as features.
X = hstack([applicant_tfidf, job_tfidf, encoded_features_sparse, skill_features_sparse])
y = merged_df['contratado'] # Variável alvo (0 para não contratado, 1 para contratado)

print(f"\nShape final da matriz de features (X): {X.shape}")
print(f"Shape final do vetor de rótulos (y): {y.shape}")

# --- Data Splitting ---
# Divide os dados em conjuntos de treinamento e teste.
# `test_size=0.2` significa 20% dos dados para teste.
# `random_state=42` garante reprodutibilidade.
# `stratify=y` mantém a proporção das classes (contratado/não contratado) em ambos os conjuntos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nShape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")

# --- Model Training (Logistic Regression) ---
print("\nIniciando treinamento do modelo Logistic Regression (sem SMOTE)...")

# Definindo o modelo Logistic Regression
# `solver='saga'` é adequado para grandes datasets e matrizes esparsas.
# `max_iter` aumentado para garantir convergência.
# `n_jobs=-1` utiliza todos os núcleos da CPU disponíveis.
# `class_weight='balanced'` RE-ADICIONADO para lidar com o desequilíbrio de classes, já que SMOTE foi removido.
model = LogisticRegression(solver='saga', max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')

# Treinando o modelo com os dados ORIGINAIS (sem SMOTE)
model.fit(X_train, y_train)

print("\nTreinamento do modelo Logistic Regression concluído.")

# --- Model Evaluation ---
# Avaliação do desempenho do modelo.
print("\nIniciando avaliação do modelo...")

# Faz previsões no conjunto de teste.
# predict_proba retorna as probabilidades para ambas as classes, pegamos a probabilidade da classe positiva (índice 1)
y_proba = model.predict_proba(X_test)[:, 1]
# O THRESHOLD DE 0.7 será mantido para avaliação
y_pred = (y_proba >= 0.7).astype(int) # Converte probabilidades para classes binárias (threshold 0.7)

print("\n--- Relatório de Classificação ---")
# Exibe o relatório de classificação com Precision, Recall e F1-Score.
print(classification_report(y_test, y_pred))

# Calcula a área sob a curva ROC (AUC-ROC).
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nAUC-ROC: {roc_auc:.4f}")

# Plotting Precision-Recall Curve
# Calcula a curva Precision-Recall.
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision) # Calcula a área sob a curva Precision-Recall.

# Gera e salva o gráfico da curva Precision-Recall.
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve.png') # Salva a imagem do gráfico.
print("Curva Precision-Recall salva como 'precision_recall_curve.png'.")

# --- Save the Trained Model and Preprocessing Objects ---
# Salva o modelo treinado e os objetos de pré-processamento (vetorizadores TF-IDF).
# Isso é crucial para usar o modelo em produção sem retreiná-lo.
print("\nSalvando o modelo treinado e os objetos de pré-processamento...")
try:
    # Salva o modelo Logistic Regression usando joblib
    joblib.dump(model, '../../models/logistic_regression_model.pkl')
    joblib.dump(tfidf_vectorizer_applicant, '../../models/tfidf_vectorizer_applicant.pkl')
    joblib.dump(tfidf_vectorizer_job, '../../models/tfidf_vectorizer_job.pkl')
    joblib.dump(existing_categorical_cols, '../../models/categorical_cols.pkl')
    joblib.dump(encoded_features.columns.tolist(), '../../models/encoded_feature_names.pkl')
    # Salva a lista de habilidades para uso na inferência
    joblib.dump(common_skills, '../../models/common_skills.pkl')

    print("Modelo Logistic Regression, vetorizadores TF-IDF, informações de colunas categóricas e habilidades comuns salvos com sucesso.")
except Exception as e:
    print(f"Erro ao salvar os objetos: {e}")

print("\n--- Exemplo de Recomendação ---")
print("\nO modelo de classificação foi treinado. Para fazer recomendações, você precisaria:")
print("1. Carregar o modelo e os objetos de pré-processamento salvos.")
print("2. Selecionar uma vaga (por exemplo, uma nova vaga ou uma vaga existente para a qual você quer encontrar candidatos).")
print("3. Para cada candidato disponível (que ainda não foi contratado para essa vaga):")
print("   a. Criar uma entrada de dados combinando as características do candidato e as características da vaga (similar à forma como o 'X' foi construído).")
print("   b. Pré-processar esta entrada usando os *mesmos* vetorizadores TF-IDF, a lógica de codificação categórica E a extração de habilidades usados no treinamento.")
print("   c. Usar o modelo `model.predict_proba()` para obter a probabilidade de contratação.")
print("4. Classificar os candidatos com base nessas probabilidades para gerar uma lista de recomendação.")

print("\nAs principais métricas do modelo são: Precision, Recall, F1-Score e AUC-ROC, que estão no relatório de classificação acima e na curva Precision-Recall. Estas métricas ajudam a entender a capacidade do modelo em identificar candidatos que serão contratados.")
