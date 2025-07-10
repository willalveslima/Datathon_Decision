import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib # Importar joblib para salvar e carregar modelos

# --- Caminhos dos dados pré processados  ---
processed_data_pasta = '../data/processed/'
merged_data_processed = processed_data_pasta + 'merged_data_processed.csv' 


#local de salvamento do modelo
pasta_modelo = '../models/'
logistic_regression_model_pkl = pasta_modelo + 'logistic_regression_model.pkl'
tfidf_vectorizer_applicant_pkl = pasta_modelo + 'tfidf_vectorizer_applicant.pkl'
tfidf_vectorizer_job_pkl = pasta_modelo + 'tfidf_vectorizer_job.pkl'
categorical_cols_pkl = pasta_modelo + 'categorical_cols.pkl'
encoded_feature_names_pkl = pasta_modelo + 'encoded_feature_names.pkl'
precision_recall_curve_png = pasta_modelo + 'precision_recall_curve.png'

# Load the processed data
# Carrega o DataFrame processado que foi salvo na etapa anterior.
# Este DataFrame já deve ter as colunas indesejadas removidas conforme sua última solicitação.
try:
    merged_df = pd.read_csv(merged_data_processed)
    print("DataFrame processado 'merged_data_processed.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: 'merged_data_processed.csv' não encontrado. Por favor, execute as etapas anteriores para criá-lo.")
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
    'tivemos', 'tiveram', 'todas', 'todo', 'todos', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês', 'vos'
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

# --- Categorical Feature Encoding (One-Hot Encoding) ---
# Codificação de features categóricas usando One-Hot Encoding.
print("\nIniciando One-Hot Encoding para features categóricas...")

# Lista de colunas categóricas a serem codificadas.
# Esta lista foi atualizada para refletir as colunas que foram mantidas após a remoção.
categorical_cols = [
    'modalidade_vaga_prospect',
    'nivel profissional', # Atenção ao espaço no nome da coluna
    'nivel_academico',
    'nivel_ingles',
    'nivel_espanhol',
    'pcd',
    'faixa_etaria',
    'vaga_especifica_para_pcd'
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
# Combina todas as features (TF-IDF e categóricas codificadas) em uma única matriz.
# Converte o DataFrame de features codificadas para uma matriz esparsa.
encoded_features_sparse = csr_matrix(encoded_features)

# Empilha horizontalmente todas as features.
X = hstack([applicant_tfidf, job_tfidf, encoded_features_sparse])
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

# --- Model Training (Logistic Regression as a baseline) ---
# Treinamento do modelo de Regressão Logística.
print("\nIniciando treinamento do modelo Logistic Regression...")

# `solver='saga'` é adequado para grandes datasets e matrizes esparsas.
# `max_iter` aumentado para garantir convergência.
# `n_jobs=-1` utiliza todos os núcleos da CPU disponíveis.
log_reg_model = LogisticRegression(solver='saga', max_iter=1000, random_state=42, n_jobs=-1)
log_reg_model.fit(X_train, y_train)

print("\nTreinamento do modelo concluído.")

# --- Model Evaluation ---
# Avaliação do desempenho do modelo.
print("\nIniciando avaliação do modelo...")

# Faz previsões no conjunto de teste.
y_pred = log_reg_model.predict(X_test)
# Obtém as probabilidades da classe positiva (contratado).
y_proba = log_reg_model.predict_proba(X_test)[:, 1]

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
plt.savefig(precision_recall_curve_png) # Salva a imagem do gráfico.
print(f"Curva Precision-Recall salva como {precision_recall_curve_png}.")

# --- Save the Trained Model and Preprocessing Objects ---
# Salva o modelo treinado e os objetos de pré-processamento (vetorizadores TF-IDF).
# Isso é crucial para usar o modelo em produção sem retreiná-lo.
print("\nSalvando o modelo treinado e os objetos de pré-processamento...")
try:
    joblib.dump(log_reg_model, logistic_regression_model_pkl)
    joblib.dump(tfidf_vectorizer_applicant, tfidf_vectorizer_applicant_pkl)
    joblib.dump(tfidf_vectorizer_job, tfidf_vectorizer_job_pkl)
    # Para as features categóricas, como usamos pd.get_dummies diretamente,
    # não há um objeto OneHotEncoder separado para salvar.
    # No entanto, é importante que as colunas categóricas e seus valores únicos
    # sejam consistentes entre o treinamento e a inferência.
    # Você pode salvar a lista de colunas categóricas e os valores únicos se necessário para validação.
    joblib.dump(existing_categorical_cols, categorical_cols_pkl)
    joblib.dump(encoded_features.columns.tolist(), encoded_feature_names_pkl) # Salva os nomes das colunas após o OHE

    print("Modelo, vetorizadores TF-IDF e informações de colunas categóricas salvos com sucesso.")
except Exception as e:
    print(f"Erro ao salvar os objetos: {e}")

print("\n--- Exemplo de Recomendação ---")
# Demonstração conceitual de como as recomendações seriam feitas.
# Como 'codigo_vaga' foi removida, o exemplo de seleção de vaga será conceitual.

print("\nO modelo de classificação foi treinado. Para fazer recomendações, você precisaria:")
print("1. Carregar o modelo e os objetos de pré-processamento salvos.")
print("2. Selecionar uma vaga (por exemplo, uma nova vaga ou uma vaga existente para a qual você quer encontrar candidatos).")
print("3. Para cada candidato disponível (que ainda não foi contratado para essa vaga):")
print("   a. Criar uma entrada de dados combinando as características do candidato e as características da vaga (similar à forma como o 'X' foi construído).")
print("   b. Pré-processar esta entrada usando os *mesmos* vetorizadores TF-IDF e a lógica de codificação categórica usados no treinamento.")
print("   c. Usar o modelo `log_reg_model.predict_proba()` para obter a probabilidade de contratação.")
print("4. Classificar os candidatos com base nessas probabilidades para gerar uma lista de recomendação.")

print("\nAs principais métricas do modelo são: Precision, Recall, F1-Score e AUC-ROC, que estão no relatório de classificação acima e na curva Precision-Recall. Estas métricas ajudam a entender a capacidade do modelo em identificar candidatos que serão contratados.")
