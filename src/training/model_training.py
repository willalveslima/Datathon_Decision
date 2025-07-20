"""Treinamento do Modelo."""

import os

import joblib  # Importar joblib para salvar e carregar modelos
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Importações para Regressão Logística
from sklearn.linear_model import LogisticRegression  # Importar LogisticRegression
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_VERSION = "1.0"  # Versão do modelo, pode ser alterada conforme necessário


def load_processed_data(data_path="../../data/processed/merged_data_processed.csv"):
    """
    Carrega o DataFrame processado que foi salvo na etapa anterior.

    Args:
        data_path (str): Caminho para o arquivo CSV processado

    Returns:
        pd.DataFrame: DataFrame carregado
    """
    try:
        merged_df = pd.read_csv(data_path)
        print(f"DataFrame processado '{data_path}' carregado com sucesso.")
        return merged_df
    except FileNotFoundError:
        print(
            (
                f"Erro: '{data_path}' não encontrado. "
                "Por favor, execute as etapas anteriores "
                "para criá-lo."
            )
        )
        raise


def get_portuguese_stop_words():
    """
    Retorna uma lista de stop words comuns em português.

    Returns:
        list: Lista de stop words em português
    """
    return [
        "a",
        "ao",
        "aos",
        "aquela",
        "aquelas",
        "aquele",
        "aqueles",
        "aquilo",
        "as",
        "às",
        "até",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "delas",
        "dele",
        "deles",
        "depois",
        "do",
        "dos",
        "e",
        "é",
        "ela",
        "elas",
        "ele",
        "eles",
        "em",
        "entre",
        "era",
        "eram",
        "essa",
        "essas",
        "esse",
        "esses",
        "esta",
        "estas",
        "este",
        "estes",
        "estou",
        "está",
        "estão",
        "eu",
        "foi",
        "fomos",
        "for",
        "fora",
        "foram",
        "fosse",
        "fossem",
        "fui",
        "há",
        "isso",
        "isto",
        "já",
        "lhe",
        "lhes",
        "mais",
        "mas",
        "me",
        "mesmo",
        "meu",
        "meus",
        "minha",
        "minhas",
        "muito",
        "na",
        "nas",
        "nem",
        "no",
        "nos",
        "nossa",
        "nossas",
        "nosso",
        "nossos",
        "num",
        "numa",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "porque",
        "qual",
        "quando",
        "que",
        "quem",
        "se",
        "sem",
        "ser",
        "será",
        "serão",
        "serei",
        "seremos",
        "seria",
        "seriam",
        "seu",
        "seus",
        "só",
        "somos",
        "sou",
        "sua",
        "suas",
        "também",
        "te",
        "tem",
        "tém",
        "temos",
        "tenho",
        "terá",
        "terão",
        "terei",
        "teremos",
        "teria",
        "teriam",
        "teu",
        "teus",
        "tive",
        "tivemos",
        "tiver",
        "tivera",
        "tiveram",
        "tivesse",
        "tivessem",
        "todas",
        "todo",
        "todos",
        "tu",
        "tua",
        "tuas",
        "um",
        "uma",
        "você",
        "vocês",
        "vos",
    ]


def preprocess_text_features(merged_df):
    """
    Trata valores NaN nas colunas de texto e converte para string.

    Args:
        merged_df (pd.DataFrame): DataFrame com features de texto

    Returns:
        pd.DataFrame: DataFrame com features de texto processadas
    """
    print("Verificando e tratando valores NaN nas features de texto...")
    merged_df = merged_df.copy()

    # Tratamento de valores NaN nas colunas de texto
    merged_df["applicant_text_features"] = merged_df["applicant_text_features"].fillna("")
    merged_df["job_text_features"] = merged_df["job_text_features"].fillna("")

    # Converte para string explicitamente para garantir compatibilidade
    merged_df["applicant_text_features"] = merged_df["applicant_text_features"].astype(str)
    merged_df["job_text_features"] = merged_df["job_text_features"].astype(str)

    print(
        "Número de valores vazios em applicant_text_features: "
        f"{(merged_df['applicant_text_features'] == '').sum()}"
    )
    print(
        "Número de valores vazios em job_text_features: "
        f"{(merged_df['job_text_features'] == '').sum()}"
    )

    return merged_df


def create_tfidf_features(merged_df, max_features=5000):
    """
    Cria features TF-IDF para textos de candidatos e vagas.

    Args:
        merged_df (pd.DataFrame): DataFrame com features de texto processadas
        max_features (int): Número máximo de features TF-IDF

    Returns:
        tuple: (applicant_tfidf, job_tfidf, tfidf_vectorizer_applicant, tfidf_vectorizer_job)
    """
    print("\nIniciando vetorização TF-IDF para features de texto com stop words em português...")

    portuguese_stop_words = get_portuguese_stop_words()

    # Inicializa os vetorizadores TF-IDF
    tfidf_vectorizer_applicant = TfidfVectorizer(
        stop_words=portuguese_stop_words, max_features=max_features, lowercase=True
    )
    tfidf_vectorizer_job = TfidfVectorizer(
        stop_words=portuguese_stop_words, max_features=max_features, lowercase=True
    )

    # Aplica o TF-IDF nas features de texto
    applicant_tfidf = tfidf_vectorizer_applicant.fit_transform(merged_df["applicant_text_features"])
    print(f"Shape do TF-IDF para candidatos: {applicant_tfidf.shape}")

    job_tfidf = tfidf_vectorizer_job.fit_transform(merged_df["job_text_features"])
    print(f"Shape do TF-IDF para vagas: {job_tfidf.shape}")

    return applicant_tfidf, job_tfidf, tfidf_vectorizer_applicant, tfidf_vectorizer_job


def get_common_skills():
    """
    Retorna uma lista de habilidades/tecnologias comuns.

    Returns:
        list: Lista de habilidades comuns
    """
    return [
        "python",
        "java",
        "javascript",
        "c#",
        ".net",
        "sql",
        "nosql",
        "django",
        "flask",
        "react",
        "angular",
        "vue",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "machine learning",
        "deep learning",
        "ci/cd",
        "agile",
        "scrum",
        "git",
        "postgresql",
        "mongodb",
        "mysql",
        "linux",
        "cloud",
        "devops",
        "api",
        "rest",
    ]


def extract_skills_from_text(text, skills_list):
    """
    Verifica a presença de habilidades em um texto.

    Args:
        text (str): Texto para analisar
        skills_list (list): Lista de habilidades para procurar

    Returns:
        list: Lista de habilidades encontradas
    """
    found_skills = []
    text_lower = str(text).lower()  # Garante que o texto seja string e em minúsculas
    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills


def create_skill_features(merged_df):
    """
    Cria features baseadas em habilidades/tecnologias.

    Args:
        merged_df (pd.DataFrame): DataFrame com features de texto

    Returns:
        pd.DataFrame: DataFrame com features de habilidades adicionadas
    """
    print("\nIniciando extração de features de habilidades/tecnologias...")

    merged_df = merged_df.copy()
    common_skills = get_common_skills()

    # Aplica a função para extrair habilidades para candidatos e vagas
    merged_df["applicant_skills"] = merged_df["applicant_text_features"].apply(
        lambda x: extract_skills_from_text(x, common_skills)
    )
    merged_df["job_skills"] = merged_df["job_text_features"].apply(
        lambda x: extract_skills_from_text(x, common_skills)
    )

    # Cria features binárias para cada habilidade (One-Hot Encoding manual para habilidades)
    for skill in common_skills:
        merged_df[f"applicant_has_{skill}"] = merged_df["applicant_skills"].apply(
            lambda x, s=skill: 1 if s in x else 0
        )
        merged_df[f"job_requires_{skill}"] = merged_df["job_skills"].apply(
            lambda x, s=skill: 1 if s in x else 0
        )

    # Cria uma feature de contagem de habilidades em comum
    if len(merged_df) > 0:
        # Para DataFrames não vazios, usa apply normalmente
        merged_df["common_skills_count"] = merged_df.apply(
            lambda row: len(set(row["applicant_skills"]).intersection(set(row["job_skills"]))),
            axis=1,
        )
    else:
        # Para DataFrames vazios, cria a coluna vazia com o tipo correto
        merged_df["common_skills_count"] = pd.Series([], dtype=int)

    print(f"Features de habilidades criadas. Total de {len(common_skills) * 2 + 1} novas features.")

    return merged_df


def create_categorical_features(merged_df):
    """
    Cria features categóricas usando One-Hot Encoding.

    Args:
        merged_df (pd.DataFrame): DataFrame com colunas categóricas

    Returns:
        tuple: (encoded_features, existing_categorical_cols)
    """
    print("\nIniciando One-Hot Encoding para features categóricas...")
    # pylint: disable=duplicate-code
    # Lista de colunas categóricas a serem codificadas
    categorical_cols = [
        "nivel profissional",  # Atenção ao espaço no nome da coluna
        "nivel_academico",
        "nivel_ingles",
        "nivel_espanhol",
        "local",  # Do candidato
        "pais",  # Da vaga
        "estado",  # Da vaga
        "cidade",  # Da vaga
        "regiao",  # Da vaga
    ]
    # pylint: enable=duplicate-code
    # Filtra as colunas que realmente existem no DataFrame mesclado
    existing_categorical_cols = [col for col in categorical_cols if col in merged_df.columns]
    missing_categorical_cols = [col for col in categorical_cols if col not in merged_df.columns]

    if missing_categorical_cols:
        print(
            "Aviso: As seguintes colunas categóricas não foram encontradas no "
            f"DataFrame mesclado e serão ignoradas: {missing_categorical_cols}"
        )

    # Converte todos os valores das colunas categóricas existentes para string e
    # preenche NaNs com string vazia
    merged_df_copy = merged_df.copy()
    for col in existing_categorical_cols:
        merged_df_copy[col] = merged_df_copy[col].astype(str).fillna("")

    # Realiza o One-Hot Encoding
    encoded_features = pd.get_dummies(merged_df_copy[existing_categorical_cols], dummy_na=False)
    print(f"Shape das features categóricas codificadas: {encoded_features.shape}")

    return encoded_features, existing_categorical_cols


def combine_all_features(applicant_tfidf, job_tfidf, encoded_features, merged_df):
    """
    Combina todas as features em uma única matriz.

    Args:
        applicant_tfidf: Matriz TF-IDF dos candidatos
        job_tfidf: Matriz TF-IDF das vagas
        encoded_features: Features categóricas codificadas
        merged_df: DataFrame com features de habilidades

    Returns:
        tuple: (x, y) - Matriz de features e vetor de labels
    """
    print("\nCombinando todas as features...")

    common_skills = get_common_skills()

    # Converte as features categóricas para matriz esparsa
    encoded_features_sparse = csr_matrix(encoded_features)

    # Cria um DataFrame com as features de habilidade para converter em sparse matrix
    skill_features_df = merged_df[
        [f"applicant_has_{skill}" for skill in common_skills]
        + [f"job_requires_{skill}" for skill in common_skills]
        + ["common_skills_count"]
    ]
    skill_features_sparse = csr_matrix(skill_features_df.values)

    # Empilha horizontalmente todas as features
    x = hstack([applicant_tfidf, job_tfidf, encoded_features_sparse, skill_features_sparse])
    y = merged_df["contratado"]  # Variável alvo (0 para não contratado, 1 para contratado)

    print(f"Shape final da matriz de features (x): {x.shape}")
    print(f"Shape final do vetor de rótulos (y): {y.shape}")

    return x, y


def split_data(x, y, test_size=0.2, random_state=42):
    """
    Divide os dados em conjuntos de treinamento e teste.

    Args:
        x: Matriz de features
        y: Vetor de labels
        test_size (float): Proporção dos dados para teste
        random_state (int): Seed para reprodutibilidade

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    print(
        f"\nDividindo dados em treinamento ({100-test_size*100:.0f}%) "
        f"e teste ({test_size*100:.0f}%)..."
    )

    try:
        # Tenta primeiro com estratificação
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        # Se a estratificação falhar (dataset muito pequeno), usa divisão simples
        print(f"Aviso: Estratificação falhou ({str(e)}). Usando divisão simples.")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=None
        )

    print(f"Shape de x_train: {x_train.shape}")
    print(f"Shape de x_test: {x_test.shape}")
    print(f"Shape de y_train: {y_train.shape}")
    print(f"Shape de y_test: {y_test.shape}")

    return x_train, x_test, y_train, y_test


def train_logistic_regression(x_train, y_train):
    """
    Treina um modelo de Regressão Logística.

    Args:
        x_train: Dados de treinamento
        y_train: Labels de treinamento

    Returns:
        LogisticRegression: Modelo treinado
    """
    print("\nIniciando treinamento do modelo Logistic Regression (sem SMOTE)...")

    # Definindo o modelo Logistic Regression
    model = LogisticRegression(
        solver="saga", max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"
    )

    # Treinando o modelo
    model.fit(x_train, y_train)
    print("Treinamento do modelo Logistic Regression concluído.")

    return model


def evaluate_model(model, x_test, y_test, threshold=0.7):
    """
    Avalia o desempenho do modelo.

    Args:
        model: Modelo treinado
        x_test: Dados de teste
        y_test: Labels verdadeiros de teste
        threshold (float): Threshold para classificação

    Returns:
        tuple: (y_pred, y_proba, roc_auc, pr_auc)
    """
    print("\nIniciando avaliação do modelo...")

    # Faz previsões no conjunto de teste
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n--- Relatório de Classificação (threshold={threshold}) ---")
    print(classification_report(y_test, y_pred))

    # Calcula a área sob a curva ROC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nAUC-ROC: {roc_auc:.4f}")

    # Calcula a curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    return y_pred, y_proba, roc_auc, pr_auc, precision, recall


def plot_precision_recall_curve(precision, recall, pr_auc, save_path="precision_recall_curve.png"):
    """
    Gera e salva o gráfico da curva Precision-Recall.

    Args:
        precision: Array de precision values
        recall: Array de recall values
        pr_auc: Área sob a curva Precision-Recall
        save_path (str): Caminho para salvar o gráfico
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Precision-Recall curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()  # Fecha a figura para liberar memória
    print(f"Curva Precision-Recall salva como '{save_path}'.")


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def save_model_and_preprocessors(
    model,
    tfidf_vectorizer_applicant,
    tfidf_vectorizer_job,
    existing_categorical_cols,
    encoded_features,
    models_dir="models/",
    model_version: str = "1.0",
):
    """
    Salva o modelo treinado e os objetos de pré-processamento.

    Args:
        model: Modelo treinado
        tfidf_vectorizer_applicant: Vetorizador TF-IDF para candidatos
        tfidf_vectorizer_job: Vetorizador TF-IDF para vagas
        existing_categorical_cols: Lista de colunas categóricas
        encoded_features: Features categóricas codificadas
        models_dir (str): Diretório para salvar os modelos
    """
    print("\nSalvando o modelo treinado e os objetos de pré-processamento...")

    # Cria o diretório se não existir
    os.makedirs(models_dir, exist_ok=True)

    try:
        # Salva o modelo e preprocessadores
        joblib.dump(model, os.path.join(models_dir, "logistic_regression_model.pkl"))
        joblib.dump(
            tfidf_vectorizer_applicant, os.path.join(models_dir, "tfidf_vectorizer_applicant.pkl")
        )
        joblib.dump(tfidf_vectorizer_job, os.path.join(models_dir, "tfidf_vectorizer_job.pkl"))
        joblib.dump(existing_categorical_cols, os.path.join(models_dir, "categorical_cols.pkl"))
        joblib.dump(
            encoded_features.columns.tolist(), os.path.join(models_dir, "encoded_feature_names.pkl")
        )
        joblib.dump(get_common_skills(), os.path.join(models_dir, "common_skills.pkl"))

        print(
            "Modelo Logistic Regression, vetorizadores TF-IDF, informações de colunas categóricas"
            " e habilidades comuns salvos com sucesso."
        )
        with open(os.path.join(models_dir, "model_version.txt"), "w", encoding="utf-8") as f:
            f.write(model_version)
    except Exception as e:
        print(f"Erro ao salvar os objetos: {e}")
        raise


# pylint: enable=too-many-arguments
# pylint: enable=too-many-positional-arguments


def print_recommendation_guide():
    """Imprime um guia sobre como usar o modelo para recomendações."""
    print("\n--- Exemplo de Recomendação ---")
    print("\nO modelo de classificação foi treinado. Para fazer recomendações, você precisaria:")
    print("1. Carregar o modelo e os objetos de pré-processamento salvos.")
    print(
        "2. Selecionar uma vaga (por exemplo, uma nova vaga ou uma vaga existente para a qual "
        "você quer encontrar candidatos)."
    )
    print("3. Para cada candidato disponível (que ainda não foi contratado para essa vaga):")
    print(
        "   a. Criar uma entrada de dados combinando as características do candidato e as "
        "características da vaga (similar à forma como o 'X' foi construído)."
    )
    print(
        "   b. Pré-processar esta entrada usando os *mesmos* vetorizadores TF-IDF, a lógica de "
        "codificação categórica E a extração de habilidades usados no treinamento."
    )
    print("   c. Usar o modelo `model.predict_proba()` para obter a probabilidade de contratação.")
    print(
        "4. Classificar os candidatos com base nessas probabilidades para gerar uma lista de "
        "recomendação."
    )
    print(
        "\nAs principais métricas do modelo são: Precision, Recall, F1-Score e AUC-ROC, que estão "
        "no relatório de classificação acima e na curva Precision-Recall. Estas métricas ajudam a "
        "entender a capacidade do modelo em identificar candidatos que serão contratados."
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def train_complete_model(
    data_path="data/processed/merged_data_processed.csv",
    models_dir="models/",
    max_features=5000,
    test_size=0.2,
    threshold=0.7,
    random_state=42,
):  # pylint: disable=too-many-locals
    """
    Função principal que executa todo o pipeline de treinamento do modelo.

    Args:
        data_path (str): Caminho para os dados processados
        models_dir (str): Diretório para salvar os modelos
        max_features (int): Número máximo de features TF-IDF
        test_size (float): Proporção dos dados para teste
        threshold (float): Threshold para classificação
        random_state (int): Seed para reprodutibilidade

    Returns:
        dict: Dicionário com os resultados e objetos do treinamento
    """
    # 1. Carrega os dados
    merged_df = load_processed_data(data_path)

    # 2. Pré-processa features de texto
    merged_df = preprocess_text_features(merged_df)

    # 3. Cria features TF-IDF
    applicant_tfidf, job_tfidf, tfidf_vectorizer_applicant, tfidf_vectorizer_job = (
        create_tfidf_features(merged_df, max_features)
    )

    # 4. Cria features de habilidades
    merged_df = create_skill_features(merged_df)

    # 5. Cria features categóricas
    encoded_features, existing_categorical_cols = create_categorical_features(merged_df)

    # 6. Combina todas as features
    x, y = combine_all_features(applicant_tfidf, job_tfidf, encoded_features, merged_df)

    # 7. Divide os dados
    x_train, x_test, y_train, y_test = split_data(x, y, test_size, random_state)

    # 8. Treina o modelo
    model = train_logistic_regression(x_train, y_train)

    # 9. Avalia o modelo
    y_pred, y_proba, roc_auc, pr_auc, precision, recall = evaluate_model(
        model, x_test, y_test, threshold
    )

    # 10. Plota curva Precision-Recall
    plot_precision_recall_curve(precision, recall, pr_auc)

    # 11. Salva modelo e preprocessadores
    save_model_and_preprocessors(
        model,
        tfidf_vectorizer_applicant,
        tfidf_vectorizer_job,
        existing_categorical_cols,
        encoded_features,
        models_dir,
        MODEL_VERSION,
    )

    # 12. Mostra guia de recomendação
    print_recommendation_guide()

    # Retorna resultados
    return {
        "model": model,
        "tfidf_vectorizer_applicant": tfidf_vectorizer_applicant,
        "tfidf_vectorizer_job": tfidf_vectorizer_job,
        "existing_categorical_cols": existing_categorical_cols,
        "encoded_features": encoded_features,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "X_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# pylint: enable=too-many-arguments
# pylint: enable=too-many-positional-arguments

if __name__ == "__main__":
    # Executa o treinamento completo quando o script é executado diretamente
    results = train_complete_model()
    print("\nTreinamento concluído com sucesso!")
    print(f"AUC-ROC: {results['roc_auc']:.4f}")
    print(f"AUC-PR: {results['pr_auc']:.4f}")
