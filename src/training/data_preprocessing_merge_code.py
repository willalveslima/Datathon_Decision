"""Módulo de Pré-processamento e Fusão de Dados.

Este módulo contém funções para carregar os dados brutos de candidatos, vagas e
prospects a partir de arquivos JSON, realizar a limpeza, normalização, fusão
e engenharia de features, e salvar o resultado em um único arquivo CSV
processado, pronto para ser utilizado na etapa de treinamento do modelo de Machine Learning.

O processo consiste nas seguintes etapas:
1.  **Carregamento de Dados**: Carrega os arquivos `applicants.json`,
   `prospects.json` e `vagas.json`     do diretório `data/raw/`.
2.  **Normalização para DataFrames**: Converte os dados JSON, que possuem estruturas aninhadas,
    em DataFrames do pandas, achatando as informações e tratando campos ausentes.
3.  **Fusão de DataFrames**: Realiza o merge dos DataFrames de prospects, candidatos e vagas
    para criar um conjunto de dados unificado, onde cada linha representa a interação de um
    candidato com uma vaga.
4.  **Filtragem de Registros**: Remove registros de candidatos cuja situação na vaga
    (ex: "Desistiu", "Inscrito") não é relevante para o treinamento do modelo.
5.  **Criação da Variável Alvo**: Cria a coluna 'contratado' (variável alvo) com base na
    coluna 'situacao_candidado', mapeando status como "Proposta Aceita" e "Aprovado" para 1
    e os demais para 0.
6.  **Engenharia de Features**:
    - Preenche valores nulos em colunas de texto importantes.
    - Combina múltiplos campos de texto de candidatos e vagas em duas colunas unificadas:
      `applicant_text_features` e `job_text_features`, para facilitar a vetorização.
7.  **Limpeza de Colunas**: Remove um grande número de colunas desnecessárias que não serão
    utilizadas como features no modelo, mantendo apenas as informações relevantes.
8.  **Salvamento do Resultado**: Salva o DataFrame final e processado no arquivo
    `data/processed/merged_data_processed.csv`.
"""

import json
import sys
from typing import Any, Dict, Tuple

import pandas as pd

# --- Caminhos dos arquivos ---
# Ajustado para que o script possa ser executado a partir de src/training/
RAW_DATA_PASTA = "data/raw/"
APPLICANTS_FILE = RAW_DATA_PASTA + "applicants.json"
VAGAS_FILE = RAW_DATA_PASTA + "vagas.json"
PROSPECTS_FILE = RAW_DATA_PASTA + "prospects.json"

# --- Caminhos do arquivo resultante ---
# Ajustado para que o script possa ser executado a partir de src/training/
PROCESSED_DATA_PASTA = "data/processed/"
MERGED_DATA_PROCESSED_FILE = PROCESSED_DATA_PASTA + "merged_data_processed.csv"


def load_json_files(
    applicants_file: str, prospects_file: str, vagas_file: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Carrega os arquivos JSON necessários para o processamento.

    Args:
        applicants_file: Caminho para o arquivo applicants.json
        prospects_file: Caminho para o arquivo prospects.json
        vagas_file: Caminho para o arquivo vagas.json

    Returns:
        Tupla contendo os dados carregados dos três arquivos JSON

    Raises:
        FileNotFoundError: Se algum arquivo não for encontrado
        json.JSONDecodeError: Se houver erro na decodificação do JSON
    """
    try:
        with open(applicants_file, "r", encoding="utf-8") as f:
            applicants_data = json.load(f)
        with open(prospects_file, "r", encoding="utf-8") as f:
            prospects_data = json.load(f)
        with open(vagas_file, "r", encoding="utf-8") as f:
            vagas_data = json.load(f)

        print("Arquivos JSON carregados com sucesso.")
        return applicants_data, prospects_data, vagas_data

    except FileNotFoundError as e:
        error_msg = (
            f"Erro ao carregar o arquivo: {e}. "
            "Certifique-se de que os arquivos "
            "estão no diretório correto."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg) from e
    except json.JSONDecodeError as e:
        error_msg = f"Erro ao decodificar JSON: {e}. Verifique a integridade dos arquivos JSON."
        print(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e


def normalize_applicants_data(applicants_data: Dict[str, Any]) -> pd.DataFrame:
    """Normaliza os dados dos candidatos para um DataFrame.

    Args:
        applicants_data: Dados dos candidatos em formato de dicionário

    Returns:
        DataFrame com os dados dos candidatos normalizados
    """
    applicants_list = []
    for code, data in applicants_data.items():
        applicant_info = {"codigo_profissional": code}
        if "infos_basicas" in data:
            applicant_info.update(data["infos_basicas"])
        if "informacoes_pessoais" in data:
            applicant_info.update(data["informacoes_pessoais"])

        if "cv" in data:
            applicant_info["cv_completo"] = data["cv"]
        else:
            applicant_info["cv_completo"] = ""

        if "historico_profissional" in data and isinstance(data["historico_profissional"], list):
            applicant_info["historico_profissional_texto"] = " ".join(
                [exp.get("descricao_atividades", "") for exp in data["historico_profissional"]]
            )
        else:
            applicant_info["historico_profissional_texto"] = ""

        applicants_list.append(applicant_info)

    return pd.DataFrame(applicants_list)


def normalize_prospects_data(prospects_data: Dict[str, Any]) -> pd.DataFrame:
    """Normaliza os dados dos prospects para um DataFrame.

    Args:
        prospects_data: Dados dos prospects em formato de dicionário

    Returns:
        DataFrame com os dados dos prospects normalizados
    """
    prospects_list = []
    for job_code, job_data in prospects_data.items():
        if "prospects" in job_data and isinstance(job_data["prospects"], list):
            for prospect in job_data["prospects"]:
                prospect_info = {
                    "codigo_vaga": job_code,
                    "titulo_vaga_prospect": job_data.get("titulo", ""),
                    "modalidade_vaga_prospect": job_data.get("modalidade", ""),
                }
                prospect_info.update(prospect)
                prospects_list.append(prospect_info)

    return pd.DataFrame(prospects_list)


def normalize_vagas_data(vagas_data: Dict[str, Any]) -> pd.DataFrame:
    """Normaliza os dados das vagas para um DataFrame.

    Args:
        vagas_data: Dados das vagas em formato de dicionário

    Returns:
        DataFrame com os dados das vagas normalizados
    """
    vagas_list = []
    for code, data in vagas_data.items():
        vaga_info = {"codigo_vaga": code}
        if "informacoes_basicas" in data:
            vaga_info.update(data["informacoes_basicas"])
        if "perfil_vaga" in data:
            vaga_info.update(data["perfil_vaga"])
        vagas_list.append(vaga_info)

    return pd.DataFrame(vagas_list)


def merge_dataframes(
    df_prospects: pd.DataFrame, df_applicants: pd.DataFrame, df_vagas: pd.DataFrame
) -> pd.DataFrame:
    """Realiza o merge dos DataFrames de prospects, candidatos e vagas.

    Args:
        df_prospects: DataFrame dos prospects
        df_applicants: DataFrame dos candidatos
        df_vagas: DataFrame das vagas

    Returns:
        DataFrame mesclado
    """
    df_prospects = df_prospects.rename(columns={"codigo": "codigo_profissional"})
    merged_df = pd.merge(
        df_prospects,
        df_applicants,
        on="codigo_profissional",
        how="left",
        suffixes=("_prospect", "_applicant"),
    )
    merged_df["codigo_vaga"] = merged_df["codigo_vaga"].astype(str)
    df_vagas["codigo_vaga"] = df_vagas["codigo_vaga"].astype(str)
    merged_df = pd.merge(
        merged_df,
        df_vagas,
        on="codigo_vaga",
        how="left",
        suffixes=("_prospect_job", "_job_details"),
    )

    print("Merge dos DataFrames concluído.")
    return merged_df


def filter_records_by_status(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra registros com base na situação do candidato.

    Args:
        df: DataFrame a ser filtrado

    Returns:
        DataFrame filtrado
    """
    situacoes_a_excluir = [
        "Desistiu",
        "Desistiu da Contratação",
        "Em avaliação pelo RH",
        "Encaminhado ao Requisitante",
        "Encaminhar Proposta",
        "Inscrito",
        "Sem interesse nesta vaga",
    ]

    initial_rows = len(df)
    filtered_df = df[~df["situacao_candidado"].isin(situacoes_a_excluir)]
    rows_removed = initial_rows - len(filtered_df)
    print(f"Removidos {rows_removed} registros com situações a excluir.")
    print(f"Total de registros após filtragem: {len(filtered_df)}")

    return filtered_df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo 'contratado' baseada na situação do candidato.

    Args:
        df: DataFrame para adicionar a variável alvo

    Returns:
        DataFrame com a variável alvo adicionada
    """
    hired_statuses = [
        "Proposta Aceita",
        "Aprovado",
        "Contratado como Hunting",
        "Contratado pela Decision",
        "Entrevista Técnica",
        "Entrevista com Cliente",
        "Documentação CLT",
        "Documentação Cooperado",
        "Documentação PJ",
    ]
    df_copy = df.copy()
    df_copy["contratado"] = df_copy["situacao_candidado"].isin(hired_statuses).astype(int)

    print("\nVariável alvo 'contratado' criada e corrigida.")
    return df_copy


def handle_missing_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche valores nulos em colunas de texto com string vazia.

    Args:
        df: DataFrame para tratar valores ausentes

    Returns:
        DataFrame com valores nulos preenchidos
    """
    text_cols_applicants = [
        "objetivo_profissional",
        "cv_completo",
        "historico_profissional_texto",
    ]
    text_cols_vagas = [
        "titulo_vaga",
        "principais_atividades",
        "competencia_tecnicas_e_comportamentais",
        "demais_observacoes",
        "areas_atuacao",
    ]

    df_copy = df.copy()

    for col in text_cols_applicants:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna("")
        else:
            print(f"Aviso: Coluna '{col}' não encontrada no DataFrame mesclado.")

    for col in text_cols_vagas:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna("")
        else:
            print(f"Aviso: Coluna '{col}' não encontrada no DataFrame mesclado.")

    print("\nValores NaN em colunas de texto preenchidos com string vazia.")
    return df_copy


def create_combined_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combina campos de texto para candidatos e vagas em features unificadas.

    Args:
        df: DataFrame para adicionar features de texto combinadas

    Returns:
        DataFrame com features de texto combinadas
    """
    df_copy = df.copy()

    # Colunas para features de candidatos
    applicant_cols = ["objetivo_profissional", "historico_profissional_texto", "cv_completo"]
    applicant_parts = []
    for col in applicant_cols:
        if col in df_copy.columns:
            applicant_parts.append(df_copy[col].astype(str))
        else:
            applicant_parts.append(pd.Series([""] * len(df_copy), index=df_copy.index))

    df_copy["applicant_text_features"] = applicant_parts[0]
    for part in applicant_parts[1:]:
        df_copy["applicant_text_features"] = df_copy["applicant_text_features"] + " " + part

    # Colunas para features de vagas
    job_cols = [
        "titulo_vaga_prospect",
        "titulo_vaga",
        "principais_atividades",
        "competencia_tecnicas_e_comportamentais",
        "demais_observacoes",
        "areas_atuacao",
    ]
    job_parts = []
    for col in job_cols:
        if col in df_copy.columns:
            job_parts.append(df_copy[col].astype(str))
        else:
            job_parts.append(pd.Series([""] * len(df_copy), index=df_copy.index))

    df_copy["job_text_features"] = job_parts[0]
    for part in job_parts[1:]:
        df_copy["job_text_features"] = df_copy["job_text_features"] + " " + part

    # Limpar espaços extras
    df_copy["applicant_text_features"] = (
        df_copy["applicant_text_features"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df_copy["job_text_features"] = (
        df_copy["job_text_features"].str.replace(r"\s+", " ", regex=True).str.strip()
    )

    print("\nCampos de texto combinados para candidatos e vagas.")
    return df_copy


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas desnecessárias do DataFrame.

    Args:
        df: DataFrame para remoção de colunas

    Returns:
        DataFrame com colunas desnecessárias removidas
    """
    # Lista de colunas a serem excluídas.
    columns_to_drop = [
        "codigo_vaga",
        "codigo_profissional",
        "data_candidatura",
        "ultima_atualizacao",
        "recrutador",
        "telefone_recado",
        "telefone_prospect_job",
        "data_criacao",
        "inserido_por",
        "email",
        "sabendo_de_nos_por",
        "data_atualizacao",
        "nome_applicant",
        "data_aceite",
        "cpf",
        "fonte_indicacao",
        "email_secundario",
        "data_nascimento",
        "telefone_celular",
        "endereco",
        "skype",
        "url_linkedin",
        "facebook",
        "download_cv",
        "data_requicisao",
        "cliente",
        "solicitante_cliente",
        "empresa_divisao",
        "requisitante",
        "analista_responsavel",
        "prazo_contratacao",
        "prioridade_vaga",
        "superior_imediato",
        "nome",
        "telefone_job_details",
        "local_trabalho",
        "horario_trabalho",
        "viagens_requeridas",
        "equipamentos_necessarios",
        "data_inicial",
        "data_final",
        "nome_substituto",
        "sexo",
        "estado_civil",
        "tipo_contratacao",
        "origem_vaga",
        "pcd",
        "modalidade_vaga_prospect",
        "faixa_etaria",
        "vaga_especifica_para_pcd",
    ]

    # Colunas que são usadas como features categóricas e NÃO devem ser removidas.
    categorical_cols_to_keep = [
        "nivel profissional",
        "nivel_academico",
        "nivel_ingles",
        "nivel_espanhol",
        "local",
        "pais",
        "estado",
        "cidade",
        "regiao",
    ]

    # Colunas que são usadas para criar features de texto combinadas e NÃO devem ser removidas.
    text_cols_to_keep = [
        "objetivo_profissional",
        "cv_completo",
        "historico_profissional_texto",
        "titulo_vaga_prospect",
        "titulo_vaga",
        "principais_atividades",
        "competencia_tecnicas_e_comportamentais",
        "demais_observacoes",
        "areas_atuacao",
    ]

    # Filtra a lista de colunas a serem removidas, excluindo as que devem ser mantidas.
    final_columns_to_drop = [
        col
        for col in columns_to_drop
        if col not in categorical_cols_to_keep and col not in text_cols_to_keep
    ]

    df_copy = df.copy()
    for col in final_columns_to_drop:
        if col in df_copy.columns:
            df_copy = df_copy.drop(columns=[col])
            print(f"Coluna '{col}' removida.")
        else:
            print(f"Aviso: Coluna '{col}' não encontrada e não pode ser removida.")

    return df_copy


def save_processed_data(df: pd.DataFrame, output_file: str) -> None:
    """Salva o DataFrame processado em arquivo CSV.

    Args:
        df: DataFrame a ser salvo
        output_file: Caminho do arquivo de saída
    """
    df.to_csv(output_file, index=False)
    print(f"\nDataFrame processado salvo como '{output_file}'.")


def display_dataframe_info(df: pd.DataFrame) -> None:
    """Exibe informações do DataFrame para verificação.

    Args:
        df: DataFrame para exibir informações
    """
    print("\n--- DataFrame Mesclado (merged_df) - Primeiras 5 linhas ---")
    print(df.head())

    print("\n--- Informações do DataFrame Mesclado ---")
    print(df.info())

    print("\n--- Contagem de valores na coluna 'contratado' ---")
    print(df["contratado"].value_counts())


def process_data_pipeline(
    applicants_file: str = APPLICANTS_FILE,
    prospects_file: str = PROSPECTS_FILE,
    vagas_file: str = VAGAS_FILE,
    output_file: str = MERGED_DATA_PROCESSED_FILE,
) -> pd.DataFrame:
    """Pipeline completo de processamento de dados.

    Args:
        applicants_file: Caminho do arquivo de candidatos
        prospects_file: Caminho do arquivo de prospects
        vagas_file: Caminho do arquivo de vagas
        output_file: Caminho do arquivo de saída

    Returns:
        DataFrame processado final
    """
    # 1. Carregar dados
    applicants_data, prospects_data, vagas_data = load_json_files(
        applicants_file, prospects_file, vagas_file
    )

    # 2. Normalizar dados para DataFrames
    df_applicants = normalize_applicants_data(applicants_data)
    df_prospects = normalize_prospects_data(prospects_data)
    df_vagas = normalize_vagas_data(vagas_data)
    print("\nDataFrames iniciais criados.")

    # 3. Fazer merge dos DataFrames
    merged_df = merge_dataframes(df_prospects, df_applicants, df_vagas)

    # 4. Filtrar registros por situação
    merged_df = filter_records_by_status(merged_df)

    # 5. Criar variável alvo
    merged_df = create_target_variable(merged_df)

    # 6. Tratar valores ausentes em texto
    merged_df = handle_missing_text_values(merged_df)

    # 7. Criar features de texto combinadas
    merged_df = create_combined_text_features(merged_df)

    # 8. Remover colunas desnecessárias
    merged_df = drop_unnecessary_columns(merged_df)

    # 9. Salvar dados processados
    save_processed_data(merged_df, output_file)

    # 10. Exibir informações finais
    display_dataframe_info(merged_df)

    return merged_df


def main():
    """Função principal para executar o pipeline de processamento."""
    try:
        process_data_pipeline()
    except FileNotFoundError as e:
        print(f"Erro de arquivo não encontrado: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Erro de decodificação JSON: {e}")
        sys.exit(1)
    # Removido o tratamento genérico de Exception para evitar captura excessiva de erros.


if __name__ == "__main__":
    main()
