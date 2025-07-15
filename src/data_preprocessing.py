import pandas as pd
import json

# --- Caminhos dos arquivos  ---
raw_data_pasta = '../data/raw/'
applicants_file = raw_data_pasta + 'applicants.json'
vagas_file = raw_data_pasta + 'vagas.json'
prospects_file = raw_data_pasta + 'prospects.json'

# --- Caminhos do arquivo resultante  ---
processed_data_pasta = '../data/processed/'
merged_data_processed = processed_data_pasta + 'merged_data_processed.csv'

# Carregar os arquivos JSON
# Tenta carregar os arquivos JSON fornecidos pelo usuário.
# Em caso de erro (arquivo não encontrado ou JSON inválido), exibe uma mensagem e encerra.
try:
    with open(applicants_file, 'r', encoding='utf-8') as f:
        applicants_data = json.load(f)
    with open(prospects_file, 'r', encoding='utf-8') as f:
        prospects_data = json.load(f)
    with open(vagas_file, 'r', encoding='utf-8') as f:
        vagas_data = json.load(f)

    print("Arquivos JSON carregados com sucesso.")

except FileNotFoundError as e:
    print(
        f"Erro ao carregar o arquivo: {e}. Certifique-se de que os arquivos estão no diretório correto.")
    exit()
except json.JSONDecodeError as e:
    print(
        f"Erro ao decodificar JSON: {e}. Verifique a integridade dos arquivos JSON.")
    exit()

# Normalizar e criar DataFrames
# Processa o dicionário de candidatos para criar um DataFrame plano.
# Combina informações básicas e pessoais, e concatena o histórico profissional em uma única string.
applicants_list = []
for code, data in applicants_data.items():
    applicant_info = {'codigo_profissional': code}
    # Combina informações de 'infos_basicas' e 'informacoes_pessoais'
    if 'infos_basicas' in data:
        applicant_info.update(data['infos_basicas'])
    if 'informacoes_pessoais' in data:
        applicant_info.update(data['informacoes_pessoais'])

    # Adiciona o CV completo se existir, caso contrário, string vazia.
    if 'cv' in data:
        applicant_info['cv_completo'] = data['cv']
    else:
        applicant_info['cv_completo'] = ''

    # Concatena as descrições de atividades do histórico profissional em uma única string.
    if 'historico_profissional' in data and isinstance(data['historico_profissional'], list):
        applicant_info['historico_profissional_texto'] = " ".join(
            [exp.get('descricao_atividades', '') for exp in data['historico_profissional']])
    else:
        applicant_info['historico_profissional_texto'] = ''

    applicants_list.append(applicant_info)

df_applicants = pd.DataFrame(applicants_list)

# Processa o dicionário de prospects para criar um DataFrame onde cada linha é uma aplicação.
# Expande a lista de 'prospects' para ter uma linha por aplicação.
prospects_list = []
for job_code, job_data in prospects_data.items():
    if 'prospects' in job_data:
        for prospect in job_data['prospects']:
            prospect_info = {
                'codigo_vaga': job_code,
                'titulo_vaga_prospect': job_data.get('titulo', ''),
                'modalidade_vaga_prospect': job_data.get('modalidade', '')
            }
            prospect_info.update(prospect)
            prospects_list.append(prospect_info)

df_prospects = pd.DataFrame(prospects_list)

# Processa o dicionário de vagas para criar um DataFrame plano.
# Combina informações básicas e de perfil da vaga.
vagas_list = []
for code, data in vagas_data.items():
    vaga_info = {'codigo_vaga': code}
    if 'informacoes_basicas' in data:
        vaga_info.update(data['informacoes_basicas'])
    if 'perfil_vaga' in data:
        vaga_info.update(data['perfil_vaga'])
    vagas_list.append(vaga_info)

df_vagas = pd.DataFrame(vagas_list)

print("\nDataFrames iniciais criados.")

# --- Merging DataFrames ---
# Renomeia a coluna 'codigo' em df_prospects para 'codigo_profissional' para corresponder a df_applicants.
df_prospects = df_prospects.rename(columns={'codigo': 'codigo_profissional'})

# Mescla df_prospects com df_applicants usando 'codigo_profissional' como chave.
# 'how='left'' garante que todas as linhas de df_prospects sejam mantidas.
merged_df = pd.merge(df_prospects, df_applicants, on='codigo_profissional',
                     how='left', suffixes=('_prospect', '_applicant'))

# Converte a coluna 'codigo_vaga' para string em ambos os DataFrames antes da mesclagem para evitar problemas de tipo.
merged_df['codigo_vaga'] = merged_df['codigo_vaga'].astype(str)
df_vagas['codigo_vaga'] = df_vagas['codigo_vaga'].astype(str)

# Mescla o DataFrame resultante com df_vagas usando 'codigo_vaga' como chave.
merged_df = pd.merge(merged_df, df_vagas, on='codigo_vaga',
                     how='left', suffixes=('_prospect_job', '_job_details'))

print("Merge dos DataFrames concluído.")

# --- Define the Target Variable ---
# Define a variável alvo 'contratado'.
# Um candidato é considerado 'contratado' se a 'situacao_candidado' for "Contratado pela Decision" ou "Contratado como Hunting".
hired_statuses = ["Contratado pela Decision", "Contratado como Hunting"]
merged_df['contratado'] = merged_df['situacao_candidado'].isin(
    hired_statuses).astype(int)

print("\nVariável alvo 'contratado' criada e corrigida.")

# --- Handling Missing Values for Text Columns ---
# Preenche valores NaN em colunas de texto relevantes com strings vazias para evitar erros no TF-IDF.
text_cols_applicants = ['objetivo_profissional',
                        'cv_completo', 'historico_profissional_texto']
text_cols_vagas = ['titulo_vaga', 'principais_atividades',
                   'competencia_tecnicas_e_comportamentais', 'demais_observacoes', 'areas_atuacao']

for col in text_cols_applicants:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna('')
    else:
        print(f"Aviso: Coluna '{col}' não encontrada no DataFrame mesclado.")

for col in text_cols_vagas:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna('')
    else:
        print(f"Aviso: Coluna '{col}' não encontrada no DataFrame mesclado.")

print("\nValores NaN em colunas de texto preenchidos com string vazia.")

# --- Feature Engineering: Combine Text Fields for Applicants and Vagas ---
# Cria uma feature de texto combinada para os candidatos.
merged_df['applicant_text_features'] = merged_df['objetivo_profissional'] + ' ' + \
    merged_df['historico_profissional_texto'] + ' ' + \
    merged_df['cv_completo']

# Cria uma feature de texto combinada para as vagas.
merged_df['job_text_features'] = merged_df['titulo_vaga_prospect'] + ' ' + \
    merged_df['titulo_vaga'] + ' ' + \
    merged_df['principais_atividades'] + ' ' + \
    merged_df['competencia_tecnicas_e_comportamentais'] + ' ' + \
    merged_df['demais_observacoes'] + ' ' + \
    merged_df['areas_atuacao']

print("\nCampos de texto combinados para candidatos e vagas.")

# --- Drop Unnecessary Columns ---
# Lista de colunas a serem excluídas, conforme solicitado pelo usuário.
columns_to_drop = [
    'codigo_vaga',
    'codigo_profissional',
    'data_candidatura',
    'ultima_atualizacao',
    'recrutador',
    'telefone_recado',
    'telefone_prospect_job',
    'data_criacao',
    'inserido_por',
    'email',
    'sabendo_de_nos_por',
    'data_atualizacao',
    'nome_applicant',
    'data_aceite',
    'cpf',
    'fonte_indicacao',
    'email_secundario',
    'data_nascimento',
    'telefone_celular',
    'endereco',
    'skype',
    'url_linkedin',
    'facebook',
    'download_cv',
    'data_requicisao',
    'cliente',
    'solicitante_cliente',
    'empresa_divisao',
    'requisitante',
    'analista_responsavel',
    'prazo_contratacao',
    'prioridade_vaga',
    'superior_imediato',
    'nome',
    'telefone_job_details',
    'local_trabalho',
    'horario_trabalho',
    'viagens_requeridas',
    'equipamentos_necessarios',
    'data_inicial',
    'data_final',
    'nome_substituto',
    # Novas colunas a serem removidas conforme solicitação do usuário
    'sexo',
    'estado_civil',
    'tipo_contratacao',
    'origem_vaga',
    'pais',
    'estado',
    'cidade',
    'regiao'
]

# Colunas que são usadas como features categóricas e NÃO devem ser removidas:
# Atualizado para remover as colunas que agora serão descartadas.
categorical_cols_to_keep = [
    'modalidade_vaga_prospect',
    'nivel profissional',
    'nivel_academico',
    'nivel_ingles',
    'nivel_espanhol',
    'pcd',
    'faixa_etaria',
    'vaga_especifica_para_pcd'
]

# Colunas que são usadas para criar features de texto combinadas e NÃO devem ser removidas:
text_cols_to_keep = [
    'objetivo_profissional', 'cv_completo', 'historico_profissional_texto',
    'titulo_vaga_prospect', 'titulo_vaga', 'principais_atividades',
    'competencia_tecnicas_e_comportamentais', 'demais_observacoes', 'areas_atuacao'
]

# Filtra a lista de colunas a serem removidas, excluindo as que devem ser mantidas.
final_columns_to_drop = [
    col for col in columns_to_drop if col not in categorical_cols_to_keep and col not in text_cols_to_keep]


for col in final_columns_to_drop:
    if col in merged_df.columns:
        merged_df = merged_df.drop(columns=[col])
        print(f"Coluna '{col}' removida.")
    else:
        print(f"Aviso: Coluna '{col}' não encontrada e não pode ser removida.")


# Salva o DataFrame processado para uso posterior no treinamento do modelo.
merged_df.to_csv(merged_data_processed, index=False)
print(f"\nDataFrame processado salvo como {merged_data_processed}.")

# Exibe as primeiras linhas e informações do DataFrame mesclado para verificação.
print("\n--- DataFrame Mesclado (merged_df) - Primeiras 5 linhas ---")
print(merged_df.head())

print("\n--- Informações do DataFrame Mesclado ---")
print(merged_df.info())

print("\n--- Contagem de valores na coluna 'contratado' ---")
print(merged_df['contratado'].value_counts())
