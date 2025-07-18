#!/bin/bash

# Este script configura o ambiente virtual Python e os hooks de pré-commit.

# --- 1. Navegar para a raiz do projeto ---
# Garante que o script seja executado a partir da raiz do projeto,
# independentemente de onde ele foi chamado.
# O script está em .conf/, então precisamos subir um nível.
cd "$(dirname "$0")/.."

echo "Navegando para a raiz do projeto: $(pwd)"

# --- 2. Criar e Ativar o Ambiente Virtual (.venv) ---
VENV_PATH=".venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Criando ambiente virtual em $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
else
    echo "Ambiente virtual $VENV_PATH já existe."
fi

echo "Ativando ambiente virtual..."
source "$VENV_PATH/bin/activate"

# --- 3. Instalar pacotes do requirements.txt (se existir) ---
REQUIREMENTS_FILE="requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Instalando pacotes de $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Aviso: Arquivo $REQUIREMENTS_FILE não encontrado. Pulando instalação de pacotes principais."
fi

# --- 4. Instalar ferramentas para os hooks de pre-commit (garantir que estejam disponíveis) ---
# Estas ferramentas são necessárias para os hooks 'language: system' no .pre-commit-config.yaml
# ATUALIZADO: Incluído 'black'
echo "Instalando ferramentas para hooks de pre-commit (isort, black, pydocstyle, pylint, pytest)..."
pip install isort black pydocstyle pylint pytest

# --- 5. Instalar pre-commit ---
echo "Instalando a ferramenta pre-commit..."
pip install pre-commit

# --- 6. Criar arquivo .pre-commit-config.yaml ---
PRE_COMMIT_CONFIG_FILE=".pre-commit-config.yaml"

echo "Criando o arquivo $PRE_COMMIT_CONFIG_FILE..."
cat <<EOF > "$PRE_COMMIT_CONFIG_FILE"
repos:
- repo: local
  hooks:
  - id: isort
    name: Run isort
    types: [python]
    exclude: ^tests/
    entry: isort
    language: system
  - id: black # NOVO: Hook para black
    name: Run black
    types: [python]
    exclude: ^tests/
    entry: black
    language: system
    args: ["--line-length=100"] # Exemplo: define o comprimento da linha
  - id: pydocstyle
    name: Roda pydocstyle
    types: [python]
    exclude: ^tests/
    entry: pydocstyle
    language: system
  - id: pylint
    name: Roda pylint
    types: [python]
    exclude: ^tests/
    entry: pylint
    language: system
    # Opcional: Adicione argumentos para pylint, como --rcfile=.pylintrc
    # args: ["--rcfile=.pylintrc"]
  - id: pytest
    name: Roda pytest
    entry: pytest
    language: system
    pass_filenames: false
    always_run: true
EOF

echo "Arquivo $PRE_COMMIT_CONFIG_FILE criado com sucesso."

# --- 7. Instalar hooks do pré-commit ---
echo "Instalando hooks do pre-commit..."
pre-commit install

# --- 8. Executar hooks manualmente pela primeira vez ---
echo "Executando hooks do pre-commit em todos os arquivos pela primeira vez..."
pre-commit run --all-files

echo -e "\nConfiguração do ambiente e pre-commit concluída!"
echo "Para ativar o ambiente virtual em novas sessões do terminal, use: source .venv/bin/activate"
