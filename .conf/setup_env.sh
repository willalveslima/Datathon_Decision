#!/bin/bash

# Este script configura o ambiente virtual Python e os hooks de pré-commit.

# --- Configuração para sair em caso de erro ---
# 'set -e' faz com que o script saia imediatamente se um comando falhar.
set -e

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
    # CORREÇÃO: Forçar o uso do python3.11 para criar o venv
    python3.11 -m venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        echo "Erro: Falha ao criar o ambiente virtual. Certifique-se de que 'python3.11' está instalado e no seu PATH."
        exit 1
    fi
else
    echo "Ambiente virtual $VENV_PATH já existe."
fi

echo "Ativando ambiente virtual..."
source "$VENV_PATH/bin/activate"
if [ $? -ne 0 ]; then
    echo "Erro: Falha ao ativar o ambiente virtual."
    exit 1
fi

# --- 3. Instalar pacotes do requirements.txt (se existir) ---
REQUIREMENTS_FILE="requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Instalando pacotes de $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        echo "Erro: Falha ao instalar pacotes de $REQUIREMENTS_FILE."
        exit 1
    fi
else
    echo "Aviso: Arquivo $REQUIREMENTS_FILE não encontrado. Pulando instalação de pacotes principais."
fi

# --- 4. Instalar ferramentas para os hooks de pre-commit (garantir que estejam disponíveis) ---
echo "Instalando ferramentas para hooks de pre-commit (isort, black, pydocstyle, pylint, pytest)..."
pip install isort black pydocstyle pylint pytest
if [ $? -ne 0 ]; then
    echo "Erro: Falha ao instalar ferramentas de pre-commit."
    exit 1
fi

# --- 5. Instalar pre-commit ---
echo "Instalando a ferramenta pre-commit..."
pip install pre-commit
if [ $? -ne 0 ]; then
    echo "Erro: Falha ao instalar a ferramenta pre-commit."
    exit 1
fi

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
  - id: black
    name: Run black
    types: [python]
    exclude: ^tests/
    entry: black
    language: system
    args: ["--line-length=100"]
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
  - id: pytest
    name: Roda pytest
    entry: bash -c "PYTHONPATH=. python -m pytest"
    language: system
    pass_filenames: false
    always_run: true
EOF

if [ $? -ne 0 ]; then
    echo "Erro: Falha ao criar o arquivo $PRE_COMMIT_CONFIG_FILE."
    exit 1
fi
echo "Arquivo $PRE_COMMIT_CONFIG_FILE criado com sucesso."

# --- 7. Criar arquivo pyproject.toml para configuração de isort e black ---
PYPROJECT_TOML_FILE="pyproject.toml"

echo "Criando o arquivo $PYPROJECT_TOML_FILE para configuração de isort e black..."
cat <<EOF > "$PYPROJECT_TOML_FILE"
[tool.isort]
profile = "black"
line_length = 100

[tool.black]
line-length = 100
target-version = ['py311']
EOF

if [ $? -ne 0 ]; then
    echo "Erro: Falha ao criar o arquivo $PYPROJECT_TOML_FILE."
    exit 1
fi
echo "Arquivo $PYPROJECT_TOML_FILE criado com sucesso."


# --- 8. Instalar hooks do pré-commit ---
echo "Instalando hooks do pre-commit..."
pre-commit install
if [ $? -ne 0 ]; then
    echo "Erro: Falha ao instalar hooks do pre-commit."
    exit 1
fi

# --- 9. Executar hooks manualmente pela primeira vez ---
echo "Executando hooks do pre-commit em todos os arquivos pela primeira vez..."
pre-commit run --all-files
if [ $? -ne 0 ]; then
    echo "Erro: Hooks do pre-commit falharam na execução inicial."
    exit 1
fi

echo -e "\nConfiguração do ambiente e pre-commit concluída!"
echo "Para ativar o ambiente virtual em novas sessões do terminal, use: source .venv/bin/activate"
