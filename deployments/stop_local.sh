#!/bin/bash

# Este script para e remove os serviços da stack Docker Compose.
# Ele assume que o arquivo docker-compose.yml está em 'deployments/docker-compose.yml'.

echo "Parando e removendo os serviços da stack Docker Compose..."

# Executa o Docker Compose a partir da raiz do projeto, especificando o arquivo compose.
# down: Para e remove os contêineres, redes e volumes criados pelo 'up'.
docker compose -f deployments/docker-compose.yml down

# Verifica se os contêineres foram parados e removidos com sucesso
if [ $? -eq 0 ]; then
    echo -e "\nServiços parados e removidos com sucesso!"
else
    echo -e "\nErro: Falha ao parar e remover os serviços do Docker Compose."
    echo "Verifique as mensagens de erro acima para mais detalhes."
fi
