#!/bin/bash

# Este script faz o deploy local da stack Docker Compose.
# Ele assume que o arquivo docker-compose.yml está na raiz do projeto.

echo "Iniciando o deploy local da stack Docker Compose..."

# Executa o Docker Compose a partir da raiz do projeto.
# --build: Garante que as imagens sejam construídas (ou reconstruídas se houver alterações).
# -d: Executa os contêineres em modo 'detached' (em segundo plano).
docker compose -f docker-compose.yml up --build -d # Caminho atualizado

# Verifica se os contêineres foram iniciados com sucesso
if [ $? -eq 0 ]; then
    echo -e "\nDeploy local concluído com sucesso!"
    echo "Sua API estará disponível em: http://localhost:8000"
    echo "O Monitor de Drift estará disponível em: http://localhost:8001"
    echo "O Prometheus estará disponível em: http://localhost:9090"
    echo "O Grafana estará disponível em: http://localhost:3000 (usuário: admin, senha: grafana-api)"
    echo -e "\nPara ver os logs dos serviços, use: docker compose -f docker-compose.yml logs -f"
    echo "Para parar e remover os serviços, use: docker compose -f docker-compose.yml down"
else
    echo -e "\nErro: Falha ao executar o deploy local do Docker Compose."
    echo "Verifique as mensagens de erro acima para mais detalhes."
fi
