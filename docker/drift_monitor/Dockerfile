# Use a mesma imagem base do Python da sua API para consistência
FROM python:3.11-slim-buster

# Define o fuso horário para o contêiner.
# 'America/Sao_Paulo' é um fuso horário comum no Brasil (UTC-3).
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Copia os requisitos específicos do serviço de drift
COPY docker/drift_monitor/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia o diretório do serviço de drift
COPY src/drift_monitor_service/ ./src/drift_monitor_service/

# Copia os modelos e transformadores que podem ser necessários para carregar o baseline
# ou para pré-processar dados para cálculo de drift (se o modelo for usado para feature extraction)
# Embora o modelo principal não seja carregado aqui, os transformadores (TF-IDF, OHE info) podem ser.
# Por simplicidade, vamos copiar a pasta models inteira.
COPY models/ ./models/

# Expõe a porta em que o serviço de drift será executado
# Usaremos uma porta diferente da API principal
EXPOSE 8001 

# Comando para iniciar o serviço
# O nome do módulo será src.drift_monitor_service.main e a instância da aplicação 'app'
CMD ["uvicorn", "src.drift_monitor_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
