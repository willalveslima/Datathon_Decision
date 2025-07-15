# Use uma imagem base oficial do Python. Python 3.11 é uma boa escolha.
# 'slim-buster' é uma versão mais leve da imagem base.
FROM python:3.11-slim-buster

# Define o diretório de trabalho dentro do contêiner.
# Todos os comandos subsequentes serão executados a partir deste diretório.
WORKDIR /app

# Copia o arquivo requirements.txt ESPECÍFICO DA API para o diretório de trabalho.
# Assumimos que este Dockerfile está em 'docker/api/' e o requirements.txt está no mesmo diretório.
# O contexto de build (o '.' no comando docker build) deve ser a raiz do seu projeto.
COPY docker/api/requirements.txt .

# Instala as dependências Python.
# O --no-cache-dir é usado para evitar o armazenamento de pacotes em cache, o que reduz o tamanho da imagem.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o diretório 'src' que contém o código da sua API.
# O ponto final '.' indica o diretório de trabalho atual (/app).
COPY src/ ./src/

# Copia o diretório 'models' que contém os arquivos .pkl do modelo.
# O ponto final '.' indica o diretório de trabalho atual (/app).
COPY models/ ./models/

# Expõe a porta em que a aplicação FastAPI será executada.
# Por padrão, sua API está configurada para rodar na porta 8000.
EXPOSE 8000

# Comando para iniciar a aplicação usando Uvicorn.
# --host 0.0.0.0 permite que a aplicação seja acessível de fora do contêiner.
# src.prediction_service:app refere-se ao módulo 'prediction_service' dentro da pasta 'src',
# e 'app' é a instância do FastAPI dentro desse módulo.
CMD ["uvicorn", "src.prediction_service:app", "--host", "0.0.0.0", "--port", "8000"]
