#!/bin/bash

# Este script envia uma requisição POST para a API de recomendação de candidatos usando curl.
# Certifique-se de que a API esteja rodando em http://127.0.0.1:8000 antes de executar este script.

# --- Dados da Requisição (JSON) ---
# O corpo da requisição JSON é armazenado em uma variável para melhor legibilidade.
# Certifique-se de que este JSON corresponde aos modelos Pydantic da sua API.
REQUEST_BODY='{
  "job": {
    "titulo_vaga_prospect": "Desenvolvedor Python Sênior",
    "titulo_vaga": "Vaga para Desenvolvedor Backend com Python e Django",
    "principais_atividades": "Desenvolvimento e manutenção de APIs, integração com sistemas externos, testes unitários e de integração.",
    "competencia_tecnicas_e_comportamentais": "Python, Django, REST APIs, SQL, Git, Metodologias Ágeis. Proatividade, comunicação e trabalho em equipe.",
    "demais_observacoes": "Experiência com cloud computing (AWS/GCP) é um diferencial.",
    "areas_atuacao": "TI - Desenvolvimento/Programação",
    "modalidade_vaga_prospect": "CLT Full",
    "nivel_profissional": "Sênior",
    "nivel_academico": "Ensino Superior Completo",
    "nivel_ingles": "Avançado",
    "nivel_espanhol": "Básico",
    "faixa_etaria": "De: 25 Até: 40",
    "horario_trabalho": "Comercial",
    "vaga_especifica_para_pcd": "Não"
  },
  "applicants": [
    {
      "objetivo_profissional": "Desenvolvedor Backend Sênior com foco em Python e Django.",
      "historico_profissional_texto": "Liderança de equipes no desenvolvimento de APIs RESTful robustas com Python e Django. Experiência em arquitetura de microsserviços, testes automatizados e implantação em nuvem (AWS).",
      "cv_completo": "Currículo detalhado com 8 anos de experiência em Python, Django, DRF, PostgreSQL, Docker, Kubernetes, AWS. Inglês fluente. Forte atuação em otimização de performance e segurança.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Desenvolvedor Backend Júnior buscando crescimento.",
      "historico_profissional_texto": "2 anos de experiência com Python e Flask, desenvolvendo pequenas APIs e scripts de automação.",
      "cv_completo": "Conhecimento em Python, Flask, MongoDB. Inglês intermediário. Buscando oportunidade para aprender e crescer.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Arquiteto de Software com foco em sistemas distribuídos.",
      "historico_profissional_texto": "Mais de 10 anos de experiência em arquitetura de sistemas Java, Spring Boot e microserviços.",
      "cv_completo": "Experiência sólida em Java, Spring, Kafka, Docker. Inglês fluente.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Estagiário em TI.",
      "historico_profissional_texto": "Cursando Análise e Desenvolvimento de Sistemas. Conhecimento básico em lógica de programação e pacote Office.",
      "cv_completo": "Interesse em aprender novas tecnologias. Inglês básico.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Cientista de Dados com Python.",
      "historico_profissional_texto": "3 anos de experiência em análise de dados, machine learning com Python (Pandas, Scikit-learn).",
      "cv_completo": "Projetos em análise preditiva, visualização de dados. Inglês avançado.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Engenheiro de Software Python Sênior.",
      "historico_profissional_texto": "Desenvolvimento de sistemas escaláveis em Python, com foco em performance e segurança. Experiência com APIs RESTful e bancos de dados relacionais (PostgreSQL).",
      "cv_completo": "Proficiência em Python, SQL, AWS. Experiência com testes automatizados e integração contínua. Inglês avançado.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Desenvolvedor Python Júnior.",
      "historico_profissional_texto": "Recém-formado em Ciência da Computação. Projeto de TCC em Python para web scraping.",
      "cv_completo": "Conhecimento em Python, HTML, CSS. Inglês intermediário.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Engenheiro DevOps com Python.",
      "historico_profissional_texto": "Automação de infraestrutura com Python, Ansible e Terraform. Experiência com CI/CD e Docker.",
      "cv_completo": "Foco em automação e otimização de processos. Inglês fluente.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Desenvolvedor .NET Sênior.",
      "historico_profissional_texto": "Mais de 8 anos de experiência em desenvolvimento de aplicações web com C# e .NET Core.",
      "cv_completo": "Conhecimento em ASP.NET, SQL Server, Azure. Inglês avançado.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Analista de Sistemas.",
      "historico_profissional_texto": "Suporte a sistemas, levantamento de requisitos e documentação. Conhecimento em diversas ferramentas de gestão.",
      "cv_completo": "Habilidades de comunicação e resolução de problemas. Inglês básico.",
      "pcd": "Não"
    },
    {
      "objetivo_profissional": "Especialista em Backend Python/Django, buscando desafios em nuvem.",
      "historico_profissional_texto": "Comprovada experiência de 7 anos no desenvolvimento e otimização de sistemas críticos em Python/Django, incluindo design de banco de dados (PostgreSQL) e integração contínua. Proficiência em AWS e GCP para escalabilidade e resiliência.",
      "cv_completo": "Portfólio com projetos complexos em Python, Django REST Framework, microsserviços, Docker, Kubernetes, AWS (EC2, Lambda, S3, RDS), GCP (Compute Engine, Cloud SQL). Inglês fluente. Forte capacidade analítica e de resolução de problemas.",
      "pcd": "Não"
    }
  ]
}'

# --- Enviar a Requisição Curl ---
echo "Enviando requisição POST para http://127.0.0.1:8000/predict..."

# Imprime o comando curl completo no console para depuração
echo "Comando a ser executado:"
echo "curl -X POST \"http://127.0.0.1:8000/predict\" -H \"Content-Type: application/json\" -d '$REQUEST_BODY'"
echo "" # Linha em branco para melhor legibilidade

curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "$REQUEST_BODY"

echo -e "\nRequisição concluída."
