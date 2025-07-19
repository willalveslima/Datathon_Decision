"""Testes unitários para o módulo de pré-processamento de dados."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open

import pandas as pd
import pytest

# Adiciona o diretório src ao path para permitir importações
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.training.data_preprocessing_merge_code import (
    load_json_files,
    normalize_applicants_data,
    normalize_prospects_data,
    normalize_vagas_data,
    merge_dataframes,
    filter_records_by_status,
    create_target_variable,
    handle_missing_text_values,
    create_combined_text_features,
    drop_unnecessary_columns,
    save_processed_data,
    process_data_pipeline
)


class TestDataPreprocessing(unittest.TestCase):
    """Classe de testes para funções de pré-processamento de dados."""

    def setUp(self):
        """Configuração inicial para os testes."""
        # Dados mock para testes
        self.mock_applicants_data = {
            "001": {
                "infos_basicas": {
                    "nome": "João Silva",
                    "objetivo_profissional": "Desenvolvedor Python"
                },
                "informacoes_pessoais": {
                    "idade": 30,
                    "cidade": "São Paulo"
                },
                "cv": "Experiência em desenvolvimento web",
                "historico_profissional": [
                    {"descricao_atividades": "Desenvolvimento de APIs"},
                    {"descricao_atividades": "Análise de dados"}
                ]
            },
            "002": {
                "infos_basicas": {
                    "nome": "Maria Santos"
                },
                "informacoes_pessoais": {
                    "idade": 25
                }
            }
        }

        self.mock_prospects_data = {
            "V001": {
                "titulo": "Desenvolvedor Backend",
                "modalidade": "Remoto",
                "prospects": [
                    {
                        "codigo": "001",
                        "situacao_candidado": "Proposta Aceita",
                        "data_candidatura": "2024-01-01"
                    },
                    {
                        "codigo": "002",
                        "situacao_candidado": "Inscrito",
                        "data_candidatura": "2024-01-02"
                    }
                ]
            }
        }

        self.mock_vagas_data = {
            "V001": {
                "informacoes_basicas": {
                    "titulo_vaga": "Desenvolvedor Backend Senior",
                    "local": "São Paulo"
                },
                "perfil_vaga": {
                    "principais_atividades": "Desenvolvimento de microserviços",
                    "competencia_tecnicas_e_comportamentais": "Python, FastAPI"
                }
            }
        }

    def test_load_json_files_success(self):
        """Testa o carregamento bem-sucedido de arquivos JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar arquivos temporários
            applicants_file = os.path.join(temp_dir, "applicants.json")
            prospects_file = os.path.join(temp_dir, "prospects.json")
            vagas_file = os.path.join(temp_dir, "vagas.json")

            with open(applicants_file, 'w') as f:
                json.dump(self.mock_applicants_data, f)
            with open(prospects_file, 'w') as f:
                json.dump(self.mock_prospects_data, f)
            with open(vagas_file, 'w') as f:
                json.dump(self.mock_vagas_data, f)

            # Testar função
            applicants, prospects, vagas = load_json_files(
                applicants_file, prospects_file, vagas_file
            )

            self.assertEqual(applicants, self.mock_applicants_data)
            self.assertEqual(prospects, self.mock_prospects_data)
            self.assertEqual(vagas, self.mock_vagas_data)

    def test_load_json_files_file_not_found(self):
        """Testa erro quando arquivo não é encontrado."""
        with self.assertRaises(FileNotFoundError):
            load_json_files("inexistente.json", "inexistente.json", "inexistente.json")

    def test_load_json_files_invalid_json(self):
        """Testa erro quando arquivo JSON é inválido."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = os.path.join(temp_dir, "invalid.json")
            with open(invalid_file, 'w') as f:
                f.write("invalid json content")

            with self.assertRaises(json.JSONDecodeError):
                load_json_files(invalid_file, invalid_file, invalid_file)

    def test_normalize_applicants_data(self):
        """Testa a normalização de dados de candidatos."""
        df = normalize_applicants_data(self.mock_applicants_data)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("codigo_profissional", df.columns)
        self.assertIn("cv_completo", df.columns)
        self.assertIn("historico_profissional_texto", df.columns)

        # Verificar primeiro candidato
        first_row = df.iloc[0]
        self.assertEqual(first_row["codigo_profissional"], "001")
        self.assertEqual(first_row["nome"], "João Silva")
        self.assertEqual(first_row["cv_completo"], "Experiência em desenvolvimento web")
        self.assertIn("Desenvolvimento de APIs", first_row["historico_profissional_texto"])

        # Verificar segundo candidato (dados incompletos)
        second_row = df.iloc[1]
        self.assertEqual(second_row["codigo_profissional"], "002")
        self.assertEqual(second_row["cv_completo"], "")
        self.assertEqual(second_row["historico_profissional_texto"], "")

    def test_normalize_prospects_data(self):
        """Testa a normalização de dados de prospects."""
        df = normalize_prospects_data(self.mock_prospects_data)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("codigo_vaga", df.columns)
        self.assertIn("titulo_vaga_prospect", df.columns)
        self.assertIn("modalidade_vaga_prospect", df.columns)

        # Verificar dados
        self.assertEqual(df.iloc[0]["codigo_vaga"], "V001")
        self.assertEqual(df.iloc[0]["titulo_vaga_prospect"], "Desenvolvedor Backend")
        self.assertEqual(df.iloc[0]["modalidade_vaga_prospect"], "Remoto")

    def test_normalize_vagas_data(self):
        """Testa a normalização de dados de vagas."""
        df = normalize_vagas_data(self.mock_vagas_data)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn("codigo_vaga", df.columns)
        self.assertIn("titulo_vaga", df.columns)
        self.assertIn("principais_atividades", df.columns)

        # Verificar dados
        first_row = df.iloc[0]
        self.assertEqual(first_row["codigo_vaga"], "V001")
        self.assertEqual(first_row["titulo_vaga"], "Desenvolvedor Backend Senior")
        self.assertEqual(first_row["principais_atividades"], "Desenvolvimento de microserviços")

    def test_merge_dataframes(self):
        """Testa o merge de DataFrames."""
        df_applicants = normalize_applicants_data(self.mock_applicants_data)
        df_prospects = normalize_prospects_data(self.mock_prospects_data)
        df_vagas = normalize_vagas_data(self.mock_vagas_data)

        merged_df = merge_dataframes(df_prospects, df_applicants, df_vagas)

        self.assertIsInstance(merged_df, pd.DataFrame)
        self.assertEqual(len(merged_df), 2)
        
        # Verificar se colunas importantes estão presentes
        expected_columns = [
            "codigo_vaga", "codigo_profissional", "situacao_candidado",
            "nome", "titulo_vaga", "principais_atividades"
        ]
        for col in expected_columns:
            self.assertIn(col, merged_df.columns)

    def test_filter_records_by_status(self):
        """Testa a filtragem de registros por situação."""
        # Criar DataFrame mock
        df = pd.DataFrame({
            "situacao_candidado": [
                "Proposta Aceita",
                "Inscrito",
                "Desistiu",
                "Aprovado"
            ],
            "codigo_profissional": ["001", "002", "003", "004"]
        })

        filtered_df = filter_records_by_status(df)

        # Deve manter apenas registros não excluídos
        self.assertEqual(len(filtered_df), 2)
        remaining_statuses = filtered_df["situacao_candidado"].tolist()
        self.assertIn("Proposta Aceita", remaining_statuses)
        self.assertIn("Aprovado", remaining_statuses)
        self.assertNotIn("Inscrito", remaining_statuses)
        self.assertNotIn("Desistiu", remaining_statuses)

    def test_create_target_variable(self):
        """Testa a criação da variável alvo."""
        df = pd.DataFrame({
            "situacao_candidado": [
                "Proposta Aceita",
                "Aprovado",
                "Rejeitado",
                "Entrevista Técnica"
            ]
        })

        df_with_target = create_target_variable(df)

        self.assertIn("contratado", df_with_target.columns)
        
        # Verificar mapeamento correto
        expected_values = [1, 1, 0, 1]  # Entrevista Técnica é considerada contratado
        actual_values = df_with_target["contratado"].tolist()
        self.assertEqual(actual_values, expected_values)

    def test_handle_missing_text_values(self):
        """Testa o tratamento de valores ausentes em colunas de texto."""
        df = pd.DataFrame({
            "objetivo_profissional": ["Desenvolver software", None],
            "cv_completo": [None, "CV completo"],
            "titulo_vaga": ["Vaga 1", None],
            "principais_atividades": ["Atividade 1", "Atividade 2"]
        })

        df_filled = handle_missing_text_values(df)

        # Verificar se NaN foram substituídos por string vazia
        self.assertEqual(df_filled["objetivo_profissional"].iloc[1], "")
        self.assertEqual(df_filled["cv_completo"].iloc[0], "")
        self.assertEqual(df_filled["titulo_vaga"].iloc[1], "")
        
        # Verificar se valores existentes foram mantidos
        self.assertEqual(df_filled["objetivo_profissional"].iloc[0], "Desenvolver software")
        self.assertEqual(df_filled["cv_completo"].iloc[1], "CV completo")

    def test_create_combined_text_features(self):
        """Testa a criação de features de texto combinadas."""
        df = pd.DataFrame({
            "objetivo_profissional": ["Objetivo 1", "Objetivo 2"],
            "historico_profissional_texto": ["Historico 1", "Historico 2"],
            "cv_completo": ["CV 1", "CV 2"],
            "titulo_vaga_prospect": ["Titulo P1", "Titulo P2"],
            "titulo_vaga": ["Titulo V1", "Titulo V2"],
            "principais_atividades": ["Atividades 1", "Atividades 2"],
            "competencia_tecnicas_e_comportamentais": ["Comp 1", "Comp 2"],
            "demais_observacoes": ["Obs 1", "Obs 2"],
            "areas_atuacao": ["Area 1", "Area 2"]
        })

        df_with_features = create_combined_text_features(df)

        self.assertIn("applicant_text_features", df_with_features.columns)
        self.assertIn("job_text_features", df_with_features.columns)

        # Verificar se textos foram combinados corretamente
        applicant_text = df_with_features["applicant_text_features"].iloc[0]
        self.assertIn("Objetivo 1", applicant_text)
        self.assertIn("Historico 1", applicant_text)
        self.assertIn("CV 1", applicant_text)

        job_text = df_with_features["job_text_features"].iloc[0]
        self.assertIn("Titulo P1", job_text)
        self.assertIn("Titulo V1", job_text)
        self.assertIn("Atividades 1", job_text)

    def test_drop_unnecessary_columns(self):
        """Testa a remoção de colunas desnecessárias."""
        df = pd.DataFrame({
            "codigo_vaga": ["V001", "V002"],
            "codigo_profissional": ["001", "002"],
            "email": ["test@test.com", "test2@test.com"],
            "telefone_celular": ["11999999999", "11888888888"],
            "nivel_academico": ["Superior", "Médio"],  # Deve ser mantida
            "objetivo_profissional": ["Obj1", "Obj2"],  # Deve ser mantida
            "contratado": [1, 0]  # Deve ser mantida
        })

        df_cleaned = drop_unnecessary_columns(df)

        # Verificar se colunas desnecessárias foram removidas
        self.assertNotIn("codigo_vaga", df_cleaned.columns)
        self.assertNotIn("codigo_profissional", df_cleaned.columns)
        self.assertNotIn("email", df_cleaned.columns)
        self.assertNotIn("telefone_celular", df_cleaned.columns)

        # Verificar se colunas importantes foram mantidas
        self.assertIn("nivel_academico", df_cleaned.columns)
        self.assertIn("objetivo_profissional", df_cleaned.columns)
        self.assertIn("contratado", df_cleaned.columns)

    def test_save_processed_data(self):
        """Testa o salvamento de dados processados."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.csv")
            save_processed_data(df, output_file)

            # Verificar se arquivo foi criado
            self.assertTrue(os.path.exists(output_file))

            # Verificar conteúdo
            loaded_df = pd.read_csv(output_file)
            pd.testing.assert_frame_equal(df, loaded_df)

    @patch('src.training.data_preprocessing_merge_code.load_json_files')
    def test_process_data_pipeline_integration(self, mock_load):
        """Testa o pipeline completo de processamento (teste de integração)."""
        # Configurar mock
        mock_load.return_value = (
            self.mock_applicants_data,
            self.mock_prospects_data,
            self.mock_vagas_data
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_pipeline_output.csv")
            
            result_df = process_data_pipeline(
                applicants_file="dummy_applicants.json",
                prospects_file="dummy_prospects.json",
                vagas_file="dummy_vagas.json",
                output_file=output_file
            )

            # Verificar resultado
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertIn("contratado", result_df.columns)
            self.assertIn("applicant_text_features", result_df.columns)
            self.assertIn("job_text_features", result_df.columns)

            # Verificar se arquivo foi salvo
            self.assertTrue(os.path.exists(output_file))


if __name__ == "__main__":
    unittest.main()
