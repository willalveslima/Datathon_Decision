"""Testes unitários para o módulo model_training.

Este módulo contém testes para todas as funções do modelo de treinamento,
incluindo carregamento de dados, processamento de features, treinamento
e avaliação do modelo.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open
from typing import List
import numpy as np
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Importa as funções do módulo a ser testado
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'training'))

from model_training import (
    load_processed_data,
    get_portuguese_stop_words,
    preprocess_text_features,
    create_tfidf_features,
    get_common_skills,
    extract_skills_from_text,
    create_skill_features,
    create_categorical_features,
    combine_all_features,
    split_data,
    train_logistic_regression,
    evaluate_model,
    plot_precision_recall_curve,
    save_model_and_preprocessors,
    train_complete_model,
    print_recommendation_guide
)


class TestModelTraining:
    """Classe de testes para o módulo model_training."""

    @pytest.fixture
    def sample_dataframe(self):
        """Fixture que cria um DataFrame de exemplo para testes."""
        data = {
            'applicant_text_features': [
                'python developer with django experience',
                'java programmer with spring framework',
                'javascript developer react angular'
            ],
            'job_text_features': [
                'python django developer needed',
                'java spring developer position',
                'frontend developer javascript react'
            ],
            'nivel profissional': ['senior', 'junior', 'pleno'],
            'nivel_academico': ['superior', 'superior', 'tecnico'],
            'nivel_ingles': ['avancado', 'basico', 'intermediario'],
            'local': ['SP', 'RJ', 'MG'],
            'contratado': [1, 0, 1]
        }
        return pd.DataFrame(data)

    def test_get_portuguese_stop_words(self):
        """Testa se a função retorna uma lista válida de stop words."""
        stop_words = get_portuguese_stop_words()
        
        assert isinstance(stop_words, list)
        assert len(stop_words) > 0
        assert 'de' in stop_words
        assert 'para' in stop_words
        assert 'com' in stop_words

    def test_get_common_skills(self):
        """Testa se a função retorna uma lista válida de habilidades."""
        skills = get_common_skills()
        
        assert isinstance(skills, list)
        assert len(skills) > 0
        assert 'python' in skills
        assert 'java' in skills
        assert 'javascript' in skills

    def test_extract_skills_from_text(self):
        """Testa a extração de habilidades de um texto."""
        text = "I am a Python developer with Django and React experience"
        skills_list = ['python', 'django', 'react', 'java']
        
        found_skills = extract_skills_from_text(text, skills_list)
        
        assert isinstance(found_skills, list)
        assert 'python' in found_skills
        assert 'django' in found_skills
        assert 'react' in found_skills
        assert 'java' not in found_skills

    def test_extract_skills_from_text_empty(self):
        """Testa a extração de habilidades de um texto vazio."""
        text = ""
        skills_list = ['python', 'java']
        
        found_skills = extract_skills_from_text(text, skills_list)
        
        assert isinstance(found_skills, list)
        assert len(found_skills) == 0

    def test_create_tfidf_features(self):
        """Testa a criação de features TF-IDF."""
        # Cria um DataFrame de exemplo
        sample_df = pd.DataFrame({
            'applicant_text_features': ['python developer', 'java programmer'],
            'job_text_features': ['python needed', 'java position'],
            'contratado': [1, 0]
        })
        
        # Pré-processa o DataFrame
        processed_df = preprocess_text_features(sample_df)
        
        applicant_tfidf, job_tfidf, vec_applicant, vec_job = create_tfidf_features(
            processed_df, max_features=100
        )
        
        assert isinstance(vec_applicant, TfidfVectorizer)
        assert isinstance(vec_job, TfidfVectorizer)
        assert isinstance(applicant_tfidf, csr_matrix)
        assert isinstance(job_tfidf, csr_matrix)
        assert applicant_tfidf.shape[0] == len(sample_df)
        assert job_tfidf.shape[0] == len(sample_df)

    def test_preprocess_and_tfidf_features(self, sample_dataframe, capsys):
        """Testa o pré-processamento e vetorização de features de texto."""
        # Pré-processa features de texto
        processed_df = preprocess_text_features(sample_dataframe)
        
        # Cria features TF-IDF
        applicant_tfidf, job_tfidf, vec_applicant, vec_job = create_tfidf_features(
            processed_df, max_features=100
        )
        
        assert isinstance(applicant_tfidf, csr_matrix)
        assert isinstance(job_tfidf, csr_matrix)
        assert applicant_tfidf.shape[0] == len(sample_dataframe)
        assert job_tfidf.shape[0] == len(sample_dataframe)
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Iniciando vetorização TF-IDF" in captured.out

    def test_preprocess_text_features_with_nan(self, capsys):
        """Testa o pré-processamento com valores NaN."""
        # DataFrame com valores NaN
        df_with_nan = pd.DataFrame({
            'applicant_text_features': ['python developer', None, np.nan, ''],
            'job_text_features': [None, 'java position', 'react developer', np.nan],
            'contratado': [1, 0, 1, 0]
        })
        
        # Pré-processa features de texto
        processed_df = preprocess_text_features(df_with_nan)
        
        # Cria features TF-IDF - não deve gerar erro mesmo com valores NaN
        applicant_tfidf, job_tfidf, vec_applicant, vec_job = create_tfidf_features(
            processed_df, max_features=50
        )
        
        assert isinstance(applicant_tfidf, csr_matrix)
        assert isinstance(job_tfidf, csr_matrix)
        assert applicant_tfidf.shape[0] == len(df_with_nan)
        assert job_tfidf.shape[0] == len(df_with_nan)
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Verificando e tratando valores NaN" in captured.out

    def test_create_skill_features(self, sample_dataframe, capsys):
        """Testa a criação de features de habilidades."""
        # Pré-processa o DataFrame primeiro
        processed_df = preprocess_text_features(sample_dataframe.copy())
        
        result_df = create_skill_features(processed_df)
        
        # Verifica se as novas colunas foram criadas
        assert 'applicant_skills' in result_df.columns
        assert 'job_skills' in result_df.columns
        assert 'common_skills_count' in result_df.columns
        
        # Verifica features binárias para algumas habilidades comuns
        common_skills = get_common_skills()
        for skill in common_skills[:3]:  # Testa apenas as primeiras 3 para não sobrecarregar
            assert f'applicant_has_{skill}' in result_df.columns
            assert f'job_requires_{skill}' in result_df.columns
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Iniciando extração de features" in captured.out

    def test_create_categorical_features(self, sample_dataframe, capsys):
        """Testa a codificação de features categóricas."""
        encoded_features, existing_cols = create_categorical_features(sample_dataframe.copy())
        
        assert isinstance(encoded_features, pd.DataFrame)
        assert isinstance(existing_cols, list)
        assert len(encoded_features) == len(sample_dataframe)
        assert len(existing_cols) > 0
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Iniciando One-Hot Encoding" in captured.out

    def test_combine_all_features(self, sample_dataframe):
        """Testa a combinação de todas as features."""
        # Prepara os dados necessários
        processed_df = preprocess_text_features(sample_dataframe.copy())
        
        # Cria features TF-IDF
        applicant_tfidf, job_tfidf, vec_applicant, vec_job = create_tfidf_features(
            processed_df, max_features=100
        )
        
        # Cria features de habilidades
        df_with_skills = create_skill_features(processed_df)
        
        # Codifica features categóricas
        encoded_features, _ = create_categorical_features(df_with_skills)
        
        # Combina features
        X, y = combine_all_features(
            applicant_tfidf, job_tfidf, encoded_features, df_with_skills
        )
        
        assert isinstance(X, csr_matrix)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == len(sample_dataframe)
        assert len(y) == len(sample_dataframe)

    def test_split_data(self, capsys):
        """Testa a divisão dos dados em treino e teste."""
        # Cria um dataset maior para permitir estratificação
        np.random.seed(42)
        n_samples = 20
        X = csr_matrix(np.random.rand(n_samples, 10))
        # Cria target balanceado com pelo menos 2 amostras por classe
        y = pd.Series([0] * 8 + [1] * 12)  # 8 zeros e 12 uns
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)
        
        assert isinstance(X_train, csr_matrix)
        assert isinstance(X_test, csr_matrix)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Verifica que os dados foram divididos
        assert len(y_train) + len(y_test) == n_samples
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        
        # Verifica aproximadamente as proporções (pode haver pequenas variações devido à estratificação)
        expected_test_size = int(n_samples * 0.3)
        assert abs(len(y_test) - expected_test_size) <= 2  # Tolerância de 2 amostras
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Shape de x_train" in captured.out

    def test_train_logistic_regression(self, capsys):
        """Testa o treinamento do modelo."""
        # Cria dados de exemplo
        X_train = csr_matrix(np.random.rand(100, 20))
        y_train = pd.Series(np.random.choice([0, 1], 100))
        
        model = train_logistic_regression(X_train, y_train)
        
        assert isinstance(model, LogisticRegression)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Iniciando treinamento do modelo" in captured.out
        assert "Treinamento do modelo Logistic Regression concluído" in captured.out

    @patch('matplotlib.pyplot.savefig')
    def test_evaluate_model(self, mock_savefig, capsys):
        """Testa a avaliação do modelo."""
        # Cria modelo e dados de exemplo
        model = LogisticRegression(random_state=42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.choice([0, 1], 100)
        model.fit(X_train, y_train)
        
        X_test = csr_matrix(np.random.rand(30, 10))
        y_test = pd.Series(np.random.choice([0, 1], 30))
        
        # Testa a função evaluate_model
        y_pred, y_proba, roc_auc, pr_auc, precision, recall = evaluate_model(
            model, X_test, y_test, threshold=0.5
        )
        
        # Verifica se os valores retornados estão corretos
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(y_proba, np.ndarray)
        assert isinstance(roc_auc, float)
        assert isinstance(pr_auc, float)
        assert isinstance(precision, np.ndarray)
        assert isinstance(recall, np.ndarray)
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Iniciando avaliação do modelo" in captured.out
        assert "AUC-ROC" in captured.out
        
        # Testa a função plot_precision_recall_curve separadamente
        plot_precision_recall_curve(precision, recall, pr_auc, "test_curve.png")
        
        # Verifica se o gráfico foi salvo
        mock_savefig.assert_called_once_with("test_curve.png")

    @patch('matplotlib.pyplot.savefig')
    def test_plot_precision_recall_curve(self, mock_savefig, capsys):
        """Testa a função de plotagem da curva Precision-Recall."""
        # Dados de exemplo para precision e recall
        precision = np.array([0.8, 0.7, 0.6, 0.5])
        recall = np.array([0.2, 0.4, 0.6, 0.8])
        pr_auc = 0.65
        
        # Testa a função
        plot_precision_recall_curve(precision, recall, pr_auc, "test_pr_curve.png")
        
        # Verifica se as mensagens foram impressas
        captured = capsys.readouterr()
        assert "Curva Precision-Recall salva como 'test_pr_curve.png'" in captured.out
        
        # Verifica se o gráfico foi salvo
        mock_savefig.assert_called_once_with("test_pr_curve.png")

    def test_save_model_and_preprocessors(self):
        """Testa o salvamento dos artefatos do modelo."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cria objetos de exemplo
            model = LogisticRegression()
            vec_applicant = TfidfVectorizer()
            vec_job = TfidfVectorizer()
            categorical_cols = ['col1', 'col2']
            encoded_features = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
            
            # Salva artefatos
            save_model_and_preprocessors(
                model, vec_applicant, vec_job, categorical_cols,
                encoded_features, tmp_dir
            )
            
            # Verifica se os arquivos foram criados
            expected_files = [
                'logistic_regression_model.pkl',
                'tfidf_vectorizer_applicant.pkl',
                'tfidf_vectorizer_job.pkl',
                'categorical_cols.pkl',
                'encoded_feature_names.pkl',
                'common_skills.pkl'
            ]
            
            for file_name in expected_files:
                file_path = os.path.join(tmp_dir, file_name)
                assert os.path.exists(file_path)

    @patch('pandas.read_csv')
    def test_load_processed_data_success(self, mock_read_csv, capsys):
        """Testa o carregamento bem-sucedido de dados."""
        # Configura o mock
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        
        result = load_processed_data("test_path.csv")
        
        assert isinstance(result, pd.DataFrame)
        assert result.equals(mock_df)
        mock_read_csv.assert_called_once_with("test_path.csv")
        
        # Verifica se a mensagem de sucesso foi impressa
        captured = capsys.readouterr()
        assert "carregado com sucesso" in captured.out

    @patch('pandas.read_csv')
    def test_load_processed_data_file_not_found(self, mock_read_csv):
        """Testa o comportamento quando o arquivo não é encontrado."""
        mock_read_csv.side_effect = FileNotFoundError()
        
        with pytest.raises(FileNotFoundError):
            load_processed_data("nonexistent.csv")

    @patch('pandas.read_csv')
    def test_load_processed_data_default_path(self, mock_read_csv):
        """Testa o carregamento com o caminho padrão."""
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        
        result = load_processed_data()  # Usa caminho padrão
        
        assert isinstance(result, pd.DataFrame)
        assert result.equals(mock_df)
        mock_read_csv.assert_called_once_with("data/processed/merged_data_processed.csv")

    def test_print_recommendation_guide(self, capsys):
        """Testa a função que imprime o exemplo de uso."""
        print_recommendation_guide()
        
        captured = capsys.readouterr()
        assert "Exemplo de Recomendação" in captured.out
        assert "modelo de classificação foi treinado" in captured.out
        assert "Carregar o modelo" in captured.out

    @patch('model_training.load_processed_data')
    @patch('model_training.save_model_and_preprocessors')
    @patch('matplotlib.pyplot.savefig')
    def test_train_complete_model_integration(
        self, mock_savefig, mock_save, mock_load, sample_dataframe
    ):
        """Teste de integração da função principal."""
        # Expande o sample_dataframe para ter mais amostras
        expanded_df = pd.concat([sample_dataframe] * 4, ignore_index=True)  # 12 amostras total
        
        # Configura mocks
        mock_load.return_value = expanded_df
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = train_complete_model(
                data_path="test.csv",
                models_dir=tmp_dir,
                max_features=50,
                test_size=0.25,  # 25% para teste (3 amostras)
                random_state=42
            )
            
            assert isinstance(result, dict)
            assert 'model' in result
            assert isinstance(result['model'], LogisticRegression)
            mock_load.assert_called_once()
            mock_save.assert_called_once()


class TestModelTrainingEdgeCases:
    """Testes para casos extremos e edge cases."""

    def test_extract_skills_with_none_text(self):
        """Testa extração de habilidades com texto None."""
        skills = extract_skills_from_text(None, ['python', 'java'])
        assert skills == []

    def test_extract_skills_with_nan_text(self):
        """Testa extração de habilidades com texto NaN."""

        skills = extract_skills_from_text(np.nan, ['python', 'java'])
        assert skills == []
        
    def test_extract_skills_with_numeric_text(self):
        """Testa extração de habilidades com entrada numérica."""
        skills = extract_skills_from_text(123, ['python', 'java'])
        assert skills == []

    def test_extract_skills_case_insensitive(self):
        """Testa se a extração é case-insensitive."""
        text = "I know PYTHON and Django"
        skills_list = ['python', 'django']
        
        found_skills = extract_skills_from_text(text, skills_list)
        
        assert 'python' in found_skills
        assert 'django' in found_skills

    def test_empty_dataframe_handling(self):
        """Testa o comportamento com DataFrame vazio."""
        empty_df = pd.DataFrame(columns=['applicant_text_features', 'job_text_features', 'contratado'])
        
        # Pré-processa o DataFrame vazio
        processed_df = preprocess_text_features(empty_df.copy())
        
        # Deve funcionar sem erro
        result_df = create_skill_features(processed_df)
        
        assert len(result_df) == 0
        assert 'applicant_skills' in result_df.columns
        assert 'job_skills' in result_df.columns
        assert 'common_skills_count' in result_df.columns
        
        # Verifica se as features binárias foram criadas
        common_skills = get_common_skills()
        for skill in common_skills[:2]:  # Testa apenas 2 skills
            assert f'applicant_has_{skill}' in result_df.columns
            assert f'job_requires_{skill}' in result_df.columns

    def test_dataframe_with_nan_values(self):
        """Testa o comportamento com valores NaN nos textos."""
        df_with_nan = pd.DataFrame({
            'applicant_text_features': ['python developer', None, ''],
            'job_text_features': [None, 'java position', 'react developer'],
            'contratado': [1, 0, 1]
        })
        
        # Pré-processa o DataFrame com NaN
        processed_df = preprocess_text_features(df_with_nan.copy())
        result_df = create_skill_features(processed_df)
        
        # Verifica se a função lidou com valores None/NaN sem erro
        assert len(result_df) == 3
        assert 'applicant_skills' in result_df.columns
        assert 'job_skills' in result_df.columns
        assert 'common_skills_count' in result_df.columns
        
        # Verifica se os valores NaN foram tratados como listas vazias
        assert isinstance(result_df.iloc[0]['applicant_skills'], list)
        assert isinstance(result_df.iloc[1]['applicant_skills'], list)
        assert isinstance(result_df.iloc[2]['applicant_skills'], list)

    def test_create_skill_features_no_common_skills(self):
        """Testa quando não há habilidades em comum."""
        test_df = pd.DataFrame({
            'applicant_text_features': ['python developer'],
            'job_text_features': ['java position'],
            'contratado': [0]
        })
        
        # Pré-processa o DataFrame
        processed_df = preprocess_text_features(test_df.copy())
        result_df = create_skill_features(processed_df)
        
        # Deve ter 0 habilidades em comum
        assert result_df.iloc[0]['common_skills_count'] == 0
        
        # Mas as features binárias devem estar corretas
        assert result_df.iloc[0]['applicant_has_python'] == 1
        assert result_df.iloc[0]['applicant_has_java'] == 0
        assert result_df.iloc[0]['job_requires_python'] == 0
        assert result_df.iloc[0]['job_requires_java'] == 1

    def test_split_data_small_dataset(self, capsys):
        """Testa divisão com dataset pequeno onde estratificação pode falhar."""
        # Para datasets muito pequenos, pode ser necessário não usar estratificação
        X = csr_matrix(np.random.rand(3, 10))
        y = pd.Series([1, 0, 1])
        
        try:
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)
            
            # Verifica que os dados foram divididos corretamente
            assert isinstance(X_train, csr_matrix)
            assert isinstance(X_test, csr_matrix)
            assert isinstance(y_train, pd.Series)
            assert isinstance(y_test, pd.Series)
            
            # Verifica que o total de amostras é preservado
            assert len(y_train) + len(y_test) == 3
            assert X_train.shape[0] == len(y_train)
            assert X_test.shape[0] == len(y_test)
            
        except ValueError as e:
            # Se a estratificação falhar devido ao dataset pequeno, isso é esperado
            assert "stratify" in str(e).lower() or "split" in str(e).lower()

    @patch('model_training.os.makedirs')
    @patch('model_training.joblib.dump')
    def test_save_model_and_preprocessors_error_handling(self, mock_dump, mock_makedirs):
        """Testa o tratamento de erro no salvamento."""
        mock_dump.side_effect = Exception("Erro de salvamento")
        
        # Deve gerar exceção
        with pytest.raises(Exception):
            save_model_and_preprocessors(
                LogisticRegression(), TfidfVectorizer(), TfidfVectorizer(),
                [], pd.DataFrame({'col': [1, 2]}), "/tmp/test"
            )
        
        mock_makedirs.assert_called_once()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configuração do ambiente de teste."""
    # Configura qualquer setup necessário para os testes
    yield
    # Cleanup após os testes
    pass


if __name__ == "__main__":
    # Executa os testes se o arquivo for executado diretamente
    pytest.main([__file__, "-v"])
