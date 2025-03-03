import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["FLUENTAI_CONFIG_PATH"] = "config.yaml"  # noqa

from fluentai.constants.config import config
from fluentai.services.mnemonic.semantic.compute import SemanticSimilarity

model_name = config.get("SEMANTIC_SIM").get("MODEL").lower()


@pytest.fixture
def mock_config(mocker):
    """
    Fixture to mock the config.get method.
    """
    return mocker.patch("fluentai.services.mnemonic.semantic.compute.config")


@pytest.fixture
def mock_sentence_transformer():
    """
    Fixture to mock the SentenceTransformer model.
    """
    mock_model = MagicMock()
    # Mock the encode method to return a mock tensor
    mock_model.encode.return_value = MagicMock()
    # Mock the similarity method to return a mock value with .item()
    mock_similarity = MagicMock()
    mock_similarity.item.return_value = 0.92
    mock_model.similarity.return_value = mock_similarity
    return mock_model


def test_compute_similarity_transformer(mock_config, mock_sentence_transformer):
    """
    Test the compute_similarity method using a Transformer model.
    """
    # Setup mock config to return the Transformer model name
    mock_config.get.return_value = {
        "MODEL": model_name,
        "EVAL": {"MODELS": [model_name]},
    }

    # Patch 'SentenceTransformer' to return the mock transformer model
    with patch(
        "fluentai.services.mnemonic.semantic.compute.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        # Initialize SemanticSimilarity
        semantic_sim = SemanticSimilarity()

        # Call compute_similarity
        similarity = semantic_sim.compute_similarity("king", "queen")

        # Assertions
        mock_sentence_transformer.encode.assert_any_call(
            "king", convert_to_tensor=True, normalize_embeddings=True
        )
        mock_sentence_transformer.encode.assert_any_call(
            "queen", convert_to_tensor=True, normalize_embeddings=True
        )
        mock_sentence_transformer.similarity.assert_called_once()
        assert similarity == 0.92


def test_compute_similarity_word_not_in_transformer(
    mock_config, mock_sentence_transformer
):
    """
    Test that compute_similarity raises a ValueError when a word is not in the Transformer model.
    """
    # Setup mock config to return the Transformer model name
    mock_config.get.return_value = {
        "MODEL": model_name,
        "EVAL": {"MODELS": [model_name]},
    }

    # Configure the mock to raise a ValueError when similarity is called
    mock_sentence_transformer.similarity.side_effect = ValueError(
        "Word not found in transformer model"
    )

    # Patch 'SentenceTransformer' to return the mock transformer model
    with patch(
        "fluentai.services.mnemonic.semantic.compute.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        # Initialize SemanticSimilarity
        semantic_sim = SemanticSimilarity()

        # Call compute_similarity and expect a ValueError
        with pytest.raises(ValueError) as exc_info:
            semantic_sim.compute_similarity("unknown_word1", "unknown_word2")

        assert "Word not found in transformer model" in str(exc_info.value)


def test_load_semantic_model_transformer(mock_config, mock_sentence_transformer):
    """
    Test that the SemanticSimilarity class loads a Transformer model correctly.
    """
    # Setup mock config to return the Transformer model name
    mock_config.get.return_value = {
        "MODEL": model_name,
        "EVAL": {"MODELS": [model_name]},
    }

    # Patch 'SentenceTransformer' to return the mock transformer model
    with patch(
        "fluentai.services.mnemonic.semantic.compute.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        # Initialize SemanticSimilarity
        semantic_sim = SemanticSimilarity()

        # Assertions
        # Encoding happens during compute_similarity
        mock_sentence_transformer.encode.assert_not_called()
        mock_sentence_transformer.similarity.assert_not_called()
        assert semantic_sim.model_name == model_name
        assert semantic_sim.model == mock_sentence_transformer


def test_example_function(mocker, mock_config, mock_sentence_transformer, caplog):
    """
    Test the example function to ensure it logs similarities correctly.
    """
    from fluentai.services.mnemonic.semantic.compute import example

    # Setup mock config to return models
    mock_config.get.return_value = {
        "MODEL": model_name,
        "EVAL": {"MODELS": [model_name]},
    }

    # Patch 'SentenceTransformer' to return the mock transformer model
    with patch(
        "fluentai.services.mnemonic.semantic.compute.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        # Configure the mock models
        mock_sentence_transformer.encode.return_value = MagicMock()
        mock_sentence_transformer.similarity.return_value = MagicMock(
            item=MagicMock(return_value=0.92)
        )

        # Run the example function
        example()

        # Assertions
        assert (
            f"Similarity between 'train' and 'brain' using '{model_name}': 0.9200"
            in caplog.text
        )
