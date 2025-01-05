import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

os.environ["FLUENTAI_CONFIG_PATH"] = "config.yaml"  # noqa

# Import the top_phonetic function
from fluentai.services.mnemonic.phonetic.compute import top_phonetic


@pytest.fixture
def mock_config(mocker):
    """
    Fixture to mock the config.get method.
    """
    return mocker.patch("fluentai.services.mnemonic.phonetic.compute.config")


@pytest.fixture
def mock_word2ipa(mocker):
    """
    Fixture to mock the word2ipa function.
    """
    return mocker.patch("fluentai.services.mnemonic.phonetic.compute.word2ipa")


@pytest.fixture
def mock_load_from_cache(mocker):
    """
    Fixture to mock the load_from_cache function.
    """
    return mocker.patch(
        "fluentai.services.mnemonic.phonetic.utils.cache.load_from_cache"
    )


@pytest.fixture
def mock_pad_vectors(mocker):
    """
    Fixture to mock the pad_vectors function.
    """
    return mocker.patch("fluentai.services.mnemonic.phonetic.compute.pad_vectors")


@pytest.fixture
def mock_convert_to_matrix(mocker):
    """
    Fixture to mock the convert_to_matrix function.
    """
    return mocker.patch("fluentai.services.mnemonic.phonetic.compute.convert_to_matrix")


@pytest.fixture
def mock_faiss_normalize_L2(mocker):
    """
    Fixture to mock the faiss.normalize_L2 function.
    """
    return mocker.patch(
        "fluentai.services.mnemonic.phonetic.compute.faiss.normalize_L2"
    )


@pytest.fixture
def mock_faiss_IndexFlatIP(mocker):
    """
    Fixture to mock the faiss.IndexFlatIP class.

    Returns a tuple of (constructor_mock, instance_mock).
    """
    instance_mock = MagicMock()
    constructor_mock = mocker.patch(
        "fluentai.services.mnemonic.phonetic.compute.faiss.IndexFlatIP",
        return_value=instance_mock,
    )
    return constructor_mock, instance_mock


def test_top_phonetic_success(
    mock_config,
    mock_word2ipa,
    mock_load_from_cache,
    mock_pad_vectors,
    mock_convert_to_matrix,
    mock_faiss_normalize_L2,
    mock_faiss_IndexFlatIP,
):
    """
    Test the top_phonetic function for a successful case.
    """
    # Unpack the fixture
    constructor_mock, instance_mock = mock_faiss_IndexFlatIP

    # Setup mock config.get to return necessary configuration
    mock_config.get.return_value = {
        # or "soundvec" depending on your test
        "EMBEDDINGS": {"METHOD": "panphon"},
        "PHONETIC_SIM": {"IPA_REPO": "mock_repo", "IPA_FILE": "mock_file.tsv"},
    }

    # Mock the word2ipa function to return a specific IPA string
    mock_word2ipa.return_value = "kʊtʃɪŋ"

    # Create a mock dataset DataFrame
    mock_dataset = pd.DataFrame(
        {
            "token_ort": ["kucing", "cutting", "couching"],
            "token_ipa": ["kʊtʃɪŋ", "kʊtɪŋ", "kɔtʃɪŋ"],
            "flattened_vectors": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
        }
    )

    # Mock the load_from_cache function to return the mock dataset
    mock_load_from_cache.return_value = mock_dataset

    # Mock pad_vectors to return padded vectors (assuming dimension=3 for simplicity)
    mock_pad_vectors.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    # Mock convert_to_matrix to return a NumPy array
    mock_convert_to_matrix.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )

    # Mock FAISS normalize_L2 to do nothing (since we're using normalized vectors in this test)
    mock_faiss_normalize_L2.side_effect = lambda x: x  # No operation

    # Setup the FAISS index mock's search method
    instance_mock.search.return_value = (
        np.array([[0.99, 0.95, 0.90]]),  # distances
        np.array([[0, 1, 2]]),  # indices
    )

    # Create a mock vectorizer function (panphon_vec or soundvec)
    with patch(
        "fluentai.services.mnemonic.phonetic.compute.panphon_vec",
        return_value=[[0.1, 0.2, 0.3]],
    ):
        # Initialize a mock g2p_model with a g2p method
        mock_g2p_model = MagicMock()
        mock_g2p_model.g2p.return_value = "kʊtʃɪŋ"

        # Call the top_phonetic function
        result, _ = top_phonetic(
            input_word="kucing",
            language_code="eng-us",
            top_n=3,
            g2p_model=mock_g2p_model,
        )

        # Assertions
        # Ensure word2ipa was called correctly
        mock_word2ipa.assert_called_once_with("kucing", "eng-us", mock_g2p_model)

        # Ensure load_from_cache was called with the correct method
        mock_load_from_cache.assert_called_once_with("panphon")

        # Ensure pad_vectors was called with the correct data
        mock_pad_vectors.assert_called_once_with(
            mock_dataset["flattened_vectors"].tolist()
        )

        # Ensure convert_to_matrix was called with the padded vectors
        mock_convert_to_matrix.assert_called_once_with(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )

        # Ensure FAISS normalize_L2 was called twice
        assert mock_faiss_normalize_L2.call_count == 2

        # Ensure FAISS IndexFlatIP constructor was called once with the correct dimension
        constructor_mock.assert_called_once_with(3)

        # Ensure the FAISS index's add method was called correctly
        # Retrieve the actual call arguments
        args, kwargs = instance_mock.add.call_args

        # Define the expected array
        expected_array = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )

        # Use NumPy's testing utility to compare arrays
        np.testing.assert_array_equal(args[0], expected_array)

        # Ensure no keyword arguments were passed
        assert kwargs == {}

        # Verify the result DataFrame
        expected_result = pd.DataFrame(
            {
                "token_ort": ["kucing", "cutting", "couching"],
                "token_ipa": ["kʊtʃɪŋ", "kʊtɪŋ", "kɔtʃɪŋ"],
                "distance": [0.99, 0.95, 0.90],
            }
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_result.reset_index(drop=True)
        )


def test_top_phonetic_no_results(
    mock_config,
    mock_word2ipa,
    mock_load_from_cache,
    mock_pad_vectors,
    mock_convert_to_matrix,
    mock_faiss_normalize_L2,
    mock_faiss_IndexFlatIP,
):
    """
    Test the top_phonetic function when no similar words are found.
    """
    # Unpack the fixture
    constructor_mock, instance_mock = mock_faiss_IndexFlatIP

    # Setup mock config.get to return necessary configuration
    mock_config.get.return_value = {
        # or "soundvec" depending on your test
        "EMBEDDINGS": {"METHOD": "panphon"},
        "PHONETIC_SIM": {"IPA_REPO": "mock_repo", "IPA_FILE": "mock_file.tsv"},
    }

    # Mock the word2ipa function to return a specific IPA string
    mock_word2ipa.return_value = "tɛst"

    # Create an empty mock dataset DataFrame
    mock_dataset = pd.DataFrame(
        {"token_ort": [], "token_ipa": [], "flattened_vectors": []}
    )

    # Mock the load_from_cache function to return the empty dataset
    mock_load_from_cache.return_value = mock_dataset

    # Mock pad_vectors to return empty list
    mock_pad_vectors.return_value = []

    # Mock convert_to_matrix to return an empty NumPy array
    mock_convert_to_matrix.return_value = np.array([]).reshape(
        0, 3
    )  # Assuming dimension=3

    # Mock FAISS normalize_L2 to do nothing
    mock_faiss_normalize_L2.side_effect = lambda x: x  # No operation

    # Setup the FAISS index mock's search method
    instance_mock.search.return_value = (
        np.array([[]]),  # distances
        np.array([[]]),  # indices
    )

    # Create a mock vectorizer function (panphon_vec or soundvec)
    with patch(
        "fluentai.services.mnemonic.phonetic.compute.panphon_vec",
        return_value=[[]],
    ):
        # Initialize a mock g2p_model with a g2p method
        mock_g2p_model = MagicMock()
        mock_g2p_model.g2p.return_value = "tɛst"

        # Call the top_phonetic function
        result, _ = top_phonetic(
            input_word="test", language_code="eng-us", top_n=3, g2p_model=mock_g2p_model
        )

        # Assertions
        # Ensure word2ipa was called correctly
        mock_word2ipa.assert_called_once_with("test", "eng-us", mock_g2p_model)

        # Ensure load_from_cache was called with the correct method
        mock_load_from_cache.assert_called_once_with("panphon")

        # Ensure pad_vectors was called with the correct data
        mock_pad_vectors.assert_called_once_with(
            mock_dataset["flattened_vectors"].tolist()
        )

        # Ensure convert_to_matrix was called with the padded vectors
        mock_convert_to_matrix.assert_called_once_with([])

        # Ensure FAISS normalize_L2 was called twice
        assert mock_faiss_normalize_L2.call_count == 2

        # Ensure FAISS IndexFlatIP constructor was called once with the correct dimension
        constructor_mock.assert_called_once_with(3)

        # Ensure the FAISS index's add method was called correctly
        # Retrieve the actual call arguments
        args, kwargs = instance_mock.add.call_args

        # Define the expected array
        expected_array = np.array([]).reshape(0, 3)

        # Use NumPy's testing utility to compare arrays
        np.testing.assert_array_equal(args[0], expected_array)

        # Ensure no keyword arguments were passed
        assert kwargs == {}

        # Verify the result DataFrame is empty
        expected_result = pd.DataFrame(
            {"token_ort": [], "token_ipa": [], "distance": []}
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_result.reset_index(drop=True)
        )


def test_top_phonetic_invalid_language_code(
    mock_config,
    mock_word2ipa,
    mock_load_from_cache,
    mock_pad_vectors,
    mock_convert_to_matrix,
    mock_faiss_normalize_L2,
    # This is now a tuple (constructor_mock, instance_mock)
    mock_faiss_IndexFlatIP,
):
    """
    Test the top_phonetic function with an unsupported language code.
    """
    # Unpack the fixture
    constructor_mock, instance_mock = mock_faiss_IndexFlatIP

    # Setup mock config.get to return necessary configuration
    mock_config.get.return_value = {
        # Testing with a different vectorizer
        "EMBEDDINGS": {"METHOD": "clts"},
        "PHONETIC_SIM": {"IPA_REPO": "mock_repo", "IPA_FILE": "mock_file.tsv"},
    }

    # Mock the word2ipa function to return a specific IPA string
    mock_word2ipa.return_value = "ɲaŋgʊŋ"

    # Create a mock dataset DataFrame
    mock_dataset = pd.DataFrame(
        {
            "token_ort": ["nyangang", "nyanung", "nyatung"],
            "token_ipa": ["ɲaŋgʊŋ", "ɲaːnʊŋ", "ɲaːtʊŋ"],
            "flattened_vectors": [
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0],
            ],
        }
    )

    # Mock the load_from_cache function to return the mock dataset
    mock_load_from_cache.return_value = mock_dataset

    # Mock pad_vectors to return padded vectors (assuming dimension=3 for simplicity)
    mock_pad_vectors.return_value = [
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0],
    ]

    # Mock convert_to_matrix to return a NumPy array
    mock_convert_to_matrix.return_value = np.array(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]
    )

    # Mock FAISS normalize_L2 to do nothing (since we're using normalized vectors in this test)
    mock_faiss_normalize_L2.side_effect = lambda x: x  # No operation

    # Setup the FAISS index mock's search method
    instance_mock.search.return_value = (
        np.array([[0.98, 0.96, 0.93]]),  # distances
        np.array([[0, 1, 2]]),  # indices
    )

    # Create a mock vectorizer function (soundvec)
    with patch(
        "fluentai.services.mnemonic.phonetic.compute.soundvec",
        return_value=[[0.2, 0.3, 0.4]],
    ):
        # Initialize a mock g2p_model with a g2p method
        mock_g2p_model = MagicMock()
        mock_g2p_model.g2p.return_value = "ɲaŋgʊŋ"

        # Call the top_phonetic function with an unsupported language code
        result, _ = top_phonetic(
            input_word="nyangang",
            language_code="mal",  # Assuming 'mal' is unsupported
            top_n=3,
            g2p_model=mock_g2p_model,
        )

        # Assertions
        # Ensure word2ipa was called correctly
        mock_word2ipa.assert_called_once_with("nyangang", "mal", mock_g2p_model)

        # Ensure load_from_cache was called with the correct method
        mock_load_from_cache.assert_called_once_with("soundvec")

        # Ensure pad_vectors was called with the correct data
        mock_pad_vectors.assert_called_once_with(
            mock_dataset["flattened_vectors"].tolist()
        )

        # Ensure convert_to_matrix was called with the padded vectors
        mock_convert_to_matrix.return_value = np.array(
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32
        )
        # Ensure FAISS normalize_L2 was called twice
        assert mock_faiss_normalize_L2.call_count == 2

        # Ensure FAISS IndexFlatIP constructor was called once with the correct dimension
        constructor_mock.assert_called_once_with(3)

        # Ensure the FAISS index's add method was called correctly
        # Retrieve the actual call arguments
        args, kwargs = instance_mock.add.call_args

        # Define the expected array
        expected_array = np.array(
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float64
        )

        # Use NumPy's testing utility to compare arrays
        np.testing.assert_array_equal(args[0], expected_array)

        # Ensure no keyword arguments were passed
        assert kwargs == {}

        # Verify the result DataFrame
        expected_result = pd.DataFrame(
            {
                "token_ort": ["nyangang", "nyanung", "nyatung"],
                "token_ipa": ["ɲaŋgʊŋ", "ɲaːnʊŋ", "ɲaːtʊŋ"],
                "distance": [0.98, 0.96, 0.93],
            }
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_result.reset_index(drop=True)
        )
