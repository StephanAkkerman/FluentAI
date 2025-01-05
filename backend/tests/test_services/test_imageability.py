import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

os.environ["FLUENTAI_CONFIG_PATH"] = "config.yaml"  # noqa

# Import the functions and classes to be tested
from backend.fluentai.services.mnemonic.imageability.predictor import (
    ImageabilityPredictor,
    make_predictions,
)


@pytest.fixture
def mock_hf_hub_download(mocker):
    """
    Fixture to mock hf_hub_download function.
    """
    return mocker.patch(
        "fluentai.services.mnemonic.imageability.predictor.hf_hub_download"
    )


@pytest.fixture
def mock_joblib_load(mocker):
    """
    Fixture to mock joblib.load function.
    """
    return mocker.patch("fluentai.services.mnemonic.imageability.predictor.joblib.load")


@pytest.fixture
def mock_pd_read_csv(mocker):
    """
    Fixture to mock pandas.read_csv function.
    """
    return mocker.patch("fluentai.services.mnemonic.imageability.predictor.pd.read_csv")


@pytest.fixture
def mock_ImageabilityEmbeddings(mocker):
    """
    Fixture to mock ImageabilityEmbeddings class.
    """
    mock_class = mocker.patch(
        "fluentai.services.mnemonic.imageability.predictor.ImageabilityEmbeddings"
    )
    mock_instance = MagicMock()
    mock_instance.get_embedding.side_effect = lambda word: np.array([1.0, 2.0, 3.0])
    mock_class.return_value = mock_instance
    return mock_class


def test_make_predictions(
    mock_hf_hub_download,
    mock_joblib_load,
    mock_pd_read_csv,
    mock_ImageabilityEmbeddings,
    mocker,
):
    """
    Test the make_predictions function.
    """
    # Mock the configuration
    mock_config = mocker.patch(
        "fluentai.services.mnemonic.imageability.predictor.config"
    )
    mock_config.get.side_effect = lambda key: {
        "IMAGEABILITY": {
            "PREDICTOR": {
                "REPO": "imageability_predictor_repo",
                "FILE": "predictor.joblib",
            },
            "PREDICTIONS": {
                "REPO": "imageability_predictions_repo",
                "FILE": "predictions.csv",
            },
        },
        "PHONETIC_SIM": {
            "IPA": {"REPO": "ipa_dataset_repo", "FILE": "ipa_dataset.csv"}
        },
    }[key]

    # Mock hf_hub_download to return dummy paths
    mock_hf_hub_download.side_effect = (
        lambda repo_id, filename, cache_dir, repo_type=None: f"{cache_dir}/{filename}"
    )

    # Mock joblib.load to return a mock regression model
    mock_regression_model = MagicMock()
    mock_regression_model.predict.return_value = [0.75]
    mock_joblib_load.return_value = mock_regression_model

    # Mock pandas.read_csv to return a dummy IPA dataset
    dummy_ipa_data = pd.DataFrame(
        # 'apple' is duplicated
        {"token_ort": ["apple", "banana", "cherry", "apple"]}
    )
    mock_pd_read_csv.return_value = dummy_ipa_data

    # Mock tqdm to just return the iterator
    mocker.patch(
        "fluentai.services.mnemonic.imageability.predictor.tqdm",
        side_effect=lambda x, total=None: x,
    )

    # Call the function
    make_predictions()

    # Assertions
    mock_hf_hub_download.assert_any_call(
        repo_id="imageability_predictor_repo",
        filename="predictor.joblib",
        cache_dir="models",
    )
    mock_hf_hub_download.assert_any_call(
        repo_id="ipa_dataset_repo",
        filename="ipa_dataset.csv",
        cache_dir="datasets",
        repo_type="dataset",
    )
    mock_joblib_load.assert_called_once_with("models/predictor.joblib")
    mock_pd_read_csv.assert_called_once_with("datasets/ipa_dataset.csv")
    mock_ImageabilityEmbeddings.assert_called_once_with(model_name="fasttext")
    mock_regression_model.predict.assert_called()


def test_ImageabilityPredictor_get_prediction(
    mock_hf_hub_download, mock_pd_read_csv, mocker
):
    """
    Test the get_prediction method of ImageabilityPredictor.
    """
    # Mock the configuration
    mock_config = mocker.patch(
        "fluentai.services.mnemonic.imageability.predictor.config"
    )
    mock_config.get.side_effect = lambda key: {
        "IMAGEABILITY": {
            "PREDICTIONS": {
                "REPO": "imageability_predictions_repo",
                "FILE": "predictions.csv",
            }
        }
    }[key]

    # Mock hf_hub_download to return dummy path
    mock_hf_hub_download.return_value = "datasets/predictions.csv"

    # Mock pandas.read_csv to return a dummy predictions dataframe
    dummy_predictions = pd.DataFrame(
        {
            "word": ["apple", "banana", "cherry"],
            "imageability_score": [0.75, 0.65, 0.85],
        }
    )
    mock_pd_read_csv.return_value = dummy_predictions

    # Initialize the predictor
    predictor = ImageabilityPredictor()

    # Test get_prediction
    prediction = predictor.get_prediction("banana")
    assert prediction == 0.65

    # Test that accessing a non-existent word raises an error
    with pytest.raises(IndexError):
        predictor.get_prediction("durian")


def test_ImageabilityPredictor_get_predictions(
    mock_hf_hub_download, mock_pd_read_csv, mocker
):
    """
    Test the get_predictions method of ImageabilityPredictor.
    """
    # Mock the configuration
    mock_config = mocker.patch(
        "fluentai.services.mnemonic.imageability.predictor.config"
    )
    mock_config.get.side_effect = lambda key: {
        "IMAGEABILITY": {
            "PREDICTIONS": {
                "REPO": "imageability_predictions_repo",
                "FILE": "predictions.csv",
            }
        }
    }[key]

    # Mock hf_hub_download to return dummy path
    mock_hf_hub_download.return_value = "datasets/predictions.csv"

    # Mock pandas.read_csv to return a dummy predictions dataframe
    dummy_predictions = pd.DataFrame(
        {
            "word": ["apple", "banana", "cherry"],
            "imageability_score": [0.75, 0.65, 0.85],
        }
    )
    mock_pd_read_csv.return_value = dummy_predictions

    # Initialize the predictor
    predictor = ImageabilityPredictor()

    # Test get_predictions
    words = ["apple", "cherry"]
    predictions = predictor.get_predictions(words)
    assert predictions == [0.75, 0.85]


def test_ImageabilityPredictor_get_column_imageability(
    mock_hf_hub_download, mock_pd_read_csv, mocker
):
    """
    Test the get_column_imageability method of ImageabilityPredictor.
    """
    # Mock the configuration
    mock_config = mocker.patch(
        "fluentai.services.mnemonic.imageability.predictor.config"
    )
    mock_config.get.side_effect = lambda key: {
        "IMAGEABILITY": {
            "PREDICTIONS": {
                "REPO": "imageability_predictions_repo",
                "FILE": "predictions.csv",
            }
        }
    }[key]

    # Mock hf_hub_download to return dummy path
    mock_hf_hub_download.return_value = "datasets/predictions.csv"

    # Mock pandas.read_csv to return a dummy predictions dataframe
    dummy_predictions = pd.DataFrame(
        {
            "word": ["apple", "banana", "cherry"],
            "imageability_score": [0.75, 0.65, 0.85],
        }
    )
    mock_pd_read_csv.return_value = dummy_predictions

    # Initialize the predictor
    predictor = ImageabilityPredictor()

    # Create a dummy dataframe
    df = pd.DataFrame(
        # 'durian' not in predictions
        {"tokens": ["apple", "banana", "durian"]}
    )

    # Test get_column_imageability
    # Note: This will raise an error for 'durian' since it's not in the predictions
    with pytest.raises(IndexError):
        predictor.get_column_imageability(df, "tokens")

    # To handle missing words gracefully, you might want to modify your class methods
