# imageability_predictor.py

import warnings

import joblib
import numpy as np
import pandas as pd
from gensim.downloader import load as gensim_load
from lightgbm import LGBMRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ImageabilityPredictor:
    def __init__(
        self,
        embedding_model_name="fasttext-wiki-news-subwords-300",
        regression_model_path="models/best_model_LGBMRegressor.joblib",
    ):
        """
        Initialize the ImageabilityPredictor by loading the embedding model and regression model.

        Args:
            embedding_model_name (str, optional): Name of the embedding model to load from Gensim.
                                                  Defaults to "fasttext-wiki-news-subwords-300".
            regression_model_path (str, optional): Path to the trained regression model (.joblib file).
                                                   Defaults to "models/best_model_LGBMRegressor.joblib".
        """
        print("Initializing ImageabilityPredictor...")

        # Load the embedding model
        print(f"Loading embedding model '{embedding_model_name}'...")
        self.embedding_model = gensim_load(embedding_model_name)
        print("Embedding model loaded successfully.")

        # Load the regression model
        print(f"Loading regression model from '{regression_model_path}'...")
        self.regression_model = joblib.load(regression_model_path)
        print("Regression model loaded successfully.")

    def get_embedding(self, word):
        """
        Retrieve the embedding vector for a given word.

        Args:
            word (str): The word to retrieve the embedding for.

        Returns:
            np.ndarray: Embedding vector for the word.
        """
        try:
            embedding = self.embedding_model.get_vector(word)
        except KeyError:
            # Handle out-of-vocabulary (OOV) words by returning a zero vector
            embedding = np.zeros(self.embedding_model.vector_size, dtype=np.float32)
        return embedding

    def predict_imageability(self, embedding):
        """
        Predict the imageability score based on the embedding.

        Args:
            embedding (np.ndarray): Embedding vector of the word.

        Returns:
            float: Predicted imageability score.
        """
        # Reshape embedding for prediction (1 sample)
        embedding = embedding.reshape(1, -1)
        imageability = self.regression_model.predict(embedding)[0]
        return imageability

    def get_imageability(self, word):
        """
        Generate the imageability score for a given word.

        Args:
            word (str): The word to evaluate.

        Returns:
            float: Predicted imageability score.
        """
        embedding = self.get_embedding(word)
        imageability = self.predict_imageability(embedding)
        return imageability


# Example Usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = ImageabilityPredictor(
        embedding_model_name="fasttext-wiki-news-subwords-300",
        regression_model_path="models/best_model_LGBMRegressor.joblib",
    )

    # Example words
    words_to_predict = ["apple", "banana", "orange", "unknownword"]

    for word in words_to_predict:
        score = predictor.get_imageability(word)
        print(f"Word: '{word}' | Predicted Imageability: {score:.4f}")
