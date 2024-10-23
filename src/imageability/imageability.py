import os
import sys
import warnings

import joblib
import numpy as np
from gensim.models.fasttext import FastTextKeyedVectors, load_facebook_vectors

# append the path of the parent directory
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from fasttext_download import download_fasttext

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def download_and_save_model(
    model_name="cc.en.300.bin", save_path="models/cc.en.300.model"
):
    """
    Download the specified embedding model and save it locally.

    Args:
        model_name (str): Name of the model to download.
        save_path (str): Path to save the downloaded model.
    """
    # Check if the .bin file already exists
    if not os.path.exists(f"data/fasttext_embeddings/{model_name}"):
        # Download the model .bin file
        download_fasttext(model_name)

    # Load the model from the .bin file
    print("Loading FastText embeddings...")
    embedding_model = load_facebook_vectors(f"data/fasttext_embeddings/{model_name}")
    embedding_model.save(save_path)
    print(f"Model saved locally at '{save_path}'.")


class ImageabilityPredictor:
    def __init__(
        self,
        embedding_model_path="models/cc.en.300.model",
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
        # Check if the embedding model exists
        if not os.path.exists(embedding_model_path):
            download_and_save_model()

        # Load the embedding model
        print(f"Loading embedding model from '{embedding_model_path}'...")
        self.embedding_model = FastTextKeyedVectors.load(embedding_model_path)
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

    def get_column_imageability(self, dataframe, column_name):
        """
        Generate the imageability score for a given column in a DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the column to evaluate.
            column_name (str): The name of the column to evaluate.

        Returns:
            pd.Series: Predicted imageability scores for the column.
        """
        embeddings = dataframe[column_name].apply(self.get_embedding)
        imageabilities = embeddings.apply(self.predict_imageability)
        return imageabilities


# Example Usage
if __name__ == "__main__":
    predictor = ImageabilityPredictor()

    # Example words
    words_to_predict = ["apple", "banana", "orange", "unknownword"]

    for word in words_to_predict:
        score = predictor.get_imageability(word)
        print(f"Word: '{word}' | Predicted Imageability: {score:.4f}")
