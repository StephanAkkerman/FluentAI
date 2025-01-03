import joblib
import numpy as np
from huggingface_hub import hf_hub_download

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.fasttext import fasttext_model
from fluentai.services.card_gen.utils.logger import logger


class ImageabilityPredictor:
    def __init__(
        self,
    ):
        """
        Initialize the ImageabilityPredictor by loading the embedding model and regression model.

        Args:
            embedding_model_name (str, optional): Name of the embedding model to load from Gensim.
                                                  Defaults to "fasttext-wiki-news-subwords-300".
            regression_model_path (str, optional): Path to the trained regression model (.joblib file).
                                                   Defaults to "models/best_model_LGBMRegressor.joblib".
        """
        # Load the embedding model
        self.embedding_model = fasttext_model

        self.regression_model = joblib.load(
            hf_hub_download(
                repo_id=config.get("IMAGEABILITY").get("MODEL"),
                filename=config.get("IMAGEABILITY").get("MODEL_FILE"),
                cache_dir="models",
            )
        )
        logger.info("Regression model loaded successfully.")

    def get_embedding(self, word):
        """
        Retrieve the embedding vector for a given word.

        Args:
            word (str): The word to retrieve the embedding for.

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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

        Returns
        -------
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
        logger.info(f"Word: '{word}' | Predicted Imageability: {score:.4f}")
