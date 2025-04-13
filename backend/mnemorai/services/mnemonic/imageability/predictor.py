import os

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from mnemorai.constants.config import config
from mnemorai.services.mnemonic.imageability.embeddings import (
    ImageabilityEmbeddings,
)


def make_predictions():
    """
    Generate imageability score predictions for the IPA dataset and save them to a CSV file.
    """
    embedding_model = ImageabilityEmbeddings(model_name="fasttext")
    regression_model = joblib.load(
        hf_hub_download(
            repo_id=config.get("IMAGEABILITY").get("PREDICTOR").get("REPO"),
            filename=config.get("IMAGEABILITY").get("PREDICTOR").get("FILE"),
            cache_dir="models",
        )
    )

    ipa_dataset = pd.read_csv(
        hf_hub_download(
            repo_id=config.get("PHONETIC_SIM").get("IPA").get("REPO"),
            filename=config.get("PHONETIC_SIM").get("IPA").get("FILE"),
            cache_dir="datasets",
            repo_type="dataset",
        )
    )

    # Only keep the unique words in the column "token_ort"
    ipa_dataset = ipa_dataset.drop_duplicates(subset=["token_ort"])

    predictions = []

    # Generate embeddings and predictions for the IPA dataset
    for idx, row in tqdm(ipa_dataset.iterrows(), total=len(ipa_dataset)):
        word = row["token_ort"]
        embedding = embedding_model.get_embedding(word)
        prediction = regression_model.predict(embedding.reshape(1, -1))[0]
        predictions.append((word, prediction))

    # Convert the predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["word", "imageability_score"])

    # Create a directory to save the predictions
    os.makedirs("local_data/imageability", exist_ok=True)

    # Save the predictions
    predictions_df.to_csv("local_data/imageability/predictions.csv", index=False)


class ImageabilityPredictor:
    def __init__(self):
        self.predictions_df = pd.read_csv(
            hf_hub_download(
                repo_id=config.get("IMAGEABILITY").get("PREDICTIONS").get("REPO"),
                filename=config.get("IMAGEABILITY").get("PREDICTIONS").get("FILE"),
                cache_dir="datasets",
                repo_type="dataset",
            )
        )

    def get_prediction(self, word):
        """
        Get the imageability score prediction for a given word.

        Args:
            word (str): The word to get the prediction for.

        Returns
        -------
            float: Predicted imageability score.
        """
        prediction = self.predictions_df[self.predictions_df["word"] == word][
            "imageability_score"
        ].values[0]
        return prediction

    def get_predictions(self, words):
        """
        Get the imageability score predictions for a list of words.

        Args:
            words (List[str]): List of words to get the predictions for.

        Returns
        -------
            List[float]: Predicted imageability scores.
        """
        predictions = [self.get_prediction(word) for word in words]
        return predictions

    def get_column_imageability(self, dataframe, column_name):
        """
        Generate the imageability score for a given column in a DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the words.
            column_name (str): The name of the column containing the words.

        Returns
        -------
            List[float]: Predicted imageability scores.
        """
        predictions = self.get_predictions(dataframe[column_name].tolist())
        return predictions


if __name__ == "__main__":
    predictor = ImageabilityPredictor()
    print(predictor.get_prediction("apple"))
