import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from fluentai.constants.config import config
from fluentai.logger import logger

EMBEDDING_MODEL = None


class ImageabilityEmbeddings:
    def __init__(
        self, model_name=config.get("IMAGEABILITY").get("EMBEDDINGS").get("MODEL")
    ):
        self.model_name = model_name
        self.model = self.load_embedding_model()

    def load_embedding_model(self):
        """
        Load the specified embedding model.
        """
        if self.model_name == "fasttext":
            logger.info("Loading FastText model for imageability embeddings...")
            from fluentai.utils.fasttext import fasttext_model

            return fasttext_model

        return SentenceTransformer(
            self.model_name, trust_remote_code=True, cache_folder="models"
        )

    def get_embedding(self, word):
        """
        Retrieve the embedding vector for a given word.

        Args:
            word (str): The word to retrieve the embedding for.

        Returns
        -------
            np.ndarray: Embedding vector for the word.
        """
        if self.model_name == "fasttext":
            return self.model.get_vector(word)
        # Ensure the output is a NumPy array
        return self.model.encode(
            word, convert_to_tensor=False, normalize_embeddings=True
        )


def initializer(model_name: str):
    """
    Initializer function for each worker process to load the embedding model.
    """
    global EMBEDDING_MODEL
    EMBEDDING_MODEL = ImageabilityEmbeddings(model_name)


def get_embedding(word):
    """
    Global function to retrieve the embedding for a given word using the global EMBEDDING_MODEL.

    Args:
        word (str): The word to retrieve the embedding for.

    Returns
    -------
        np.ndarray: Embedding vector for the word.
    """
    global EMBEDDING_MODEL
    return EMBEDDING_MODEL.get_embedding(word)


def generate_embeddings(
    n_jobs=-1,
    verbose=True,
):
    """
    Generate word embeddings from a dataset and save them to a .parquet file.

    Args:
        n_jobs (int, optional): Number of CPU cores to use. -1 means all available cores. Defaults to -1.
        verbose (bool, optional): Enable verbose output for progress bars. Defaults to True.

    Raises
    ------
        ValueError: If required columns are missing or unsupported model is specified.
    """
    model = config.get("IMAGEABILITY").get("EMBEDDINGS").get("MODEL")
    # Remove any / from the output_parquet path
    model_name = model.replace("/", "_")
    output_parquet = f"local_data/imageability/{model_name}_embeddings.parquet"

    # Load your dataset
    df = load_dataset(
        config.get("IMAGEABILITY").get("EMBEDDINGS").get("EVAL").get("DATASET"),
        cache_dir="datasets",
        split="train",
    ).to_pandas()

    # Separate features and target
    y = df["score"].values
    words = df["word"].astype(str).values  # Ensure all words are strings

    logger.info(f"Number of unique words to process: {len(words)}")

    # Determine the number of CPU cores to use
    if n_jobs == -1:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = n_jobs
    logger.info(f"Number of CPU cores to use: {num_cores}")

    embeddings = []

    # Use ProcessPoolExecutor for parallel processing with initializer
    with ProcessPoolExecutor(
        max_workers=num_cores, initializer=initializer, initargs=(model,)
    ) as executor:
        # Initialize tqdm progress bar if verbose
        if verbose:
            progress_bar = tqdm(
                total=len(words), desc="Generating Embeddings", unit="word"
            )
        else:
            progress_bar = None

        # Use executor.map to preserve order
        for emb in executor.map(get_embedding, words):
            embeddings.append(emb)
            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

    # Convert list of embeddings to a NumPy array
    try:
        embeddings = np.vstack(embeddings)
    except ValueError as ve:
        logger.error(f"Error stacking embeddings: {ve}")
        raise

    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    # Create a DataFrame for words and scores
    df_output = pd.DataFrame({"word": words, "score": y})

    # Determine embedding dimensions
    embedding_dim = embeddings.shape[1]
    embedding_columns = [f"emb_{i+1}" for i in range(embedding_dim)]

    # Create a DataFrame for embeddings
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)

    # Concatenate the words, scores, and embeddings into a single DataFrame
    df_output = pd.concat([df_output, embeddings_df], axis=1)

    logger.info(f"Combined DataFrame shape: {df_output.shape}")

    # Save the DataFrame to a .parquet file
    df_output.to_parquet(output_parquet, index=False)
    logger.info(f"Embeddings saved successfully to '{output_parquet}'.")

    # Upload the embeddings to Hugging Face Hub
    upload_embeddings(output_parquet)


def upload_embeddings(output_parquet: str):
    """
    Upload embeddings to Hugging Face Hub.
    """
    file_name = output_parquet.split("/")[-1]
    api = HfApi()
    api.upload_file(
        path_or_fileobj=output_parquet,
        path_in_repo=file_name,
        repo_id=config.get("IMAGEABILITY").get("EMBEDDINGS").get("REPO"),
        repo_type="dataset",
    )
    logger.info("Embeddings uploaded to Hugging Face Hub")


# Example usage
if __name__ == "__main__":
    generate_embeddings(
        n_jobs=-1,  # Use desired number of CPU cores; -1 for all available
        verbose=True,  # Enable progress bar
    )
