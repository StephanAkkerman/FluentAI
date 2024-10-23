# embed_generator.py

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.models.fasttext import FastTextKeyedVectors
from tqdm import tqdm

from imageability import download_and_save_model

# Global variable to hold the embedding model in each worker process
embedding_model = None


def load_embedding_model(model_name):
    """
    Load the specified embedding model.
    """
    if model_name.lower() == "fasttext":
        print("Loading FastText embeddings...")
        embedding_model = FastTextKeyedVectors.load("models/cc.en.300.model")
    elif model_name.lower() == "glove":
        print("Loading GloVe embeddings...")
        embedding_model = api.load("glove-wiki-gigaword-300")
    else:
        raise ValueError("Unsupported model. Choose 'fasttext' or 'glove'.")
    print(f"{model_name.capitalize()} model loaded successfully.")
    return embedding_model


def initializer(model_name):
    """
    Initializer function for each worker process to load the embedding model.
    """
    global embedding_model
    embedding_model = load_embedding_model(model_name)


def get_embedding(word):
    """
    Retrieve the embedding vector for a given word.
    Note: This function runs in separate worker processes.
    """
    global embedding_model
    try:
        return embedding_model.get_vector(word)
    except KeyError:
        # For FastText, this might not be necessary, but kept as a fallback
        return np.zeros(embedding_model.vector_size, dtype=np.float32)


def generate_embeddings(
    input_csv,
    output_parquet,
    model="fasttext",
    word_column="word",
    score_column="score",
    n_jobs=-1,
    verbose=True,
):
    """
    Generate word embeddings from a dataset and save them to a .parquet file.

    Args:
        input_csv (str): Path to the input CSV file containing words and scores.
        output_parquet (str): Path to the output .parquet file to save embeddings and words.
        model (str, optional): Embedding model to use ('fasttext' or 'glove'). Defaults to 'fasttext'.
        word_column (str, optional): Name of the column containing words in the CSV. Defaults to 'word'.
        score_column (str, optional): Name of the column containing scores in the CSV. Defaults to 'score'.
        n_jobs (int, optional): Number of CPU cores to use. -1 means all available cores. Defaults to -1.
        verbose (bool, optional): Enable verbose output for progress bars. Defaults to True.

    Raises:
        FileNotFoundError: If the input CSV file is not found.
        ValueError: If required columns are missing or unsupported model is specified.
    """
    # Load your dataset
    try:
        print(f"Loading dataset from '{input_csv}'...")
        df = pd.read_csv(input_csv)
        print(f"Dataset loaded successfully with {len(df)} rows.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{input_csv}' was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{input_csv}' is empty.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading the dataset: {e}")

    # Validate required columns
    if word_column not in df.columns:
        raise ValueError(
            f"The specified word column '{word_column}' does not exist in the dataset."
        )
    if score_column not in df.columns:
        raise ValueError(
            f"The specified score column '{score_column}' does not exist in the dataset."
        )

    if model.lower() == "fasttext":
        if not os.path.exists("models/cc.en.300.model"):
            print("Fasttext model not found, downloading / saving it...")
            download_and_save_model()

    # Separate features and target
    y = df[score_column].values
    words = df[word_column].astype(str).values  # Ensure all words are strings

    print(f"Number of unique words to process: {len(words)}")

    # Determine the number of CPU cores to use
    if n_jobs == -1:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = n_jobs
    print(f"Number of CPU cores to use: {num_cores}")

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
    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Create a DataFrame for words and scores
    df_output = pd.DataFrame({"word": words, "score": y})

    # Determine embedding dimensions
    embedding_dim = embeddings.shape[1]
    embedding_columns = [f"emb_{i+1}" for i in range(embedding_dim)]

    # Create a DataFrame for embeddings
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)

    # Concatenate the words, scores, and embeddings into a single DataFrame
    df_output = pd.concat([df_output, embeddings_df], axis=1)

    print(f"Combined DataFrame shape: {df_output.shape}")

    # Save the DataFrame to a .parquet file
    try:
        df_output.to_parquet(output_parquet, index=False)
        print(f"Embeddings saved successfully to '{output_parquet}'.")
    except Exception as e:
        raise Exception(f"An error occurred while saving to parquet: {e}")


# Example usage
if __name__ == "__main__":
    model = "fasttext"  # Change to 'glove' to use GloVe embeddings

    generate_embeddings(
        input_csv="data/imageability/data.csv",
        output_parquet=f"data/imageability/{model}_embeddings.parquet",
        model=model,
        word_column="word",
        score_column="score",
        n_jobs=2,  # Use all available CPU cores = -1
        verbose=True,  # Enable progress bar
    )
