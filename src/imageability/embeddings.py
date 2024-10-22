# embed_generator.py

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from tqdm import tqdm


def load_embedding_model(model_name):
    """
    Load the specified embedding model.

    Args:
        model_name (str): Name of the model to load ('fasttext' or 'glove').

    Returns:
        gensim.models.KeyedVectors: Loaded embedding model.
    """
    if model_name.lower() == "fasttext":
        print("Loading FastText embeddings...")
        # embedding_model = api.load(
        #     "fasttext-wiki-news-subwords-300"
        # )  # 300-dim FastText
        embedding_model = KeyedVectors.load("models/fasttext.model")
        # embedding_model = load_facebook_vectors("data/wiki-news-300d-1M-subword.bin")
    elif model_name.lower() == "glove":
        print("Loading GloVe embeddings...")
        embedding_model = api.load("glove-wiki-gigaword-300")  # 300-dim GloVe
    else:
        raise ValueError("Unsupported model. Choose 'fasttext' or 'glove'.")
    print(f"{model_name.capitalize()} model loaded successfully.")
    return embedding_model


def get_embedding(word, embedding_model):
    """
    Retrieve the embedding vector for a given word.

    Args:
        word (str): The word to retrieve the embedding for.
        embedding_model (gensim.models.KeyedVectors): The embedding model.

    Returns:
        np.ndarray: Embedding vector for the word.
    """
    try:
        return embedding_model.get_vector(word)
    except KeyError:
        # Handle out-of-vocabulary (OOV) words
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

    # Preprocess the dataset
    initial_count = len(df)
    df = df.drop_duplicates(subset=word_column)
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows based on '{word_column}'.")

    initial_count = len(df)
    df = df.dropna(subset=[word_column, score_column])
    na_removed = initial_count - len(df)
    if na_removed > 0:
        print(f"Removed {na_removed} rows with NaN values.")

    # Separate features and target
    y = df[score_column].values
    words = df[word_column].astype(str).values  # Ensure all words are strings

    print(f"Number of unique words to process: {len(words)}")

    # Load the specified embedding model
    embedding_model = load_embedding_model(model)

    # Determine the number of CPU cores to use
    if n_jobs == -1:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = n_jobs
    print(f"Number of CPU cores to use: {num_cores}")

    embeddings = []

    # Define the worker function
    def worker(word):
        return get_embedding(word, embedding_model)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(worker, word) for word in words]

        # Initialize tqdm progress bar
        if verbose:
            progress_bar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating Embeddings",
                unit="word",
            )
        else:
            progress_bar = None

        # Iterate over futures as they complete
        for future in as_completed(futures):
            try:
                emb = future.result()
                embeddings.append(emb)
                if progress_bar:
                    progress_bar.update(1)
            except Exception as e:
                print(f"Error processing word: {e}")
                embeddings.append(
                    np.zeros(embedding_model.vector_size, dtype=np.float32)
                )
                if progress_bar:
                    progress_bar.update(1)

    if verbose:
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

    embedding_model = KeyedVectors.load("models/fasttext.model")
    print(embedding_model["a"])

    df = pd.read_parquet(f"data/imageability/{model}_embeddings.parquet")
    print(df.head())

    quit()
    generate_embeddings(
        input_csv="data/imageability/data.csv",
        output_parquet=f"data/imageability/{model}_embeddings4.parquet",
        model=model,
        word_column="word",
        score_column="score",
        n_jobs=-1,  # Use all available CPU cores
        verbose=True,  # Enable progress bar
    )
