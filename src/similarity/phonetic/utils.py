import ast
import functools
import logging
import os
import time

import numpy as np
import pandas as pd
from ipa2vec import panphon_vec, soundvec

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def timer(func):
    """
    Decorator to measure the execution time of functions.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logging.info(f"--- Starting '{func.__name__}' ---")
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"--- Finished '{func.__name__}' in {run_time:.2f} seconds ---\n")
        return value

    return wrapper_timer


@timer
def pad_vectors(vectors):
    """
    Pad all vectors to the maximum length with zeros.

    Parameters:
    - vectors: List of NumPy arrays

    Returns:
    - List of padded NumPy arrays
    """
    max_len = max(vec.shape[0] for vec in vectors)
    padded_vectors = [
        np.pad(vec, (0, max_len - len(vec)), "constant") for vec in vectors
    ]
    print(f"Padded vectors to maximum length {max_len}.")
    return padded_vectors


@timer
def convert_to_matrix(padded_vectors):
    """
    Convert list of padded vectors to a NumPy matrix.

    Parameters:
    - padded_vectors: List of NumPy arrays

    Returns:
    - NumPy array matrix
    """
    dataset_matrix = np.array(padded_vectors, dtype=np.float32)
    print(f"Converted padded vectors to matrix with shape {dataset_matrix.shape}.")
    return dataset_matrix


@timer
def load_dataset(vectorizer, vector_column="vectors", max_rows=None):
    """
    Load the dataset based on the vectorizer.

    Parameters:
    - vectorizer: Function used for vectorizing (panphon_vec or soundvec)
    - vector_column: String, name of the column containing vectors
    - max_rows: Integer or None, number of top rows to load

    Returns:
    - DataFrame containing the dataset
    """
    vector_file = (
        "data/eng_latn_us_broad_vectors_panphon.csv"
        if vectorizer == panphon_vec
        else "data/eng_latn_us_broad_vectors.csv"
    )
    try:
        df = pd.read_csv(vector_file, nrows=max_rows)
        logging.info(f"Dataset loaded from '{vector_file}' with {len(df)} rows.")
    except FileNotFoundError:
        logging.error(f"Vector file '{vector_file}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise
    return df


@timer
def parse_vectors(dataset, vector_column="vectors"):
    """
    Parse the 'vectors' column from strings to actual lists using ast.literal_eval.

    Parameters:
    - dataset: DataFrame with 'vectors' column as strings
    - vector_column: String, name of the column containing vectors

    Returns:
    - DataFrame with 'vectors' column as lists
    """
    try:
        dataset[vector_column] = dataset[vector_column].apply(ast.literal_eval)
        logging.info(f"Parsed '{vector_column}' column from strings to lists.")
    except Exception as e:
        logging.error(f"Error parsing '{vector_column}' column: {e}")
        raise
    return dataset


def save_cache(dataset, cache_file="cache/parsed_dataset.parquet"):
    """
    Save the processed dataset to a cache file.

    Parameters:
    - dataset: DataFrame to save
    - cache_file: String, path to the cache file

    Returns:
    - None
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    dataset.to_parquet(cache_file, index=False)
    logging.info(f"Saved parsed dataset to '{cache_file}'.")


def load_cache(method: str = "panphon"):
    """
    Load the processed dataset from a cache file.

    Parameters:
    - cache_file: String, path to the cache file

    Returns:
    - DataFrame containing the cached dataset
    """
    if method == "clts":
        cache_file = "cache/parsed_dataset_clts.parquet"
    elif method == "panphon":
        cache_file = "cache/parsed_dataset_panphon.parquet"

    if os.path.exists(cache_file):
        dataset = pd.read_parquet(cache_file)
        logging.info(f"Loaded parsed dataset from '{cache_file}'.")
        return dataset
    else:
        logging.info(f"No cache found at '{cache_file}'.")
        return None


def flatten_vector(vec):
    """
    Flatten a nested list of vectors into a single 1D NumPy array.

    Parameters:
    - vec: List of lists or NumPy arrays

    Returns:
    - 1D NumPy array
    """
    if not vec:
        logging.warning("Encountered an empty vector. Assigning a default value of 0.")
        return np.array([0.0], dtype=np.float32)
    try:
        flattened = np.hstack(vec).astype(np.float32)
        return flattened
    except Exception as e:
        logging.error(f"Error flattening vector: {e}")
        return np.array([0.0], dtype=np.float32)


@timer
def flatten_vectors(dataset, vector_column="vectors"):
    """
    Flatten each vector in the 'vectors' column using np.hstack, handling empty vectors.

    Parameters:
    - dataset: DataFrame with 'vectors' column as lists of lists
    - vector_column: String, name of the column containing vectors

    Returns:
    - DataFrame with an additional 'flattened_vectors' column
    """
    dataset["flattened_vectors"] = dataset[vector_column].apply(flatten_vector)
    logging.info(
        f"Flattened all vectors. Total vectors: {len(dataset['flattened_vectors'])}."
    )
    return dataset