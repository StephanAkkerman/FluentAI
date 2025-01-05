import ast

import numpy as np

from fluentai.logger import logger


def pad_vectors(vectors):
    """
    Pad all vectors to the maximum length with zeros.

    Parameters
    ----------
    - vectors: List of NumPy arrays

    Returns
    -------
    - List of padded NumPy arrays
    """
    max_len = max(vec.shape[0] for vec in vectors)
    padded_vectors = [
        np.pad(vec, (0, max_len - len(vec)), "constant") for vec in vectors
    ]
    return padded_vectors


def convert_to_matrix(padded_vectors):
    """
    Convert list of padded vectors to a NumPy matrix.

    Parameters
    ----------
    - padded_vectors: List of NumPy arrays

    Returns
    -------
    - NumPy array matrix
    """
    dataset_matrix = np.array(padded_vectors, dtype=np.float32)
    return dataset_matrix


def parse_vectors(dataset, vector_column="vectors"):
    """
    Parse the 'vectors' column from strings to actual lists using ast.literal_eval.

    Parameters
    ----------
    - dataset: DataFrame with 'vectors' column as strings
    - vector_column: String, name of the column containing vectors

    Returns
    -------
    - DataFrame with 'vectors' column as lists
    """
    try:
        dataset[vector_column] = dataset[vector_column].apply(ast.literal_eval)
        logger.info(f"Parsed '{vector_column}' column from strings to lists.")
    except Exception as e:
        logger.error(f"Error parsing '{vector_column}' column: {e}")
        raise
    return dataset


def flatten_vector(vec):
    """
    Flatten a nested list of vectors into a single 1D NumPy array.

    Parameters
    ----------
    - vec: List of lists or NumPy arrays

    Returns
    -------
    - 1D NumPy array
    """
    if not vec:
        logger.warning("Encountered an empty vector. Assigning a default value of 0.")
        return np.array([0.0], dtype=np.float32)
    try:
        flattened = np.hstack(vec).astype(np.float32)
        return flattened
    except Exception as e:
        logger.error(f"Error flattening vector: {e}")
        return np.array([0.0], dtype=np.float32)


def flatten_vectors(dataset, vector_column="vectors"):
    """
    Flatten each vector in the 'vectors' column using np.hstack, handling empty vectors.

    Parameters
    ----------
    - dataset: DataFrame with 'vectors' column as lists of lists
    - vector_column: String, name of the column containing vectors

    Returns
    -------
    - DataFrame with an additional 'flattened_vectors' column
    """
    dataset["flattened_vectors"] = dataset[vector_column].apply(flatten_vector)
    logger.info(
        f"Flattened all vectors. Total vectors: {len(dataset['flattened_vectors'])}."
    )
    return dataset
