import ast
import functools
import time

import faiss
import numpy as np
import pandas as pd
from ipa2vec import panphon_vec, soundvec


def timer(func):
    """
    Decorator to measure the execution time of functions.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        print(f"--- Starting '{func.__name__}' ---")
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"--- Finished '{func.__name__}' in {run_time:.2f} seconds ---\n")
        return value

    return wrapper_timer


@timer
def load_dataset(vectorizer, vector_column="vectors"):
    """
    Load the dataset based on the vectorizer.

    Parameters:
    - vectorizer: Function used for vectorizing (panphon_vec or soundvec)
    - vector_column: String, name of the column containing vectors

    Returns:
    - DataFrame containing the dataset
    """
    vector_file = (
        "data/eng_latn_us_broad_vectors_panphon.csv"
        if vectorizer == panphon_vec
        else "data/eng_latn_us_broad_vectors.csv"
    )
    try:
        df = pd.read_csv(vector_file)
        print(f"Dataset loaded from {vector_file}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Vector file {vector_file} not found.")
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
        print(f"Parsed '{vector_column}' column from strings to lists.")
    except Exception as e:
        raise ValueError(f"Error parsing '{vector_column}' column: {e}")
    return dataset


@timer
def flatten_vectors(dataset, vector_column="vectors"):
    """
    Flatten each vector in the 'vectors' column using np.hstack, handling empty vectors.

    Parameters:
    - dataset: DataFrame with 'vectors' column as lists of lists
    - vector_column: String, name of the column containing vectors

    Returns:
    - List of flattened vectors
    """

    def safe_hstack(vec):
        if len(vec) > 0:
            return np.hstack(vec)
        else:
            # Determine sub-vector size if possible
            return np.array([0.0])  # Adjust this as needed

    dataset_vectors_flat = [safe_hstack(vec) for vec in dataset[vector_column]]
    print(f"Flattened {len(dataset_vectors_flat)} vectors.")
    return dataset_vectors_flat


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
def normalize_vectors(matrix):
    """
    Normalize vectors to have unit length for cosine similarity.

    Parameters:
    - matrix: NumPy array

    Returns:
    - Normalized NumPy array
    """
    faiss.normalize_L2(matrix)
    print("Normalized dataset vectors for cosine similarity.")
    return matrix


@timer
def build_faiss_index(matrix):
    """
    Build a FAISS index for Inner Product similarity search.

    Parameters:
    - matrix: Normalized NumPy array

    Returns:
    - FAISS index
    """
    dimension = matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)
    print(f"Built FAISS index with dimension {dimension} and {index.ntotal} vectors.")
    return index


@timer
def vectorize_input(ipa_input, vectorizer, dimension):
    """
    Vectorize the input IPA string and pad to match dataset vector dimensions.

    Parameters:
    - ipa_input: String, IPA representation of the input word
    - vectorizer: Function to vectorize the IPA string
    - dimension: Integer, dimension of the dataset vectors

    Returns:
    - Padded and reshaped input vector as NumPy array
    """
    input_vector = np.hstack(vectorizer(ipa_input)).astype(np.float32)
    input_length = len(input_vector)
    if input_length > dimension:
        input_vector_padded = input_vector[:dimension]
        print(f"Padded input vector by truncating to {dimension} elements.")
    else:
        padding_length = dimension - input_length
        input_vector_padded = np.pad(input_vector, (0, padding_length), "constant")
        print(f"Padded input vector with {padding_length} zeros.")
    input_vector_padded = input_vector_padded.reshape(1, -1)
    return input_vector_padded


@timer
def normalize_input_vector(input_vector):
    """
    Normalize the input vector for cosine similarity.

    Parameters:
    - input_vector: NumPy array

    Returns:
    - Normalized input vector
    """
    faiss.normalize_L2(input_vector)
    print("Normalized input vector for cosine similarity.")
    return input_vector


@timer
def perform_search(index, input_vector, top_n=5):
    """
    Perform similarity search using FAISS index.

    Parameters:
    - index: FAISS index
    - input_vector: Normalized input vector as NumPy array
    - top_n: Number of top similar vectors to retrieve

    Returns:
    - distances: NumPy array of similarity scores
    - indices: NumPy array of indices of similar vectors
    """
    distances, indices = index.search(input_vector, top_n)
    print(f"Performed search and retrieved top {top_n} closest vectors.")
    return distances, indices


@timer
def retrieve_closest_words(dataset, indices, top_n=5):
    """
    Retrieve the closest words from the dataset based on indices.

    Parameters:
    - dataset: DataFrame
    - indices: NumPy array of indices
    - top_n: Number of top similar words to retrieve

    Returns:
    - DataFrame of closest words with 'token_ort' and 'token_ipa'
    """
    closest_words = dataset.iloc[indices[0]][["token_ort", "token_ipa"]]
    print(f"Retrieved top {top_n} closest words from the dataset.")
    return closest_words


def main(ipa_input, top_n=5, vectorizer=panphon_vec, vector_column="vectors"):
    """
    Main function to find top_n closest phonetically similar words to the input IPA.

    Parameters:
    - ipa_input: String, IPA representation of the input word
    - top_n: Integer, number of top similar words to retrieve
    - vectorizer: Function used for vectorizing IPA input
    - vector_column: String, name of the column containing vectors
    """
    # Load dataset
    dataset = load_dataset(vectorizer, vector_column)

    # Parse vectors
    dataset = parse_vectors(dataset, vector_column)

    # Flatten vectors
    dataset_vectors_flat = flatten_vectors(dataset, vector_column)

    # Pad vectors
    dataset_vectors_padded = pad_vectors(dataset_vectors_flat)

    # Convert to matrix
    dataset_matrix = convert_to_matrix(dataset_vectors_padded)

    # Normalize dataset vectors
    dataset_matrix = normalize_vectors(dataset_matrix)

    # Build FAISS index
    index = build_faiss_index(dataset_matrix)

    # Vectorize input
    input_vector_padded = vectorize_input(
        ipa_input, vectorizer, dataset_matrix.shape[1]
    )

    # Normalize input vector
    input_vector_padded = normalize_input_vector(input_vector_padded)

    # Perform search
    distances, indices = perform_search(index, input_vector_padded, top_n)

    # Retrieve closest words
    closest_words = retrieve_closest_words(dataset, indices, top_n)

    # Display the results
    print(f"Top {top_n} phonetically similar words to '{ipa_input}':")
    print(closest_words.to_string(index=False))


if __name__ == "__main__":
    # Example usage
    ipa_input = "mɝəkə"  # "kˈut͡ʃiŋ"
    top_n = 5
    vectorizer = panphon_vec  # or soundvec

    main(ipa_input, top_n, vectorizer)
