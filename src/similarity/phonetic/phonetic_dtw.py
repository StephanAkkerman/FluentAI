import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from fastdtw import fastdtw
from ipa2vec import panphon_vec, soundvec
from tqdm import tqdm
from utils import (
    flatten_vector,
    flatten_vectors,
    load_cache,
    load_dataset,
    parse_vectors,
    save_cache,
    timer,
)


def compute_dtw_distance(v1, v2, dist=2):
    """
    Compute the DTW distance between two vectors.

    Parameters:
    - v1: 1D NumPy array
    - v2: 1D NumPy array
    - dist: Integer, distance metric (default=2 for Euclidean)

    Returns:
    - Float, DTW distance
    """
    try:
        distance, _ = fastdtw(v1, v2, dist=dist)
        return distance
    except Exception as e:
        logging.error(f"Error computing DTW distance: {e}")
        return np.inf


def compute_distance(args, dist=2):
    """
    Wrapper function to compute DTW distance. Needed for multiprocessing.

    Parameters:
    - args: Tuple containing (input_vector, vec)
    - dist: Integer, distance metric

    Returns:
    - Float, DTW distance
    """
    input_vector, vec = args
    return compute_dtw_distance(input_vector, vec, dist)


@timer
def find_closest_words_dtw(ipa_word, vectorizer, dataset, top_n=5, n_jobs=4):
    """
    Find the top N closest words to the input IPA word using DTW with parallel processing.

    Parameters:
    - ipa_word: String, IPA representation of the input word
    - vectorizer: Function to vectorize the IPA word
    - dataset: DataFrame containing the dataset with 'flattened_vectors'
    - top_n: Integer, number of top similar words to retrieve
    - n_jobs: Integer, number of parallel processes

    Returns:
    - DataFrame of top N closest words with their 'token_ort' and 'token_ipa'
    """
    # Vectorize the input IPA word
    input_vector = flatten_vector(vectorizer(ipa_word))
    logging.info(f"Vectorized input IPA word '{ipa_word}'.")

    # Extract all vectors as a list
    vectors = dataset["flattened_vectors"].tolist()

    # Prepare arguments for parallel processing
    args = [(input_vector, vec) for vec in vectors]

    # Initialize a list to store distances
    distances = []

    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Use executor.map which preserves the order of the input
        # Wrap with tqdm for progress tracking
        for distance in tqdm(
            executor.map(partial(compute_distance, dist=2), args),
            total=len(args),
            desc="Computing DTW distances",
        ):
            distances.append(distance)

    # Add distances to the dataset
    dataset = dataset.copy()
    dataset["dtw_distance"] = distances

    # Replace only in 'dtw_distance' column
    dataset["dtw_distance"] = dataset["dtw_distance"].replace([np.inf, -np.inf], np.nan)

    # Drop rows where 'dtw_distance' is NaN
    dataset = dataset.dropna(subset=["dtw_distance"])

    # Get the top N words with the smallest DTW distances
    top_words = dataset.nsmallest(top_n, "dtw_distance")

    logging.info(f"Retrieved top {top_n} closest words.")

    return top_words.reset_index(drop=True)


def main(
    ipa_input,
    top_n=5,
    vectorizer=soundvec,
    vector_column="vectors",
    max_rows=100,
    n_jobs=4,
    cache_file="cache/parsed_dataset.parquet",
):
    """
    Main function to find top N closest phonetically similar words to the input IPA.

    Parameters:
    - ipa_input: String, IPA representation of the input word
    - top_n: Integer, number of top similar words to retrieve
    - vectorizer: Function used for vectorizing IPA input
    - vector_column: String, name of the column containing vectors
    - max_rows: Integer or None, number of top rows to load from the dataset
    - n_jobs: Integer, number of parallel processes

    Returns:
    - None (prints the closest words)
    """
    # Attempt to load from cache
    dataset = load_cache(cache_file)

    if dataset is None:
        # Load dataset
        dataset = load_dataset(vectorizer, vector_column, max_rows)

        # Parse vectors
        dataset = parse_vectors(dataset, vector_column)

        # Flatten vectors
        dataset = flatten_vectors(dataset, vector_column)

        # Save to cache
        save_cache(dataset, cache_file)
    else:
        # If cached, skip loading and parsing
        pass

    # Find closest words using DTW
    closest_words = find_closest_words_dtw(
        ipa_input, vectorizer, dataset, top_n, n_jobs
    )

    # Display the closest words
    logging.info(f"Top {top_n} phonetically similar words to '{ipa_input}':")
    print(closest_words[["token_ort", "token_ipa"]])
    # except Exception as e:
    #     logging.error(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    # Example usage
    ipa_input = "kˈut͡ʃiŋ"
    top_n = 5
    vectorizer = panphon_vec  # or soundvec
    vector_column = "vectors"
    max_rows = None  # Load only top 100 rows
    n_jobs = 8  # Number of parallel processes

    main(ipa_input, top_n, vectorizer, vector_column, max_rows, n_jobs)
