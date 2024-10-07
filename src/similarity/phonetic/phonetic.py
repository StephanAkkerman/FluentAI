import faiss
import numpy as np
from ipa2vec import panphon_vec, soundvec
from utils import convert_to_matrix, load_cache, pad_vectors, timer


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


def main(
    ipa_input, top_n=5, method: str = "panphon", dataset: str = "eng_latn_us_broad"
):
    """
    Main function to find top_n closest phonetically similar words to the input IPA.

    Parameters:
    - ipa_input: String, IPA representation of the input word
    - top_n: Integer, number of top similar words to retrieve
    - vectorizer: Function used for vectorizing IPA input
    - vector_column: String, name of the column containing vectors
    """
    if method == "clts":
        vectorizer = soundvec
    elif method == "panphon":
        vectorizer = panphon_vec

    # Attempt to load from cache
    dataset = load_cache(method, dataset)

    dataset_vectors_flat = dataset["flattened_vectors"].tolist()

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
    ipa_input = "kˈut͡ʃiŋ"
    top_n = 15
    method = "panphon"  # or clts

    # main(ipa_input, top_n, method)
