import faiss
import numpy as np

from similarity.phonetic.g2p import g2p
from similarity.phonetic.ipa2vec import panphon_vec, soundvec
from similarity.phonetic.utils import convert_to_matrix, load_cache, pad_vectors
from similarity.phonetic.vectorizer import load_data


def word2ipa(
    word: str,
    language_code: str = "eng-us",
) -> str:

    # Try searching in the dataset
    if "eng-us" in language_code:
        # First try lookup in the .tsv file
        eng_ipa = load_data("data/phonological/en_US.txt")

        # Check if the word is in the dataset
        ipa = eng_ipa[eng_ipa["token_ort"] == word]["token_ipa"]

        if not ipa.empty:
            return ipa.values[0].replace(" ", "")

    # Use the g2p model
    return g2p([f"<{language_code}>:{word}"])


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
    return index


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
    else:
        padding_length = dimension - input_length
        input_vector_padded = np.pad(input_vector, (0, padding_length), "constant")
    input_vector_padded = input_vector_padded.reshape(1, -1)
    return input_vector_padded


def top_phonetic(
    input_word: str,
    language_code: str,
    top_n=15,
    method: str = "panphon",
    dataset: str = "en_US",
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

    # Convert the input word to IPA representation
    ipa = word2ipa(input_word, language_code)

    # Attempt to load from cache
    dataset = load_cache(method, dataset)

    dataset_vectors_flat = dataset["flattened_vectors"].tolist()

    # Pad vectors
    dataset_vectors_padded = pad_vectors(dataset_vectors_flat)

    # Convert to matrix
    dataset_matrix = convert_to_matrix(dataset_vectors_padded)

    # Normalize dataset vectors
    faiss.normalize_L2(dataset_matrix)

    # Build FAISS index
    index = build_faiss_index(dataset_matrix)

    # Vectorize input
    input_vector_padded = vectorize_input(ipa, vectorizer, dataset_matrix.shape[1])

    # Normalize input vector
    faiss.normalize_L2(input_vector_padded)

    # Perform search
    distances, indices = index.search(input_vector_padded, top_n)

    # Retrieve closest words
    closest_words = dataset.iloc[indices[0]][["token_ort", "token_ipa"]]

    # Add the distance column
    closest_words["distance"] = distances[0]

    return closest_words


if __name__ == "__main__":
    # Example usage
    word_input = "kucing"
    language_code = "ind"
    top_n = 15
    method = "panphon"  # or clts
    dataset = "en_US"  # "en_US" or eng_latn_us_broad

    # Temporary fix
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    print(top_phonetic(word_input, language_code, top_n, method, dataset))
