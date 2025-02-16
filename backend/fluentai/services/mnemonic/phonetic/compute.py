import faiss
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.services.mnemonic.phonetic.ipa2vec import panphon_vec, soundvec
from fluentai.services.mnemonic.phonetic.utils.cache import load_from_cache
from fluentai.services.mnemonic.phonetic.utils.vectors import (
    convert_to_matrix,
    pad_vectors,
)


def word2ipa(
    word: str,
    language_code: str,
    g2p_model,
) -> str:
    """
    Get the IPA representation of a word.

    Parameters
    ----------
    word : str
        The word to convert to IPA
    language_code : str, optional
        The language code of the word, by default "eng-us"

    Returns
    -------
    str
        The IPA representation of the word
    """
    # Try searching in the dataset
    if "eng-us" in language_code:
        # First try lookup in the .tsv file
        logger.debug("Loading the IPA dataset")
        eng_ipa = pd.read_csv(
            hf_hub_download(
                repo_id=config.get("PHONETIC_SIM").get("IPA").get("REPO"),
                filename=config.get("PHONETIC_SIM").get("IPA").get("FILE"),
                cache_dir="datasets",
                repo_type="dataset",
            )
        )

        # Check if the word is in the dataset
        ipa = eng_ipa[eng_ipa["token_ort"] == word]["token_ipa"]

        if not ipa.empty:
            return ipa.values[0].replace(" ", "")

    # Use the g2p model
    return g2p_model.g2p([f"<{language_code}>:{word}"])


def build_faiss_index(matrix: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for Inner Product similarity search.

    Parameters
    ----------
    - matrix: Normalized NumPy array

    Returns
    -------
    - FAISS index
    """
    dimension = matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)
    return index


def vectorize_input(ipa_input: str, vectorizer, dimension: int) -> np.ndarray:
    """
    Vectorize the input IPA string and pad to match dataset vector dimensions.

    Parameters
    ----------
    - ipa_input: String, IPA representation of the input word
    - vectorizer: Function to vectorize the IPA string
    - dimension: Integer, dimension of the dataset vectors

    Returns
    -------
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


class Phonetic_Similarity:
    def __init__(self):
        method = config.get("PHONETIC_SIM").get("EMBEDDINGS").get("METHOD")

        # Default to panphon if method is not specified
        self.vectorizer = panphon_vec

        if method == "clts":
            self.vectorizer = soundvec

        # Attempt to load from cache
        self.dataset = load_from_cache(method)

        # Pad the flattened vectors
        dataset_vectors_padded = pad_vectors(self.dataset["flattened_vectors"].tolist())

        # Convert to matrix
        dataset_matrix = convert_to_matrix(dataset_vectors_padded)
        self.dimension = dataset_matrix.shape[1]

        # Normalize dataset vectors
        faiss.normalize_L2(dataset_matrix)

        # Build FAISS index
        self.index = build_faiss_index(dataset_matrix)

    def top_phonetic(
        self, input_word: str, language_code: str, top_n: int, g2p_model
    ) -> tuple[pd.DataFrame, str]:
        """
        Main function to find top_n closest phonetically similar words to the input IPA.

        Parameters
        ----------
        - ipa_input: String, IPA representation of the input word
        - top_n: Integer, number of top similar words to retrieve
        - vectorizer: Function used for vectorizing IPA input
        - vector_column: String, name of the column containing vectors

        Returns
        -------
        - DataFrame containing the top_n similar words and their distances
        - The IPA representation of the input word
        """
        # Convert the input word to IPA representation
        ipa = word2ipa(input_word, language_code, g2p_model)

        # Vectorize input
        input_vector_padded = vectorize_input(
            ipa,
            self.vectorizer,
            self.dimension,
        )

        # Normalize input vector
        faiss.normalize_L2(input_vector_padded)

        # Perform search
        distances, indices = self.index.search(input_vector_padded, top_n)

        # Check if no similar words found
        if len(indices) == 0:
            logger.error("No similar words found.")
            return pd.DataFrame({"token_ort": [], "token_ipa": [], "distance": []})

        # Retrieve closest words
        closest_words = self.dataset.iloc[indices[0]][["token_ort", "token_ipa"]]

        # Add the distance column
        closest_words["distance"] = distances[0]

        return closest_words, ipa


if __name__ == "__main__":
    # Example usage
    word_input = "ratatouille"
    language_code = "eng-us"
    top_n = 15

    # Temporary fix
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # Load the G2P model
    from fluentai.services.mnemonic.phonetic.grapheme2phoneme import Grapheme2Phoneme

    phon_sim = Phonetic_Similarity()

    result = phon_sim.top_phonetic(word_input, language_code, top_n, Grapheme2Phoneme())
    print(result)
