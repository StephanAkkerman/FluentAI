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

    def __init__(self, g2p_model):
        self.g2p_model = g2p_model
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

    def top_phonetic_old(
        self,
        input_word: str,
        language_code: str,
        top_n: int,
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
        ipa = word2ipa(input_word, language_code, self.g2p_model)

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

    def top_phonetic(
        self,
        input_word: str,
        language_code: str,
        top_n: int,
        min_seg_length: int = 3,
        top_k: int = 3,
    ) -> tuple[pd.DataFrame, str]:
        """
        Find the top_n closest phonetically similar words to the input IPA.

        In addition to a full-word search, this function tries every possible split
        of the IPA transcription (respecting a minimum segment length) and retrieves
        top_k candidates for each segment. For each candidate pair, the similarity scores
        are averaged to create a combined candidate.

        Parameters
        ----------
        input_word : str
            The input word.
        language_code : str
            The language code of the word.
        top_n : int
            Number of top similar matches to retrieve.
        min_seg_length : int, optional
            Minimum length for each IPA segment when splitting (default is 3).
        top_k : int, optional
            Number of candidates to retrieve for each segment (default is 3).

        Returns
        -------
        tuple[pd.DataFrame, str]
            A tuple containing:
            - A DataFrame with columns "token_ort", "token_ipa", "distance", "match_type", and "split_position"
            - The IPA transcription of the input word.
        """
        # Convert the input word to IPA.
        ipa = word2ipa(input_word, language_code, self.g2p_model)

        # Full–word search.
        input_vector = vectorize_input(ipa, self.vectorizer, self.dimension)
        faiss.normalize_L2(input_vector)
        full_dists, full_indices = self.index.search(input_vector, top_n)
        single_results = self.dataset.iloc[full_indices[0]][
            ["token_ort", "token_ipa"]
        ].copy()
        single_results["distance"] = full_dists[0]
        single_results["match_type"] = "full"
        single_results["split_position"] = None
        # single_results["score"] =

        # Prepare to store split–based candidates.
        split_candidates = []

        # Only try splitting if IPA is long enough.
        if len(ipa) >= 2 * min_seg_length:
            # Try every possible split position that respects the min_seg_length requirement.
            for i in range(min_seg_length, len(ipa) - min_seg_length + 1):
                prefix_ipa = ipa[:i]
                suffix_ipa = ipa[i:]

                # Vectorize both segments.
                prefix_vec = vectorize_input(
                    prefix_ipa, self.vectorizer, self.dimension
                )
                suffix_vec = vectorize_input(
                    suffix_ipa, self.vectorizer, self.dimension
                )
                faiss.normalize_L2(prefix_vec)
                faiss.normalize_L2(suffix_vec)

                # Get top_k candidates for each segment.
                prefix_dists, prefix_indices = self.index.search(prefix_vec, top_k)
                suffix_dists, suffix_indices = self.index.search(suffix_vec, top_k)

                # For each candidate pair, average the scores.
                for j in range(top_k):
                    for k in range(top_k):
                        cand_prefix = self.dataset.iloc[prefix_indices[0][j]]
                        cand_suffix = self.dataset.iloc[suffix_indices[0][k]]
                        avg_score = (prefix_dists[0][j] + suffix_dists[0][k]) / 2.0
                        # avg_score -= penalty

                        combined_word = (
                            f"{cand_prefix['token_ort']}+{cand_suffix['token_ort']}"
                        )
                        combined_ipa = (
                            f"{cand_prefix['token_ipa']}+{cand_suffix['token_ipa']}"
                        )
                        split_candidates.append(
                            (combined_word, combined_ipa, avg_score, "split", i)
                        )

        # Convert split candidates to DataFrame.
        if split_candidates:
            split_results = pd.DataFrame(
                split_candidates,
                columns=[
                    "token_ort",
                    "token_ipa",
                    "distance",
                    "match_type",
                    "split_position",
                ],
            )
        else:
            split_results = pd.DataFrame(
                columns=[
                    "token_ort",
                    "token_ipa",
                    "distance",
                    "match_type",
                    "split_position",
                ]
            )

        # Combine full-word and split results.
        combined_results = pd.concat([single_results, split_results], ignore_index=True)
        # Drop duplicates based on token_ort and token_ipa.
        combined_results = combined_results.drop_duplicates(
            subset=["token_ort", "token_ipa"]
        )
        combined_results = combined_results.sort_values(by="distance", ascending=False)
        final_results = combined_results.head(top_n).reset_index(drop=True)

        return final_results, ipa


if __name__ == "__main__":
    # Example usage
    word_input = "kucing"  # "ratatouille"
    language_code = "ind"  # "eng-us"
    top_n = 25

    # Temporary fix
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # Load the G2P model
    from fluentai.services.mnemonic.phonetic.grapheme2phoneme import Grapheme2Phoneme

    phon_sim = Phonetic_Similarity(Grapheme2Phoneme())

    result = phon_sim.top_phonetic(word_input, language_code, top_n)
    print(result)
