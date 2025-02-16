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


def build_faiss_index(matrix):
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


def vectorize_input(ipa_input, vectorizer, dimension):
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


def find_top_k_candidates_both(
    input_word: str,
    language_code: str,
    g2p_model,
    dataset: pd.DataFrame,
    index,
    vectorizer,
    dimension: int,
    top_k: int = 5,
    min_seg_length: int = 1,
) -> list:
    """
    For a given input word, search for the best candidate either as a single word or as a
    combination of two words. Returns the top_k candidates ranked by similarity score.

    Parameters
    ----------
    input_word : str
        The input word (e.g., "ratatouille").
    language_code : str
        Language code for IPA conversion.
    g2p_model : object
        Grapheme-to-phoneme model.
    dataset : pd.DataFrame
        DataFrame containing the single-word dataset.
    index : faiss.Index
        FAISS index built over the normalized single-word vectors.
    vectorizer : function
        Function to convert an IPA string to a vector (e.g., panphon_vec or soundvec).
    dimension : int
        The fixed dimension of the dataset vectors.
    top_k : int, optional
        Number of top candidate pairs to return (default is 5).
    min_seg_length : int, optional
        Minimum number of IPA characters per segment for splitting (default is 3).

    Returns
    -------
    list of dict
        Each dict represents a candidate match and includes keys such as:
          - "type": "single" or "split"
          - "candidate" (for non-split) or "prefix_candidate" and "suffix_candidate" (for split)
          - "score": similarity score
          - additional segmentation details.
    """
    candidate_pairs = []
    # Convert the input word to IPA once.
    ipa_full = word2ipa(input_word, language_code, g2p_model)

    # --------------------
    # Non-split search:
    # Search for the best matching word using the entire IPA string.
    full_vec = vectorize_input(ipa_full, vectorizer, dimension)
    full_dists, full_indices = index.search(full_vec, top_k)
    for i in range(top_k):
        candidate = dataset.iloc[full_indices[0][i]]["token_ort"]
        candidate_pairs.append(
            {
                "type": "single",
                "split_index": None,
                "candidate": candidate,
                "score": full_dists[0][i],
                "ipa": ipa_full,
            }
        )

    # --------------------
    # Split search:
    # Try every possible split position that respects the min_seg_length requirement.
    for i in range(min_seg_length, len(ipa_full) - min_seg_length):
        prefix_ipa = ipa_full[:i]
        suffix_ipa = ipa_full[i:]

        # Vectorize both segments.
        prefix_vec = vectorize_input(prefix_ipa, vectorizer, dimension)
        suffix_vec = vectorize_input(suffix_ipa, vectorizer, dimension)

        # Get top_k candidates for each segment.
        prefix_dists, prefix_indices = index.search(prefix_vec, top_k)
        suffix_dists, suffix_indices = index.search(suffix_vec, top_k)

        # Combine each prefix candidate with each suffix candidate.
        for j in range(top_k):
            for k in range(top_k):
                score = prefix_dists[0][j] + suffix_dists[0][k]
                candidate_prefix = dataset.iloc[prefix_indices[0][j]]["token_ort"]
                candidate_suffix = dataset.iloc[suffix_indices[0][k]]["token_ort"]
                candidate_pairs.append(
                    {
                        "type": "split",
                        "split_index": i,
                        "prefix_candidate": candidate_prefix,
                        "suffix_candidate": candidate_suffix,
                        "score": score,
                        "prefix_ipa": prefix_ipa,
                        "suffix_ipa": suffix_ipa,
                    }
                )

    # Sort all candidates by their score (higher is better) and return the top_k.
    candidate_pairs_sorted = sorted(
        candidate_pairs, key=lambda x: x["score"], reverse=True
    )
    return candidate_pairs_sorted[:top_k]


# Example usage:
if __name__ == "__main__":
    input_word = "latitude"
    language_code = "eng-us"

    from fluentai.services.mnemonic.phonetic.grapheme2phoneme import Grapheme2Phoneme

    g2p_model = Grapheme2Phoneme()

    method = config.get("PHONETIC_SIM").get("EMBEDDINGS").get("METHOD")
    dataset = load_from_cache(method)

    # Prepare the dataset vectors and FAISS index as in your existing pipeline.
    dataset_vectors_flat = dataset["flattened_vectors"].tolist()
    padded_vectors = pad_vectors(dataset_vectors_flat)
    dataset_matrix = convert_to_matrix(padded_vectors)
    faiss.normalize_L2(dataset_matrix)
    index = build_faiss_index(dataset_matrix)

    vectorizer = panphon_vec if method == "panphon" else soundvec
    dimension = dataset_matrix.shape[1]

    top_candidates = find_top_k_candidates_both(
        input_word,
        language_code,
        g2p_model,
        dataset,
        index,
        vectorizer,
        dimension,
        top_k=1,
    )

    for candidate in top_candidates:
        if candidate["type"] == "single":
            print(
                f"Whole word match: '{candidate['candidate']}' "
                f"(score: {candidate['score']:.4f}) for IPA '{candidate['ipa']}'"
            )
        else:
            print(
                f"Split at {candidate['split_index']}: "
                f"'{candidate['prefix_candidate']}' + '{candidate['suffix_candidate']}' "
                f"(score: {candidate['score']:.4f}) "
                f"from IPA '{candidate['prefix_ipa']}' + '{candidate['suffix_ipa']}'"
            )
