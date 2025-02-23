import asyncio

import faiss
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, util

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.services.mnemonic.phonetic.ipa2vec import panphon_vec, soundvec
from fluentai.services.mnemonic.phonetic.utils.cache import load_from_cache
from fluentai.services.mnemonic.phonetic.utils.vectors import (
    convert_to_matrix,
    pad_vectors,
)
from fluentai.services.mnemonic.semantic.translator import translate_word


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
        self.dataset = pd.read_parquet(
            "datasets/mnemonics.parquet"
        )  # load_from_cache(method)

        # Pad the flattened vectors
        # TODO: move this (and other steps) to the dataset creation
        dataset_vectors_padded = pad_vectors(self.dataset["flattened_vectors"].tolist())

        # Convert to matrix
        dataset_matrix = convert_to_matrix(dataset_vectors_padded)
        self.dimension = dataset_matrix.shape[1]

        # Normalize dataset vectors
        faiss.normalize_L2(dataset_matrix)

        # Build FAISS index
        self.index = build_faiss_index(dataset_matrix)

        model_name = config.get("SEMANTIC_SIM").get("MODEL").lower()
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, cache_folder="models"
        )

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

    def semantic_sim(self, embedding, single_results: pd.DataFrame) -> list:
        # Shape should be [x, embedding_dim]

        # Stack the numpy arrays into one array and convert it to a tensor
        corpus_embeddings = torch.tensor(
            np.vstack(single_results["word_embedding"].tolist())
        )

        # Move to same device
        corpus_embeddings = corpus_embeddings.to(embedding.device)

        # Compute cosine similarity between the query and all corpus embeddings.
        cos_scores = util.cos_sim(embedding, corpus_embeddings)

        return cos_scores.squeeze(0).cpu().tolist()

    def top_phonetic(
        self,
        input_word: str,
        language_code: str,
        top_n: int,
        min_seg_length: int = 3,
        top_k: int = 10,
    ) -> tuple[pd.DataFrame, str]:
        """
        Find the top_n closest phonetically similar words to the input IPA.

        This function performs a full–word search and, if possible, splits the IPA transcription
        to generate additional candidate pairs using vectorized operations.
        """
        logger.debug(f"Finding top {top_n} phonetically similar words to {input_word}.")
        penalty = 0  # TODO: add penalty to config

        # Step 1: Translate and embed the input word.
        translated, _ = asyncio.run(translate_word(input_word, language_code))
        embedding = self._get_embedding(translated)

        # Step 2: Convert the input word to IPA.
        ipa = word2ipa(input_word, language_code, self.g2p_model)

        # Step 3: Perform a full–word search.
        full_results = self._full_word_search(ipa, top_n, embedding)

        # Step 4: Compute split–based candidates if the IPA is long enough.
        split_results = self._split_candidates_search(
            ipa, embedding, min_seg_length, top_k, penalty
        )

        # Step 5: Combine the full–word and split candidates, remove duplicates, and sort.
        combined_results = self._combine_results(full_results, split_results, top_n)
        return combined_results, ipa

    def _get_embedding(self, translated: str):
        """Encode the translated word and ensure the embedding is 2D."""
        embedding = self.model.encode(
            translated, convert_to_tensor=True, normalize_embeddings=True
        )
        return embedding.unsqueeze(0) if embedding.dim() == 1 else embedding

    def _full_word_search(self, ipa: str, top_n: int, embedding) -> pd.DataFrame:
        """Perform a full–word search using FAISS and compute the full search scores."""
        input_vector = vectorize_input(ipa, self.vectorizer, self.dimension)
        faiss.normalize_L2(input_vector)
        full_dists, full_indices = self.index.search(input_vector, top_n)

        results = self.dataset.iloc[full_indices[0]][
            [
                "token_ort",
                "token_ipa",
                "norm_freq",
                "scaled_aoa",
                "imageability_score",
                "word_embedding",
            ]
        ].copy()
        results["distance"] = full_dists[0]
        results["semantic_similarity"] = self.semantic_sim(embedding, results)
        results["score"] = (
            results["distance"]
            + results["norm_freq"]
            + results["scaled_aoa"]
            + results["imageability_score"]
            + results["semantic_similarity"]
        )
        return results

    def _split_candidates_search(
        self, ipa: str, embedding, min_seg_length: int, top_k: int, penalty: float
    ) -> pd.DataFrame:
        """Compute candidate pairs based on splitting the IPA transcription."""
        if len(ipa) < 2 * min_seg_length:
            return pd.DataFrame(
                columns=[
                    "token_ort",
                    "token_ipa",
                    "distance",
                    "norm_freq",
                    "scaled_aoa",
                    "imageability_score",
                    "semantic_similarity",
                    "score",
                ]
            )

        candidates = []
        # Iterate over all valid split positions.
        for i in range(min_seg_length, len(ipa) - min_seg_length + 1):
            prefix_ipa = ipa[:i]
            suffix_ipa = ipa[i:]

            # Vectorize both segments.
            prefix_vec = vectorize_input(prefix_ipa, self.vectorizer, self.dimension)
            suffix_vec = vectorize_input(suffix_ipa, self.vectorizer, self.dimension)
            faiss.normalize_L2(prefix_vec)
            faiss.normalize_L2(suffix_vec)

            # Search FAISS for top_k candidates for each segment.
            prefix_dists, prefix_indices = self.index.search(prefix_vec, top_k)
            suffix_dists, suffix_indices = self.index.search(suffix_vec, top_k)

            prefix_results = self.dataset.iloc[prefix_indices[0]][
                [
                    "token_ort",
                    "token_ipa",
                    "norm_freq",
                    "scaled_aoa",
                    "imageability_score",
                    "word_embedding",
                ]
            ].copy()
            suffix_results = self.dataset.iloc[suffix_indices[0]][
                [
                    "token_ort",
                    "token_ipa",
                    "norm_freq",
                    "scaled_aoa",
                    "imageability_score",
                    "word_embedding",
                ]
            ].copy()

            # Calculate semantic similarity for each segment.
            prefix_semantic = self.semantic_sim(embedding, prefix_results)
            suffix_semantic = self.semantic_sim(embedding, suffix_results)

            # Vectorized computations for candidate pair scores.
            prefix_dists_arr = prefix_dists[0]  # shape: (top_k,)
            suffix_dists_arr = suffix_dists[0]  # shape: (top_k,)
            avg_distance = (prefix_dists_arr[:, None] + suffix_dists_arr[None, :]) / 2.0

            freq = (
                prefix_results["norm_freq"].values[:, None]
                + suffix_results["norm_freq"].values[None, :]
            ) / 2
            aoa = (
                prefix_results["scaled_aoa"].values[:, None]
                + suffix_results["scaled_aoa"].values[None, :]
            ) / 2
            imageability = (
                prefix_results["imageability_score"].values[:, None]
                + suffix_results["imageability_score"].values[None, :]
            ) / 2

            semantic_similarity_1d = np.array(
                [(a + b) / 2 for a, b in zip(prefix_semantic, suffix_semantic)]
            )
            semantic_similarity = np.repeat(
                semantic_similarity_1d[:, None], top_k, axis=1
            )
            score_matrix = (
                avg_distance + freq + aoa + imageability + semantic_similarity
            ) * (1 - penalty)

            # Combine token strings for candidates.
            prefix_words = np.array(
                prefix_results["token_ort"].astype(str).values, dtype="<U50"
            )
            suffix_words = np.array(
                suffix_results["token_ort"].astype(str).values, dtype="<U50"
            )
            combined_words = np.char.add(
                np.char.add(prefix_words[:, None], "+"), suffix_words[None, :]
            )

            prefix_ipas = np.array(
                prefix_results["token_ipa"].astype(str).values, dtype="<U50"
            )
            suffix_ipas = np.array(
                suffix_results["token_ipa"].astype(str).values, dtype="<U50"
            )
            combined_ipas = np.char.add(
                np.char.add(prefix_ipas[:, None], "+"), suffix_ipas[None, :]
            )

            candidate_tuples = list(
                zip(
                    combined_words.flatten(),
                    combined_ipas.flatten(),
                    avg_distance.flatten(),
                    freq.flatten(),
                    aoa.flatten(),
                    imageability.flatten(),
                    semantic_similarity.flatten(),
                    score_matrix.flatten(),
                )
            )
            candidates.extend(candidate_tuples)

        return pd.DataFrame(
            candidates,
            columns=[
                "token_ort",
                "token_ipa",
                "distance",
                "norm_freq",
                "scaled_aoa",
                "imageability_score",
                "semantic_similarity",
                "score",
            ],
        )

    def _combine_results(
        self, full_results: pd.DataFrame, split_results: pd.DataFrame, top_n: int
    ) -> pd.DataFrame:
        """Combine full–word and split candidate results, remove duplicates, and sort by score."""
        combined = pd.concat([full_results, split_results], ignore_index=True)
        combined = combined.drop_duplicates(subset=["token_ort", "token_ipa"])
        combined = combined.sort_values(by="score", ascending=False)
        return combined.head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    word_input = "ratatouille"
    language_code = "eng-us"
    top_n = 25

    # Load the G2P model
    from fluentai.services.mnemonic.phonetic.grapheme2phoneme import Grapheme2Phoneme

    phon_sim = Phonetic_Similarity(Grapheme2Phoneme())

    result = phon_sim.top_phonetic(word_input, language_code, top_n)
    print(result)
