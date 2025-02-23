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
        logger.debug(f"Finding top {top_n} phonetically similar words to {input_word}.")

        # The penalty is used to avoid overly long splits.
        # TODO: add penalty to config
        penalty = 0

        # Translate word to English
        translated, _ = asyncio.run(translate_word(input_word, language_code))

        # Convert the translated word to a embedding
        embedding = self.model.encode(
            translated, convert_to_tensor=True, normalize_embeddings=True
        )

        # Ensure the query embedding is 2D (shape: [1, embedding_dim])
        embedding = embedding.unsqueeze(0) if len(embedding.shape) == 1 else embedding

        # Convert the input word to IPA.
        ipa = word2ipa(input_word, language_code, self.g2p_model)

        # Full–word search.
        input_vector = vectorize_input(ipa, self.vectorizer, self.dimension)
        faiss.normalize_L2(input_vector)
        full_dists, full_indices = self.index.search(input_vector, top_n)
        single_results = self.dataset.iloc[full_indices[0]][
            [
                "token_ort",
                "token_ipa",
                "norm_freq",
                "scaled_aoa",
                "imageability_score",
                "word_embedding",
            ]
        ].copy()
        single_results["distance"] = full_dists[0]

        # Assign the similarity scores to the dataframe (converting to list if necessary)
        single_results["semantic_similarity"] = self.semantic_sim(
            embedding, single_results
        )

        # Combine all scores into a single score
        single_results["score"] = (
            single_results["distance"]
            + single_results["norm_freq"]
            + single_results["scaled_aoa"]
            + single_results["imageability_score"]
            + single_results["semantic_similarity"]
        )

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

                # Calculate the semantic similarity for each candidate pair.
                prefix_semantic = self.semantic_sim(embedding, prefix_results)
                suffix_semantic = self.semantic_sim(embedding, suffix_results)

                # Assume these are your top_k arrays extracted from FAISS (shape: (1, top_k))
                prefix_dists_arr = prefix_dists[0]  # shape: (top_k,)
                suffix_dists_arr = suffix_dists[0]  # shape: (top_k,)

                # Vectorized average distance: result shape (top_k, top_k)
                avg_distance = (
                    prefix_dists_arr[:, None] + suffix_dists_arr[None, :]
                ) / 2.0

                # For other scores, first extract arrays from the DataFrames:
                freq_prefix = prefix_results["norm_freq"].values  # shape: (top_k,)
                freq_suffix = suffix_results["norm_freq"].values  # shape: (top_k,)
                freq = (freq_prefix[:, None] + freq_suffix[None, :]) / 2

                aoa_prefix = prefix_results["scaled_aoa"].values
                aoa_suffix = suffix_results["scaled_aoa"].values
                aoa = (aoa_prefix[:, None] + aoa_suffix[None, :]) / 2

                img_prefix = prefix_results["imageability_score"].values
                img_suffix = suffix_results["imageability_score"].values
                imageability = (img_prefix[:, None] + img_suffix[None, :]) / 2

                # For semantic similarity, you already computed:
                # prefix_semantic = self.semantic_sim(embedding, prefix_results)  (1D array, length top_k)
                # suffix_semantic = self.semantic_sim(embedding, suffix_results)  (1D array, length top_k)
                # And then you averaged them elementwise:
                semantic_similarity_1d = np.array(
                    [(a + b) / 2 for a, b in zip(prefix_semantic, suffix_semantic)]
                )
                # In your original loop, you only used semantic_similarity[j] (i.e. by prefix candidate)
                # so we broadcast this over the second axis:
                semantic_similarity = np.repeat(
                    semantic_similarity_1d[:, None], top_k, axis=1
                )

                # Combine all scores into one matrix (same shape)
                score_matrix = (
                    avg_distance + freq + aoa + imageability + semantic_similarity
                ) * (1 - penalty)

                # If you also need the combined words and IPA from the candidates:
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

                # If a flat list of candidate tuples is desired, you can flatten the matrices:
                split_candidates = list(
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
                print(split_candidates)

        # Convert split candidates to DataFrame.
        if split_candidates:
            split_results = pd.DataFrame(
                split_candidates,
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
        else:
            split_results = pd.DataFrame(
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

        # Combine full-word and split results.
        combined_results = pd.concat([single_results, split_results], ignore_index=True)
        # Drop duplicates based on token_ort and token_ipa.
        combined_results = combined_results.drop_duplicates(
            subset=["token_ort", "token_ipa"]
        )
        combined_results = combined_results.sort_values(by="score", ascending=False)
        final_results = combined_results.head(top_n).reset_index(drop=True)

        return final_results, ipa


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
