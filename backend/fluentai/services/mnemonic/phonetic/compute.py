import asyncio

import faiss
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, util

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.services.mnemonic.orthographic.compute import (
    compute_damerau_levenshtein_similarity,
)
from fluentai.services.mnemonic.phonetic.grapheme2phoneme import Grapheme2Phoneme
from fluentai.services.mnemonic.phonetic.ipa2vec import panphon_vec, soundvec
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
        self.g2p_model = Grapheme2Phoneme()

        method = config.get("PHONETIC_SIM").get("EMBEDDINGS").get("METHOD")

        # Default to panphon if method is not specified
        self.vectorizer = panphon_vec

        if method == "clts":
            self.vectorizer = soundvec

        # Todo: move to config
        file = "en_US_filtered"
        file = f"{file}_{method}_mnemonics.parquet"

        self.dataset = pd.read_parquet(
            hf_hub_download(
                repo_id="StephanAkkerman/mnemonics",
                filename=file,
                cache_dir="datasets",
                repo_type="dataset",
            )
        )

        # We don't really care about the matrix later on
        self.dataset_columns = [
            "token_ort",
            "token_ipa",
            "freq",
            "aoa",
            "imageability",
            "word_embedding",
        ]
        self.final_columns = [
            "token_ort",
            "token_ipa",
            "distance",
            "freq",
            "aoa",
            "imageability",
            "semantic",
            "orthographic",
            "score",
        ]

        # Get the matrix from the dataset
        dataset_matrix = np.array(self.dataset["matrix"].tolist())
        self.dimension = dataset_matrix.shape[1]

        # IndexFlatIP = Cosine similarity / Inner product
        self.index = faiss.IndexFlatIP(self.dimension)
        # faiss.IndexFlatL2 = Euclidean distance
        # faiss.IndexIVFFlat for faster query times
        self.index.add(dataset_matrix)

        model_name = config.get("SEMANTIC_SIM").get("MODEL").lower()
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, cache_folder="models"
        )

        # Load the weights
        weights = config.get("WEIGHTS")
        self.distance_weights = weights.get("PHONETIC")
        self.imageability_weights = weights.get("IMAGEABILITY")
        self.semantic_weights = weights.get("SEMANTIC")
        self.orthographic_weights = weights.get("ORTHOGRAPHIC")
        self.freq_weights = weights.get("FREQUENCY")
        self.aoa_weights = weights.get("AOA")

        self.penalty = config.get("PENALTY")

    def semantic_sim(self, embedding, single_results: pd.DataFrame) -> list:
        """Calculate the semantic similarity between the input word and the corpus embeddings.

        Parameters
        ----------
        embedding : tensor
            The embedding of the input word
        single_results : pd.DataFrame
            The results of the single word search

        Returns
        -------
        list
            The cosine similarity scores between the input word and the corpus embeddings
        """
        # Shape should be [x, embedding_dim]
        logger.debug("Generating semantic similarity...")

        # Stack the numpy arrays into one array and convert it to a tensor
        corpus_embeddings = torch.tensor(
            np.vstack(single_results["word_embedding"].tolist())
        )

        # Move to same device
        corpus_embeddings = corpus_embeddings.to(embedding.device)

        # Compute cosine similarity between the query and all corpus embeddings.
        cos_scores = util.cos_sim(embedding, corpus_embeddings)

        return cos_scores.squeeze(0).cpu().tolist()

    def orthographic_sim(self, transliterated_word: str, words: list) -> list:
        """Calculate the orthographic similarity between the input word and the corpus words."""
        # Orthographic similarity, use damerau levenshtein
        logger.info("Generating orthographic similarity...")
        return [
            compute_damerau_levenshtein_similarity(transliterated_word, word)
            for word in words
        ]

    def top_phonetic(
        self,
        input_word: str,
        language_code: str,
        top_n: int,
        min_seg_length: int = 3,
    ) -> tuple[pd.DataFrame, str]:
        """
        Find the top_n closest phonetically similar words to the input IPA.

        This function performs a full–word search and, if possible, splits the IPA transcription
        to generate additional candidate pairs using vectorized operations.
        """
        logger.debug(f"Finding top {top_n} phonetically similar words to {input_word}.")

        # Step 1: Translate and embed the input word.
        translated, transliterated = asyncio.run(
            translate_word(input_word, language_code)
        )
        embedding = self._get_embedding(translated)

        # Step 2: Convert the input word to IPA.
        ipa = word2ipa(input_word, language_code, self.g2p_model)

        # Step 3: Perform a full–word search.
        full_results = self._full_word_search(ipa, top_n, embedding, transliterated)

        # Step 4: Compute split–based candidates if the IPA is long enough.
        split_results = self._split_candidates_search(
            ipa, embedding, min_seg_length, top_n, transliterated
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

    def _full_word_search(
        self, ipa: str, top_n: int, embedding, transliterated: str
    ) -> pd.DataFrame:
        """Perform a full–word search using FAISS and compute the full search scores."""
        input_vector = vectorize_input(ipa, self.vectorizer, self.dimension)
        faiss.normalize_L2(input_vector)
        full_dists, full_indices = self.index.search(input_vector, top_n)

        results = self.dataset.iloc[full_indices[0]][self.dataset_columns].copy()
        results["distance"] = full_dists[0]
        results["semantic"] = self.semantic_sim(embedding, results)
        results["orthographic"] = self.orthographic_sim(
            transliterated, results["token_ort"].to_list()
        )
        results["score"] = (
            results["distance"] * self.distance_weights
            + results["freq"] * self.freq_weights
            + results["aoa"] * self.aoa_weights
            + results["imageability"] * self.imageability_weights
            + results["semantic"] * self.semantic_weights
            + results["orthographic"] * self.orthographic_weights
        )
        return results

    def _split_candidates_search(
        self, ipa: str, embedding, min_seg_length: int, top_k: int, transliterated: str
    ) -> pd.DataFrame:
        """Compute candidate pairs based on splitting the IPA transcription."""
        if len(ipa) < 2 * min_seg_length:
            return pd.DataFrame(columns=self.final_columns)

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
                self.dataset_columns
            ].copy()
            suffix_results = self.dataset.iloc[suffix_indices[0]][
                self.dataset_columns
            ].copy()

            # Calculate semantic similarity for each segment.
            prefix_semantic = self.semantic_sim(embedding, prefix_results)
            suffix_semantic = self.semantic_sim(embedding, suffix_results)

            # Vectorized computations for candidate pair scores.
            prefix_dists_arr = prefix_dists[0]  # shape: (top_k,)
            suffix_dists_arr = suffix_dists[0]  # shape: (top_k,)
            avg_distance = (prefix_dists_arr[:, None] + suffix_dists_arr[None, :]) / 2.0

            freq = (
                prefix_results["freq"].values[:, None]
                + suffix_results["freq"].values[None, :]
            ) / 2
            aoa = (
                prefix_results["aoa"].values[:, None]
                + suffix_results["aoa"].values[None, :]
            ) / 2
            imageability = (
                prefix_results["imageability"].values[:, None]
                + suffix_results["imageability"].values[None, :]
            ) / 2

            semantic_similarity_1d = np.array(
                [(a + b) / 2 for a, b in zip(prefix_semantic, suffix_semantic)]
            )
            semantic_similarity = np.repeat(
                semantic_similarity_1d[:, None], top_k, axis=1
            )

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

            # combined_words is a 2D array of shape (top_k, top_k) containing strings like "prefix+suffix"
            combined_words_flat = combined_words.flatten()

            # Compute the similarity for each candidate pair using a list comprehension.
            orthographic_sim_flat = np.array(
                [
                    compute_damerau_levenshtein_similarity(transliterated, word)
                    for word in combined_words_flat
                ]
            )

            # Reshape the flat array back to the 2D shape.
            orthographic_similarity = orthographic_sim_flat.reshape(
                combined_words.shape
            )

            score_matrix = (
                avg_distance * self.distance_weights
                + freq * self.freq_weights
                + aoa * self.aoa_weights
                + imageability * self.imageability_weights
                + semantic_similarity * self.semantic_weights
                + orthographic_similarity * self.orthographic_weights
            ) * (1 - self.penalty)

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
                    orthographic_similarity.flatten(),
                    score_matrix.flatten(),
                )
            )
            candidates.extend(candidate_tuples)
            # TODO: dont recalculate when it has been done in a previous loop

        return pd.DataFrame(
            candidates,
            columns=self.final_columns,
        )

    def _combine_results(
        self, full_results: pd.DataFrame, split_results: pd.DataFrame, top_n: int
    ) -> pd.DataFrame:
        """Combine full–word and split candidate results, remove duplicates, and sort by score."""
        combined = pd.concat([full_results, split_results], ignore_index=True)
        combined = combined.drop_duplicates(subset=["token_ort", "token_ipa"])
        combined = combined.sort_values(by="score", ascending=False)
        # Drop word_embedding column
        combined = combined.drop(columns=["word_embedding"])
        return combined.reset_index(drop=True)

    def test(self, input_word: str, language_code: str):
        import numpy as np

        ipa = word2ipa(input_word, language_code, self.g2p_model)

        # For loop

        input_vector = vectorize_input(ipa, self.vectorizer, self.dimension)
        faiss.normalize_L2(input_vector)
        # Add the input vector to the index
        self.index.add(input_vector)

        search_word = "tattoo"

        # Search for the word "rat" in the dataset
        id1 = int(self.dataset[self.dataset["token_ort"] == search_word].index[0])
        id2 = len(self.dataset)  # This will be the input word

        # Replace id1 and id2 with the indices of your vectors.
        vec1 = self.index.reconstruct(id1)
        vec2 = self.index.reconstruct(id2)

        word1 = self.dataset.iloc[id1]
        # word2 = self.dataset.iloc[id2]
        print("Word 1:", word1["token_ort"], word1["token_ipa"])
        # print("Word 2:", word2["token_ort"], word2["token_ipa"])

        # Compute the Euclidean (L2) distance
        distance = np.linalg.norm(vec1 - vec2)
        print("Euclidean distance:", distance)

        # Squared Euclidean distance, matching what IndexFlatL2 returns
        squared_distance = np.sum((vec1 - vec2) ** 2)
        print("Squared L2 distance:", squared_distance)

        inner_product = np.dot(vec1, vec2)
        print("Inner product:", inner_product)


if __name__ == "__main__":
    # Example usage
    word_input = "ratatouille"
    language_code = "eng-us"
    top_n = 250

    # Load the G2P model
    phon_sim = Phonetic_Similarity()
    # phon_sim.test(word_input, language_code)

    result = phon_sim.top_phonetic(word_input, language_code, top_n)
    print(result)

    # Print where token_ort == "rat+tattoo"
    print(result[0][result[0]["token_ort"] == "rat+tattoo"])
