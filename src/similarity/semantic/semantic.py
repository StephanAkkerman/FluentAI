import logging
import os
import pickle
from typing import Tuple

import gensim
import gensim.downloader as api
import nltk
import spacy
from gensim.models import KeyedVectors
from nltk.corpus import words
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from imageability.imageability import download_and_save_model

# Ensure the 'words' corpus is downloaded
nltk.download("words", quiet=True)
WORD_LIST = words.words()

# Initialize global variables for models to ensure they are loaded only once
GLOVE_MODEL = None
FASTTEXT_MODEL = None
MINILM_MODEL = None
MINILM_EMBEDDINGS = None
MINILM_WORDS = None
SPACY_MODEL = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SemanticSimilarity:
    def __init__(
        self,
        embedding_model_path="models/fasttext.model",
    ):
        # Check if the embedding model exists
        if not os.path.exists(embedding_model_path):
            download_and_save_model()
        self.model = KeyedVectors.load(embedding_model_path)

    def compute_similarity(self, word1: str, word2: str) -> float:
        """
        Computes the semantic similarity between two words using the FastText model.

        Parameters:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            float: Cosine similarity score between -1 and 1.

        Raises:
            ValueError: If either word is not in the FastText vocabulary.
        """
        if word1 not in self.model.key_to_index or word2 not in self.model.key_to_index:
            return 0
        similarity = self.model.similarity(word1, word2)
        return similarity


def load_glove_model() -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads the GloVe model using gensim's downloader.

    Returns:
        gensim.models.keyedvectors.KeyedVectors: The loaded GloVe model.
    """
    global GLOVE_MODEL
    if GLOVE_MODEL is None:
        print("Loading GloVe model. This may take a while...")
        GLOVE_MODEL = api.load("glove-wiki-gigaword-100")  # You can choose other models
        print("GloVe model loaded successfully.")
    return GLOVE_MODEL


def load_fasttext_model() -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads the FastText model using gensim's downloader.

    Returns:
        gensim.models.keyedvectors.KeyedVectors: The loaded FastText model.
    """
    global FASTTEXT_MODEL

    if not os.path.exists("models/fasttext.model"):
        print("Loading FastText model. This may take a while...")
        download_and_save_model()
    FASTTEXT_MODEL = KeyedVectors.load("models/fasttext.model")
    print("FastText model loaded successfully.")
    return FASTTEXT_MODEL


def load_minilm_model() -> Tuple[SentenceTransformer, list, list]:
    """
    Loads the MiniLM SentenceTransformer model and precomputes embeddings for the word list.

    Returns:
        Tuple containing the SentenceTransformer model, list of words, and their embeddings.
    """
    global MINILM_MODEL, MINILM_EMBEDDINGS, MINILM_WORDS
    if MINILM_MODEL is None:
        print("Loading MiniLM model. This may take a while...")
        MINILM_MODEL = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="models")
        print("MiniLM model loaded successfully.")

    if MINILM_EMBEDDINGS is None or MINILM_WORDS is None:
        embeddings_path = "minilm_word_embeddings.pkl"
        if not os.path.exists(embeddings_path):
            print(
                "Creating MiniLM embeddings for the word list. This may take a while..."
            )
            MINILM_WORDS = list(set([word.lower() for word in WORD_LIST]))
            MINILM_EMBEDDINGS = MINILM_MODEL.encode(
                MINILM_WORDS, batch_size=64, show_progress_bar=True
            )
            with open(embeddings_path, "wb") as f:
                pickle.dump({"words": MINILM_WORDS, "embeddings": MINILM_EMBEDDINGS}, f)
            print("MiniLM embeddings created and saved.")
        else:
            print("Loading precomputed MiniLM embeddings...")
            with open(embeddings_path, "rb") as f:
                data = pickle.load(f)
                MINILM_WORDS = data["words"]
                MINILM_EMBEDDINGS = data["embeddings"]
            print("MiniLM embeddings loaded.")

    return MINILM_MODEL, MINILM_WORDS, MINILM_EMBEDDINGS


def load_spacy_model():
    """
    Loads the spaCy model with vectors.

    Returns:
        spacy.lang.en.English: The loaded spaCy model.
    """
    global SPACY_MODEL
    if SPACY_MODEL is None:
        print("Loading spaCy model. This may take a while...")
        try:
            SPACY_MODEL = spacy.load("en_core_web_md")  # or "en_core_web_lg"
        except OSError:
            print("spaCy model not found. Downloading 'en_core_web_md'...")
            from spacy.cli import download

            download("en_core_web_md")
            SPACY_MODEL = spacy.load("en_core_web_md")
        print("spaCy model loaded successfully.")
    return SPACY_MODEL


def compute_glove_similarity(word1: str, word2: str) -> float:
    """
    Computes the semantic similarity between two words using the GloVe model.

    Parameters:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        float: Cosine similarity score between -1 and 1.

    Raises:
        ValueError: If either word is not in the GloVe vocabulary.
    """
    model = load_glove_model()
    if word1 not in model.key_to_index:
        raise ValueError(f"The word '{word1}' is not in the GloVe vocabulary.")
    if word2 not in model.key_to_index:
        raise ValueError(f"The word '{word2}' is not in the GloVe vocabulary.")
    similarity = model.similarity(word1, word2)
    return similarity


def compute_fasttext_similarity(word1: str, word2: str) -> float:
    """
    Computes the semantic similarity between two words using the FastText model.

    Parameters:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        float: Cosine similarity score between -1 and 1.

    Raises:
        ValueError: If either word is not in the FastText vocabulary.
    """
    model = load_fasttext_model()
    if word1 not in model.key_to_index:
        raise ValueError(f"The word '{word1}' is not in the FastText vocabulary.")
    if word2 not in model.key_to_index:
        raise ValueError(f"The word '{word2}' is not in the FastText vocabulary.")
    similarity = model.similarity(word1, word2)
    return similarity


def compute_minilm_similarity(word1: str, word2: str) -> float:
    """
    Computes the semantic similarity between two words using the MiniLM model.

    Parameters:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        float: Cosine similarity score between 0 and 1.

    Raises:
        ValueError: If either word is not in the MiniLM word list.
    """
    model, word_list, embeddings = load_minilm_model()
    word1 = word1.lower()
    word2 = word2.lower()
    if word1 not in word_list:
        raise ValueError(f"The word '{word1}' is not in the MiniLM word list.")
    if word2 not in word_list:
        raise ValueError(f"The word '{word2}' is not in the MiniLM word list.")

    idx1 = word_list.index(word1)
    idx2 = word_list.index(word2)
    emb1 = embeddings[idx1].reshape(1, -1)
    emb2 = embeddings[idx2].reshape(1, -1)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity


def compute_spacy_similarity(word1: str, word2: str) -> float:
    """
    Computes the semantic similarity between two words using the spaCy model.

    Parameters:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        float: Similarity score between 0 and 1.

    Raises:
        ValueError: If either word does not have a vector representation in spaCy.
    """
    nlp = load_spacy_model()
    token1 = nlp(word1)
    token2 = nlp(word2)
    if not token1.has_vector:
        raise ValueError(
            f"The word '{word1}' does not have a vector representation in spaCy."
        )
    if not token2.has_vector:
        raise ValueError(
            f"The word '{word2}' does not have a vector representation in spaCy."
        )
    similarity = token1.similarity(token2)
    return similarity


def compute_similarity(word1: str, word2: str, method: str) -> float:
    """
    Computes the semantic similarity between two words using the specified method.

    Parameters:
        word1 (str): The first word.
        word2 (str): The second word.
        method (str): The similarity method to use. Options:
            - 'glove'
            - 'fasttext'
            - 'minilm'
            - 'spacy'

    Returns:
        float: Similarity score. The scale depends on the method:
            - 'glove', 'fasttext', 'spacy': -1 to 1
            - 'minilm': 0 to 1

    Raises:
        ValueError: If an unsupported method is provided or words are not in the vocabulary.
    """
    method = method.lower()
    if method == "glove":
        return compute_glove_similarity(word1, word2)
    elif method == "fasttext":
        return compute_fasttext_similarity(word1, word2)
    elif method == "minilm":
        return compute_minilm_similarity(word1, word2)
    elif method == "spacy":
        return compute_spacy_similarity(word1, word2)
    else:
        raise ValueError(
            f"Unsupported similarity method: '{method}'. "
            f"Choose from 'glove', 'fasttext', 'minilm', 'spacy'."
        )


def example():
    """
    Runs predefined examples to compare semantic similarity across different methods.
    """
    examples = [
        ("train", "brain", "glove"),
        ("train", "brain", "fasttext"),
        ("train", "brain", "minilm"),
        ("train", "brain", "spacy"),
        ("king", "queen", "glove"),
        ("king", "queen", "fasttext"),
        ("king", "queen", "minilm"),
        ("king", "queen", "spacy"),
        ("happy", "joyful", "glove"),
        ("happy", "joyful", "fasttext"),
        ("happy", "joyful", "minilm"),
        ("happy", "joyful", "spacy"),
    ]

    for word1, word2, method in examples:
        try:
            similarity = compute_similarity(word1, word2, method)
            print(
                f"Similarity between '{word1}' and '{word2}' using '{method}': {similarity:.4f}"
            )
        except ValueError as e:
            print(
                f"Error computing similarity between '{word1}' and '{word2}' using '{method}': {e}"
            )


# Example usage (for testing purposes only; remove or comment out in production)
if __name__ == "__main__":
    example()
