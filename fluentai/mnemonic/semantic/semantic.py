from gensim.models.fasttext import FastTextKeyedVectors
from sentence_transformers import SentenceTransformer, util

from fluentai.constants.config import config
from fluentai.utils.logger import logger

# Initialize global variables for models to ensure they are loaded only once
FASTTEXT_MODEL = None
MINILM_MODEL = None


class SemanticSimilarity:
    def __init__(self):
        from fluentai.utils.fasttext import fasttext_model

        self.model = fasttext_model

    def compute_similarity(self, word1: str, word2: str) -> float:
        """
        Computes the semantic similarity between two words using the FastText model.

        Parameters
        ----------
            word1 (str): The first word.
            word2 (str): The second word.

        Returns
        -------
            float: Cosine similarity score between -1 and 1.

        Raises
        ------
            ValueError: If either word is not in the FastText vocabulary.
        """
        if word1 not in self.model.key_to_index or word2 not in self.model.key_to_index:
            return 0
        similarity = self.model.similarity(word1, word2)
        return similarity


def load_fasttext_model() -> FastTextKeyedVectors:
    """
    Loads the FastText model using gensim's downloader.

    Returns
    -------
        gensim.models.keyedvectors.KeyedVectors: The loaded FastText model.
    """
    global FASTTEXT_MODEL
    from fluentai.utils.fasttext import fasttext_model

    FASTTEXT_MODEL = fasttext_model
    return FASTTEXT_MODEL


# Turn this into a class!
def load_sentence_transformer(
    model: str = "intfloat/multilingual-e5-small",
) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model and precomputes embeddings for the word list.

    Returns
    -------
        Tuple containing the SentenceTransformer model, list of words, and their embeddings.
    """
    global MINILM_MODEL
    if MINILM_MODEL is None:
        logger.info(f"Loading {model} model. This may take a while...")
        MINILM_MODEL = SentenceTransformer(model, cache_folder="models")
        logger.info(f"{model} model loaded successfully.")

    return MINILM_MODEL


def compute_fasttext_similarity(word1: str, word2: str) -> float:
    """
    Computes the semantic similarity between two words using the FastText model.

    Parameters
    ----------
        word1 (str): The first word.
        word2 (str): The second word.

    Returns
    -------
        float: Cosine similarity score between -1 and 1.

    Raises
    ------
        ValueError: If either word is not in the FastText vocabulary.
    """
    model = load_fasttext_model()
    if word1 not in model.key_to_index:
        raise ValueError(f"The word '{word1}' is not in the FastText vocabulary.")
    if word2 not in model.key_to_index:
        raise ValueError(f"The word '{word2}' is not in the FastText vocabulary.")
    similarity = model.similarity(word1, word2)
    return similarity


def compute_transformer_similarity(
    word1: str, word2: str, model: SentenceTransformer
) -> float:
    """
    Computes the semantic similarity between two words using the MiniLM model.

    Parameters
    ----------
        word1 (str): The first word.
        word2 (str): The second word.
        model (SentenceTransformer): The MiniLM model.

    Returns
    -------
        float: Cosine similarity score between 0 and 1.

    Raises
    ------
        ValueError: If either word is not in the MiniLM word list.
    """
    # word1 = word1.lower()
    # word2 = word2.lower()
    # emb1 = model.encode([word1])
    # emb2 = model.encode([word2])
    # similarity = cosine_similarity(emb1, emb2)[0][0]

    # Compute embedding for both lists
    embedding_1 = model.encode(word1, convert_to_tensor=True)
    embedding_2 = model.encode(word2, convert_to_tensor=True)

    sim = util.pytorch_cos_sim(embedding_1, embedding_2)

    # Get sim as float
    similarity = sim.item()

    return similarity


def compute_similarity(word1: str, word2: str, model: str) -> float:
    """
    Computes the semantic similarity between two words using the specified method.

    Parameters
    ----------
        word1 (str): The first word.
        word2 (str): The second word.
        model (str): The embedding model to use.

    Returns
    -------
        float: Similarity score. The scale depends on the method:
            - 'fasttext': -1 to 1
            - 'minilm': 0 to 1

    Raises
    ------
        ValueError: If an unsupported method is provided or words are not in the vocabulary.
    """
    model = model.lower()
    if model == "fasttext":
        return compute_fasttext_similarity(word1, word2)

    # Get the model from huggingface
    model = load_sentence_transformer(model)

    return compute_transformer_similarity(word1, word2, model)


def example():
    """
    Runs predefined examples to compare semantic similarity across different methods.
    """
    words = ["train", "brain", "king", "queen"]
    models = config.get("SEMANTIC_SIM").get("EVAL").get("MODELS")

    # Create tuples of all possible combinations of words and models
    examples = [(w1, w2, m) for w1 in words for w2 in words for m in models]

    for word1, word2, method in examples:
        try:
            similarity = compute_similarity(word1, word2, method)
            logger.info(
                f"Similarity between '{word1}' and '{word2}' using '{method}': {similarity:.4f}"
            )
        except ValueError as e:
            logger.info(
                f"Error computing similarity between '{word1}' and '{word2}' using '{method}': {e}"
            )


# Example usage (for testing purposes only; remove or comment out in production)
if __name__ == "__main__":
    example()
    # print(compute_minilm_similarity("train", "brain"))
