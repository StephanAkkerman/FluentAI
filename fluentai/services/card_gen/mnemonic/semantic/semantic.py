from gensim.models.fasttext import FastTextKeyedVectors
from sentence_transformers import SentenceTransformer

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


class SemanticSimilarity:
    def __init__(self, model: str = config.get("SEMANTIC_SIM").get("MODEL")):
        self.model_name = model.lower()
        self.model = self.load_semantic_model()

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
        if self.model_name == "fasttext":
            return self.model.similarity(word1, word2)
        else:
            return self.compute_transformer_similarity(word1, word2)

    def compute_transformer_similarity(self, word1: str, word2: str) -> float:
        """
        Computes the semantic similarity between two words using a transformer model.

        Parameters
        ----------
            word1 (str): The first word.
            word2 (str): The second word.

        Returns
        -------
            float: Cosine similarity score between 0 and 1.

        Raises
        ------
            ValueError: If either word is not in the MiniLM word list.
        """
        # Compute embedding for both lists
        embedding_1 = self.model.encode(
            word1, convert_to_tensor=True, normalize_embeddings=True
        )
        embedding_2 = self.model.encode(
            word2, convert_to_tensor=True, normalize_embeddings=True
        )
        return self.model.similarity(embedding_1, embedding_2).item()

    def load_semantic_model(self) -> SentenceTransformer | FastTextKeyedVectors:
        """
        Load the specified semantic model.

        Returns
        -------
        _type_
            _description_
        """
        if self.model_name == "fasttext":
            from fluentai.services.card_gen.utils.fasttext import fasttext_model

            return fasttext_model

        # Get the model from huggingface
        return SentenceTransformer(
            self.model_name, trust_remote_code=True, cache_folder="models"
        )


def example():
    """
    Runs predefined examples to compare semantic similarity across different methods.
    """
    words = ["train", "brain", "king", "queen"]
    models = config.get("SEMANTIC_SIM").get("EVAL").get("MODELS")

    # Create the model objects
    semantic_models = [SemanticSimilarity(model) for model in models]

    # Create tuples of all possible combinations of words and models
    examples = [
        (w1, w2, m) for w1 in words for w2 in words for m in semantic_models if w1 != w2
    ]

    for word1, word2, semantic_model in examples:
        try:
            similarity = semantic_model.compute_similarity(word1, word2)
            logger.info(
                f"Similarity between '{word1}' and '{word2}' using '{semantic_model.model_name}': {similarity:.4f}"
            )
        except ValueError as e:
            logger.info(
                f"Error computing similarity between '{word1}' and '{word2}' using '{semantic_model.model_name}': {e}"
            )


if __name__ == "__main__":
    example()
