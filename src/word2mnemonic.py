import logging

from imageability.imageability import ImageabilityPredictor
from similarity.orthographic.orthographic import compute_damerau_levenshtein_similarity
from similarity.phonetic.phonetic import top_phonetic
from similarity.semantic.semantic import SemanticSimilarity

imageability_predictor = ImageabilityPredictor()
semantic_sim = SemanticSimilarity()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_mnemonic(word, language_code):
    """
    Generate a mnemonic for the input word using the phonetic representation.

    Parameters:
    - word: String, input word
    - language_code: String, language code of the input word
    """
    # Get the top x phonetically similar words
    logging.info(f"Generating top phonetic similarity for {word} in {language_code}...")
    top = top_phonetic(word, language_code)

    # Generate their word imageability for all token_ort in top
    logging.info("Generating imageability...")
    top["imageability"] = imageability_predictor.get_column_imageability(
        top, "token_ort"
    )

    # Orthographic similarity, use damerau levenshtein
    logging.info("Generating orthographic similarity...")
    top["orthographic_similarity"] = top.apply(
        lambda row: compute_damerau_levenshtein_similarity(word, row["token_ort"]),
        axis=1,
    )

    # Semantic similarity, use fasttext
    logging.info("Generating semantic similarity...")
    top["semantic_similarity"] = top.apply(
        lambda row: semantic_sim.compute_similarity(word, row["token_ort"]),
        axis=1,
    )

    # Calculate the mnemonic score
    print(top)


if __name__ == "__main__":
    generate_mnemonic("kucing", "ind")
