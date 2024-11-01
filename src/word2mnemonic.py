import logging

from constants import G2P_LANGCODES
from imageability.imageability import ImageabilityPredictor
from similarity.orthographic.orthographic import compute_damerau_levenshtein_similarity
from similarity.phonetic.phonetic import top_phonetic
from similarity.semantic.semantic import SemanticSimilarity
from similarity.semantic.translator import translate_word

imageability_predictor = ImageabilityPredictor()
semantic_sim = SemanticSimilarity()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_mnemonic(word: str, language_code):
    """
    Generate a mnemonic for the input word using the phonetic representation.

    Parameters:
    - word: String, foreign word to generate mnemonic for
    - language_code: String, language code of the input word
    """
    # Test if the language code is valid
    if language_code not in G2P_LANGCODES:
        logging.error(f"Invalid language code: {language_code}")
        return

    # Get the top x phonetically similar words
    logging.info(f"Generating top phonetic similarity for {word} in {language_code}...")
    top = top_phonetic(word, language_code)

    # Generate their word imageability for all token_ort in top
    logging.info("Generating imageability...")
    top["imageability"] = imageability_predictor.get_column_imageability(
        top, "token_ort"
    )

    translated_word, transliterated_word = translate_word(word, language_code)

    # Semantic similarity, use fasttext
    logging.info("Generating semantic similarity...")
    top["semantic_similarity"] = top.apply(
        lambda row: semantic_sim.compute_similarity(translated_word, row["token_ort"]),
        axis=1,
    )

    # Orthographic similarity, use damerau levenshtein
    logging.info("Generating orthographic similarity...")
    top["orthographic_similarity"] = top.apply(
        lambda row: compute_damerau_levenshtein_similarity(
            transliterated_word, row["token_ort"]
        ),
        axis=1,
    )

    # Calculate the mnemonic score
    print(top)


if __name__ == "__main__":
    # generate_mnemonic("kucing", "ind")

    generate_mnemonic("猫", "zho-s")
