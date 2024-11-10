import gc

import torch

from fluentai.constants.config import weights_percentages
from fluentai.constants.languages import G2P_LANGCODES
from fluentai.mnemonic.imageability.predictions import ImageabilityPredictor
from fluentai.mnemonic.orthographic.orthographic import (
    compute_damerau_levenshtein_similarity,
)
from fluentai.mnemonic.phonetic.g2p import G2P
from fluentai.mnemonic.phonetic.phonetic import top_phonetic
from fluentai.mnemonic.semantic.semantic import SemanticSimilarity
from fluentai.mnemonic.semantic.translator import translate_word
from fluentai.utils.logger import logger


def generate_mnemonic(word: str, language_code):
    """
    Generate a mnemonic for the input word using the phonetic representation.

    Parameters
    ----------
    - word: String, foreign word to generate mnemonic for
    - language_code: String, language code of the input word
    """
    # Initialize the models
    g2p_model = G2P()
    imageability_predictor = ImageabilityPredictor()
    semantic_sim = SemanticSimilarity()

    # Test if the language code is valid
    if language_code not in G2P_LANGCODES:
        logger.error(f"Invalid language code: {language_code}")
        return

    # Get the top x phonetically similar words
    logger.info(f"Generating top phonetic similarity for {word} in {language_code}...")
    top = top_phonetic(word, language_code, 15, g2p_model)

    # Generate their word imageability for all token_ort in top
    logger.info("Generating imageability...")
    top["imageability"] = imageability_predictor.get_column_imageability(
        top, "token_ort"
    )

    translated_word, transliterated_word = translate_word(word, language_code)

    # Semantic similarity, use fasttext
    logger.info("Generating semantic similarity...")
    top["semantic_similarity"] = top.apply(
        lambda row: semantic_sim.compute_similarity(translated_word, row["token_ort"]),
        axis=1,
    )

    # Orthographic similarity, use damerau levenshtein
    logger.info("Generating orthographic similarity...")
    top["orthographic_similarity"] = top.apply(
        lambda row: compute_damerau_levenshtein_similarity(
            transliterated_word, row["token_ort"]
        ),
        axis=1,
    )

    # Calculate weighted score using dynamic weights from config
    top["score"] = (
        top["distance"] * weights_percentages["PHONETIC"]
        + top["imageability"] * weights_percentages["IMAGEABILITY"]
        + top["semantic_similarity"] * weights_percentages["SEMANTIC"]
        + top["orthographic_similarity"] * weights_percentages["ORTHOGRAPHIC"]
    )

    # Sort by score
    top = top.sort_values(by="score", ascending=False)

    # Clean up all the models
    del g2p_model
    del imageability_predictor
    del semantic_sim

    gc.collect()

    torch.cuda.empty_cache()

    return top


if __name__ == "__main__":
    # generate_mnemonic("kucing", "ind")

    print(generate_mnemonic("猫", "zho-s"))
