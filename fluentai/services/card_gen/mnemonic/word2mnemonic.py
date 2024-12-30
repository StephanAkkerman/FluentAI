from fluentai.services.card_gen.constants.config import config, weights_percentages
from fluentai.services.card_gen.constants.languages import G2P_LANGUAGES
from fluentai.services.card_gen.mnemonic.imageability.predictions import (
    ImageabilityPredictor,
)
from fluentai.services.card_gen.mnemonic.orthographic.orthographic import (
    compute_damerau_levenshtein_similarity,
)
from fluentai.services.card_gen.mnemonic.phonetic.g2p import G2P
from fluentai.services.card_gen.mnemonic.phonetic.phonetic import top_phonetic
from fluentai.services.card_gen.mnemonic.semantic.semantic import SemanticSimilarity
from fluentai.services.card_gen.mnemonic.semantic.translator import translate_word
from fluentai.services.card_gen.utils.logger import logger


class Word2Mnemonic:
    def __init__(self):
        self.g2p_model = G2P()
        self.imageability_predictor = ImageabilityPredictor()
        self.semantic_sim = SemanticSimilarity()

    def generate_mnemonic(self, word: str, language_code: str):
        """
        Generate a mnemonic for the input word using the phonetic representation.

        Parameters
        ----------
        - word: String, foreign word to generate mnemonic for
        - language_code: string language code of the input word
        """
        logger.debug(f"Generating mnemonic for {word} and lanuage: {language_code}...")

        # Test if the language code is valid
        if language_code not in G2P_LANGUAGES:
            logger.error(f"Invalid language code: {language_code}")
            return

        # Get the top x phonetically similar words
        logger.info(
            f"Generating top phonetic similarity for {word} in {language_code}..."
        )
        top, ipa = top_phonetic(
            word, language_code, config.get("WORD_LIMIT"), self.g2p_model
        )
        logger.debug(f"Top phonetic similarity: {top}")

        # Generate their word imageability for all token_ort in top
        logger.info("Generating imageability...")
        top["imageability"] = self.imageability_predictor.get_column_imageability(
            top, "token_ort"
        )

        translated_word, transliterated_word = translate_word(word, language_code)

        # Semantic similarity, use fasttext
        logger.info("Generating semantic similarity...")
        top["semantic_similarity"] = top.apply(
            lambda row: self.semantic_sim.compute_similarity(
                translated_word, row["token_ort"]
            ),
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

        return top, translated_word, transliterated_word, ipa


if __name__ == "__main__":
    w2m = Word2Mnemonic()
    # print(generate_mnemonic("kucing", "ind"))
    # print(generate_mnemonic("kat", "dut"))
    print(w2m.generate_mnemonic("猫", "zho-s"))
