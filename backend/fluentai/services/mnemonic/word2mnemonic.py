import asyncio

from fluentai.constants.config import config, weights_percentages
from fluentai.constants.languages import G2P_LANGUAGES
from fluentai.logger import logger
from fluentai.services.card_gen.mnemonic.imageability.imageability import (
    ImageabilityPredictor,
)
from fluentai.services.mnemonic.orthographic.compute import (
    compute_damerau_levenshtein_similarity,
)
from fluentai.services.mnemonic.phonetic.compute import top_phonetic, word2ipa
from fluentai.services.mnemonic.phonetic.grapheme2phoneme import (
    Grapheme2Phoneme,
)
from fluentai.services.mnemonic.semantic.compute import SemanticSimilarity
from fluentai.services.mnemonic.semantic.translator import translate_word


class Word2Mnemonic:
    def __init__(self):
        self.g2p_model = Grapheme2Phoneme()
        self.imageability_predictor = ImageabilityPredictor()
        self.semantic_sim = SemanticSimilarity()

    def generate_mnemonic(
        self,
        word: str,
        language_code: str,
        keyword: str = None,
        key_sentence: str = None,
    ):
        """
        Generate a mnemonic for the input word using the phonetic representation.

        Parameters
        ----------
        word : str
            Foreign word to generate mnemonic for.
        language_code : str
            Language code of the input word.
        keyword : str, optional
            User-provided keyword to use in the mnemonic.
        key_sentence : str, optional
            User-provided key sentence to use as the mnemonic.

        Returns
        -------
        tuple
            A tuple containing the top matches, translated word, transliterated word, and IPA.
        """
        logger.debug(
            "Generating mnemonic for %s and language: %s...", word, language_code
        )

        if language_code not in G2P_LANGUAGES:
            logger.error(f"Invalid language code: {language_code}")
            return

        translated_word, transliterated_word = asyncio.run(
            translate_word(word, language_code)
        )

        if keyword or key_sentence:
            # If keyword is provided, use it directly for scoring
            top = None

            # Convert the input word to IPA representation
            ipa = word2ipa(word, language_code, self.g2p_model)
        else:
            # Get the top x phonetically similar words
            logger.info(
                "Generating top phonetic similarity for %s in %s ...",
                word,
                language_code,
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
    print(w2m.generate_mnemonic("kat", "dut"))
    print(w2m.generate_mnemonic("house", "eng", keyword="হাউজ"))
    print(w2m.generate_mnemonic("猫", "zho-s"))
