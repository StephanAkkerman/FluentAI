import asyncio

from mnemorai.constants.config import config
from mnemorai.constants.languages import G2P_LANGUAGES
from mnemorai.logger import logger
from mnemorai.services.mnemonic.phonetic.compute import Phonetic_Similarity
from mnemorai.services.mnemonic.semantic.translator import translate_word


class Word2Mnemonic:
    def __init__(self):
        self.phonetic_sim = Phonetic_Similarity()

    async def generate_mnemonic(
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
        logger.debug(f"Generating mnemonic for {word} and language: {language_code}...")

        if language_code not in G2P_LANGUAGES:
            logger.error(f"Invalid language code: {language_code}")
            return

        if keyword or key_sentence:
            # If keyword is provided, use it directly for scoring
            top = None

            # Convert the input word to IPA representation
            ipa = self.phonetic_sim.word2ipa(word=word, language_code=language_code)

            translated_word, transliterated_word = await translate_word(
                word, language_code
            )

        else:
            # Get the top x phonetically similar words
            (
                top,
                ipa,
                translated_word,
                transliterated_word,
            ) = await self.phonetic_sim.top_phonetic(
                word, language_code, config.get("MAX_CANDIDATES", 200)
            )

        return top, translated_word, transliterated_word, ipa


if __name__ == "__main__":
    w2m = Word2Mnemonic()
    print(asyncio.run(w2m.generate_mnemonic("kat", "dut")))
    print(asyncio.run(w2m.generate_mnemonic("house", "eng-us", keyword="হাউজ")))
    print(asyncio.run(w2m.generate_mnemonic("猫", "zho-s")))
