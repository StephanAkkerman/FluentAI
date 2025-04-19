import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, T5ForConditionalGeneration

from mnemorai.constants.config import config
from mnemorai.logger import logger


class Grapheme2Phoneme:
    def __init__(self):
        # https://github.com/lingjzhu/CharsiuG2P
        logger.debug("Loading G2P model")
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.get("G2P").get("MODEL"), device_map="cpu", cache_dir="models"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("G2P").get("TOKENIZER"), cache_dir="models"
        )
        self.offload = config.get("G2P").get("OFFLOAD")
        logger.debug("G2P model loaded")

    def g2p(self, words: list[str]) -> str:
        """
        Use the G2P model to convert a list of words to their phonetic transcription.

        Parameters
        ----------
        words : list[str]
            The list of words to convert to phonetic transcription.

        Returns
        -------
        str
            The phonetic transcription of the words.
        """
        out = self.tokenizer(
            words, padding=True, add_special_tokens=False, return_tensors="pt"
        )

        preds = self.model.generate(
            **out, num_beams=1, max_length=50
        )  # We do not find beam search helpful. Greedy decoding is enough.
        ipa = self.tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        ipa = ipa[0]
        # Remove the leading ˈ
        ipa = ipa.removeprefix("ˈ")
        return ipa

    def word2ipa(
        self,
        word: str,
        language_code: str,
    ) -> str:
        """
        Get the IPA representation of a word.

        Use the IPA dataset if available, otherwise use the G2P model.

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

        # Use the g2p models
        return self.g2p([f"<{language_code}>:{word}"])


def example():
    """
    Example usage of the G2P module. It prints the phonetic transcription of the words in Indonesian, English, and Dutch.
    """
    g2p = Grapheme2Phoneme()

    # https://en.wiktionary.org/wiki/kucing#Indonesian
    # IPA(key): /ˈkut͡ʃɪŋ/
    indonesian_word = g2p.g2p(["<ind>: Kucing"])
    # https://en.wiktionary.org/wiki/hello#English (UK)
    # IPA: /həˈləʊ/, /hɛˈləʊ/
    # https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/en_US.txt
    # /əˌbɹiviˈeɪʃən/ ->  əˌbɹiviˈeɪʃən
    # G2P removes / at the beginning but not leading ˈ
    # /ˈɡɹəmbəɫ/ -> ˈɡɹəmbəɫ
    # Wiktionary: /ˈɡɹʌmbl̩/
    english_word = g2p.g2p(["<eng-us>: grumble"])
    dutch_word = g2p.g2p(["<dut>: Koekje"])

    print(
        indonesian_word,
        english_word,
        dutch_word,
    )


if __name__ == "__main__":
    example()
