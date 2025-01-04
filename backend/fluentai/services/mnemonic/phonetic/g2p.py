from transformers import AutoTokenizer, T5ForConditionalGeneration

from fluentai.constants.config import config
from fluentai.logger import logger


class G2P:
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


def example():
    """
    Example usage of the G2P module. It prints the phonetic transcription of the words in Indonesian, English, and Dutch.
    """
    g2p = G2P()

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
