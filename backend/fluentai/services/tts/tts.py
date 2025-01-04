import os

import gtts
from gtts import gTTS

from fluentai.logger import logger
from fluentai.utils.lang_codes import map_language_code


class TTS:
    def __init__(self):
        self.languages = gtts.lang.tts_langs()
        # Create the local_data/tts directory if it doesn't exist
        os.makedirs("local_data/tts", exist_ok=True)

    def tts(self, text: str, lang: str = "en") -> str:
        """Generate a TTS audio file from the input text.

        The generated audio file will be saved in the local_data/tts directory.
        This filename can be removed after adding it to a card.
        """
        lang = map_language_code(lang)
        save_path = f"local_data/tts/{lang}.mp3"

        # Check if the language code is supported
        if lang not in self.languages:
            logger.error(
                f"Language code '{lang}' is not supported. TTS could not be generated."
            )
            return

        # TODO implement accents: https://gtts.readthedocs.io/en/latest/module.html#localized-accents
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(save_path)
        logger.info(save_path)
        return save_path


def _supported_languages():
    supported = gtts.lang.tts_langs()

    lang_list = [val for val in supported.values()]

    for lang in sorted(lang_list):
        print(lang)


if __name__ == "__main__":
    TTS().tts("Hello, world!")
