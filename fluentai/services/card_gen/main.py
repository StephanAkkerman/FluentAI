import gc

import torch

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.imagine.image_gen import generate_img
from fluentai.services.card_gen.imagine.verbal_cue import VerbalCue
from fluentai.services.card_gen.mnemonic.word2mnemonic import generate_mnemonic
from fluentai.services.card_gen.tts.tts import TTS
from fluentai.services.card_gen.utils.logger import logger


def generate_mnemonic_img(word: str, lang_code: str) -> tuple:
    """
    Generate an image for a given word using the mnemonic pipeline.

    Parameters
    ----------
    word : str
        The word to generate an image for in the language of lang_code.
    lang_code : str
        The language code for the word.

    Returns
    -------
    str
        The path to the generated image.
    str
        The verbal cue for the image.
    str
        The translated word.
    str
        The path to the generated audio file.
    str
        The IPA spelling of the best match.
    """
    best_matches, translated_word, _ = generate_mnemonic(word, lang_code)

    # Get the top phonetic match
    best_match = best_matches.iloc[0]
    ipa = best_match["token_ipa"]

    vc = VerbalCue()

    # Generate a verbal cue
    logger.debug(
        f"Generating verbal cue for '{best_match['token_ort']}'-'{translated_word}'..."
    )
    prompt = vc.generate_cue(translated_word, best_match["token_ort"])

    if config.get("LLM", {}).get("DELETE_AFTER_USE", True):
        _clean(vc)

    # Generate the image
    image_path = generate_img(prompt=prompt, word1=word, word2=best_match["token_ort"])

    # Generate TTS
    tts_model = TTS(lang_code)
    tts_path = tts_model.tts(word)
    _clean(tts_model)

    return image_path, prompt, translated_word, tts_path, ipa


def _clean(var):
    del var
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    generate_mnemonic_img("kat", "dut")
