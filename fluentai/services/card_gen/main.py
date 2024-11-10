import gc

import torch

from fluentai.services.card_gen.imagine.image_gen import generate_img
from fluentai.services.card_gen.imagine.verbal_cue import VerbalCue
from fluentai.services.card_gen.mnemonic.word2mnemonic import generate_mnemonic
from fluentai.services.card_gen.utils.logger import logger


def generate_mnemonic_img(word: str, lang_code: str):
    """
    Generate an image for a given word using the mnemonic pipeline.

    Parameters
    ----------
    word : str
        The word to generate an image for.
    lang_code : str
        The language code for the word.
    """
    best_matches = generate_mnemonic(word, lang_code)

    # Get the top phonetic match
    best_match = best_matches.iloc[0]

    vc = VerbalCue()

    # Generate a verbal cue
    logger.debug(f"Generating verbal cue for '{best_match}'-'{word}'...")
    prompt = vc.generate_cue(word, best_match["token_ort"])

    del vc

    gc.collect()

    torch.cuda.empty_cache()

    # Generate the image
    generate_img(prompt=prompt)
    logger.info("Image generated successfully!")


if __name__ == "__main__":
    generate_mnemonic_img("猫", "zho-s")
