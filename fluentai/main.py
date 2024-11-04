# This will serve as the main functionality
# It will call the word to mnemonic pipeline
# Generate a prompt
# Generate the image
from fluentai.imagine.image_gen import generate_img
from fluentai.imagine.verbal_cue import VerbalCue
from fluentai.mnemonic.word2mnemonic import generate_mnemonic

vc = VerbalCue()


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

    # Generate a verbal cue
    prompt = vc.generate_cue(word, best_match["token_ort"])

    # Generate the image
    generate_img(prompt=prompt)


if __name__ == "__main__":
    generate_mnemonic_img("猫", "zho-s")
