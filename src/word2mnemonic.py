from imageability.imageability import ImageabilityPredictor
from similarity.phonetic.phonetic import top_phonetic

imageability_predictor = ImageabilityPredictor()


def generate_mnemonic(word, language_code):
    """
    Generate a mnemonic for the input word using the phonetic representation.

    Parameters:
    - word: String, input word
    - language_code: String, language code of the input word
    """
    # Get the top x phonetically similar words
    top = top_phonetic(word, language_code)

    # Generate their word imageability for all token_ort in top
    scores = imageability_predictor.get_column_imageability(top, "token_ort")

    # Add scores to the top DataFrame
    top["imageability"] = scores

    # Orthographic similarity

    # Semantic similarity


if __name__ == "__main__":
    generate_mnemonic("kucing", "ind")
