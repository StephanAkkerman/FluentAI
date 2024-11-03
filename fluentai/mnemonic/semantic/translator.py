import unicodedata
from functools import lru_cache

import pycountry
from googletrans import Translator

from fluentai.utils.logger import logger

translator = Translator()


def is_latin_script(word: str):
    for char in word:
        if "LATIN" not in unicodedata.name(char):
            return False
    return True


def remove_diacritics(word):
    # Normalize the word to decomposed form (NFD)
    decomposed = unicodedata.normalize("NFD", word)
    # Filter out diacritic characters (those in the 'Mn' category)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def map_language_code(input_code):
    """
    Maps custom or ISO 639-2 language codes to ISO 639-1 codes used by Google Translate.

    Args:
        input_code (str): The input language code (e.g., 'ind', 'zho-s').

    Returns:
        str: The mapped ISO 639-1 language code (e.g., 'id', 'zh-cn').

    Raises:
        ValueError: If the language code cannot be mapped.
    """
    # Handle special cases
    # https://py-googletrans.readthedocs.io/en/latest/#googletrans-languages
    special_mappings = {
        "zho-s": "zh-cn",  # Chinese Simplified
        "zho-t": "zh-tw",  # Chinese Traditional
        "arm-e": "hy",  # Armenian (Eastern)
        "arm-w": "hy",  # Armenian (Western)
        "eng-uk": "en",  # English (United Kingdom)
        "eng-us": "en",  # English (United States)
        "fra-qu": "fr",  # French (Quebec)
        "lat-clas": "la",  # Latin (Classical)
        "lat-eccl": "la",  # Latin (Ecclesiastical)
        "hbs-latn": "hr",  # Serbo-Croatian (Latin) -> Croatian
        "hbs-cyrl": "sr",  # Serbo-Croatian (Cyrillic) -> Serbian
        "spa-latin": "es",  # Spanish (Latin America)
        "spa-me": "es",  # Spanish (Mexico)
        "vie-n": "vi",  # Vietnamese (Northern)
        "vie-c": "vi",  # Vietnamese (Central)
        "vie-s": "vi",  # Vietnamese (Southern)
        "wel-nw": "cy",  # Welsh (North)
        "wel-sw": "cy",  # Welsh (South)
        "por-br": "pt",  # Portuguese (Brazil)
        "por-po": "por",  # Portuguese (Portugal)
    }

    if input_code in special_mappings:
        return special_mappings[input_code]

    # Attempt to get ISO 639-1 code using pycountry
    language = pycountry.languages.get(alpha_3=input_code.lower())
    if language and hasattr(language, "alpha_2"):
        return language.alpha_2


@lru_cache(maxsize=10000)
def translate_word(word, src_lang_code, target_lang_code: str = "en"):
    """
    Translates a word from the source language to target languages.

    Args:
        word (str): The word to translate.
        src_lang_code (str): The source language code (ISO 639-1 or custom).
        target_lang_code (str): A target language codes (ISO 639-1 or custom).

    Returns:
        dict: A dictionary with target language codes as keys and translated words as values.
    """
    src = map_language_code(src_lang_code)

    # Map target language codes.
    if len(target_lang_code) > 2:
        target = map_language_code(target_lang_code)
    else:
        target = target_lang_code

    # Add transliteration
    transliterated_word = word

    try:
        if not is_latin_script(word):
            transliterated_word = translator.translate(
                word, src=src, dest=src
            ).pronunciation
            # Lower case it
            transliterated_word = transliterated_word.lower()
            # Remove diacritics
            transliterated_word = remove_diacritics(transliterated_word)

        return (
            translator.translate(word, src=src, dest=target).text,
            transliterated_word,
        )
    except Exception as e:
        logger.info(f"Error translating {word} from {src} to {target}: {e}")
        return word, word


def translate_dataframe_column(
    df,
    word_col: str,
    lang_code: str,
    translated_col: str = "translated_word",
    target_lang_code: str = "en",
):
    """
    Translates a column in the DataFrame to the target language.

    Args:
        df (pd.DataFrame): The DataFrame containing words and language codes.
        word_col (str): The name of the column containing words to translate.
        lang_code_col (str): The name of the source language codes.
        translated_col (str, optional): The name of the new column for translations. Defaults to 'translated_word'.
        target_lang_code (str, optional): The target language code. Defaults to 'en'.

    Returns:
        pd.DataFrame: The DataFrame with an added translated column.
    """
    # Apply translation with progress bar
    df[translated_col] = df.apply(
        lambda row: translate_word(row[word_col], lang_code, target_lang_code),
        axis=1,
    )

    return df


if __name__ == "__main__":
    print(translate_word("amigo", "spa-latin"))
    print(translate_word("猫", "zho-s"))
