import unicodedata
from functools import lru_cache

from googletrans import Translator

from fluentai.services.card_gen.utils.lang_codes import map_language_code
from fluentai.services.card_gen.utils.logger import logger

translator = Translator()


def is_latin_script(word: str) -> bool:
    """
    Checks if the given word is written in the Latin script.

    Parameters
    ----------
    word : str
        The word to check.

    Returns
    -------
    bool
        True if the word is written in the Latin script, False otherwise.
    """
    for char in word:
        if "LATIN" not in unicodedata.name(char):
            return False
    return True


def remove_diacritics(word: str) -> str:
    """
    Removes diacritics from a word.

    Parameters
    ----------
    word : str
        The word to remove diacritics from.

    Returns
    -------
    str
        The word without diacritics.
    """
    # Normalize the word to decomposed form (NFD)
    decomposed = unicodedata.normalize("NFD", word)
    # Filter out diacritic characters (those in the 'Mn' category)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


@lru_cache(maxsize=10000)
def translate_word(word, src_lang_code, target_lang_code: str = "en"):
    """
    Translates a word from the source language to target languages.

    Args:
        word (str): The word to translate.
        src_lang_code (str): The source language code (ISO 639-1 or custom).
        target_lang_code (str): A target language codes (ISO 639-1 or custom).

    Returns
    -------
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

    Returns
    -------
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
