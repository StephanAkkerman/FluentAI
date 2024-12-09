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


def get_transliteration(word: str, src: str) -> str:
    """
    Given a word, returns its transliteration in the same language.

    Parameters
    ----------
    word : str
        The word to transliterate.
    src : str
        The language code of the word.

    Returns
    -------
    str
        The transliterated word.
    """
    if is_latin_script(word):
        return word

    try:
        transliterated_word = translator.translate(
            word, src=src, dest=src
        ).pronunciation
    except Exception as e:
        logger.error(f"Error transliterating {word} from {src} to {src}: {e}")
        detected_lang = translator.detect(word)
        logger.info(
            f"Could not comprehend original language code ({src}), detected {detected_lang.lang} with {detected_lang.confidence} confidence."
        )
        src = detected_lang.lang

    # Lower case it
    transliterated_word = transliterated_word.lower()
    # Remove diacritics
    return remove_diacritics(transliterated_word)


def get_translation(word: str, src: str, target: str) -> str:
    """
    Translates a word from the source language to the target language.

    Parameters
    ----------
    word : str
        The word in the source (src) language.
    src : str
        The language code of the source language.
    target : str
        The language code of the target language.

    Returns
    -------
    str
        The translated word.
    """
    try:
        return translator.translate(word, src=src, dest=target).text
    except Exception as e:
        logger.error(f"Error translating {word} from {src} to {target}: {e}")

    # Detect the language of the word
    detected_lang = translator.detect(word)
    logger.info(
        f"Could not comprehend original language code ({src}), detected {detected_lang.lang} with {detected_lang.confidence} confidence."
    )

    return translator.translate(word, src=detected_lang.lang, dest=target).text


@lru_cache(maxsize=10000)
def translate_word(word, src_lang_code, target_lang_code: str = "en") -> tuple:
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
    logger.debug(f"Translating {word} from {src_lang_code} to {target_lang_code}...")

    src = map_language_code(src_lang_code)
    logger.debug(f"Mapped translation source language code: {src}")

    # Map target language codes.
    if len(target_lang_code) > 2:
        target = map_language_code(target_lang_code)
        logger.debug(f"Mapped translation target language code: {target}")
    else:
        target = target_lang_code

    return get_translation(word, src, target), get_transliteration(word, src)


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
    print(translate_word("kat", "dut"))
    print(translate_word("amigo", "spa-latin"))
    print(translate_word("猫", "zho-s"))
