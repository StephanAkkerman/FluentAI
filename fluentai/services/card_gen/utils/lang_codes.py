import pycountry


def map_language_code(input_code: str) -> str:
    """
    Maps custom or ISO 639-2 language codes to ISO 639-1 codes used by Google Translate.

    Args:
        input_code (str): The input language code (e.g., 'ind', 'zho-s').

    Returns
    -------
        str: The mapped ISO 639-1 language code (e.g., 'id', 'zh-cn').

    Raises
    ------
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
        "dut": "nl",  # Dutch
    }

    if input_code in special_mappings:
        return special_mappings[input_code]

    # Attempt to get ISO 639-1 code using pycountry
    language = pycountry.languages.get(alpha_3=input_code.lower())
    if language and hasattr(language, "alpha_2"):
        return language.alpha_2
