import pycountry

from fluentai.services.card_gen.constants.languages import (
    G2P_LANGCODES,
    TRANSLATE_LANGCODES,
)


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
    # Check if the code is supported in
    if input_code in TRANSLATE_LANGCODES.values():
        return input_code

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
        "bur": "my",  # Burmese
        "cze": "cs",  # Czech
        "geo": "ka",  # Georgian
        "ger": "de",  # German
        "gre": "el",  # Greek
        "grc": "el",  # Ancient Greek
        "mac": "mk",  # Macedonian
        "tts": "th",  # Thai
        "slo": "sk",  # Slovak
        "ice": "is",  # Icelandic
        "gle": "ga",  # Irish
        "nob": "no",  # Norwegian (Bokmål)
    }

    if input_code in special_mappings:
        return special_mappings[input_code]

    # Attempt to get ISO 639-1 code using pycountry
    language = pycountry.languages.get(alpha_3=input_code.lower())
    if language and hasattr(language, "alpha_2"):
        # Check if the code is in the list of supported languages
        if language.alpha_2 in TRANSLATE_LANGCODES.values():
            return language.alpha_2


if __name__ == "__main__":
    # Try mapping all G2p language codes
    for lang, code in G2P_LANGCODES.items():
        mapped = map_language_code(code)
        if mapped is None:
            print(f"Failed to map {code} ({lang})")
        else:
            # Check if the code is in the list of supported languages
            if mapped not in TRANSLATE_LANGCODES.values():
                print(f"Unsupported language code: {code} ({lang})")
