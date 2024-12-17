import json

from googletrans import LANGCODES, LANGUAGES

with open("data/languages.json") as f:
    G2P_LANGUAGES = json.load(f)
G2P_LANGCODES = dict(map(reversed, G2P_LANGUAGES.items()))

# Google Translate
TRANSLATE_LANGUAGES = LANGUAGES
TRANSLATE_LANGCODES = LANGCODES

# Vocab Languages
VOCAB_LANGUAGES = [
    "af",
    "ar",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "gl",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "ka",
    "kk",
    "ko",
    "lt",
    "lv",
    "mk",
    "ml",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "pt_br",  # Brazilian Portuguese
    "ro",
    "ru",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "ze_en",  # Chinese & English
    "ze_zh",  # Chinese & English
    "zh_cn",  # Simplified Chinese
    "zh_tw",  # Traditional Chinese
]
