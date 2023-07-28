import pandas as pd
from transphone import read_tokenizer

# https://github.com/xinjli/transphone/blob/main/doc/language.md
language_code = "ind"
tokenizer = read_tokenizer(language_code)

# Read the IPA data
english_ipa = pd.read_table(
    "transphoner_data/english/twl.ipa.tsv", header=None, names=["word", "ipa"]
)
print(english_ipa.head())

# See word == "hello"
print(english_ipa[english_ipa["word"] == "hello"])
print(english_ipa[english_ipa["word"] == "world"])

# Create phonemes
print(tokenizer.tokenize("kucing", language_code))
