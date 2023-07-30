import pandas as pd
from transphone import read_tokenizer
from nltk.metrics import aline

# https://github.com/xinjli/transphone/blob/main/doc/language.md
ind_tokenizer = read_tokenizer("ind")
eng_tokenizer = read_tokenizer("eng")

# Read the IPA data
english_ipa = pd.read_table(
    "transphoner_data/english/twl.ipa.tsv", header=None, names=["word", "ipa"]
)

# See word == "hello"
english_word = english_ipa[english_ipa["word"] == "hello"]["ipa"].to_list()[0]
print(english_word)

english_word = "".join(eng_tokenizer.tokenize("hello", "eng"))

# Create phonemes
indonesian_word = "".join(ind_tokenizer.tokenize("kucing", "ind"))
print(english_word, indonesian_word)

alignment = aline.align(english_word, indonesian_word)
print(alignment)
