import pandas as pd
from transphone import read_tokenizer

from src.similarity.phonetic.phonetic_sim import word_similarity

# https://github.com/xinjli/transphone/blob/main/doc/language.md
ind_tokenizer = read_tokenizer("ind")
eng_tokenizer = read_tokenizer("eng")

# Read the IPA data
english_ipa = pd.read_table(
    "transphoner_data/english/twl.ipa.tsv", header=None, names=["word", "ipa"]
)

# See word == "hello"
# https://en.wiktionary.org/wiki/hello#English (UK)
# IPA: /həˈləʊ/, /hɛˈləʊ/
english_word = english_ipa[english_ipa["word"] == "hello"]["ipa"].to_list()[0]
# print(english_word)

# english_word = "".join(eng_tokenizer.tokenize("hello", "eng"))

# Create phonemes
# https://en.wiktionary.org/wiki/kucing#Indonesian
# IPA(key): /ˈkut͡ʃɪŋ/
indonesian_word = "".join(ind_tokenizer.tokenize("kucing", "ind"))
print(english_word, indonesian_word)

score = word_similarity(english_word, indonesian_word)
print(score)
