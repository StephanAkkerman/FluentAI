# List of supported languages
# https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?gid=557940309#gid=557940309
from transformers import AutoTokenizer, T5ForConditionalGeneration

from phonetic_sim import word_similarity

# https://github.com/lingjzhu/CharsiuG2P
model = T5ForConditionalGeneration.from_pretrained(
    "charsiu/g2p_multilingual_byT5_small_100", cache_dir="models"
)
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", cache_dir="models")


def g2p(words):

    out = tokenizer(words, padding=True, add_special_tokens=False, return_tensors="pt")

    preds = model.generate(
        **out, num_beams=1, max_length=50
    )  # We do not find beam search helpful. Greedy decoding is enough.
    phones = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
    return phones


# https://en.wiktionary.org/wiki/kucing#Indonesian
# IPA(key): /ˈkut͡ʃɪŋ/
indonesian_word = g2p(["<ind>: Kucing"])
# https://en.wiktionary.org/wiki/hello#English (UK)
# IPA: /həˈləʊ/, /hɛˈləʊ/
english_word = g2p(["<eng-uk>: Hello"])
dutch_word = g2p(["<dut>: Koekje"])

print(dutch_word[0], indonesian_word[0])
score = word_similarity(dutch_word[0], indonesian_word[0])
print(score)
