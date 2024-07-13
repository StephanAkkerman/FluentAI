# This could be a different approach compared to transphoner.py
from transformers import AutoTokenizer, T5ForConditionalGeneration

from phonetic_sim import word_similarity

model = T5ForConditionalGeneration.from_pretrained(
    "charsiu/g2p_multilingual_byT5_small_100"
)
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

# tokenized English words
# words = ["Char", "siu", "is", "a", "Cantonese", "style", "of", "barbecued", "pork"]
# words = ["<eng-us>: " + i for i in words]

# Indonesian
# words = ["Kucing", "adalah", "hewan", "peliharaan", "yang", "lucu"]
# words = ["<ind>: " + i for i in words]


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
print(english_word[0], indonesian_word[0])
score = word_similarity(english_word[0], indonesian_word[0])
print(score)
