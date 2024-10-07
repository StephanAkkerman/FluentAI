# List of supported languages
# https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?gid=557940309#gid=557940309
from transformers import AutoTokenizer, T5ForConditionalGeneration

# https://github.com/lingjzhu/CharsiuG2P
model = T5ForConditionalGeneration.from_pretrained(
    "charsiu/g2p_multilingual_byT5_small_100", cache_dir="models"
)
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", cache_dir="models")


def g2p(words: list[str]) -> str:

    out = tokenizer(words, padding=True, add_special_tokens=False, return_tensors="pt")

    preds = model.generate(
        **out, num_beams=1, max_length=50
    )  # We do not find beam search helpful. Greedy decoding is enough.
    ipa = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
    ipa = ipa[0]
    # Remove the leading ˈ
    ipa = ipa.removeprefix("ˈ")
    return ipa


def example():
    # https://en.wiktionary.org/wiki/kucing#Indonesian
    # IPA(key): /ˈkut͡ʃɪŋ/
    indonesian_word = g2p(["<ind>: Kucing"])
    # https://en.wiktionary.org/wiki/hello#English (UK)
    # IPA: /həˈləʊ/, /hɛˈləʊ/
    # https://raw.githubusercontent.com/open-dict-data/ipa-dict/refs/heads/master/data/en_US.txt
    # /əˌbɹiviˈeɪʃən/ ->  əˌbɹiviˈeɪʃən
    # G2P removes / at the beginning but not leading ˈ
    # /ˈɡɹəmbəɫ/ -> ˈɡɹəmbəɫ
    # Wiktionary: /ˈɡɹʌmbl̩/
    english_word = g2p(["<eng-us>: grumble"])
    dutch_word = g2p(["<dut>: Koekje"])

    print(
        indonesian_word,
        english_word,
        dutch_word,
    )


if __name__ == "__main__":
    example()
