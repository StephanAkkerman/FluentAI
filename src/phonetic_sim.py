# Create phoneme categories
phoneme_categories = {
    "iyeøɛœæaɑɒɔəɜɪʊʌɘɵɤʉɨʘuo": 0,  # vowels
    "wɥɹjɰ": 1,  # semivowels
    "bfpv": 2,  # bilabial and labiodental plosives/fricatives
    "m": 3,  # bilabial nasal
    "θð": 4,  # dental fricatives
    "tdʈɖ": 5,  # alveolar and post-alveolar plosives
    "n": 6,  # alveolar nasal
    "ɳ": 7,  # retroflex nasal
    "ʃʒ": 8,  # post-alveolar fricatives
    "ɕʑ": 9,  # alveolo-palatal fricatives
    "rl": 10,  # alveolar and post-alveolar liquids
    "ɾɽ": 11,  # flaps
    "cɟkɡqɢ": 12,  # velar and uvular plosives
    "ŋɴ": 13,  # velar, uvular, and pharyngeal nasals
    "xɣχʁħʕ": 14,  # velar, uvular, and pharyngeal fricatives
    "ʔ": 15,  # glottal stop
    "ʍh": 16,  # other glottal consonants
    "ʙrʀ": 17,  # trills
    "ɬɮ": 18,  # lateral fricatives
    "ʎʟ": 19,  # lateral approximants
    "ʦʣʧʤ": 20,  # affricates
    "ʘǀǃǂǁ": 21,  # clicks
}

# Flatten the dictionary to map each phoneme to its category
phoneme_to_category = {
    phoneme: category
    for phonemes, category in phoneme_categories.items()
    for phoneme in phonemes
}


def phoneme_similarity(phoneme1, phoneme2):
    """Calculates the similarity between two phonemes."""
    # Test if the phoneme exist in the dictionary
    if phoneme1 not in phoneme_to_category:
        print(f"{phoneme1} is not a known phoneme")
    if phoneme2 not in phoneme_to_category:
        print(f"{phoneme2} is not a known phoneme")

    if phoneme1 == phoneme2:
        return 1
    elif phoneme_to_category.get(phoneme1) == phoneme_to_category.get(phoneme2):
        return 0.5
    else:
        return 0


def word_similarity(word1, word2):
    """Calculates the similarity between two words based on their phonemes."""
    len1 = len(word1)
    len2 = len(word2)
    match_table = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match_table[i][j] = max(
                match_table[i - 1][j - 1]
                + phoneme_similarity(word1[i - 1], word2[j - 1]),
                match_table[i - 1][j],
                match_table[i][j - 1],
            )
    return match_table[len1][len2] / (len1 + len2) * 2.0
