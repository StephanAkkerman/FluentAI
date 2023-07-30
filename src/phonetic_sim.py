# Create phoneme categories
phoneme_categories = {
    "aehijouwæɑɒɔəɛɜɪʊʌʔ": 0,  # vowels, semivowels, and weak consonants
    "bfpv": 1,  # labial consonants
    "ksxzɡʃʒ": 2,  # velars and sibilant consonants
    "dtðθ": 3,  # dental consonants
    "l": 4,  # lateral consonant
    "mnŋ": 5,  # nasals
    "ɹ": 6,  # rhotic
}

# Flatten the dictionary to map each phoneme to its category
phoneme_to_category = {
    phoneme: category
    for category, phonemes in phoneme_categories.items()
    for phoneme in phonemes
}


def phoneme_similarity(phoneme1, phoneme2):
    """Calculates the similarity between two phonemes."""
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
