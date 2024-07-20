import numpy as np
from sklearn.preprocessing import LabelEncoder

# Example phoneme set (IPA)
phonemes = [
    "a",
    "b",
    "k",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "z",
]

# Encode phonemes as unique vectors (one-hot encoding for simplicity)
encoder = LabelEncoder()
encoder.fit(phonemes)
phoneme_vectors = np.eye(len(phonemes))

phoneme_to_vector = dict(zip(encoder.classes_, phoneme_vectors))


def word_to_vector(word, phoneme_to_vector):
    vectors = [
        phoneme_to_vector[phoneme] for phoneme in word if phoneme in phoneme_to_vector
    ]
    if vectors:
        return np.mean(vectors, axis=0)  # Average vectors for simplicity
    else:
        return np.zeros(len(phoneme_to_vector))


# Example word list with IPA transcriptions
words = ["cat", "dog", "bat"]
ipa_transcriptions = ["kæt", "dɔg", "bæt"]

word_vectors = np.array(
    [word_to_vector(ipa, phoneme_to_vector) for ipa in ipa_transcriptions]
)

import faiss

# Initialize Faiss index for Inner Product search
d = word_vectors.shape[1]  # Dimension of vectors
index = faiss.IndexFlatIP(d)

# Add word vectors to the index
index.add(word_vectors)

# Given IPA transcription to find similar words
foreign_word_ipa = "kætɔt"  # Example foreign word IPA transcription
foreign_word_vector = word_to_vector(foreign_word_ipa, phoneme_to_vector).reshape(1, -1)

# Perform search
k = 1  # Number of closest words to find
distances, indices = index.search(foreign_word_vector, k)

# Get closest words
closest_words = [words[i] for i in indices[0]]
print(closest_words)
