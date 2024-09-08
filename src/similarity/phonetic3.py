import numpy as np
import pandas as pd
from pyclts import CLTS
from sklearn.metrics.pairwise import cosine_similarity
from soundvectors import SoundVectors

# Load CLTS data and SoundVectors
bipa = CLTS("data/clts-2.3.0").bipa
sv = SoundVectors(ts=bipa)

# Load the dataset
ds = pd.read_csv("data/eng_latn_us_broad_vectors.csv")

# Parse the vectors from string to actual lists
ds["token_vectors"] = ds["token_vectors"].apply(lambda x: eval(x))


# Function to vectorize the input IPA word
def vectorize_word(ipa_word, sv):
    word_vector = []
    for letter in ipa_word:
        try:
            word_vector.append(sv.get_vec(letter))
        except ValueError:
            word_vector.append(np.zeros(len(sv.get_vec("a"))))  # Handle unknown letters
    return word_vector


# Function to pad vectors to the maximum length
def pad_vectors(vectors):
    # Find the maximum length of any vector
    max_len = max(len(vec) for vec in vectors)

    # Pad each vector with zeros to match the maximum length
    padded_vectors = [
        np.pad(vec, (0, max_len - len(vec)), "constant") for vec in vectors
    ]

    return padded_vectors


# Function to find the top 5 closest words
def find_closest_words(ipa_word, sv, dataset, top_n=5):
    # Vectorize the input IPA word
    input_vector = vectorize_word(ipa_word, sv)

    # Flatten the input vector
    input_vector_flat = np.hstack(input_vector)

    # Flatten the dataset vectors
    dataset_vectors_flat = [np.hstack(vec) for vec in dataset["token_vectors"]]

    # Pad both input vector and dataset vectors
    all_vectors = [input_vector_flat] + dataset_vectors_flat
    padded_vectors = pad_vectors(all_vectors)

    # Separate the input vector from the dataset
    input_vector_padded = padded_vectors[0].reshape(1, -1)  # Reshape to (1, n_features)
    dataset_vectors_padded = np.array(
        padded_vectors[1:]
    )  # Ensure it's 2D (n_samples, n_features)

    # Compute cosine similarity between input vector and all dataset vectors
    similarities = cosine_similarity(input_vector_padded, dataset_vectors_padded)[0]

    # Get the top N most similar words
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_words = dataset.iloc[top_indices]

    return top_words


# Example usage
# You need to pass `sv` (SoundVectors object) that you used to vectorize the IPA in your dataset
ipa_input = "mɝəkə"  # Input IPA
closest_words = find_closest_words(ipa_input, sv, ds, top_n=5)

# Display the closest words
print(closest_words[["token_ort", "token_ipa"]])
