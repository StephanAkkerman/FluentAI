from ast import literal_eval

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from ipa2vec import panphon_vec, soundvec
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = panphon_vec
vector_file = "data/eng_latn_us_broad_vectors_panphon.csv"

# Load the dataset and only the top 100 rows
ds = pd.read_csv(vector_file).head(100)

# Parse the "token_vectors" column from string to actual lists
ds["token_vectors"] = ds["token_vectors"].apply(literal_eval)


# Function to flatten nested lists (if needed)
def flatten_vector(vec):
    return np.hstack(vec).astype(np.float32)


# Apply flattening to each token vector
ds["token_vectors"] = ds["token_vectors"].apply(flatten_vector)


def dtw_distance(v1, v2):
    # Ensure both input vectors are 1D using np.ravel()
    v1 = np.ravel(v1)
    v2 = np.ravel(v2)

    # Check if both vectors are 1D
    assert v1.ndim == 1, "v1 is not 1D"
    assert v2.ndim == 1, "v2 is not 1D"

    distance, path = fastdtw(v1, v2, dist=2)
    return distance


# Function to find the top 5 closest words using DTW
def find_closest_words_dtw(ipa_word, dataset, top_n=5):
    # Vectorize the input IPA word
    input_vector = np.hstack(vectorizer(ipa_word)).astype(np.float32)

    # Compute DTW distance between input vector and all dataset vectors
    distances = [
        dtw_distance(input_vector, np.hstack(vec)) for vec in dataset["token_vectors"]
    ]

    # Get the top N words with the smallest DTW distances
    top_indices = np.argsort(distances)[:top_n]
    top_words = dataset.iloc[top_indices]

    return top_words


# Example usage
# You need to pass `sv` (SoundVectors object) that you used to vectorize the IPA in your dataset
ipa_input = "mɝəkə"  # Input IPA
print(
    dtw_distance(
        np.hstack(vectorizer(ipa_input)).astype(np.float32), ds["token_vectors"].iloc[0]
    )
)
closest_words = find_closest_words_dtw(ipa_input, ds, top_n=5)

# Display the closest words
print(closest_words)
