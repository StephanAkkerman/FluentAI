import ast

import numpy as np
import pandas as pd
from ipa2vec import panphon_vec, soundvec
from sklearn.metrics.pairwise import cosine_similarity

# Select the appropriate vectorizer and file
vectorizer = panphon_vec
vector_file = (
    "data/eng_latn_us_broad_vectors_panphon.csv"
    if vectorizer == panphon_vec
    else "data/eng_latn_us_broad_vectors.csv"
)

# Load only necessary columns and the top 100 rows
necessary_columns = ["token_ort", "token_ipa", "vectors"]
ds = pd.read_csv(vector_file, usecols=necessary_columns)

# Parse the vectors using ast.literal_eval and convert to NumPy arrays
ds["vectors"] = (
    ds["vectors"].apply(ast.literal_eval).apply(lambda vec: np.array(vec).flatten())
)

# Remove entries with empty vectors
ds_cleaned = ds[ds["vectors"].apply(len) > 0].reset_index(drop=True)
print(f"Removed {len(ds) - len(ds_cleaned)} entries with empty vectors.")

# Determine the maximum vector length after cleaning
max_len = ds_cleaned["vectors"].apply(len).max()
print(f"Maximum vector length: {max_len}")


# Function to pad a single vector
def pad_vector(vec, max_length):
    try:
        if len(vec) < max_length:
            return np.pad(vec, (0, max_length - len(vec)), "constant")
        elif len(vec) > max_length:
            # Optionally, truncate vectors longer than max_length
            return vec[:max_length]
        else:
            return vec
    except Exception as e:
        print(f"Error padding vector {vec}: {e}")
        return np.zeros(max_length)


# Apply padding with the corrected function
ds_cleaned["padded_vectors"] = ds_cleaned["vectors"].apply(
    lambda vec: pad_vector(vec, max_len)
)

# Verify all vectors are of the same length
vector_lengths = ds_cleaned["padded_vectors"].apply(len)
unique_lengths = vector_lengths.unique()
print(f"Unique vector lengths after padding: {unique_lengths}")

assert (
    len(unique_lengths) == 1 and unique_lengths[0] == max_len
), "Padding failed: Not all vectors have the same length."

# Stack all padded vectors into a 2D NumPy array for efficient computation
dataset_vectors_padded = np.stack(ds_cleaned["padded_vectors"].values).astype(
    np.float32
)


def find_closest_words(ipa_word, dataset_vectors, dataset_df, top_n=5):
    # Vectorize the input IPA word
    input_vector = np.hstack(vectorizer(ipa_word)).astype(np.float32)

    # Pad the input vector to match the dataset vectors
    padded_input_vector = pad_vector(input_vector, max_len).reshape(1, -1)

    # Compute cosine similarity between the input vector and all dataset vectors
    similarities = cosine_similarity(padded_input_vector, dataset_vectors)[0]

    # Find the indices of the top N most similar vectors
    top_indices = np.argpartition(similarities, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    # Retrieve the top N closest words
    top_words = dataset_df.iloc[top_indices]

    return top_words


# Example usage
ipa_input = "kˈut͡ʃiŋ"  # Input IPA
closest_words = find_closest_words(
    ipa_input, dataset_vectors_padded, ds_cleaned, top_n=5
)

# Display the closest words
print(closest_words[["token_ort", "token_ipa"]])
