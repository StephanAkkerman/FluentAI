import faiss
import numpy as np
import pandas as pd
from ipa2vec import panphon_vec, soundvec

vectorizer = panphon_vec
vector_file = (
    "data/eng_latn_us_broad_vectors_panphon.csv"
    if vectorizer == panphon_vec
    else "data/eng_latn_us_broad_vectors.csv"
)

# Load the dataset
ds = pd.read_csv(vector_file)

# Parse the vectors from string to actual lists
ds["token_vectors"] = ds["token_vectors"].apply(lambda x: eval(x))


# Function to pad vectors to the maximum length
def pad_vectors(vectors):
    # Find the maximum length of any vector
    max_len = max(len(vec) for vec in vectors)

    # Pad each vector with zeros to match the maximum length
    padded_vectors = [
        np.pad(vec, (0, max_len - len(vec)), "constant") for vec in vectors
    ]

    return padded_vectors


# Handle empty vectors before flattening
def safe_hstack(vec):
    if len(vec) > 0:
        return np.hstack(vec)
    else:
        # Handle case where the vector is empty
        return np.zeros(
            1
        )  # You can adjust this based on how you want to handle empty vectors


# Flatten the dataset vectors and handle any empty ones
dataset_vectors_flat = [safe_hstack(vec) for vec in ds["token_vectors"]]
dataset_vectors_padded = pad_vectors(dataset_vectors_flat)

# Convert the padded vectors into a NumPy array
dataset_matrix = np.array(dataset_vectors_padded, dtype=np.float32)

# Build a FAISS index for Maximum-Inner-Product-Search (MIPS)
dimension = dataset_matrix.shape[1]  # Number of features (vector length)
index = faiss.IndexFlatIP(dimension)  # Index for Inner Product (dot product)
index.add(dataset_matrix)  # Add the dataset vectors to the index

# Input word (IPA format)
ipa_input = "kˈut͡ʃiŋ"

# Vectorize the input word
input_vector = np.hstack(vectorizer(ipa_input)).astype(np.float32)

# Pad the input vector to match the length of dataset vectors
input_vector_padded = np.pad(
    input_vector, (0, dataset_matrix.shape[1] - len(input_vector))
)

# Reshape input vector for FAISS (needs to be 2D)
input_vector_padded = input_vector_padded.reshape(1, -1)

# Perform the Maximum-Inner-Product-Search (MIPS)
top_n = 5  # Number of closest words to retrieve
distances, indices = index.search(input_vector_padded, top_n)

# Retrieve the closest words
closest_words = ds.iloc[indices[0]]

# Display the closest words and their IPA
print(closest_words[["token_ort", "token_ipa"]])
