import numpy as np
import pandas as pd
from ipa2vec import panphon_vec, soundvec
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = soundvec
vector_file = "data/eng_latn_us_broad_vectors.csv"

# Load the dataset and only the top 100 rows
ds = pd.read_csv(vector_file).head(100)

# Parse the vectors from string to actual lists
ds["token_vectors"] = ds["token_vectors"].apply(lambda x: eval(x))


# Function to apply average pooling
def average_pool(vectors):
    return np.mean(vectors, axis=0)


# Example usage
input_vector = np.hstack(vectorizer("mɝəkə")).astype(np.float32)
dataset_vectors_flat = [np.hstack(vec) for vec in ds["token_vectors"]]

# Apply average pooling to each vector
pooled_input_vector = average_pool(input_vector)
pooled_dataset_vectors = [average_pool(vec) for vec in dataset_vectors_flat]

# Compute cosine similarity
similarities = cosine_similarity([pooled_input_vector], pooled_dataset_vectors)[0]
top_indices = np.argsort(similarities)[-5:][::-1]
closest_words = ds.iloc[top_indices]
print(closest_words[["token_ort", "token_ipa"]])
