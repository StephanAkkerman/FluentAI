import gensim.downloader as api
import numpy as np
import pandas as pd

# ================================
# 1. Load and Preprocess Data
# ================================

# Load your dataset
df = pd.read_csv("data/imageability/data.csv")

# Extract the list of words
words = df["word"].tolist()

# ================================
# 2. Load FastText Embeddings
# ================================

print("Loading FastText embeddings...")
embedding_model = api.load(
    "fasttext-wiki-news-subwords-300"
)  # 300-dim FastText embeddings
print("FastText embeddings loaded.")

# ================================
# 3. Generate Embeddings in Bulk
# ================================

print("Generating embeddings in bulk...")
# Retrieve embeddings for all words at once
embeddings = embedding_model[words]  # This returns a 2D NumPy array
print("Embeddings generated.")

# ================================
# 4. Save the Embeddings Efficiently
# ================================

# Save words and embeddings using NumPy's savez_compressed for efficiency
np.savez_compressed(
    "data/imageability/fasttext_word_embeddings.npz",
    words=np.array(words),
    embeddings=embeddings,
)
print("Embeddings saved using NumPy's savez_compressed as 'word_embeddings.npz'.")
