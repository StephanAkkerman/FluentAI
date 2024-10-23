import multiprocessing

import gensim.downloader as api
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# Load your dataset
df = pd.read_csv("data/imageability/data.csv")

# Separate features and target
y = df["score"]

# Load pre-trained FastText embeddings
print("Loading FastText embeddings...")
embedding_model = api.load(
    "fasttext-wiki-news-subwords-300"
)  # 300-dim FastText embeddings


def get_embedding(word):
    try:
        return embedding_model.get_vector(word)
    except KeyError:
        # FastText can handle OOV by default, but include this as a fallback
        print(word)
        return np.zeros(embedding_model.vector_size)


# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cores}")

# Initialize tqdm_joblib context manager to integrate tqdm with joblib
with tqdm_joblib(tqdm(desc="Generating Embeddings", total=len(df))):
    # Generate embeddings in parallel
    embeddings = Parallel(n_jobs=num_cores)(
        delayed(get_embedding)(word) for word in df["word"]
    )

# Convert list of embeddings to a NumPy array
embeddings = np.vstack(embeddings)

# Convert list of embeddings to a NumPy array
embeddings = np.vstack(embeddings)
print(f"Generated embeddings shape: {embeddings.shape}")

# Create a DataFrame for words and scores
df_output = pd.DataFrame({"word": df["word"].values, "score": y})

# Determine embedding dimensions
embedding_dim = embeddings.shape[1]
embedding_columns = [f"emb_{i+1}" for i in range(embedding_dim)]

# Create a DataFrame for embeddings
embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)

# Concatenate the words, scores, and embeddings into a single DataFrame
df_output = pd.concat([df_output, embeddings_df], axis=1)

print(f"Combined DataFrame shape: {df_output.shape}")

output_parquet = "data/imageability/fasttext_embeddings_v3.parquet"

# Save the DataFrame to a .parquet file
try:
    df_output.to_parquet(output_parquet, index=False)
    print(f"Embeddings saved successfully to '{output_parquet}'.")
except Exception as e:
    raise Exception(f"An error occurred while saving to parquet: {e}")
