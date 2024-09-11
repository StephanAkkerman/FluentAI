from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pyclts import CLTS
from soundvectors import SoundVectors
from tqdm import tqdm

# Load CLTS data and SoundVectors
bipa = CLTS("data/clts-2.3.0").bipa
sv = SoundVectors(ts=bipa)

# Load dataset
ds = pd.read_csv(
    "data/eng_latn_us_broad.tsv", names=["token_ort", "token_ipa"], sep="\t"
)

# Remove spaces in token_ipa
ds["token_ipa"] = ds["token_ipa"].apply(lambda x: x.replace(" ", ""))


# Function to vectorize a word (i.e., a sequence of IPA symbols) using multi-threading
def vectorize_word(word, max_workers=4):
    word_vector = []

    # Define a helper function to get vector for each letter
    def vectorize_letter(letter):
        try:
            return sv.get_vec(letter)
        except ValueError:
            return None  # Return None for unknown characters

    # Use ThreadPoolExecutor to parallelize the letter vectorization
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(vectorize_letter, letter) for letter in word]
        for future in futures:
            result = future.result()
            if result is not None:
                word_vector.append(result)

    return word_vector


# Function to parallelize the vectorization using multi-threading and tqdm for progress
def vectorize_in_parallel(token_ipa_list, max_workers=8):
    vectors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress while submitting tasks
        futures = []
        for word in tqdm(token_ipa_list, desc="Submitting tasks"):
            futures.append(executor.submit(vectorize_word, word))

        # As the tasks complete, append the results and update the progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Vectorizing words"
        ):
            vectors.append(future.result())

    return vectors


print("Starting vectorization...")
# Apply multi-threaded vectorization process to the entire dataset
ds["token_vectors"] = vectorize_in_parallel(ds["token_ipa"].tolist())

# Save the result as a new CSV file
output_file = "data/eng_latn_us_broad_vectors.csv"
ds.to_csv(output_file, index=False)

print(f"Word vectors saved to {output_file}")
