from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import panphon
from pyclts import CLTS
from soundvectors import SoundVectors
from tqdm import tqdm


def load_data(file_path):
    """
    Loads the dataset from a TSV file and cleans the IPA tokens by removing spaces.

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'token_ort' and 'token_ipa' columns.
    """
    print("Loading dataset...")
    ds = pd.read_csv(
        file_path, names=["token_ort", "token_ipa"], sep="\t", encoding="utf-8"
    )
    # Remove spaces in token_ipa
    ds["token_ipa"] = ds["token_ipa"].str.replace(" ", "", regex=False)
    return ds


def initialize_clts(clts_path="data/clts-2.3.0"):
    """
    Initializes the CLTS and SoundVectors objects.

    Args:
        clts_path (str): Path to the CLTS data directory.

    Returns:
        SoundVectors: Initialized SoundVectors object.
    """
    print("Initializing CLTS and SoundVectors...")
    bipa = CLTS(clts_path).bipa
    sv = SoundVectors(ts=bipa)
    return sv


def initialize_panphon():
    """
    Initializes the Panphon FeatureTable.

    Returns:
        panphon.FeatureTable: Initialized FeatureTable object.
    """
    print("Initializing Panphon FeatureTable...")
    ft = panphon.FeatureTable()
    return ft


def vectorize_word_clts(word, sv):
    """
    Vectorizes a word using CLTS by retrieving vectors for each IPA character.

    Args:
        word (str): The IPA transcription of the word.
        sv (SoundVectors): Initialized SoundVectors object.

    Returns:
        list: Concatenated list of vectors for the word.
    """
    word_vector = []
    # Vectorize each letter
    for letter in word:
        try:
            vec = sv.get_vec(letter)
            if vec is not None:
                word_vector.extend(vec)  # Flatten the vectors
        except ValueError:
            continue  # Skip unknown characters
    return word_vector


def vectorize_word_panphon(word, ft):
    """
    Vectorizes a word using Panphon by converting it to a list of feature vectors.

    Args:
        word (str): The IPA transcription of the word.
        ft (panphon.FeatureTable): Initialized FeatureTable object.

    Returns:
        list: List of Panphon feature vectors for the word.
    """
    return ft.word_to_vector_list(word, numeric=True)


def vectorize_in_parallel(
    token_ipa_list,
    vectorize_func,
    func_args,
    max_workers=8,
    description="Vectorization",
):
    """
    Vectorizes a list of IPA tokens in parallel using the specified vectorization function.

    Args:
        token_ipa_list (list): List of IPA transcriptions.
        vectorize_func (function): Function to vectorize a single word.
        func_args (tuple): Arguments to pass to the vectorize_func.
        max_workers (int): Number of worker threads.
        description (str): Description for the progress bar.

    Returns:
        list: List of vectorized representations.
    """
    vectors = [None] * len(token_ipa_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, word in tqdm(
            enumerate(token_ipa_list),
            total=len(token_ipa_list),
            desc=f"Submitting {description} tasks",
        ):
            futures[executor.submit(vectorize_func, word, *func_args)] = idx

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Vectorizing {description} words",
        ):
            idx = futures[future]
            vectors[idx] = future.result()

    return vectors


def pad_vector(vector, max_length, padding_value=0):
    """
    Pads a vector to the specified maximum length with a padding value.

    Args:
        vector (list): The original vector.
        max_length (int): The desired maximum length.
        padding_value (int or float): The value to use for padding.

    Returns:
        list: Padded vector.
    """
    if not vector:
        return [padding_value] * max_length
    if len(vector) >= max_length:
        return vector[:max_length]
    else:
        padding = [padding_value] * (max_length - len(vector))
        return vector + padding


def main(method="clts"):
    """
    Main function to orchestrate the vectorization process based on the selected method.

    Args:
        method (str): Vectorization method to use ('clts' or 'panphon').
    """
    # Configuration
    data_file = "data/eng_latn_us_broad.tsv"
    clts_path = "data/clts-2.3.0"
    output_file = "data/eng_latn_us_broad_vectors.csv"
    max_workers = 8  # Adjust based on your system's capabilities

    # Validate method
    if method not in ["clts", "panphon"]:
        raise ValueError("Invalid method selected. Choose either 'clts' or 'panphon'.")

    # Load data
    ds = load_data(data_file)

    # Initialize vectorizers
    sv = initialize_clts(clts_path) if method == "clts" else None
    ft = initialize_panphon() if method == "panphon" else None

    # Perform CLTS vectorization
    if method == "clts":
        print("Starting CLTS vectorization...")
        ds["vectors"] = vectorize_in_parallel(
            token_ipa_list=ds["token_ipa"].tolist(),
            vectorize_func=vectorize_word_clts,
            func_args=(sv,),
            max_workers=max_workers,
            description="CLTS",
        )

    # Perform Panphon vectorization
    elif method == "panphon":
        print("Starting Panphon vectorization...")
        ds["vectors"] = vectorize_in_parallel(
            token_ipa_list=ds["token_ipa"].tolist(),
            vectorize_func=vectorize_word_panphon,
            func_args=(ft,),
            max_workers=max_workers,
            description="Panphon",
        )

    # Padding Vectors
    print("Calculating maximum vector length for padding...")
    max_length = ds["vectors"].apply(len).max()
    print(f"Maximum vector length: {max_length}")
    print("Padding vectors...")
    ds["vectors_padded"] = ds["vectors"].apply(
        lambda x: pad_vector(x, max_length, padding_value=0) if x else []
    )
    # Convert to string for CSV compatibility
    ds["vectors_padded"] = ds["vectors_padded"].apply(lambda x: ";".join(map(str, x)))

    # Optionally rename columns based on method
    if method == "clts":
        ds.rename(columns={"vectors_padded": "clts_vectors_padded"}, inplace=True)
    elif method == "panphon":
        ds.rename(columns={"vectors_padded": "panphon_vectors_padded"}, inplace=True)

    # Save the result as a new CSV file
    print(f"Saving the vectors to {output_file}...")
    ds.to_csv(output_file, index=False)
    print(f"Word vectors successfully saved to {output_file}")


if __name__ == "__main__":
    # Configuration: Set the desired method here.
    # Options:
    #   - "clts"    : Generate only CLTS vectors.
    #   - "panphon" : Generate only Panphon vectors.
    method = "clts"  # Change this to "clts" or "panphon" as needed.

    main(method)
