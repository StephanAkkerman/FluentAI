import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.mnemonic.phonetic.ipa2vec import ft, sv
from fluentai.services.card_gen.mnemonic.phonetic.utils import flatten_vectors
from fluentai.services.card_gen.utils.logger import logger


def vectorize_word_clts(word, sv):
    """
    Vectorizes a word using CLTS by retrieving vectors for each IPA character.

    Args:
        word (str): The IPA transcription of the word.
        sv (SoundVectors): Initialized SoundVectors object.

    Returns
    -------
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
        except IndexError:
            logger.info(f"Error processing letter '{letter}' in '{word}'")
            continue
    return word_vector


def vectorize_word_panphon(word, ft):
    """
    Vectorizes a word using Panphon by converting it to a list of feature vectors.

    Args:
        word (str): The IPA transcription of the word.
        ft (panphon.FeatureTable): Initialized FeatureTable object.

    Returns
    -------
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

    Returns
    -------
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

    Returns
    -------
        list: Padded vector.
    """
    if not vector:
        return [padding_value] * max_length
    if len(vector) >= max_length:
        return vector[:max_length]
    else:
        padding = [padding_value] * (max_length - len(vector))
        return vector + padding


def main(
    method: str = "clts",
    save_loc: str = "data/phonological/embeddings",
):
    """
    Main function to orchestrate the vectorization process based on the selected method.

    Args:
        method (str): Vectorization method to use ('clts' or 'panphon').
    """
    data_file_name = config.get("PHONETIC_SIM").get("IPA_FILE")

    if method == "clts":
        output_file = f"{save_loc}/{data_file_name}_clts.parquet"
    elif method == "panphon":
        output_file = f"{save_loc}/{data_file_name}_panphon.parquet"
    max_workers = 8  # Adjust based on your system's capabilities

    # Validate method
    if method not in ["clts", "panphon"]:
        raise ValueError("Invalid method selected. Choose either 'clts' or 'panphon'.")

    # Load data
    ds = pd.read_csv(
        hf_hub_download(
            repo_id=config.get("PHONETIC_SIM").get("IPA_REPO"),
            filename=config.get("PHONETIC_SIM").get("IPA_FILE"),
            cache_dir="datasets",
            repo_type="dataset",
        )
    )

    # Perform CLTS vectorization
    if method == "clts":
        logger.info("Starting CLTS vectorization...")
        ds["vectors"] = vectorize_in_parallel(
            token_ipa_list=ds["token_ipa"].tolist(),
            vectorize_func=vectorize_word_clts,
            func_args=(sv,),
            max_workers=max_workers,
            description="CLTS",
        )

    # Perform Panphon vectorization
    elif method == "panphon":
        logger.info("Starting Panphon vectorization...")
        ds["vectors"] = vectorize_in_parallel(
            token_ipa_list=ds["token_ipa"].tolist(),
            vectorize_func=vectorize_word_panphon,
            func_args=(ft,),
            max_workers=max_workers,
            description="Panphon",
        )

    # Flatten vectors
    ds = flatten_vectors(ds, "vectors")

    os.makedirs(save_loc, exist_ok=True)

    ds.to_parquet(output_file, index=False)
    logger.info(f"Word vectors successfully saved to {output_file}")


if __name__ == "__main__":
    # Configuration: Set the desired method here.
    # Options:
    #   - "clts"    : Generate only CLTS vectors.
    #   - "panphon" : Generate only Panphon vectors.
    method = "clts"  # Change this to "clts" or "panphon" as needed.
    data_file1 = "data/phonological/eng_latn_us_broad.tsv"
    data_file2 = "data/phonological/en_US.txt"

    main(method, data_file=data_file1)
