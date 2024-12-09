import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from git import GitCommandError, RemoteProgress, Repo
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


def check_directory_exists(directory_path):
    """
    Checks if a directory exists at the specified path.

    Args:
        directory_path (str or Path): The path to the directory.

    Returns
    -------
        bool: True if the directory exists, False otherwise.
    """
    return Path(directory_path).is_dir()


def clone_repository(repo_url, clone_path):
    """
    Clones a GitHub repository to the specified path with a progress bar.

    Args:
        repo_url (str): The HTTPS or SSH URL of the GitHub repository.
        clone_path (str or Path): The local path where the repository will be cloned.
    """
    try:
        logger.info(f"Cloning repository from {repo_url} to {clone_path}...")
        # Initialize CloneProgress with descriptive parameters
        Repo.clone_from(repo_url, clone_path, progress=CloneProgress())
        logger.info("Repository cloned successfully.")
    except GitCommandError as e:
        logger.info(f"Error cloning repository: {e}")
        sys.exit(1)


class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=""):
        """
        Update the progress bar with the current operation and counts.
        """
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


def get_clts():
    """
    Downloads the Concepticon and CLTS repositories to the /local_data directory.
    """
    # Configuration
    data_directory = Path("local_data")  # Change this to your desired data directory
    repo_url = (
        "https://github.com/cldf-clts/clts.git"  # Replace with your repository URL
    )
    repo_name = (
        repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    )  # Extract repository name

    # Define the full path where the repository will be cloned
    clone_path = data_directory / repo_name

    # Check if the directory already exists
    if check_directory_exists(clone_path):
        logger.debug(f"The directory '{clone_path}' already exists. Skipping clone.")
    else:
        # Ensure the /local_data directory exists
        if not data_directory.exists():
            try:
                data_directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {data_directory}")
            except Exception as e:
                logger.info(f"Failed to create directory '{data_directory}': {e}")
                sys.exit(1)

        # Clone the repository
        clone_repository(repo_url, clone_path)


def pad_vectors(vectors):
    """
    Pad all vectors to the maximum length with zeros.

    Parameters
    ----------
    - vectors: List of NumPy arrays

    Returns
    -------
    - List of padded NumPy arrays
    """
    max_len = max(vec.shape[0] for vec in vectors)
    padded_vectors = [
        np.pad(vec, (0, max_len - len(vec)), "constant") for vec in vectors
    ]
    return padded_vectors


def convert_to_matrix(padded_vectors):
    """
    Convert list of padded vectors to a NumPy matrix.

    Parameters
    ----------
    - padded_vectors: List of NumPy arrays

    Returns
    -------
    - NumPy array matrix
    """
    dataset_matrix = np.array(padded_vectors, dtype=np.float32)
    return dataset_matrix


def parse_vectors(dataset, vector_column="vectors"):
    """
    Parse the 'vectors' column from strings to actual lists using ast.literal_eval.

    Parameters
    ----------
    - dataset: DataFrame with 'vectors' column as strings
    - vector_column: String, name of the column containing vectors

    Returns
    -------
    - DataFrame with 'vectors' column as lists
    """
    try:
        dataset[vector_column] = dataset[vector_column].apply(ast.literal_eval)
        logger.info(f"Parsed '{vector_column}' column from strings to lists.")
    except Exception as e:
        logger.error(f"Error parsing '{vector_column}' column: {e}")
        raise
    return dataset


def load_cache(method: str = "panphon"):
    """
    Load the processed dataset from a cache file.

    Parameters
    ----------
    - cache_file: String, path to the cache file

    Returns
    -------
    - DataFrame containing the cached dataset
    """
    repo = config.get("PHONETIC_SIM").get("EMBEDDINGS").get("REPO")
    # Remove the file extension to get the dataset name
    dataset = config.get("PHONETIC_SIM").get("IPA").get("FILE").split(".")[0]
    file = f"{dataset}_{method}.parquet"

    dataset = pd.read_parquet(
        hf_hub_download(
            repo_id=repo,
            filename=file,
            cache_dir="datasets",
            repo_type="dataset",
        )
    )
    logger.info(f"Loaded parsed dataset from '{repo}' and file {file}.")
    return dataset


def flatten_vector(vec):
    """
    Flatten a nested list of vectors into a single 1D NumPy array.

    Parameters
    ----------
    - vec: List of lists or NumPy arrays

    Returns
    -------
    - 1D NumPy array
    """
    if not vec:
        logger.warning("Encountered an empty vector. Assigning a default value of 0.")
        return np.array([0.0], dtype=np.float32)
    try:
        flattened = np.hstack(vec).astype(np.float32)
        return flattened
    except Exception as e:
        logger.error(f"Error flattening vector: {e}")
        return np.array([0.0], dtype=np.float32)


def flatten_vectors(dataset, vector_column="vectors"):
    """
    Flatten each vector in the 'vectors' column using np.hstack, handling empty vectors.

    Parameters
    ----------
    - dataset: DataFrame with 'vectors' column as lists of lists
    - vector_column: String, name of the column containing vectors

    Returns
    -------
    - DataFrame with an additional 'flattened_vectors' column
    """
    dataset["flattened_vectors"] = dataset[vector_column].apply(flatten_vector)
    logger.info(
        f"Flattened all vectors. Total vectors: {len(dataset['flattened_vectors'])}."
    )
    return dataset
