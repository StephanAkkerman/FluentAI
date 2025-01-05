import pandas as pd
from huggingface_hub import hf_hub_download

from fluentai.constants.config import config
from fluentai.logger import logger


def load_from_cache(method: str = "panphon"):
    """
    Load the processed dataset from a cache file.

    Parameters
    ----------
    - cache_file: String, path to the cache file

    Returns
    -------
    - DataFrame containing the cached dataset
    """
    logger.debug("Loading the cached dataset from Huggingface")

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
