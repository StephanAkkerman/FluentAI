import faiss
import numpy as np
import pandas as pd
from huggingface_hub import create_repo, hf_hub_download, upload_file
from sentence_transformers import SentenceTransformer

from fluentai.constants.config import config
from fluentai.services.mnemonic.phonetic.utils.vectors import (
    convert_to_matrix,
    pad_vectors,
)


def create_dataset(method: str, file: str = "en_US_filtered") -> None:
    """Creates a dataset for mnemonics.

    Parameters
    ----------
    method : str
        Options are: panphon and clts.
    file : str, optional
        Options are: en_US, en_US_filtered, and eng_latn_us_broad.
    """
    ipa = pd.read_parquet(
        hf_hub_download(
            repo_id="StephanAkkerman/english-words-IPA-embeddings",
            filename=f"{file}_{method}.parquet",
            cache_dir="datasets",
            repo_type="dataset",
            force_download=True,
        )
    )

    # Rename token_ort to word
    ipa = ipa.rename(columns={"token_ort": "word"})

    dataset_vectors_padded = pad_vectors(ipa["flattened_vectors"].tolist())

    # Convert to matrix
    dataset_matrix = convert_to_matrix(dataset_vectors_padded)

    # Normalize dataset vectors
    faiss.normalize_L2(dataset_matrix)
    ipa["matrix"] = list(dataset_matrix)

    imageability = pd.read_csv(
        hf_hub_download(
            repo_id=config.get("IMAGEABILITY").get("PREDICTIONS").get("REPO"),
            filename=config.get("IMAGEABILITY").get("PREDICTIONS").get("FILE"),
            cache_dir="datasets",
            repo_type="dataset",
        )
    )
    # Rename imageability_score to imageability
    imageability = imageability.rename(columns={"imageability_score": "imageability"})

    frequency = pd.read_csv(
        hf_hub_download(
            repo_id="StephanAkkerman/English-Age-of-Acquisition",
            filename="en.aoa.csv",
            cache_dir="datasets",
            repo_type="dataset",
        )
    )

    # Rename Word to word
    frequency = frequency.rename(columns={"Word": "word"})
    # Keep AoA_Kup_lem and Freq_pm as the columns
    frequency = frequency[["word", "AoA_Kup_lem", "Freq_pm"]]

    ipa = ipa.merge(imageability, on="word", how="left")
    ipa = ipa.merge(frequency, on="word", how="left")

    model_name = config.get("SEMANTIC_SIM").get("MODEL").lower()
    model = SentenceTransformer(
        model_name, trust_remote_code=True, cache_folder="models"
    )
    print("Computing embeddings...")
    embeddings = (
        model.encode(
            ipa["word"].tolist(),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        .cpu()
        .numpy()
    )

    ipa["word_embedding"] = list(embeddings)

    # Rename word to token_ort
    ipa = ipa.rename(columns={"word": "token_ort"})

    # Fill NaNs for AoA_Kup_lem and Freq_pm and rename
    ipa["AoA_Kup_lem"] = ipa["AoA_Kup_lem"].fillna(15)
    ipa["Freq_pm"] = ipa["Freq_pm"].fillna(0.1)

    # Scale frequency: log transform then min-max normalization
    ipa["log_freq"] = np.log(ipa["Freq_pm"] + 1)
    ipa["freq"] = (ipa["log_freq"] - ipa["log_freq"].min()) / (
        ipa["log_freq"].max() - ipa["log_freq"].min()
    )

    # Scale aoa: min-max normalization and invert (lower age -> higher score)
    ipa["aoa"] = 1 - (
        (ipa["AoA_Kup_lem"] - ipa["AoA_Kup_lem"].min())
        / (ipa["AoA_Kup_lem"].max() - ipa["AoA_Kup_lem"].min())
    )

    ipa = ipa[
        [
            "token_ort",
            "token_ipa",
            "matrix",
            "freq",
            "aoa",
            "imageability",
            "word_embedding",
        ]
    ]

    file_name = f"{file}_{method}_mnemonics.parquet"
    path = f"datasets/{file_name}"
    ipa.to_parquet(path, index=False)

    print(ipa.head())


def upload_dataset(method: str, file: str):
    file_name = f"{file}_{method}_mnemonics.parquet"
    path = f"datasets/{file_name}"
    repo_id = "StephanAkkerman/mnemonics"

    # Create the repository (if it doesn't already exist).
    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload the Parquet file to the repository.
    upload_file(
        path_or_fileobj=path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    # Options are: panphon and clts.
    METHOD = "panphon"
    # Options are: en_US, en_US_filtered, and eng_latn_us_broad.
    file = "en_US_filtered"

    create_dataset(METHOD, file)
    upload_dataset(METHOD, file)
