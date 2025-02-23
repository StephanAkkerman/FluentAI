import faiss
import numpy as np
import pandas as pd
from huggingface_hub import create_repo, hf_hub_download, upload_file
from sentence_transformers import SentenceTransformer

from fluentai.constants.config import config
from fluentai.services.mnemonic.phonetic.utils.cache import load_ipa
from fluentai.services.mnemonic.phonetic.utils.vectors import (
    convert_to_matrix,
    pad_vectors,
)


def create_dataset(method: str):

    # Combine the IPA dataset with the frequency and imageability dataset
    ipa = load_ipa(method)

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
    ipa["aoa"] = ipa["AoA_Kup_lem"].fillna(15)
    ipa["freq"] = ipa["Freq_pm"].fillna(0.1)

    # Scale frequency: log transform then min-max normalization
    ipa["log_freq"] = np.log(ipa["freq"] + 1)
    ipa["norm_freq"] = (ipa["log_freq"] - ipa["log_freq"].min()) / (
        ipa["log_freq"].max() - ipa["log_freq"].min()
    )

    # Scale aoa: min-max normalization and invert (lower age -> higher score)
    ipa["scaled_aoa"] = 1 - (
        (ipa["aoa"] - ipa["aoa"].min()) / (ipa["aoa"].max() - ipa["aoa"].min())
    )

    ipa = ipa[
        [
            "token_ort",
            "token_ipa",
            "matrix",
            "norm_freq",
            "scaled_aoa",
            "imageability_score",
            "word_embedding",
        ]
    ]

    file_name = f"{method}_mnemonics.parquet"
    path = f"datasets/{file_name}"
    ipa.to_parquet(path, index=False)

    print(ipa.head())


def upload_dataset(method):
    file_name = f"{method}_mnemonics.parquet"
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
    METHOD = "panphon"

    # create_dataset(METHOD)
    upload_dataset(METHOD)
