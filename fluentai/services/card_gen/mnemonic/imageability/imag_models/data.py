import hashlib
import io
import json
import os
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


def upload_model(model_path: str):
    """
    Upload model to Hugging Face Hub.

    Parameters
    ----------
    model_path : str
        The path to the model file to upload.
    """
    file_name = os.path.basename(model_path)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=file_name,
        repo_id=config.get("IMAGEABILITY").get("PREDICTOR").get("REPO"),
        repo_type="model",
    )
    logger.info("Model uploaded to Hugging Face Hub")


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Computes a unique hash for the given DataFrame based on its content.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to hash.

    Returns
    -------
    str
        The MD5 hash of the DataFrame.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return hashlib.md5(buffer.getvalue().encode("utf-8")).hexdigest()


def ensure_logs_directory(logs_dir: str):
    """
    Ensures that the logs directory exists.

    Parameters
    ----------
    logs_dir : str
        The path to the logs directory.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logger.debug(f"Ensured that the logs directory '{logs_dir}' exists.")


def load_existing_logs(log_file_path: str) -> pd.DataFrame:
    """
    Loads existing evaluation logs if the log file exists.

    Parameters
    ----------
    log_file_path : str
        The path to the log file.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing existing logs.
    """
    if os.path.exists(log_file_path):
        try:
            existing_logs = pd.read_csv(log_file_path)
            logger.info(f"Loaded existing logs from '{log_file_path}'.")
            return existing_logs
        except Exception as e:
            logger.error(f"Failed to read log file '{log_file_path}': {e}")
            return pd.DataFrame()
    else:
        logger.info(f"No existing log file found at '{log_file_path}'. Starting fresh.")
        return pd.DataFrame()


def append_to_log(log_file_path: str, new_entry: dict):
    """
    Appends a new evaluation entry to the log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file.
    new_entry : dict
        The evaluation results to append.
    """
    try:
        df_new = pd.DataFrame([new_entry])
        if os.path.exists(log_file_path):
            df_new.to_csv(log_file_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(log_file_path, mode="w", header=True, index=False)
        logger.debug(f"Appended new entry to log file '{log_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to append to log file '{log_file_path}': {e}")


def load_existing_hyperparameters_log(log_file_path: str) -> dict:
    """
    Loads existing hyperparameter trials from the log file.

    Parameters
    ----------
    log_file_path : str
        The path to the hyperparameter log file.

    Returns
    -------
    dict
        A dictionary mapping model names to their tried hyperparameter hashes and objective values.
    """
    if os.path.exists(log_file_path):
        try:
            df = pd.read_csv(log_file_path)
            # Create a dict: model_name -> {hyperparam_hash: objective_value}
            log_dict = {}
            for _, row in df.iterrows():
                model = row["model_name"]
                hyperparam_hash = row["hyperparam_hash"]
                objective_value = row["objective_value"]
                if model not in log_dict:
                    log_dict[model] = {}
                log_dict[model][hyperparam_hash] = objective_value
            logger.info(f"Loaded existing hyperparameter log from '{log_file_path}'.")
            return log_dict
        except Exception as e:
            logger.error(f"Failed to read hyperparameter log '{log_file_path}': {e}")
            return {}
    else:
        logger.info(
            f"No existing hyperparameter log found at '{log_file_path}'. Starting fresh."
        )
        return {}


def append_hyperparameters_log(
    log_file_path: str, model_name: str, hyperparams: dict, objective_value: float
):
    """
    Appends a new hyperparameter trial to the hyperparameter log file.

    Parameters
    ----------
    log_file_path : str
        The path to the hyperparameter log file.
    model_name : str
        The name of the model.
    hyperparams : dict
        The hyperparameters of the trial.
    objective_value : float
        The objective value of the trial.
    """
    try:
        # Compute hyperparam hash
        sorted_hyperparams = sorted(hyperparams.items())
        hyperparam_str = json.dumps(sorted_hyperparams, sort_keys=True)
        hyperparam_hash = hashlib.md5(hyperparam_str.encode("utf-8")).hexdigest()

        # Create a new entry
        new_entry = {
            "model_name": model_name,
            "hyperparam_hash": hyperparam_hash,
            "hyperparameters": json.dumps(hyperparams),
            "objective_value": objective_value,
            "timestamp": datetime.now().isoformat(),
        }
        df_new = pd.DataFrame([new_entry])
        if os.path.exists(log_file_path):
            df_new.to_csv(log_file_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(log_file_path, mode="w", header=True, index=False)
        logger.debug(f"Appended hyperparameter trial to log file '{log_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to append hyperparameter log to '{log_file_path}': {e}")


def load_data(local_file: bool = False, filename: str = None):
    """
    Load words, embeddings, and scores from a single file.

    Returns
    -------
        tuple: (embeddings, scores, dataset_hash)
    """
    if local_file:
        if filename is None:
            raise ValueError("Filename must be provided when local_file is True.")
        df = pd.read_parquet(filename)
    else:
        if filename is None:
            model = config.get("IMAGEABILITY").get("EMBEDDINGS").get("MODEL")
            filename = f"{model.replace('/', '_')}_embeddings.parquet"
        logger.info(f"Loading embeddings from {filename}...")
        df = pd.read_parquet(
            hf_hub_download(
                config.get("IMAGEABILITY").get("EMBEDDINGS").get("REPO"),
                filename=filename,
                cache_dir="datasets",
                repo_type="dataset",
            )
        )

    # Verify required columns exist
    required_columns = {"word", "score"}
    embedding_columns = [col for col in df.columns if col.startswith("emb_")]
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns in parquet file: {missing}")
    if not embedding_columns:
        raise ValueError(
            "No embedding columns found. Ensure embeddings are named starting with 'emb_'."
        )

    # Extract embeddings and scores
    embeddings = df[embedding_columns].values
    scores = df["score"].values

    logger.info(
        f"Loaded {len(scores)} words with embeddings shape {embeddings.shape} and scores shape {scores.shape}."
    )

    # Compute dataset hash
    dataset_hash = compute_dataset_hash(df)
    logger.debug(f"Computed dataset hash: {dataset_hash}")

    # TODO: add data validation to check if the embeddings shape == imageabillity dataset shape (6090 rows)

    return embeddings, scores, dataset_hash


def split_dataset(embeddings, scores):
    """
    Preprocess the data for training.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        scores (np.ndarray): Array of scores.

    Returns
    -------
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, scores, test_size=0.2, random_state=42
    )
    logger.info("Data split into training and testing sets.")
    logger.info(f"Training set size: {X_train.shape[0]} samples.")
    logger.info(f"Testing set size: {X_test.shape[0]} samples.")

    return X_train, X_test, y_train, y_test
