import hashlib
import io
import os
import time  # Import the time module for tracking execution time
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.services.card_gen.mnemonic.semantic.semantic import SemanticSimilarity


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


def evaluate_models():
    """
    Evaluates all semantic similarity models on a given dataset and reports performance metrics.

    Utilizes logging to avoid redundant evaluations and stores results for future reference.
    Additionally tracks and logs the execution time of each model evaluation.

    Returns
    -------
    None
    """
    # Configuration and paths
    logs_dir = "logs"
    log_file_path = os.path.join(logs_dir, "semantic_evaluation_results.csv")
    ensure_logs_directory(logs_dir)

    # Load the dataset
    dataset_name = config.get("SEMANTIC_SIM").get("EVAL").get("DATASET")
    df = load_dataset(
        dataset_name,
        cache_dir="datasets",
        split="train",
    ).to_pandas()
    logger.info(f"Loaded dataset '{dataset_name}' with {len(df)} entries.")

    # Ensure necessary columns exist
    required_columns = {"word1", "word2", "similarity", "dataset"}
    if not required_columns.issubset(df.columns):
        logger.error(f"Dataset must contain columns: {required_columns}")
        return

    # Compute a unique hash for the dataset
    dataset_hash = compute_dataset_hash(df)
    dataset_size = len(df)
    logger.debug(f"Computed dataset hash: {dataset_hash}")

    # Load existing logs
    existing_logs = load_existing_logs(log_file_path)

    # Initialize a list to store results
    results_list = []

    models = config.get("SEMANTIC_SIM").get("EVAL").get("MODELS")

    for model_name in models:
        model_name = model_name.lower()
        logger.info(f"Processing model: {model_name}")

        # Check if this model and dataset_hash combination exists in logs
        if not existing_logs.empty:
            logger.debug("Found existing logs, checking for matching entry...")
            mask = (existing_logs["dataset_hash"] == dataset_hash) & (
                existing_logs["model_name"] == model_name
            )
            existing_entry = existing_logs[mask]

        if not existing_logs.empty and not existing_entry.empty:
            # Retrieve existing results
            pearson_corr = existing_entry.iloc[0]["pearson_corr"]
            spearman_corr = existing_entry.iloc[0]["spearman_corr"]
            time_seconds = existing_entry.iloc[0].get(
                "time_seconds", None
            )  # Handle if time_seconds is missing
            results_list.append(
                {
                    "model": model_name,
                    "pearson_corr": pearson_corr,
                    "spearman_corr": spearman_corr,
                    "time_seconds": time_seconds,
                    "source": "log",
                }
            )
            logger.info(
                f"Skipped evaluation for model '{model_name}'. Loaded results from logs."
            )
        else:
            # Perform evaluation
            logger.info(f"Evaluating model '{model_name}'...")

            # Initialize the model
            model = SemanticSimilarity(model_name)

            computed_similarities = []
            valid_indices = []

            start_time = time.time()  # Start timing

            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Evaluating {model.model_name}"
            ):
                word1 = row["word1"]
                word2 = row["word2"]

                try:
                    sim = model.compute_similarity(word1, word2)
                    # Assuming all methods are scaled to 0-1
                    computed_similarities.append(sim)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Failed to compute similarity for index {idx}: {e}")

            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time  # Calculate elapsed time in seconds

            if not valid_indices:
                logger.warning(
                    f"No valid similarities computed for model '{model.model_name}'. Skipping."
                )
                continue

            # Create a DataFrame with valid entries
            evaluation_df = df.loc[valid_indices].copy()
            evaluation_df["computed_similarity"] = computed_similarities

            # Compute Pearson and Spearman correlations
            pearson_corr, _ = pearsonr(
                evaluation_df["similarity"], evaluation_df["computed_similarity"]
            )
            spearman_corr, _ = spearmanr(
                evaluation_df["similarity"], evaluation_df["computed_similarity"]
            )

            # Append the results to the list
            results_list.append(
                {
                    "model": model.model_name,
                    "pearson_corr": pearson_corr,
                    "spearman_corr": spearman_corr,
                    "time_seconds": elapsed_time,
                    "source": "evaluated",
                }
            )

            logger.info(
                f"Model '{model.model_name}': Pearson Correlation = {pearson_corr:.4f}, "
                f"Spearman Correlation = {spearman_corr:.4f}, "
                f"Time Taken = {elapsed_time:.2f} seconds."
            )

            # Append the new results to the log file
            new_log_entry = {
                "dataset_hash": dataset_hash,
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "model_name": model.model_name,
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
                "time_seconds": elapsed_time,  # Add the time taken
                "timestamp": datetime.now().isoformat(),
            }
            append_to_log(log_file_path, new_log_entry)

    if not results_list:
        logger.warning("No evaluation results to display.")
        return

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Display the results
    logger.info("\nEvaluation Results:")
    logger.info(results_df.to_string(index=False))

    # Determine the best model based on Pearson correlation
    best_row = results_df.loc[results_df["pearson_corr"].idxmax()]
    best_method = best_row["model"]
    best_pearson = best_row["pearson_corr"]
    best_spearman = best_row["spearman_corr"]
    best_time = best_row["time_seconds"]

    logger.info(
        f"\nConclusion: The best performing model is '{best_method}' with a Pearson correlation of {best_pearson:.4f}, "
        f"Spearman correlation of {best_spearman:.4f}, and processed the dataset in {best_time:.2f} seconds."
    )


if __name__ == "__main__":
    evaluate_models()
