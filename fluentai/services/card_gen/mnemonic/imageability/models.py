import hashlib
import io
import os
from datetime import datetime

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from huggingface_hub import HfApi, hf_hub_download
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


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


def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_hash):
    """
    Train multiple models and evaluate their performance, utilizing logging to avoid redundant evaluations.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        dataset_hash (str): Unique identifier for the dataset.

    Returns
    -------
        pd.DataFrame: DataFrame containing model performances.
    """
    # Define logs directory and log file
    log_file_path = os.path.join("logs", "imageability_evaluation_results.csv")
    embedding_model = config.get("IMAGEABILITY").get("EMBEDDINGS").get("MODEL")
    upload_to_hf = False

    # Define the models to evaluate
    models = [
        ("Linear Regression (OLS)", LinearRegression()),  # Baseline
        ("Ridge Regression", Ridge()),  # Baseline
        ("Support Vector Regression", SVR(kernel="linear")),
        ("Random Forest", RandomForestRegressor(n_estimators=100)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100)),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric="rmse",
                device="gpu",
            ),
        ),
        ("LightGBM", LGBMRegressor(n_estimators=100)),
        ("CatBoost", CatBoostRegressor(n_estimators=100)),
    ]

    # Load existing logs
    existing_logs = load_existing_logs(log_file_path)

    results = []
    new_evaluations = []
    best_model = None
    best_metric = None

    # Iterate over each model
    for name, model in tqdm(models, desc="Processing Models", unit="model"):
        # Check if this model and dataset_hash combination exists in logs
        if not existing_logs.empty:
            mask = (existing_logs["dataset_hash"] == dataset_hash) & (
                existing_logs["model_name"] == name
            )
            existing_entry = existing_logs[mask]

        if not existing_logs.empty and not existing_entry.empty:
            # Retrieve existing results
            mse = existing_entry.iloc[0]["MSE"]
            r2 = existing_entry.iloc[0]["R2 Score"]
            source = "log"
            logger.info(f"Skipped training for '{name}'. Loaded results from logs.")
        else:
            # Train and evaluate the model
            logger.info(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            logger.info(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

            # Append the results to the new evaluations list
            new_evaluations.append(
                {
                    "dataset_hash": dataset_hash,
                    "dataset_name": config.get("IMAGEABILITY")
                    .get("PREDICTOR")
                    .get("EVAL")
                    .get("DATASET"),
                    "embedding_model": embedding_model,
                    "model_name": name,
                    "MSE": mse,
                    "R2 Score": r2,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            source = "evaluated"

            # Optionally, save individual models if desired
            # For consistency, you might choose to save all evaluated models or only the best one

        # Append to the results list
        results.append({"Model": name, "MSE": mse, "R2 Score": r2, "Source": source})

        # Determine the best model based on lowest MSE
        if best_metric is None or mse < best_metric:
            best_metric = mse
            best_model = (
                (name, model) if source == "evaluated" else (name, None)
            )  # Handle if model is from log

    # Append new evaluations to the log file
    if new_evaluations:
        for entry in new_evaluations:
            append_to_log(log_file_path, entry)

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Display the results
    logger.info("\nModel Performances:")
    logger.info(results_df.to_string(index=False))

    # Determine the best model considering both logged and newly evaluated models
    if not results_df.empty:
        best_row = results_df.loc[results_df["MSE"].idxmin()]
        best_method = best_row["Model"]
        best_mse = best_row["MSE"]
        best_r2 = best_row["R2 Score"]

        logger.info(
            f"\nConclusion: The best performing model is '{best_method}' with an MSE of {best_mse:.4f} and an R2 Score of {best_r2:.4f}."
        )

        # Optionally, load the best model from logs or save the newly trained best model
        if best_row["Source"] == "evaluated":
            # Save the best model
            best_model_instance = best_model[1]
            model_name_clean = best_method.replace(" ", "_").lower()
            filename = (
                f"models/{model_name_clean}-{embedding_model.replace('/', '_')}.joblib"
            )
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model_instance, filename)
            logger.info(f"Best model '{best_method}' saved to '{filename}'.")
            upload_to_hf = True
        else:
            logger.info(
                f"The best model '{best_method}' was retrieved from existing logs."
            )

    else:
        logger.warning("No model performances to display.")

    if upload_to_hf:
        # Upload the best model to Hugging Face Hub
        upload_model(filename)


def upload_model(model_path: str):
    """
    Upload model to Hugging Face Hub.
    """
    file_name = model_path.split("/")[-1]
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=file_name,
        repo_id=config.get("IMAGEABILITY").get("PREDICTOR").get("REPO"),
        repo_type="model",
    )
    logger.info("Model uploaded to Hugging Face Hub")


if __name__ == "__main__":
    # Install the following extra dependencies:
    # pip install scikit-learn lightgbm xgboost catboost

    # Load data
    embeddings, scores, dataset_hash = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = split_dataset(embeddings, scores)

    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_hash)

    # TODO: hyperparameter optimization
    # TODO: ensemble methods
