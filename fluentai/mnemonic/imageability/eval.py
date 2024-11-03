import os

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

from fluentai.constants.config import config
from fluentai.utils.logger import logger


def load_data():
    """
    Load words, embeddings, and scores from a single file.

    Returns:
        tuple: (words, embeddings, scores)
    """

    df = pd.read_parquet(
        hf_hub_download(
            config.get("IMAGEABILITY").get("EMBEDDINGS_REPO"),
            filename=config.get("IMAGEABILITY").get("EMBEDDINGS_FILE"),
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
    return embeddings, scores


def preprocess_data(embeddings, scores):
    """
    Preprocess the data for training.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        scores (np.ndarray): Array of scores.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, scores, test_size=0.2, random_state=42
    )
    logger.info("Data split into training and testing sets.")
    logger.info(f"Training set size: {X_train.shape[0]} samples.")
    logger.info(f"Testing set size: {X_test.shape[0]} samples.")

    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate their performance.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        task (str, optional): 'classification' or 'regression'. Defaults to 'classification'.

    Returns:
        pd.DataFrame: DataFrame containing model performances.
    """
    models = [
        ("Linear Regression (OLS)", LinearRegression()),
        ("Ridge Regression", Ridge()),
        ("Support Vector Regression", SVR(kernel="linear")),
        ("Random Forest", RandomForestRegressor(n_estimators=100)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100)),
        (
            "XGBoost",
            XGBRegressor(n_estimators=100, use_label_encoder=False, eval_metric="rmse"),
        ),
        ("LightGBM", LGBMRegressor(n_estimators=100)),
    ]

    results = []
    best_model = None
    best_metric = None

    for name, model in tqdm(models, desc="Training Models", unit="model"):
        logger.info(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results.append({"Model": name, "MSE": mse, "R2 Score": r2})
        logger.info(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

        # Determine the best model based on lowest MSE
        if best_metric is None or mse < best_metric:
            best_metric = mse
            best_model = model

    results_df = pd.DataFrame(results)

    # Display the results
    logger.info("\nModel Performances:")
    logger.info(results_df)

    # Display the best model
    if best_model:
        logger.info(
            f"\nBest Model: {type(best_model).__name__} with 'MSE' of {best_metric:.4f}"
        )

    return results_df, best_model


def main():
    # Load data
    embeddings, scores = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(embeddings, scores)

    # Train and evaluate models
    results_df, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save the best model
    if best_model:
        model_name = type(best_model).__name__
        filename = f"models/best_model_{model_name}.joblib"
        # Create the models directory if it doesn't exist

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, filename)
        logger.info(f"\nBest model '{model_name}' saved to '{filename}'.")


if __name__ == "__main__":
    main()
