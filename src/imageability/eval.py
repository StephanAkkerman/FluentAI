import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_data(path):
    """
    Load words, embeddings, and scores from a single file.

    Args:
        path (str): Path to the .npz or .parquet file.

    Returns:
        tuple: (words, embeddings, scores)
    """
    print(f"Loading data from '{path}'...")

    # Load data from .npz or .parquet file
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        scores = data["scores"]
    elif path.endswith(".parquet"):
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{path}' was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the parquet file: {e}")

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

    print(
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
    print("Data split into training and testing sets.")
    print(f"Training set size: {X_train.shape[0]} samples.")
    print(f"Testing set size: {X_test.shape[0]} samples.")

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
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results.append({"Model": name, "MSE": mse, "R2 Score": r2})
        print(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

        # Determine the best model based on lowest MSE
        if best_metric is None or mse < best_metric:
            best_metric = mse
            best_model = model

    results_df = pd.DataFrame(results)

    # Display the results
    print("\nModel Performances:")
    print(results_df)

    # Display the best model
    if best_model:
        print(
            f"\nBest Model: {type(best_model).__name__} with 'MSE' of {best_metric:.4f}"
        )

    return results_df, best_model


def main():
    # This code also trains the models and evaluates their performance, but it is more general and can be used with any dataset.

    # Path to your .npz file containing words, embeddings, and scores
    # data/imageability/glove_embeddings.parquet
    path = "data/imageability/fasttext_embeddings_v2.parquet"  # Update if necessary

    # Load data
    embeddings, scores = load_data(path)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(embeddings, scores)

    # Train and evaluate models
    results_df, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save the best model
    if best_model:
        model_name = type(best_model).__name__
        filename = f"models/best_model_{model_name}.joblib"
        # Create the models directory if it doesn't exist
        import os

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, filename)
        print(f"\nBest model '{model_name}' saved to '{filename}'.")


if __name__ == "__main__":
    main()
