# train_evaluate_models.py

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_data(npz_path):
    """
    Load words, embeddings, and scores from a single .npz file.

    Args:
        npz_path (str): Path to the .npz file.

    Returns:
        tuple: (words, embeddings, scores)
    """
    print(f"Loading data from '{npz_path}'...")
    data = np.load(npz_path, allow_pickle=True)
    words = data["words"]  # Fixed-length Unicode strings
    embeddings = data["embeddings"]
    scores = data["scores"]
    print(
        f"Loaded {len(words)} words with embeddings shape {embeddings.shape} and scores shape {scores.shape}."
    )
    return embeddings, scores


def preprocess_data(embeddings, scores, task="classification"):
    """
    Preprocess the data for training.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        scores (np.ndarray): Array of scores.
        task (str, optional): 'classification' or 'regression'. Defaults to 'classification'.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if task == "classification":
        # Binarize scores based on median
        median_score = np.median(scores)
        y = (scores > median_score).astype(int)
        print(f"Binarized scores based on median value {median_score}.")
    elif task == "regression":
        y = scores
    else:
        raise ValueError("Invalid task. Choose 'classification' or 'regression'.")

    # Split the data
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=0.2, random_state=42
        )
    print("Data split into training and testing sets.")
    print(f"Training set size: {X_train.shape[0]} samples.")
    print(f"Testing set size: {X_test.shape[0]} samples.")

    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test, task="classification"):
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
    models = []
    if task == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Support Vector Machine", SVC(kernel="linear", probability=True)),
            ("Random Forest", RandomForestClassifier(n_estimators=100)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100)),
        ]
    elif task == "regression":
        models = [
            ("Ridge Regression", Ridge()),
            ("Support Vector Regression", SVR(kernel="linear")),
            ("Random Forest", RandomForestRegressor(n_estimators=100)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100)),
        ]
    else:
        raise ValueError("Invalid task. Choose 'classification' or 'regression'.")

    results = []
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if task == "classification":
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
            print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        elif task == "regression":
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results.append({"Model": name, "MSE": mse, "R2 Score": r2})
            print(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    results_df = pd.DataFrame(results)
    return results_df


def main():
    # Path to your .npz file containing words, embeddings, and scores
    npz_path = "data/imageability/fasttext_embeddings2.npz"  # Update if necessary

    # Load data
    embeddings, scores = load_data(npz_path)

    # Choose task: 'classification' or 'regression'
    task = "regression"  # Change to 'regression' if needed

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(embeddings, scores, task=task)

    # Train and evaluate models
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, task=task)

    # Display results
    print("\nModel Performance:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
