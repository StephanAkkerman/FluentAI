import os
import sys

import pandas as pd
from orthographic import compute_similarity
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

from datasets import load_dataset
from fluentai.utils.logger import logger


def scale_ratings(ratings: pd.Series) -> pd.Series:
    """
    Scales the ratings to a 0-100 range using MinMaxScaler.

    Parameters:
    - ratings (pd.Series): Original ratings.

    Returns:
    - pd.Series: Scaled ratings.
    """
    scaler = MinMaxScaler(feature_range=(0, 100))
    # Reshape ratings to a 2D array as required by MinMaxScaler
    scaled = scaler.fit_transform(ratings.values.reshape(-1, 1)).flatten()
    return pd.Series(scaled, index=ratings.index)


def compute_all_similarities(df: pd.DataFrame, methods: list) -> pd.DataFrame:
    """
    Computes similarity scores for all methods and adds them as new columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing word pairs.
    - methods (list): List of similarity methods to compute.

    Returns:
    - pd.DataFrame: DataFrame with new similarity score columns.
    """
    for method in methods:
        similarity_scores = df.apply(
            lambda row: compute_similarity(
                row["English Cognate"], row["Spanish Cognate"], method
            ),
            axis=1,
        )
        df[f"sim_{method}"] = similarity_scores
    return df


def evaluate_methods(
    df: pd.DataFrame, scaled_ratings: pd.Series, methods: list
) -> pd.DataFrame:
    """
    Evaluates each similarity method by computing Pearson and Spearman correlations.

    Parameters:
    - df (pd.DataFrame): DataFrame containing similarity scores.
    - scaled_ratings (pd.Series): Scaled human ratings.
    - methods (list): List of similarity methods evaluated.

    Returns:
    - pd.DataFrame: DataFrame containing correlation results.
    """
    results = []
    for method in methods:
        sim_column = f"sim_{method}"
        pearson_corr, pearson_p = pearsonr(df[sim_column], scaled_ratings)
        spearman_corr, spearman_p = spearmanr(df[sim_column], scaled_ratings)
        results.append(
            {
                "Method": method,
                "Pearson Correlation": pearson_corr,
                "Pearson p-value": pearson_p,
                "Spearman Correlation": spearman_corr,
                "Spearman p-value": spearman_p,
            }
        )
    results_df = pd.DataFrame(results)
    return results_df


def determine_best_method(evaluation_df: pd.DataFrame) -> str:
    """
    Determines the best similarity method based on the highest Pearson correlation.
    In case of a tie, uses Spearman correlation as a secondary criterion.

    Parameters:
    - evaluation_df (pd.DataFrame): DataFrame containing correlation results.

    Returns:
    - str: The name of the best similarity method.
    """
    # Sort methods by Pearson Correlation descending
    sorted_df = evaluation_df.sort_values(
        by=["Pearson Correlation", "Spearman Correlation"], ascending=False
    )

    # The top method is the first row after sorting
    best_method = sorted_df.iloc[0]["Method"]
    return best_method


def main():
    from fluentai.utils.constants import config

    # Load the dataset
    df = load_dataset(
        config.get("ORTHOGRAPHIC_SIM").get("EVAL"),
        cache_dir="datasets",
        split="train",
    ).to_pandas()

    # Check required columns
    required_columns = ["English Cognate", "Spanish Cognate", "Mean Rating"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Handle missing data by dropping rows with missing values in required columns
    df = df.dropna(subset=required_columns)

    # Scale the mean ratings to 0-100
    scaled_ratings = scale_ratings(df["Mean Rating"])
    df["Scaled Mean Rating"] = scaled_ratings

    # Define similarity methods to evaluate
    methods = [
        "difflib",
        "rapidfuzz_ratio",
        "rapidfuzz_partial_ratio",
        "damerau_levenshtein",
        "levenshtein",
    ]

    # Compute similarity scores
    logger.info("Computing similarity scores...")
    df = compute_all_similarities(df, methods)
    logger.info("Similarity scores computed.\n")

    # Evaluate methods
    logger.info("Evaluating similarity methods against human ratings...")
    evaluation_results = evaluate_methods(df, scaled_ratings, methods)
    logger.info("Evaluation completed.\n")

    # Print the evaluation results
    logger.info("=== Evaluation Results ===")
    logger.info(evaluation_results.to_string(index=False))
    logger.info()

    # Determine and print the best method
    best_method = determine_best_method(evaluation_results)
    best_metrics = evaluation_results[evaluation_results["Method"] == best_method].iloc[
        0
    ]

    logger.info("=== Conclusion ===")
    logger.info(f"The best orthographic similarity method is **{best_method}**.")
    logger.info(
        f"Pearson Correlation: {best_metrics['Pearson Correlation']:.4f} (p-value: {best_metrics['Pearson p-value']:.2e})"
    )
    logger.info(
        f"Spearman Correlation: {best_metrics['Spearman Correlation']:.4f} (p-value: {best_metrics['Spearman p-value']:.2e})"
    )


if __name__ == "__main__":
    sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..", "..")))
    main()
