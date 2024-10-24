# evaluate_similarity.py

import pandas as pd
from orthographic import compute_similarity
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler


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
    # Specify the input file path
    input_file = "data/orthographic/AWL_Data.csv"

    # Load the dataset
    df = pd.read_csv(input_file)

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
    print("Computing similarity scores...")
    df = compute_all_similarities(df, methods)
    print("Similarity scores computed.\n")

    # Evaluate methods
    print("Evaluating similarity methods against human ratings...")
    evaluation_results = evaluate_methods(df, scaled_ratings, methods)
    print("Evaluation completed.\n")

    # Print the evaluation results
    print("=== Evaluation Results ===")
    print(evaluation_results.to_string(index=False))
    print()

    # Determine and print the best method
    best_method = determine_best_method(evaluation_results)
    best_metrics = evaluation_results[evaluation_results["Method"] == best_method].iloc[
        0
    ]

    print("=== Conclusion ===")
    print(f"The best orthographic similarity method is **{best_method}**.")
    print(
        f"Pearson Correlation: {best_metrics['Pearson Correlation']:.4f} (p-value: {best_metrics['Pearson p-value']:.2e})"
    )
    print(
        f"Spearman Correlation: {best_metrics['Spearman Correlation']:.4f} (p-value: {best_metrics['Spearman p-value']:.2e})"
    )


if __name__ == "__main__":
    main()
