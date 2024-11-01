import logging

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from semantic import compute_similarity

from datasets import load_dataset


def evaluate_models():
    """
    Evaluates all semantic similarity models on a given dataset and reports performance metrics.

    Parameters:
        dataset_csv (str): Path to the merged semantic similarity dataset CSV file.

    Returns:
        None
    """
    # Load the dataset
    df = load_dataset(
        "StephanAkkerman/semantic-similarity", cache_dir="datasets", split="train"
    ).to_pandas()
    logging.info(f"Loaded dataset with {len(df)} entries.")

    # Ensure necessary columns exist
    required_columns = {"word1", "word2", "similarity", "dataset"}
    if not required_columns.issubset(df.columns):
        logging.error(f"Dataset must contain columns: {required_columns}")
        return

    # Initialize a list to store results
    results_list = []

    methods = ["glove", "fasttext", "minilm", "spacy"]

    for method in methods:
        logging.info(f"Evaluating method: {method}")
        computed_similarities = []
        valid_indices = []
        for idx, row in df.iterrows():
            word1 = row["word1"]
            word2 = row["word2"]
            # human_score = row["similarity"]
            try:
                sim = compute_similarity(word1, word2, method)
                # Assuming all methods are scaled to 0-1
                computed_similarities.append(sim)
                valid_indices.append(idx)
            except ValueError as e:
                logging.warning(
                    f"Skipping pair ('{word1}', '{word2}') for method '{method}': {e}"
                )
                continue

        if not computed_similarities:
            logging.warning(
                f"No valid similarity scores computed for method '{method}'. Skipping."
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
                "method": method,
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
            }
        )

        logging.info(
            f"Method '{method}': Pearson Correlation = {pearson_corr:.4f}, Spearman Correlation = {spearman_corr:.4f}"
        )

    if not results_list:
        logging.error(
            "No similarity scores were computed for any method. Evaluation aborted."
        )
        return

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Display the results
    print("\nEvaluation Results:")
    print(results_df)

    # Determine the best model based on Pearson correlation
    best_row = results_df.loc[results_df["pearson_corr"].idxmax()]
    best_method = best_row["method"]
    best_pearson = best_row["pearson_corr"]
    best_spearman = best_row["spearman_corr"]

    print(
        f"\nConclusion: The best performing model is '{best_method}' with a Pearson correlation of {best_pearson:.4f} and a Spearman correlation of {best_spearman:.4f}."
    )


if __name__ == "__main__":
    evaluate_models()
