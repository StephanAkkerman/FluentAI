import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

from fluentai.constants.config import config
from fluentai.mnemonic.semantic.semantic import SemanticSimilarity
from fluentai.utils.logger import logger


def evaluate_models():
    """
    Evaluates all semantic similarity models on a given dataset and reports performance metrics.

    Parameters
    ----------
        dataset_csv (str): Path to the merged semantic similarity dataset CSV file.

    Returns
    -------
        None
    """
    # Load the dataset
    df = load_dataset(
        config.get("SEMANTIC_SIM").get("EVAL"), cache_dir="datasets", split="train"
    ).to_pandas()
    logger.info(f"Loaded dataset with {len(df)} entries.")

    # Ensure necessary columns exist
    required_columns = {"word1", "word2", "similarity", "dataset"}
    if not required_columns.issubset(df.columns):
        logger.error(f"Dataset must contain columns: {required_columns}")
        return

    # Initialize a list to store results
    results_list = []

    models = config.get("SEMANTIC_SIM").get("EVAL").get("MODELS")

    # Create the model objects
    semantic_models = [SemanticSimilarity(model) for model in models]

    for model in semantic_models:
        logger.info(f"Evaluating method: {model.model_name}")
        computed_similarities = []
        valid_indices = []
        # TODO: add tqdm here
        for idx, row in df.iterrows():
            word1 = row["word1"]
            word2 = row["word2"]
            # human_score = row["similarity"]

            sim = model.compute_similarity(word1, word2)

            # Assuming all methods are scaled to 0-1
            computed_similarities.append(sim)
            valid_indices.append(idx)

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
            }
        )

        logger.info(
            f"Model '{model.model_name}': Pearson Correlation = {pearson_corr:.4f}, Spearman Correlation = {spearman_corr:.4f}"
        )

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Display the results
    logger.info("\nEvaluation Results:")
    logger.info(results_df)

    # Determine the best model based on Pearson correlation
    best_row = results_df.loc[results_df["pearson_corr"].idxmax()]
    best_method = best_row["method"]
    best_pearson = best_row["pearson_corr"]
    best_spearman = best_row["spearman_corr"]

    logger.info(
        f"\nConclusion: The best performing model is '{best_method}' with a Pearson correlation of {best_pearson:.4f} and a Spearman correlation of {best_spearman:.4f}."
    )


if __name__ == "__main__":
    evaluate_models()
