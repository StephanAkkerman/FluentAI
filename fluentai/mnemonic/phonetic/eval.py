import faiss
import numpy as np
import pandas as pd
from g2p import g2p
from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from datasets import load_dataset
from fluentai.mnemonic.phonetic.ipa2vec import panphon_vec, soundvec
from fluentai.utils.logger import logger


def word2ipa(
    word: str,
    ipa_dataset: pd.DataFrame,
    use_fallback: bool = True,
) -> str:

    if not ipa_dataset.empty:
        # Check if the word is in the dataset
        ipa = ipa_dataset[ipa_dataset["token_ort"] == word]["token_ipa"]

    if ipa.empty:
        # logger.info(f"{word} not found in dataset.")
        if use_fallback:
            # Fallback on the g2p model
            return g2p([f"<eng-us>:{word}"])[0]
        else:
            return
    # Remove whitespace and return the IPA transcription
    return ipa.values[0].replace(" ", "")


def compute_phonetic_similarity(
    word1: str, word2: str, ipa_dataset: pd.DataFrame, method: str = "panphon"
) -> float:
    """
    Computes the phonetic similarity between two words using the specified method.

    Parameters:
        word1 (str): The first word.
        word2 (str): The second word.
        method (str): The similarity method to use. Options:
            - 'panphon'
            - 'clts' (if implemented)

    Returns:
        float: Phonetic similarity score between 0 and 1.

    Raises:
        ValueError: If an unsupported method is provided or words cannot be vectorized.
    """
    method = method.lower()
    if method not in ["panphon", "clts"]:
        raise ValueError(f"Unsupported method '{method}'. Choose 'panphon' or 'clts'.")

    # Convert word to IPA transcription
    word1 = word2ipa(word1, ipa_dataset=ipa_dataset)
    word2 = word2ipa(word2, ipa_dataset=ipa_dataset)

    # Skip if either is None
    if word1 is None or word2 is None:
        return

    if method == "clts":
        vectorizer = soundvec
    elif method == "panphon":
        vectorizer = panphon_vec

    # Vectorize both words
    try:
        word1_vector = np.hstack(vectorizer(word1)).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error vectorizing word '{word1}': {e}")

    try:
        word2_vector = np.hstack(vectorizer(word2)).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error vectorizing word '{word2}': {e}")

    # Determine the required dimension (assume dataset has already been loaded)
    # For consistency, use the same dimension as the dataset vectors
    # You might need to adjust this based on your dataset's dimension
    dimension = 300  # Example dimension; replace with actual if different

    # Pad or truncate vectors to match the dataset's dimension
    def pad_or_truncate(vector, dimension):
        if len(vector) > dimension:
            return vector[:dimension]
        elif len(vector) < dimension:
            return np.pad(vector, (0, dimension - len(vector)), "constant")
        else:
            return vector

    word1_padded = pad_or_truncate(word1_vector, dimension).reshape(1, -1)
    word2_padded = pad_or_truncate(word2_vector, dimension).reshape(1, -1)

    # Normalize vectors
    faiss.normalize_L2(word1_padded)
    faiss.normalize_L2(word2_padded)

    # Compute cosine similarity
    similarity = cosine_similarity(word1_padded, word2_padded)[0][0]

    return similarity


def evaluate_phonetic_similarity(
    dataset_repo: str, ipa_repo: str, ipa_file: str, methods: list
):
    """
    Evaluates multiple phonetic similarity models on a given dataset and reports performance metrics.

    Parameters:
        dataset_csv (str): Path to the phonetic similarity dataset CSV file.
        methods (list): List of phonetic similarity methods to evaluate (e.g., ['panphon', 'clts']).

    Returns:
        None
    """
    # Load the dataset
    df = load_dataset(dataset_repo, cache_dir="datasets")["train"].to_pandas()
    logger.info(f"Loaded dataset with {len(df)} entries.")

    ipa_dataset = pd.read_csv(
        hf_hub_download(
            repo_id=ipa_repo,
            filename=ipa_file,
            cache_dir="datasets",
            repo_type="dataset",
        )
    )

    # Ensure necessary columns exist
    required_columns = {"word1", "word2", "obtained"}
    if not required_columns.issubset(df.columns):
        logger.error(f"Dataset must contain columns: {required_columns}")
        return

    # Scale the 'obtained' scores to 0-1
    scaler = MinMaxScaler()
    df["obtained_scaled"] = scaler.fit_transform(df[["obtained"]])
    logger.info("Scaled 'obtained' scores to a 0-1 range.")

    # Initialize a list to store results for each method
    results_list = []

    for method in methods:
        computed_similarities = []
        valid_indices = []

        # Iterate through each row to compute similarity
        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Computing similarities for method '{method}'",
        ):
            word1 = row["word1"]
            word2 = row["word2"]
            try:
                sim = compute_phonetic_similarity(word1, word2, ipa_dataset, method)
                if sim is None:
                    continue
                computed_similarities.append(sim)
                valid_indices.append(idx)
            except ValueError as e:
                logger.warning(
                    f"Skipping pair ('{word1}', '{word2}') for method '{method}': {e}"
                )
                continue

        if not computed_similarities:
            logger.warning(
                f"No similarity scores were computed for method '{method}'. Skipping."
            )
            continue

        # Create a DataFrame with valid entries
        evaluation_df = df.loc[valid_indices].copy()
        evaluation_df["computed_similarity"] = computed_similarities

        # Compute Pearson and Spearman correlations
        pearson_corr, _ = pearsonr(
            evaluation_df["obtained_scaled"], evaluation_df["computed_similarity"]
        )
        spearman_corr, _ = spearmanr(
            evaluation_df["obtained_scaled"], evaluation_df["computed_similarity"]
        )

        # Append the results to the list
        results_list.append(
            {
                "method": method,
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
            }
        )

        logger.info(
            f"Method '{method}': Pearson Correlation = {pearson_corr:.4f}, Spearman Correlation = {spearman_corr:.4f}"
        )

    if not results_list:
        logger.error(
            "No similarity scores were computed for any method. Evaluation aborted."
        )
        return

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Display the results
    logger.info("\nPhonetic Similarity Evaluation Results:")
    logger.info(results_df.to_string(index=False))

    # Determine the best method based on Pearson correlation
    best_method_row = results_df.loc[results_df["pearson_corr"].idxmax()]
    best_method = best_method_row["method"]
    best_pearson = best_method_row["pearson_corr"]
    best_spearman = best_method_row["spearman_corr"]

    logger.info(
        f"\nConclusion: The best performing method is '{best_method}' with a Pearson correlation of {best_pearson:.4f} and a Spearman correlation of {best_spearman:.4f}."
    )


def main():
    """
    Main function to evaluate phonetic similarity methods on a dataset.
    """
    # Define the dataset path and methods to evaluate
    dataset_repo = "StephanAkkerman/english-words-human-similarity"
    ipa_repo = "StephanAkkerman/english-words-IPA"
    ipa_file = "en_US.csv"
    methods = ["panphon", "clts"]  # Add more methods here if needed

    # Call the evaluation function
    evaluate_phonetic_similarity(dataset_repo, ipa_repo, ipa_file, methods)


if __name__ == "__main__":
    main()
