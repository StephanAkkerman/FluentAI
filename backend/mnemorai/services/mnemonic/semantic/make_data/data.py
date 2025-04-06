import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from mnemorai.logger import logger


# Function to scale similarity scores to 0-1 range
def scale_similarity(df, similarity_col="similarity"):
    """
    Scales the similarity scores in the specified column of the DataFrame to a 0-1 range.

    Parameters
    ----------
        df (pd.DataFrame): The DataFrame containing similarity scores.
        similarity_col (str): The name of the column to scale.

    Returns
    -------
        pd.DataFrame: The DataFrame with scaled similarity scores.
    """
    # Check if the similarity column exists
    if similarity_col not in df.columns:
        raise ValueError(f"Column '{similarity_col}' not found in DataFrame.")

    # Ensure the similarity scores are numeric
    if not pd.api.types.is_numeric_dtype(df[similarity_col]):
        raise TypeError(f"Column '{similarity_col}' must be numeric.")

    # Initialize a new MinMaxScaler for each dataset
    scaler = MinMaxScaler()

    # Reshape the data for scaler
    similarity_values = df[[similarity_col]].values

    # Fit and transform the similarity scores
    df[similarity_col] = scaler.fit_transform(similarity_values)

    return df


# 1. Load and Prepare SimLex-999 Dataset
simlex999 = pd.read_csv("local_data/semantic/SimLex-999.txt", sep="\t")
simlex999 = simlex999[["word1", "word2", "SimLex999"]]
simlex999 = simlex999.rename(columns={"SimLex999": "similarity"})
simlex999["dataset"] = "SimLex-999"

# Scale SimLex-999 similarity scores independently
simlex999 = scale_similarity(simlex999, similarity_col="similarity")

# 2. Load and Prepare SimVerb-3500 Dataset
simverb3500 = pd.read_csv(
    "local_data/semantic/SimVerb-3500.txt",
    sep="\t",
    names=["word1", "word2", "POS", "similarity", "relatedness"],
)
simverb3500 = simverb3500[["word1", "word2", "similarity"]]
simverb3500["dataset"] = "SimVerb-3500"

# Scale SimVerb-3500 similarity scores independently
simverb3500 = scale_similarity(simverb3500, similarity_col="similarity")

# 3. Load and Prepare WordSim-353 Dataset
wordsim353 = pd.read_csv("local_data/semantic/wordsim-353.csv")
wordsim353 = wordsim353.rename(
    columns={"Word 1": "word1", "Word 2": "word2", "Human (mean)": "similarity"}
)
wordsim353["dataset"] = "WordSim-353"

# Scale WordSim-353 similarity scores independently
wordsim353 = scale_similarity(wordsim353, similarity_col="similarity")

# 4. Merge the Datasets
merged_df = pd.concat([simlex999, simverb3500, wordsim353], ignore_index=True)

# 5. Handle Duplicates
# Check for duplicate word pairs within the same dataset
duplicate_mask = merged_df.duplicated(
    subset=["word1", "word2", "dataset"], keep="first"
)
num_duplicates = duplicate_mask.sum()
if num_duplicates > 0:
    logger.info(f"Found {num_duplicates} duplicate entries. Removing duplicates...")
    merged_df = merged_df[~duplicate_mask]
else:
    logger.info("No duplicate entries found within the same dataset.")

# Optionally, handle duplicate word pairs across different datasets
# For example, if you want to keep all instances, do nothing.
# If you want to average the similarities across datasets, you can do so.
# Here, we'll keep all instances as separate entries since they belong to different datasets.

# 6. Save the Merged Dataset to a CSV File
output_file = "local_data/semantic/semantic_similarity.csv"
merged_df.to_csv(output_file, index=False)
logger.info(f"Merged dataset saved to '{output_file}'.")

# Optional: Display a preview of the merged DataFrame
logger.info("\nPreview of the merged dataset:")
logger.info(merged_df.head())
