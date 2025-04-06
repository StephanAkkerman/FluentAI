import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ================================
# 1. Load and Preprocess Datasets
# ================================

# Load Dataset 1: ICMR 2020
df_icmr = pd.read_csv("local_data/imageability/raw/corpus_icmr2020.csv")
df_icmr.columns = df_icmr.columns.str.lower()

# Select relevant columns
df_icmr = df_icmr[["word", "visual", "phonetic", "textual"]]

# Preprocess 'word' column
df_icmr["word"] = (
    df_icmr["word"].str.lower().str.strip().str.replace(r"[^a-zA-Z\s]", "", regex=True)
)

# Remove duplicates and drop rows with missing scores
df_icmr = df_icmr.drop_duplicates(subset="word").dropna(
    subset=["visual", "phonetic", "textual"]
)

print(f"ICMR Dataset: {df_icmr.shape[0]} words loaded.")

# Load Dataset 2: MIPR 2021 (JSON format)
df_mipr = pd.read_json("local_data/imageability/raw/corpus_mipr2021.json").T
df_mipr.reset_index(inplace=True)
df_mipr.rename(columns={"index": "word"}, inplace=True)
df_mipr.columns = df_mipr.columns.str.lower()

# Select relevant columns
df_mipr = df_mipr[["word", "visual", "phonetic", "textual"]]

# Preprocess 'word' column
df_mipr["word"] = (
    df_mipr["word"].str.lower().str.strip().str.replace(r"[^a-zA-Z\s]", "", regex=True)
)

# Remove duplicates and drop rows with missing scores
df_mipr = df_mipr.drop_duplicates(subset="word").dropna(
    subset=["visual", "phonetic", "textual"]
)

print(f"MIPR Dataset: {df_mipr.shape[0]} words loaded.")

# ================================
# 2. Scale the Score Columns Individually
# ================================

# Initialize scalers for each dataset
scaler_icmr = MinMaxScaler()
scaler_mipr = MinMaxScaler()

# Scale ICMR scores
df_icmr_scaled = df_icmr.copy()
df_icmr_scaled[["visual", "phonetic", "textual"]] = scaler_icmr.fit_transform(
    df_icmr[["visual", "phonetic", "textual"]]
)

print("ICMR scores scaled using MinMaxScaler.")

# Scale MIPR scores
df_mipr_scaled = df_mipr.copy()
df_mipr_scaled[["visual", "phonetic", "textual"]] = scaler_mipr.fit_transform(
    df_mipr[["visual", "phonetic", "textual"]]
)

print("MIPR scores scaled using MinMaxScaler.")

# ================================
# 3. Merge the Datasets Using a Full Outer Join
# ================================

# Merge the two scaled datasets using a full outer join
merged_df = pd.merge(
    df_icmr_scaled, df_mipr_scaled, on="word", how="outer", suffixes=("_icmr", "_mipr")
)

print(f"Total words after full outer join: {merged_df.shape[0]}")

# ================================
# 4. Combine the Scaled Scores
# ================================


# Function to combine scores by averaging available scores
def combine_scores(row, score_type) -> float:
    """
    Combine scores by averaging available scores.

    Parameters
    ----------
    row : _type_
        The row containing the scores.
    score_type : _type_
        The type of score to combine.

    Returns
    -------
    float
        The combined score.
    """
    score_icmr = row.get(f"{score_type}_icmr", pd.NA)
    score_mipr = row.get(f"{score_type}_mipr", pd.NA)

    if pd.notna(score_icmr) and pd.notna(score_mipr):
        return (score_icmr + score_mipr) / 2
    elif pd.notna(score_icmr):
        return score_icmr
    elif pd.notna(score_mipr):
        return score_mipr
    else:
        return pd.NA


# Apply the function to each score type
for score in ["visual", "phonetic", "textual"]:
    merged_df[score] = merged_df.apply(lambda row: combine_scores(row, score), axis=1)

print("Scores combined by averaging where applicable.")

# ================================
# 5. Finalize and Save the Combined Dataset
# ================================

# Select the final columns
final_df = merged_df[["word", "visual", "phonetic", "textual"]]

# Optional: Sort the dataset alphabetically by word
final_df.sort_values(by="word", inplace=True)

print(f"Final combined dataset shape: {final_df.shape}")
print(final_df.head())

# Save the final dataset to a CSV file
final_df.to_csv("local_data/imageability/imageability_corpus.csv", index=False)
print("Combined dataset saved to 'local_data/imageability/combined_imageability.csv'.")
