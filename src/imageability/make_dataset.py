import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ================================
# 1. Load and Preprocess Datasets
# ================================

# Load Dataset 1
df1 = pd.read_csv("data/imageability/mrc2.csv")

# Preprocess df1
df1.drop_duplicates(subset="word", inplace=True)
df1.dropna(subset=["imag"], inplace=True)
df1 = df1[df1["imag"] != 0]
df1.rename(columns={"imag": "score_df1"}, inplace=True)

# Load Dataset 2 (JSON format)
df2 = pd.read_json("data/imageability/corpus_mipr2021.json").T

# Preprocess df2
df2.reset_index(inplace=True)
df2.rename(columns={"index": "word", "visual": "score_df2"}, inplace=True)

# Load Dataset 3
df3 = pd.read_csv("data/imageability/corpus_icmr2020.csv")

# Preprocess df3
df3.rename(columns={"Word": "word", "All": "score_df3"}, inplace=True)

# ================================
# 2. Inspect and Analyze Overlaps
# ================================

# Find overlapping words among all three datasets
overlap = pd.merge(df1, df2, on="word", how="inner")
overlap = pd.merge(overlap, df3, on="word", how="inner")
print(f"Number of overlapping words across all three datasets: {len(overlap)}")

# Calculate pairwise correlations between the scores
corr_df1_df2 = overlap["score_df1"].corr(overlap["score_df2"])
corr_df1_df3 = overlap["score_df1"].corr(overlap["score_df3"])
corr_df2_df3 = overlap["score_df2"].corr(overlap["score_df3"])
print(f"Correlation between df1 and df2: {corr_df1_df2:.4f}")
print(f"Correlation between df1 and df3: {corr_df1_df3:.4f}")
print(f"Correlation between df2 and df3: {corr_df2_df3:.4f}")

# ================================
# 3. Harmonize Imageability Scores
# ================================

# Combine all scores into a single 'score' column for scaling
combined_scores = pd.concat(
    [
        df1[["word", "score_df1"]].rename(columns={"score_df1": "score"}),
        df2[["word", "score_df2"]].rename(columns={"score_df2": "score"}),
        df3[["word", "score_df3"]].rename(columns={"score_df3": "score"}),
    ],
    ignore_index=True,
)

print(f"\nCombined Scores Sample:\n{combined_scores.head()}")

# Initialize the scaler
min_max_scaler = MinMaxScaler()

# Fit the scaler on all combined scores
min_max_scaler.fit(combined_scores[["score"]])

# Rename columns for consistency
df1.rename(columns={"score_df1": "score"}, inplace=True)
df2.rename(columns={"score_df2": "score"}, inplace=True)
df3.rename(columns={"score_df3": "score"}, inplace=True)

# Apply the scaler to each dataset's scores
df1["score_scaled"] = min_max_scaler.transform(df1[["score"]])
df2["score_scaled"] = min_max_scaler.transform(df2[["score"]])
df3["score_scaled"] = min_max_scaler.transform(df3[["score"]])

# ================================
# 4. Merge All Datasets into Final Dataset
# ================================

# Select relevant columns from each dataset
df1_subset = df1[["word", "score_scaled"]]
df2_subset = df2[["word", "score_scaled"]]
df3_subset = df3[["word", "score_scaled"]]

# Combine all subsets into a single DataFrame
final_df = pd.concat([df1_subset, df2_subset, df3_subset], ignore_index=True)

print(f"\nFinal Dataset Before Handling Overlaps: {final_df.shape}")
print(final_df.head())

# Handle overlapping words by averaging their scaled scores
final_df = final_df.groupby("word").agg({"score_scaled": "mean"}).reset_index()

print(f"\nTotal unique words after merging: {len(final_df)}")
print(final_df.head())

# ================================
# 5. Save the Final Dataset (Optional)
# ================================

# You may want to save the final_df for future use
final_df.to_csv("data/imageability/final_imageability_dataset.csv", index=False)
print("\nFinal dataset saved to 'data/imageability/final_imageability_dataset.csv'.")

# ================================
# 6. Proceed with Feature Engineering and Model Training
# ================================

# (This section assumes you have already set up your embedding model and preprocessing steps)
# Here, you would integrate final_df into your existing modeling pipeline.
