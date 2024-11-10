import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler

# Load datasets
imageability_corpus = load_dataset(
    "StephanAkkerman/imageability-corpus", split="train", cache_dir="datasets"
)
mrc = load_dataset(
    "StephanAkkerman/MRC-psycholinguistic-database", split="train", cache_dir="datasets"
)

# Convert to pandas DataFrames
imageability_corpus_df = imageability_corpus.to_pandas()
mrc_df = mrc.to_pandas()

# Prepare 'imageability_corpus_df' DataFrame
imageability_corpus_df["word"] = imageability_corpus_df[
    "word"
].str.lower()  # ensure words are in lowercase

# Combine the visual and phonetic scores into a single score (e.g., by averaging)
imageability_corpus_df["score"] = (
    imageability_corpus_df["visual"] + imageability_corpus_df["phonetic"]
) / 2

# Prepare 'mrc_df' DataFrame
mrc_df["word"] = mrc_df["Word"].str.lower()  # convert word column to lowercase
mrc_df["score"] = mrc_df["Imageability"]  # assign imageability score to 'score'
mrc_df.drop_duplicates(subset="word", inplace=True)
mrc_df.dropna(subset=["score"], inplace=True)
mrc_df = mrc_df[mrc_df["score"] != 0]

# Scale the scores in both datasets using MinMaxScaler
imag_scaler = MinMaxScaler()
mrc_scaler = MinMaxScaler()

# Scaling the scores in the imageability corpus dataset
imageability_corpus_df["scaled_score"] = imag_scaler.fit_transform(
    imageability_corpus_df[["score"]]
)

# Scaling the scores in the MRC dataset
mrc_df["scaled_score"] = mrc_scaler.fit_transform(mrc_df[["score"]])

# Merge the datasets using a full outer join on the 'word' column
combined_df = pd.merge(
    imageability_corpus_df[["word", "scaled_score"]],
    mrc_df[["word", "scaled_score"]],
    on="word",
    how="outer",
    suffixes=("_imageability", "_mrc"),
)

# Combine the scores: take the mean of both scores, handling missing values
combined_df["score"] = combined_df[
    ["scaled_score_imageability", "scaled_score_mrc"]
].mean(axis=1)

# Fill any missing scores with available data (if one dataset has score and the other doesn't)
combined_df["score"] = combined_df["score"].fillna(
    combined_df["scaled_score_imageability"].combine_first(
        combined_df["scaled_score_mrc"]
    )
)


# Save the final combined dataset
combined_df.to_csv("data/imageability/data.csv", index=False)

print("Dataset combined and saved")
