import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read imaginability data json file
# df = pd.read_json("transphoner_data/imagine/corpus_mipr2021.json").T

mrc2 = pd.read_csv("transphoner_data/imagine/mrc2.csv")
# Only were imag ratings are available
mrc2 = mrc2[mrc2["imag"] != 0]


# Normalize imag ratings
mrc2["imag"] = (mrc2["imag"] - mrc2["imag"].min()) / (
    mrc2["imag"].max() - mrc2["imag"].min()
)

# Train a logistic regression model to predict imag ratings
model = LogisticRegression()

# Maybe add num of letters, num of phonemes, num of syllables

# Fit on AoA, familiatrity, part of speech feature values
model.fit(mrc2[["aoa", "fam", "wtype"]], mrc2["imag"])

# Compute imaginability rating for all English words not covered by Kuperman corpus
