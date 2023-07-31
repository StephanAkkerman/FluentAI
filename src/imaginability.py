import pandas as pd
from sklearn.linear_model import LinearRegression


def create_model():
    # Read the AoA values and other features (e.g., familiarity, part of speech) from the Kuperman corpus
    kuperman_corpus = pd.read_csv("transphoner_data/imagine/en.aoa.csv")

    # Read the imageability values from the MRC Psycholinguistic Database
    imageability_values = pd.read_csv("imageability_values.csv")

    # Merge the datasets on the common word identifier
    data = pd.merge(kuperman_corpus, imageability_values, on="word")

    # Normalize the imageability values to be between 0 and 1
    data["imageability"] = (data["imageability"] - data["imageability"].min()) / (
        data["imageability"].max() - data["imageability"].min()
    )

    # Select the features for the linear regression model
    X = data[["AoA", "familiarity", "part_of_speech"]]  # Add relevant feature columns
    y = data["imageability"]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model


# Step 2: Computing Imageability of Foreign Words and Uncovered English Words
# Read the word-sense association weights from UWN
uwn_data = pd.read_csv("uwn_data.csv")


# Function to compute imageability for a given word
def compute_imageability(word):
    S_w = uwn_data[uwn_data["word"] == word]
    numerator_IW = 0
    denominator_IW = 0
    for _, row in S_w.iterrows():
        a_i = row["a_i"]
        s_i = row["s_i"]
        W_s = uwn_data[uwn_data["sense"] == s_i]
        numerator_IS = 0
        denominator_IS = 0
        for _, sub_row in W_s.iterrows():
            b_i = sub_row["b_i"]
            w_i = sub_row["word"]
            # Compute IW using the trained regression model for w_i
            IW_wi = model.predict(
                get_features(w_i)
            )  # Define `get_features` to obtain features for w_i
            numerator_IS += b_i * IW_wi
            denominator_IS += b_i
        IS_si = numerator_IS / denominator_IS
        numerator_IW += a_i * IS_si
        denominator_IW += a_i
    return numerator_IW / denominator_IW if denominator_IW != 0 else 0


# Example usage
word_imageability = compute_imageability("example_word")
