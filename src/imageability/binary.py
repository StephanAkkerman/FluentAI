# Based on: https://aclanthology.org/P14-1024.pdf
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def load_word_vectors(file_path):
    """
    Load word vectors from a .feat file where each line contains a word and a JSON-formatted vector.

    Example line:
    wrong	{"0": -0.1484, "1": -0.1638, "10": 0.10606, "11": 0.085995}
    """
    word_vectors = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Split the line into word and vector string using tab delimiter
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    print(f"Line {line_num}: Incorrect format, skipping.")
                    continue
                word, vector_str = parts

                # Parse the JSON string into a dictionary
                vector_dict = json.loads(vector_str)

                # Convert the dictionary to a list of floats sorted by index
                sorted_keys = sorted(vector_dict.keys(), key=lambda x: int(x))
                vector = np.array(
                    [vector_dict[key] for key in sorted_keys], dtype=float
                )

                word_vectors[word] = vector
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decoding error: {e}, skipping.")
            except Exception as e:
                print(f"Line {line_num}: Unexpected error: {e}, skipping.")
    return word_vectors


def load_labels(file_path):
    """
    Load labels from a labels file where each line contains a word and its label separated by whitespace.

    Example line:
    wrong	A
    extent	A
    obese	C
    """
    labels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) != 2:
                    print(f"Line {line_num}: Incorrect format, skipping.")
                    continue
                word, label = parts
                labels[word.lower()] = label.upper()
            except Exception as e:
                print(f"Line {line_num}: Unexpected error: {e}, skipping.")
    return labels


def vectors_to_df(word_vectors):
    """
    Convert word vectors dictionary to a Pandas DataFrame.
    """
    if not word_vectors:
        print("No word vectors loaded. Exiting.")
        return pd.DataFrame()
    vector_dim = len(next(iter(word_vectors.values())))
    df = pd.DataFrame.from_dict(
        word_vectors, orient="index", columns=[f"dim_{i}" for i in range(vector_dim)]
    )
    df.reset_index(inplace=True)
    df.rename(columns={"index": "word"}, inplace=True)
    return df


def main():
    # Paths to feature and label files
    train_feat_path = "data/imageability/train.feat"
    test_feat_path = "data/imageability/test.feat"
    train_labels_path = "data/imageability/train.labels"
    test_labels_path = "data/imageability/test.labels"

    # 1. Load word vectors
    print("Loading training word vectors...")
    train_word_vectors = load_word_vectors(train_feat_path)
    print(f"Total training words with vectors: {len(train_word_vectors)}\n")

    print("Loading testing word vectors...")
    test_word_vectors = load_word_vectors(test_feat_path)
    print(f"Total testing words with vectors: {len(test_word_vectors)}\n")

    # 2. Load labels
    print("Loading training labels...")
    train_labels = load_labels(train_labels_path)
    print(f"Total training labels: {len(train_labels)}\n")

    print("Loading testing labels...")
    test_labels = load_labels(test_labels_path)
    print(f"Total testing labels: {len(test_labels)}\n")

    # 3. Merge labels with word vectors
    # Convert word vectors to DataFrame
    train_df = vectors_to_df(train_word_vectors)
    test_df = vectors_to_df(test_word_vectors)

    # Add labels to DataFrame
    train_df["label"] = train_df["word"].str.lower().map(train_labels)
    test_df["label"] = test_df["word"].str.lower().map(test_labels)

    # Drop rows with missing labels
    initial_train_len = len(train_df)
    train_df = train_df.dropna(subset=["label"]).reset_index(drop=True)
    print(
        f"Training data: Dropped {initial_train_len - len(train_df)} words without labels."
    )
    initial_test_len = len(test_df)
    test_df = test_df.dropna(subset=["label"]).reset_index(drop=True)
    print(
        f"Testing data: Dropped {initial_test_len - len(test_df)} words without labels.\n"
    )

    # 4. Map labels to binary values
    label_mapping = {
        "A": 0,
        "C": 1,
    }  # A = Abstract (not imaginable), C = Concrete (imaginable)
    train_df["binary_label"] = train_df["label"].map(label_mapping)
    test_df["binary_label"] = test_df["label"].map(label_mapping)

    # Verify mapping
    if (
        train_df["binary_label"].isnull().any()
        or test_df["binary_label"].isnull().any()
    ):
        print(
            "Warning: Some labels could not be mapped. Check label_mapping and label files."
        )

    # 5. Define feature columns
    feature_cols = [col for col in train_df.columns if col.startswith("dim_")]

    # 6. Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df["binary_label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["binary_label"].values

    # 7. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 8. Train Classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    print("Training completed.\n")

    # 9. Evaluate Classifier
    y_pred = clf.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")

    # 10. Predict Posterior Probabilities for All Words (Optional)
    # If you want to propagate scores to all words, including those not in test set
    # Here, we assume 'test.feat' contains all words
    # Modify as needed based on your dataset

    # Example: Using test_df as all_words
    all_words_df = test_df.copy()

    # Predict probabilities
    all_words_df["prob_concrete"] = clf.predict_proba(X_test_scaled)[:, 1]
    all_words_df["is_concrete"] = (all_words_df["prob_concrete"] >= 0.5).astype(
        int
    )  # Threshold=0.5

    # Save results
    output_path = "word_imageability_abstractness.csv"
    all_words_df[["word", "prob_concrete", "is_concrete"]].to_csv(
        output_path, index=False
    )
    print(f"Propagation completed. Results saved to '{output_path}'.")


if __name__ == "__main__":
    main()
