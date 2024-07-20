#!/usr/bin/env python3

import csv
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def load_human_similarity_data(file_path="data/human_similarity.csv") -> list:
    with open(file_path, "r") as f:
        return list(csv.DictReader(f))


def create_batches(data_hs, embeddings) -> defaultdict:
    batches = defaultdict(list)
    for entry in data_hs:
        word1, word2, obtained = (
            entry["word1"],
            entry["word2"],
            float(entry["obtained"]),
        )
        if word1 in embeddings and word2 in embeddings:
            batches[word2].append((embeddings[word1], embeddings[word2], obtained))
    return batches


def compute_correlations(predicted, obtained) -> tuple:
    pearson = pearsonr(predicted, obtained)[0]
    spearman = spearmanr(predicted, obtained)[0]
    return pearson, spearman


def evaluate_human_similarity(embeddings, data_hs) -> dict:
    batches = create_batches(data_hs, embeddings)

    metrics = {
        "cos": {"pearson": [], "spearman": []},
        "l2": {"pearson": [], "spearman": []},
        "ip": {"pearson": [], "spearman": []},
    }

    for batch in batches.values():
        e1, e2, obtained = zip(*batch)
        e1, e2 = np.array(e1), np.array(e2)
        obtained = np.array(obtained)

        predictions = {
            "cos": cosine_distances(e1, e2).diagonal(),
            "l2": -euclidean_distances(e1, e2).diagonal(),
            "ip": np.sum(np.multiply(e1, e2), axis=1),
        }

        for metric, preds in predictions.items():
            pearson, spearman = compute_correlations(preds, obtained)
            metrics[metric]["pearson"].append(pearson)
            metrics[metric]["spearman"].append(spearman)

    results = {
        f"pearson {metric}": abs(np.mean(values["pearson"]))
        for metric, values in metrics.items()
    }
    results.update(
        {
            f"spearman {metric}": abs(np.mean(values["spearman"]))
            for metric, values in metrics.items()
        }
    )

    return results


def main():
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    data_hs = load_human_similarity_data()

    results = evaluate_human_similarity(embeddings, data_hs)

    print("Overall:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()
