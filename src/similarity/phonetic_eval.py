# human judgment: https://github.com/zouharvi/pwesuite/blob/master/suite_evaluation/eval_human_similarity.py
# articulatory distance: https://github.com/zouharvi/pwesuite/blob/master/suite_evaluation/eval_correlations.py
# According to https://github.com/zouharvi/pwesuite/blob/master/suite_evaluation/eval_all.py
#!/usr/bin/env python3

import collections
import csv
import pickle

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def evaluate_human_similarity(data_multi_hs: dict):
    with open("data/human_similarity.csv", "r") as f:
        data_hs = list(csv.DictReader(f))

    batches = collections.defaultdict(list)
    # Loop over the data and add the word pairs to the batches
    for w_hs in data_hs:
        batches[w_hs["word2"]].append(
            (
                # Get the embeddings for the word pairs
                data_multi_hs[w_hs["word1"]],
                data_multi_hs[w_hs["word2"]],
                w_hs["obtained"],
            )
        )
    corr_pearson_l2_all = []
    corr_spearman_l2_all = []
    corr_pearson_cos_all = []
    corr_spearman_cos_all = []
    corr_pearson_ip_all = []
    corr_spearman_ip_all = []
    for batch in batches.values():
        predicted_cos = [cosine_distances([e1], [e2])[0, 0] for e1, e2, _ in batch]
        predicted_l2 = [-euclidean_distances([e1], [e2])[0, 0] for e1, e2, _ in batch]
        predicted_ip = [np.multiply(e1, e2).sum() for e1, e2, _ in batch]
        obtained = [float(o) for _, _, o in batch]

        corr_pearson_ip_all.append(pearsonr(predicted_ip, obtained)[0])
        corr_pearson_l2_all.append(pearsonr(predicted_l2, obtained)[0])
        corr_pearson_cos_all.append(pearsonr(predicted_cos, obtained)[0])
        corr_spearman_ip_all.append(spearmanr(predicted_ip, obtained)[0])
        corr_spearman_l2_all.append(spearmanr(predicted_l2, obtained)[0])
        corr_spearman_cos_all.append(spearmanr(predicted_cos, obtained)[0])

    return {
        "pearson IP": abs(np.average(corr_pearson_ip_all)),
        "pearson L2": abs(np.average(corr_pearson_l2_all)),
        "pearson cos": abs(np.average(corr_pearson_cos_all)),
        "spearman IP": abs(np.average(corr_spearman_ip_all)),
        "spearman L2": abs(np.average(corr_spearman_l2_all)),
        "spearman cos": abs(np.average(corr_spearman_cos_all)),
    }


if __name__ == "__main__":
    with open("embeddings.pkl", "rb") as f:
        data_embd = pickle.load(f)

    output = evaluate_human_similarity(data_embd)
    print("Overall:")
    for key in output:
        print(f"{key}: {output[key]:.2f}")
