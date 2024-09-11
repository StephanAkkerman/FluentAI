#!/usr/bin/env python3

import random

import multiprocess as mp
import numpy as np
import tqdm
from panphon.distance import Distance
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def evaluate_correlations(data_multi, data_size=1000, jobs=20):
    # data_langs = collections.defaultdict(data_multi)

    def compute_panphon_distance(y, data):
        fed = Distance().feature_edit_distance
        return [fed(w, y) for w, _ in data]

    corr_pearson_l2_all = {}
    corr_spearman_l2_all = {}
    corr_pearson_cos_all = {}
    corr_spearman_cos_all = {}

    for lang, data in tqdm.tqdm(data_multi.items()):
        # Take only dev data
        r = random.Random(0)
        print(data)
        data = r.sample(data, k=data_size)

        with mp.Pool(jobs) as pool:
            data_dists_fed = np.array(
                pool.map(lambda y: compute_panphon_distance(y[0], data), data)
            )

        data_dists_l2 = euclidean_distances(np.array([x[1] for x in data]))
        data_dists_cos = cosine_distances(np.array([x[1] for x in data]))

        corr_pearson_l2 = []
        corr_spearman_l2 = []
        corr_pearson_cos = []
        corr_spearman_cos = []

        for dist_fed, dist_l2, dist_cos in zip(
            data_dists_fed, data_dists_l2, data_dists_cos
        ):
            corr_pearson_l2.append(pearsonr(dist_fed, dist_l2)[0])
            corr_spearman_l2.append(spearmanr(dist_fed, dist_l2)[0])

            corr_pearson_cos.append(pearsonr(dist_fed, dist_cos)[0])
            corr_spearman_cos.append(spearmanr(dist_fed, dist_cos)[0])

        corr_pearson_l2_all[lang] = abs(np.average(corr_pearson_l2))
        corr_pearson_cos_all[lang] = abs(np.average(corr_pearson_cos))
        corr_spearman_l2_all[lang] = abs(np.average(corr_spearman_l2))
        corr_spearman_cos_all[lang] = abs(np.average(corr_spearman_cos))

    corr_pearson_l2_all["all"] = abs(np.average(list(corr_pearson_l2_all.values())))
    corr_pearson_cos_all["all"] = abs(np.average(list(corr_pearson_cos_all.values())))
    corr_spearman_l2_all["all"] = abs(np.average(list(corr_spearman_l2_all.values())))
    corr_spearman_cos_all["all"] = abs(np.average(list(corr_spearman_cos_all.values())))

    return {
        "pearson L2": corr_pearson_l2_all,
        "pearson cos": corr_pearson_cos_all,
        "spearman L2": corr_spearman_l2_all,
        "spearman cos": corr_spearman_cos_all,
    }


if __name__ == "__main__":
    import pickle

    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    output = evaluate_correlations(embeddings)
    print("Overall:")
    for key in output:
        print(f"{key}: {output[key]['all']:.2f}")
