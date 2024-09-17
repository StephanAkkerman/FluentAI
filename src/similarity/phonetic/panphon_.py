import numpy as np
from panphon import FeatureTable

# Initialize panphon's FeatureTable
ft = FeatureTable()


def compute_phonetic_similarity(ipa1, ipa2, distance_metric="cosine"):
    """
    Computes the phonetic similarity between two IPA transcriptions using panphon.

    Parameters:
    - ipa1 (str): The first IPA transcription.
    - ipa2 (str): The second IPA transcription.
    - distance_metric (str): The distance metric to use ('cosine', 'euclidean', etc.).

    Returns:
    - float: Similarity score between 0 and 1, where 1 indicates identical phonetic features.
    """
    # Convert IPA transcriptions to feature vectors
    ft_vector1 = ft.word_to_vector(ipa1)
    ft_vector2 = ft.word_to_vector(ipa2)

    if ft_vector1 is None or ft_vector2 is None:
        print("One or both IPA transcriptions contain unsupported symbols.")
        return None

    # Choose distance metric
    if distance_metric == "cosine":
        # Compute cosine similarity
        dot_product = np.dot(ft_vector1, ft_vector2)
        norm1 = np.linalg.norm(ft_vector1)
        norm2 = np.linalg.norm(ft_vector2)
        similarity = dot_product / (norm1 * norm2)
    elif distance_metric == "euclidean":
        # Compute Euclidean distance and convert to similarity
        distance = np.linalg.norm(ft_vector1 - ft_vector2)
        similarity = 1 / (1 + distance)  # Invert distance to similarity
    else:
        raise ValueError("Unsupported distance metric. Choose 'cosine' or 'euclidean'.")

    return similarity
