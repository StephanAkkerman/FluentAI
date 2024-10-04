import difflib

import Levenshtein
import nltk
from nltk.corpus import words
from rapidfuzz import distance, fuzz

# Ensure the 'words' corpus is downloaded
nltk.download("words", quiet=True)
WORD_LIST = words.words()


def compute_difflib_similarity(word1: str, word2: str) -> float:
    """
    Computes the orthographic similarity between two words using difflib's SequenceMatcher.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - float: Similarity ratio between 0 and 1.
    """
    matcher = difflib.SequenceMatcher(None, word1, word2)
    similarity = matcher.ratio()
    return similarity * 100


def compute_rapidfuzz_ratio(word1: str, word2: str) -> float:
    """
    Computes the orthographic similarity between two words using rapidfuzz's fuzz.ratio.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - float: Similarity score between 0 and 100.
    """
    similarity_score = fuzz.ratio(word1.strip().lower(), word2.strip().lower())
    return similarity_score


def compute_rapidfuzz_partial_ratio(word1: str, word2: str) -> float:
    """
    Computes the orthographic similarity between two words using rapidfuzz's fuzz.partial_ratio.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - float: Similarity score between 0 and 100.
    """
    similarity_score = fuzz.partial_ratio(word1, word2)
    return similarity_score


def compute_damerau_levenshtein_similarity(word1: str, word2: str) -> float:
    """
    Computes the Damerau-Levenshtein similarity between two words.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - float: Similarity score between 0 and 100.
    """
    dl_distance = distance.DamerauLevenshtein.distance(word1, word2)
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 100.0  # Both words are empty
    similarity = (1 - dl_distance / max_len) * 100
    return similarity


def compute_levenshtein_similarity(word1: str, word2: str) -> float:
    """
    Computes the Levenshtein similarity between two words.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - float: Similarity score between 0 and 100.
    """
    lev_distance = Levenshtein.distance(word1, word2)
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 100.0  # Both words are empty
    similarity = (1 - lev_distance / max_len) * 100
    return similarity


def compute_similarity(word1: str, word2: str, method: str) -> float:
    """
    Computes the orthographic similarity between two words using the specified method.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.
    - method (str): The similarity method to use. Options:
        - 'difflib'
        - 'rapidfuzz_ratio'
        - 'rapidfuzz_partial_ratio'
        - 'damerau_levenshtein'
        - 'levenshtein'

    Returns:
    - float: Similarity score. The scale depends on the method:
        - 'difflib': 0 to 1
        - Others: 0 to 100

    Raises:
    - ValueError: If an unsupported method is provided.
    """
    method = method.lower()
    if method == "difflib":
        return compute_difflib_similarity(word1, word2)
    elif method == "rapidfuzz_ratio":
        return compute_rapidfuzz_ratio(word1, word2)
    elif method == "rapidfuzz_partial_ratio":
        return compute_rapidfuzz_partial_ratio(word1, word2)
    elif method == "damerau_levenshtein":
        return compute_damerau_levenshtein_similarity(word1, word2)
    elif method == "levenshtein":
        return compute_levenshtein_similarity(word1, word2)
    else:
        raise ValueError(
            f"Unsupported similarity method: '{method}'. "
            f"Choose from 'difflib', 'rapidfuzz_ratio', "
            f"'rapidfuzz_partial_ratio', 'damerau_levenshtein', 'levenshtein'."
        )


def example():
    examples = [
        ("train", "brain", "difflib"),
        ("train", "brain", "rapidfuzz_ratio"),
        ("train", "brain", "rapidfuzz_partial_ratio"),
        ("train", "brain", "damerau_levenshtein"),
        ("train", "brain", "levenshtein"),
    ]

    for word1, word2, method in examples:
        similarity = compute_similarity(word1, word2, method)
        print(
            f"Similarity between '{word1}' and '{word2}' using '{method}': {similarity}"
        )


# Example usage (for testing purposes only; remove or comment out in production)
if __name__ == "__main__":
    # Example words and methods
    example()

# Maybe improve this by only importing the library in a function that uses it
