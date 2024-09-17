import nltk
from rapidfuzz import fuzz, process

# Ensure the 'words' corpus is downloaded
nltk.download("words")

from nltk.corpus import words
from rapidfuzz import distance


def compute_damerau_levenshtein_similarity(word1, word2):
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


def find_orthographically_close_rapidfuzz(target_word, topn=5):
    """
    Finds the top N orthographically closest words to the target_word using rapidfuzz.

    Parameters:
    - target_word (str): The input word.
    - topn (int): Number of top similar words to return.

    Returns:
    - list of tuples: Each tuple contains a similar word and its similarity score.
    """
    word_list = words.words()
    # Using fuzz.WRatio which is a weighted similarity ratio
    matches = process.extract(target_word, word_list, scorer=fuzz.WRatio, limit=topn)
    # Exclude the target word itself
    filtered_matches = [
        (word, score) for word, score, _ in matches if word.lower() != target_word
    ]
    return filtered_matches[:topn]


def compute_rapidfuzz_similarity(word1, word2):
    """
    Computes the orthographic similarity between two words using rapidfuzz's fuzz.ratio.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - float: Similarity score between 0 and 100.
    """
    # Preprocess words: strip whitespace and convert to lowercase
    word1_processed = word1.strip().lower()
    word2_processed = word2.strip().lower()

    # Compute similarity score using fuzz.ratio
    similarity_score = fuzz.ratio(word1_processed, word2_processed)

    return similarity_score


def compute_partial_ratio_similarity(word1, word2):
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


print(find_orthographically_close_rapidfuzz("train", topn=5))
