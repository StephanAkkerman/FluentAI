import difflib

import nltk

# Ensure the 'words' corpus is downloaded
nltk.download("words")

from nltk.corpus import words


def find_orthographically_close_difflib(target_word, topn=5, cutoff=0.6):
    """
    Finds the top N orthographically closest words to the target_word using difflib.

    Parameters:
    - target_word (str): The input word.
    - topn (int): Number of top similar words to return.
    - cutoff (float): Similarity threshold between 0 and 1.

    Returns:
    - list of str: List of similar words.
    """
    word_list = words.words()
    close_matches = difflib.get_close_matches(
        target_word, word_list, n=topn, cutoff=cutoff
    )
    # Exclude the target word itself
    close_matches = [word for word in close_matches if word.lower() != target_word]
    return close_matches[:topn]


def compute_difflib_similarity(word1, word2):
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
    return similarity


print(find_orthographically_close_difflib("train", topn=5))
