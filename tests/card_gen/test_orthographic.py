# test_similarity.py

import os

import pytest

os.environ["FLUENTAI_CONFIG_PATH"] = "config.yaml"  # noqa

from fluentai.services.card_gen.mnemonic.orthographic.orthographic import \
    compute_damerau_levenshtein_similarity


@pytest.mark.parametrize(
    "word1, word2, expected_similarity",
    [
        # Both words are empty
        ("", "", 1.0),
        # One word is empty
        ("", "test", 0.0),
        ("test", "", 0.0),
        # Identical words
        ("test", "test", 1.0),
        # Single substitution
        ("test", "tent", 0.75),
        # Insertion
        ("test", "tests", 0.8),
        # Deletion
        ("test", "tes", 0.75),
        # Transposition
        ("test", "tset", 0.75),
        # Completely different words
        ("abc", "xyz", 0.0),
        # Single character identical
        ("a", "a", 1.0),
        # Single character different
        ("a", "b", 0.0),
        # Transposition in two-character words
        ("ab", "ba", 0.5),
        # Multiple operations
        ("example", "samples", 0.57142857142857),
        # Longer words with partial similarity
        ("damerau", "damarau", 0.85714285714285),
        # Case sensitivity
        ("Test", "test", 0.75),
        # Unicode characters
        ("café", "cafe", 0.75),
    ],
)
def test_compute_damerau_levenshtein_similarity(word1, word2, expected_similarity):
    """
    Test the compute_damerau_levenshtein_similarity function with various input cases.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.
        expected_similarity (float): The expected similarity score.
    """
    similarity = compute_damerau_levenshtein_similarity(word1, word2)
    assert similarity == pytest.approx(expected_similarity), (
        f"Similarity between '{word1}' and '{word2}' should be {expected_similarity}, "
        f"but got {similarity}."
    )
