import Levenshtein
import nltk
from nltk.corpus import words

# Download the words corpus if not already done
nltk.download("words")

# Get the list of English words
word_list = words.words()


def levenshtein_distance(word1, word2):
    return Levenshtein.distance(word1, word2)


def find_orthographic_neighbors(word, word_list, top_k=5):
    distances = [(w, levenshtein_distance(word, w)) for w in word_list if w != word]
    distances.sort(key=lambda x: x[1])

    nearest_neighbors = distances[:top_k]
    return nearest_neighbors


# Example usage
input_word = "train"
top_k = 5
nearest_neighbors = find_orthographic_neighbors(input_word, word_list, top_k=top_k)

print(f"Top {top_k} orthographically similar words to '{input_word}':")
for neighbor, distance in nearest_neighbors:
    print(f"{neighbor}: {distance}")
