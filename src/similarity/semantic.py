import os
import pickle

import nltk
from nltk.corpus import words
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def find_nearest_neighbors(word, words, embeddings, model, top_k=5):
    # Convert the input word to lower case
    word = word.lower()

    # Compute the embedding for the input word
    word_embedding = model.encode([word])

    # Compute cosine similarities between the input word and the pre-computed embeddings
    similarities = cosine_similarity(word_embedding, embeddings).flatten()

    # Get the indices of the top_k most similar words
    top_k_indices = similarities.argsort()[-(top_k + 1) :][
        ::-1
    ]  # Add 1 to top_k to account for the word itself

    # Remove the input word from the results
    nearest_neighbors = [
        (words[idx], similarities[idx]) for idx in top_k_indices if words[idx] != word
    ]
    nearest_neighbors = nearest_neighbors[
        :top_k
    ]  # Ensure we still return top_k results

    return nearest_neighbors


def create_embeddings(model):

    # Download the words corpus if not already done
    nltk.download("words")

    # Get the list of English words
    word_list = words.words()

    # Use lower case words
    word_list = [word.lower() for word in word_list]

    # only unique words
    word_list = list(set(word_list))

    # Generate embeddings for all words
    embeddings = model.encode(word_list, batch_size=64, show_progress_bar=True)

    # Save the embeddings and corresponding words to a file
    with open("word_embeddings.pkl", "wb") as f:
        pickle.dump({"words": word_list, "embeddings": embeddings}, f)


def semantic_top5(top_k: int = 5):

    # Load the pre-trained model
    # https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="models")

    if not os.path.exists("word_embeddings.pkl"):
        # Create embeddings for all words
        create_embeddings(model)

    # Load the embeddings from the file
    with open("word_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        words = data["words"]
        embeddings = data["embeddings"]

    # Example input word
    input_word = "cat"

    # Find the top k nearest neighbors
    nearest_neighbors = find_nearest_neighbors(
        input_word, words, embeddings, model, top_k=top_k
    )

    print(f"Top {top_k} nearest neighbors for '{input_word}':")
    for neighbor, similarity in nearest_neighbors:
        print(f"{neighbor}: {similarity:.4f}")


def words_distance(word1: str = "cat", word2: str = "dog") -> float:
    # Load the embeddings from the file
    with open("word_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        words = data["words"]
        embeddings = data["embeddings"]

    # Find the index of the first word
    # What if the word is not in the list?
    idx1 = words.index(word1.lower())

    # Find the index of the second word
    idx2 = words.index(word2.lower())

    # Compute the cosine similarity between the two words
    similarity = cosine_similarity(
        embeddings[idx1].reshape(1, -1), embeddings[idx2].reshape(1, -1)
    )[0][0]

    return similarity


print(words_distance("cat", "dog"))
