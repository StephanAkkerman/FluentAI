import gensim.downloader as api


def load_model():
    print("Loading GloVe model. This may take a while...")
    model = api.load("glove-wiki-gigaword-100")  # Options: 50, 100, 200, 300 dimensions
    print("Model loaded successfully.")
    return model


def find_top_similar_words(word, model, topn=5):
    """
    Finds the top N semantically similar words to the given word using the provided model.
    """
    try:
        similar_words = model.most_similar(word, topn=topn)
        return similar_words
    except KeyError:
        print(f"The word '{word}' is not in the vocabulary.")
        return []


def compute_similarity(word1, word2, model):
    """
    Computes the semantic similarity between two words using the provided model.
    """
    # Preprocess words
    word1_processed = word1.strip().lower()
    word2_processed = word2.strip().lower()

    # Check if both words are in the vocabulary
    missing_words = []
    if word1_processed not in model:
        missing_words.append(word1)
    if word2_processed not in model:
        missing_words.append(word2)

    if missing_words:
        print(
            f"The following word(s) are not in the vocabulary: {', '.join(missing_words)}"
        )
        return None

    # Compute similarity using gensim's built-in method
    similarity_score = model.similarity(word1_processed, word2_processed)
    return similarity_score


def main():
    model = load_model()
    while True:
        print("\nSelect an option:")
        print("1. Find top 5 semantically similar words")
        print("2. Compute semantic similarity between two words")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            word = input("\nEnter an English word: ").strip().lower()
            similar = find_top_similar_words(word, model)
            if similar:
                print(f"\nTop {len(similar)} words similar to '{word}':")
                for idx, (sim_word, score) in enumerate(similar, start=1):
                    print(f"{idx}. {sim_word} (Similarity: {score:.4f})")
        elif choice == "2":
            word1 = input("\nEnter the first English word: ").strip().lower()
            word2 = input("Enter the second English word: ").strip().lower()
            similarity = compute_similarity(word1, word2, model)
            if similarity is not None:
                print(
                    f"\nSemantic similarity between '{word1}' and '{word2}': {similarity:.4f}"
                )
        elif choice == "3" or choice.lower() == "exit":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
