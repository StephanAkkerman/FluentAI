import gensim.downloader as api
import joblib
import numpy as np

# Load the saved model and scaler
model_and_scaler = joblib.load("imageability_model.pkl")
scaler = model_and_scaler["scaler"]
model = model_and_scaler["model"]
embedding_model = model_and_scaler["embedding_model"]


def get_embedding(word, embedding_model, vector_size=300):
    try:
        return embedding_model.get_vector(word)
    except KeyError:
        return np.zeros(vector_size)


def predict_imageability(word):
    # Get embedding
    embedding = get_embedding(word, embedding_model, embedding_model.vector_size)
    embedding = embedding.reshape(1, -1)  # Reshape for scaler

    # Scale features
    embedding_scaled = scaler.transform(embedding)

    # Predict
    prediction = model.predict(embedding_scaled)

    return prediction[0]


# Example usage
new_word = "example"
predicted_score = predict_imageability(new_word)
print(f"Predicted Imageability for '{new_word}': {predicted_score}")
