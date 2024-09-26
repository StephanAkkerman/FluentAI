import multiprocessing

import gensim.downloader as api
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ================================
# 1. Configuration Parameters
# ================================

# Choose the model type: 'random_forest', 'logistic_regression', 'linear_regression'
MODEL_TYPE = "random_forest"  # Change this as needed

# If using classification, define the number of classes and how to binarize
CLASSIFY = False  # Set to True if you want to perform classification
NUM_CLASSES = 2  # Example: 2 classes for high and low imageability

# Path to save the trained model
MODEL_SAVE_PATH = "fasttext_imageability_model.pkl"

# ================================
# 2. Load and Preprocess Data
# ================================

# Load your dataset
df = pd.read_csv("data/imageability/final_imageability_dataset.csv")

# Separate features and target
y = df["score_scaled"]

# If performing classification, convert continuous scores to categorical labels
if CLASSIFY:
    # Example: Binarize based on the median
    median_imag = y.median()
    y = (y > median_imag).astype(int)
    print(
        f"Converted imageability scores to {NUM_CLASSES} classes based on median value."
    )

# Load pre-trained FastText embeddings
print("Loading FastText embeddings...")
embedding_model = api.load(
    "fasttext-wiki-news-subwords-300"
)  # 300-dim FastText embeddings


def get_embedding(word):
    try:
        return embedding_model.get_vector(word)
    except KeyError:
        # FastText can handle OOV by default, but include this as a fallback
        print(word)
        return np.zeros(embedding_model.vector_size)


# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cores}")

# Initialize tqdm_joblib context manager to integrate tqdm with joblib
with tqdm_joblib(tqdm(desc="Generating Embeddings", total=len(df))):
    # Generate embeddings in parallel
    embeddings = Parallel(n_jobs=num_cores)(
        delayed(get_embedding)(word) for word in df["word"]
    )

# Convert list of embeddings to a NumPy array
embeddings = np.vstack(embeddings)

# Maybe save this for future use
# Dump words + embeddings to a file
# joblib.dump((df["word"], embeddings), "data/imageability/word_embeddings.pkl")

# If you have additional features, include them here
# For this example, we'll assume only embeddings are used
X_combined = embeddings

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 3. Model Initialization
# ================================

# Initialize the model based on the selected MODEL_TYPE and task
if MODEL_TYPE == "random_forest" and not CLASSIFY:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif MODEL_TYPE == "linear_regression" and not CLASSIFY:
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
elif MODEL_TYPE == "logistic_regression" and CLASSIFY:
    model = LogisticRegression(max_iter=1000, random_state=42)
elif MODEL_TYPE == "random_forest" and CLASSIFY:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Invalid combination of MODEL_TYPE and CLASSIFY flag.")

print(
    f"Selected Model: {MODEL_TYPE} | Task: {'Classification' if CLASSIFY else 'Regression'}"
)

# ================================
# 4. Model Training
# ================================

model.fit(X_train, y_train)
print("Model training completed.")

# ================================
# 5. Predictions and Evaluation
# ================================

y_pred = model.predict(X_test)

if CLASSIFY:
    # For classification, predictions might need to be thresholded
    if hasattr(model, "predict_proba"):
        y_pred_classes = model.predict(X_test)
    else:
        # Some classifiers might not have predict_proba
        y_pred_classes = (y_pred > 0.5).astype(int)

    # Evaluate classification performance
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))
else:
    # For regression, evaluate using MSE and R²
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

# ================================
# 6. Save the Trained Model
# ================================

# Save both the scaler and the model to ensure consistent preprocessing during inference
model_and_scaler = {
    "scaler": scaler,
    "model": model,
    "embedding_model": embedding_model,  # Optionally save the embedding model reference
}

joblib.dump(model_and_scaler, MODEL_SAVE_PATH)
print(f"Model and scaler saved to {MODEL_SAVE_PATH}")

# ================================
# 7. Optional: Model Tuning and Cross-Validation
# ================================

# You can implement cross-validation and hyperparameter tuning as needed.
# Here's an example using cross_val_score for regression.

if not CLASSIFY and MODEL_TYPE == "random_forest":
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"Cross-Validation R² Scores: {cv_scores}")
    print(f"Mean CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
