import hashlib
import io
import json
import os
from datetime import datetime

import joblib
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from huggingface_hub import HfApi, hf_hub_download
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Computes a unique hash for the given DataFrame based on its content.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to hash.

    Returns
    -------
    str
        The MD5 hash of the DataFrame.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return hashlib.md5(buffer.getvalue().encode("utf-8")).hexdigest()


def ensure_logs_directory(logs_dir: str):
    """
    Ensures that the logs directory exists.

    Parameters
    ----------
    logs_dir : str
        The path to the logs directory.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logger.debug(f"Ensured that the logs directory '{logs_dir}' exists.")


def load_existing_logs(log_file_path: str) -> pd.DataFrame:
    """
    Loads existing evaluation logs if the log file exists.

    Parameters
    ----------
    log_file_path : str
        The path to the log file.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing existing logs.
    """
    if os.path.exists(log_file_path):
        try:
            existing_logs = pd.read_csv(log_file_path)
            logger.info(f"Loaded existing logs from '{log_file_path}'.")
            return existing_logs
        except Exception as e:
            logger.error(f"Failed to read log file '{log_file_path}': {e}")
            return pd.DataFrame()
    else:
        logger.info(f"No existing log file found at '{log_file_path}'. Starting fresh.")
        return pd.DataFrame()


def append_to_log(log_file_path: str, new_entry: dict):
    """
    Appends a new evaluation entry to the log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file.
    new_entry : dict
        The evaluation results to append.
    """
    try:
        df_new = pd.DataFrame([new_entry])
        if os.path.exists(log_file_path):
            df_new.to_csv(log_file_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(log_file_path, mode="w", header=True, index=False)
        logger.debug(f"Appended new entry to log file '{log_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to append to log file '{log_file_path}': {e}")


def load_existing_hyperparameters_log(log_file_path: str) -> dict:
    """
    Loads existing hyperparameter trials from the log file.

    Parameters
    ----------
    log_file_path : str
        The path to the hyperparameter log file.

    Returns
    -------
    dict
        A dictionary mapping model names to their tried hyperparameter hashes and objective values.
    """
    if os.path.exists(log_file_path):
        try:
            df = pd.read_csv(log_file_path)
            # Create a dict: model_name -> {hyperparam_hash: objective_value}
            log_dict = {}
            for _, row in df.iterrows():
                model = row["model_name"]
                hyperparam_hash = row["hyperparam_hash"]
                objective_value = row["objective_value"]
                if model not in log_dict:
                    log_dict[model] = {}
                log_dict[model][hyperparam_hash] = objective_value
            logger.info(f"Loaded existing hyperparameter log from '{log_file_path}'.")
            return log_dict
        except Exception as e:
            logger.error(f"Failed to read hyperparameter log '{log_file_path}': {e}")
            return {}
    else:
        logger.info(
            f"No existing hyperparameter log found at '{log_file_path}'. Starting fresh."
        )
        return {}


def append_hyperparameters_log(
    log_file_path: str, model_name: str, hyperparams: dict, objective_value: float
):
    """
    Appends a new hyperparameter trial to the hyperparameter log file.

    Parameters
    ----------
    log_file_path : str
        The path to the hyperparameter log file.
    model_name : str
        The name of the model.
    hyperparams : dict
        The hyperparameters of the trial.
    objective_value : float
        The objective value of the trial.
    """
    try:
        # Compute hyperparam hash
        sorted_hyperparams = sorted(hyperparams.items())
        hyperparam_str = json.dumps(sorted_hyperparams, sort_keys=True)
        hyperparam_hash = hashlib.md5(hyperparam_str.encode("utf-8")).hexdigest()

        # Create a new entry
        new_entry = {
            "model_name": model_name,
            "hyperparam_hash": hyperparam_hash,
            "hyperparameters": json.dumps(hyperparams),
            "objective_value": objective_value,
            "timestamp": datetime.now().isoformat(),
        }
        df_new = pd.DataFrame([new_entry])
        if os.path.exists(log_file_path):
            df_new.to_csv(log_file_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(log_file_path, mode="w", header=True, index=False)
        logger.debug(f"Appended hyperparameter trial to log file '{log_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to append hyperparameter log to '{log_file_path}': {e}")


def load_data(local_file: bool = False, filename: str = None):
    """
    Load words, embeddings, and scores from a single file.

    Returns
    -------
        tuple: (embeddings, scores, dataset_hash)
    """
    if local_file:
        if filename is None:
            raise ValueError("Filename must be provided when local_file is True.")
        df = pd.read_parquet(filename)
    else:
        if filename is None:
            model = config.get("IMAGEABILITY").get("EMBEDDINGS").get("MODEL")
            filename = f"{model.replace('/', '_')}_embeddings.parquet"
        logger.info(f"Loading embeddings from {filename}...")
        df = pd.read_parquet(
            hf_hub_download(
                config.get("IMAGEABILITY").get("EMBEDDINGS").get("REPO"),
                filename=filename,
                cache_dir="datasets",
                repo_type="dataset",
            )
        )

    # Verify required columns exist
    required_columns = {"word", "score"}
    embedding_columns = [col for col in df.columns if col.startswith("emb_")]
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns in parquet file: {missing}")
    if not embedding_columns:
        raise ValueError(
            "No embedding columns found. Ensure embeddings are named starting with 'emb_'."
        )

    # Extract embeddings and scores
    embeddings = df[embedding_columns].values
    scores = df["score"].values

    logger.info(
        f"Loaded {len(scores)} words with embeddings shape {embeddings.shape} and scores shape {scores.shape}."
    )

    # Compute dataset hash
    dataset_hash = compute_dataset_hash(df)
    logger.debug(f"Computed dataset hash: {dataset_hash}")

    # TODO: add data validation to check if the embeddings shape == imageabillity dataset shape (6090 rows)

    return embeddings, scores, dataset_hash


def split_dataset(embeddings, scores):
    """
    Preprocess the data for training.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        scores (np.ndarray): Array of scores.

    Returns
    -------
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, scores, test_size=0.2, random_state=42
    )
    logger.info("Data split into training and testing sets.")
    logger.info(f"Training set size: {X_train.shape[0]} samples.")
    logger.info(f"Testing set size: {X_test.shape[0]} samples.")

    return X_train, X_test, y_train, y_test


def get_optuna_search_space(trial, model_name):
    """
    Define hyperparameter search space based on the model.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        model_name (str): Name of the model.

    Returns:
        dict: Hyperparameter suggestions.
    """
    if model_name == "Random Forest":
        max_depth = trial.suggest_categorical("max_depth_null", [True, False])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": (
                trial.suggest_int("max_depth", 10, 50) if not max_depth else None
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    elif model_name == "Gradient Boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        }
    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        }
    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
            "max_depth": trial.suggest_int("max_depth", -1, 50),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        }
    elif model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        }
    elif model_name == "Support Vector Regression":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2),
            "epsilon": trial.suggest_float("epsilon", 1e-4, 1e-1),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    elif model_name in ["Linear Regression (OLS)", "Ridge Regression"]:
        return {}  # No hyperparameters to tune for basic linear models
    else:
        return {}


def objective(
    trial,
    model_name,
    X_train,
    y_train,
    existing_hyperparams,
    hyperparam_log_file_path,
):
    """
    Objective function for Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        model_name (str): Name of the model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        existing_hyperparams (dict): Existing hyperparameter trials.
        hyperparam_log_file_path (str): Path to the hyperparameter log file.

    Returns:
        float: Mean Squared Error (to minimize).
    """
    hyperparams = get_optuna_search_space(trial, model_name)

    # Compute hyperparam hash
    sorted_hyperparams = sorted(hyperparams.items())
    hyperparam_str = json.dumps(sorted_hyperparams, sort_keys=True)
    hyperparam_hash = hashlib.md5(hyperparam_str.encode("utf-8")).hexdigest()

    # Check if this hyperparameter configuration has been evaluated before
    if (
        model_name in existing_hyperparams
        and hyperparam_hash in existing_hyperparams[model_name]
    ):
        objective_value = existing_hyperparams[model_name][hyperparam_hash]
        logger.info(
            f"Found existing trial for model '{model_name}' with hash '{hyperparam_hash}'. Skipping evaluation."
        )
        return objective_value

    # Proceed with evaluation
    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 3),
            subsample=hyperparams.get("subsample", 1.0),
            random_state=42,
        )
    elif model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 3),
            subsample=hyperparams.get("subsample", 1.0),
            colsample_bytree=hyperparams.get("colsample_bytree", 1.0),
            objective="reg:squarederror",
            verbosity=0,
            tree_method=(
                "gpu_hist" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"
            ),
            random_state=42,
        )
    elif model_name == "LightGBM":
        model = LGBMRegressor(
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", -1),
            num_leaves=hyperparams.get("num_leaves", 31),
            subsample=hyperparams.get("subsample", 1.0),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "CatBoost":
        model = CatBoostRegressor(
            iterations=hyperparams.get("iterations", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            depth=hyperparams.get("depth", 6),
            l2_leaf_reg=hyperparams.get("l2_leaf_reg", 3),
            subsample=hyperparams.get("subsample", 1.0),
            random_state=42,
            verbose=0,
        )
    elif model_name == "Support Vector Regression":
        model = SVR(
            C=hyperparams.get("C", 1.0),
            epsilon=hyperparams.get("epsilon", 0.1),
            gamma=hyperparams.get("gamma", "scale"),
        )
    elif model_name == "Ridge Regression":
        model = Ridge(
            alpha=1.0,  # You can include alpha in hyperparams if desired
            random_state=42,
        )
    elif model_name == "Linear Regression (OLS)":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # For models sensitive to feature scaling, include a scaler in the pipeline
    if model_name in ["Support Vector Regression"]:
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", model)])
    else:
        pipeline = model

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on training data
    preds = pipeline.predict(X_train)

    # Calculate MSE
    mse = mean_squared_error(y_train, preds)

    # Log the hyperparameter trial
    append_hyperparameters_log(hyperparam_log_file_path, model_name, hyperparams, mse)

    return mse  # Optuna minimizes the objective


def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_hash):
    """
    Train multiple models with hyperparameter optimization and evaluate their performance,
    utilizing logging to avoid redundant evaluations.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        dataset_hash (str): Unique identifier for the dataset.

    Returns:
        pd.DataFrame: DataFrame containing model performances.
    """
    # Define logs directory and log files
    logs_dir = "logs"
    ensure_logs_directory(logs_dir)
    evaluation_log_file_path = os.path.join(
        logs_dir, "imageability_evaluation_results.csv"
    )
    hyperparam_log_file_path = os.path.join(
        logs_dir, "imageability_hyperparameters.csv"
    )
    embedding_model = config.get("IMAGEABILITY").get("EMBEDDINGS").get("MODEL")
    upload_to_hf = False

    # Define the base models to evaluate
    base_models = [
        ("Linear Regression (OLS)", LinearRegression()),  # Baseline
        ("Ridge Regression", Ridge()),  # Baseline
        ("Support Vector Regression", SVR(kernel="linear")),
        ("Random Forest", RandomForestRegressor(random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        (
            "XGBoost",
            XGBRegressor(
                random_state=42,
                objective="reg:squarederror",
                verbosity=0,
                tree_method=(
                    "gpu_hist" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"
                ),
            ),
        ),
        ("LightGBM", LGBMRegressor(random_state=42)),
        ("CatBoost", CatBoostRegressor(random_state=42, verbose=0)),
    ]

    # Load existing evaluation logs
    existing_evaluation_logs = load_existing_logs(evaluation_log_file_path)

    # Load existing hyperparameter logs
    existing_hyperparams = load_existing_hyperparameters_log(hyperparam_log_file_path)

    results = []
    new_evaluations = []
    best_model = None
    best_metric = None

    # Iterate over each base model
    for name, model in tqdm(base_models, desc="Processing Models", unit="model"):
        # Check if this model and dataset_hash combination exists in evaluation logs
        if not existing_evaluation_logs.empty:
            mask = (existing_evaluation_logs["dataset_hash"] == dataset_hash) & (
                existing_evaluation_logs["model_name"] == name
            )
            existing_entry = existing_evaluation_logs[mask]

        if not existing_evaluation_logs.empty and not existing_entry.empty:
            # Retrieve existing results
            mse = existing_entry.iloc[0]["MSE"]
            rmse = existing_entry.iloc[0].get(
                "RMSE", mse**0.5
            )  # Handle older logs without RMSE
            r2 = existing_entry.iloc[0]["R2 Score"]
            source = "log"
            logger.info(f"Skipped training for '{name}'. Loaded results from logs.")
        else:
            # Hyperparameter Optimization with Optuna
            logger.info(f"\nStarting hyperparameter optimization for '{name}'...")
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.MedianPruner(),
            )
            n_trials = 50  # Adjust based on computational resources
            if name in ["Linear Regression (OLS)", "Ridge Regression"]:
                n_trials = 1

            study.optimize(
                lambda trial: objective(
                    trial,
                    name,
                    X_train,
                    y_train,
                    existing_hyperparams,
                    hyperparam_log_file_path,
                ),
                n_trials=n_trials,  # Adjust based on computational resources
                timeout=3600,  # Optional: Set a timeout in seconds
                # callbacks=[SklearnPruningCallback(study, "objective")],
            )
            best_params = study.best_params
            logger.info(f"Best params for '{name}': {best_params}")

            # Instantiate the best model with optimized hyperparameters
            if name == "Random Forest":
                best_estimator = RandomForestRegressor(
                    n_estimators=best_params.get("n_estimators", 100),
                    max_depth=best_params.get("max_depth", None),
                    min_samples_split=best_params.get("min_samples_split", 2),
                    min_samples_leaf=best_params.get("min_samples_leaf", 1),
                    random_state=42,
                    n_jobs=-1,
                )
            elif name == "Gradient Boosting":
                best_estimator = GradientBoostingRegressor(
                    n_estimators=best_params.get("n_estimators", 100),
                    learning_rate=best_params.get("learning_rate", 0.1),
                    max_depth=best_params.get("max_depth", 3),
                    subsample=best_params.get("subsample", 1.0),
                    random_state=42,
                )
            elif name == "XGBoost":
                best_estimator = XGBRegressor(
                    n_estimators=best_params.get("n_estimators", 100),
                    learning_rate=best_params.get("learning_rate", 0.1),
                    max_depth=best_params.get("max_depth", 3),
                    subsample=best_params.get("subsample", 1.0),
                    colsample_bytree=best_params.get("colsample_bytree", 1.0),
                    objective="reg:squarederror",
                    verbosity=0,
                    tree_method=(
                        "gpu_hist" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"
                    ),
                    random_state=42,
                )
            elif name == "LightGBM":
                best_estimator = LGBMRegressor(
                    n_estimators=best_params.get("n_estimators", 100),
                    learning_rate=best_params.get("learning_rate", 0.1),
                    max_depth=best_params.get("max_depth", -1),
                    num_leaves=best_params.get("num_leaves", 31),
                    subsample=best_params.get("subsample", 1.0),
                    random_state=42,
                    n_jobs=-1,
                )
            elif name == "CatBoost":
                best_estimator = CatBoostRegressor(
                    iterations=best_params.get("iterations", 100),
                    learning_rate=best_params.get("learning_rate", 0.1),
                    depth=best_params.get("depth", 6),
                    l2_leaf_reg=best_params.get("l2_leaf_reg", 3),
                    subsample=best_params.get("subsample", 1.0),
                    random_state=42,
                    verbose=0,
                )
            elif name == "Support Vector Regression":
                best_estimator = SVR(
                    C=best_params.get("C", 1.0),
                    epsilon=best_params.get("epsilon", 0.1),
                    gamma=best_params.get("gamma", "scale"),
                )
            elif name == "Ridge Regression":
                best_estimator = Ridge(
                    alpha=1.0,  # You can include alpha in hyperparams if desired
                    random_state=42,
                )
            elif name == "Linear Regression (OLS)":
                best_estimator = LinearRegression()
            else:
                raise ValueError(f"Unknown model name: {name}")

            # For models sensitive to feature scaling, include a scaler in the pipeline
            if name in ["Support Vector Regression"]:
                pipeline = Pipeline(
                    [("scaler", StandardScaler()), ("regressor", best_estimator)]
                )
            else:
                pipeline = best_estimator

            # Train the best estimator on the full training data
            pipeline.fit(X_train, y_train)

            # Predict on testing data
            predictions = pipeline.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = mse**0.5
            r2 = r2_score(y_test, predictions)
            logger.info(
                f"{name} - RMSE: {rmse:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}"
            )

            # Append the results to the new evaluations list
            new_evaluations.append(
                {
                    "dataset_hash": dataset_hash,
                    "dataset_name": config.get("IMAGEABILITY")
                    .get("PREDICTOR")
                    .get("EVAL")
                    .get("DATASET"),
                    "embedding_model": embedding_model,
                    "model_name": name,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2 Score": r2,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            source = "evaluated"

            # Save the best estimator
            model_name_clean = name.replace(" ", "_").lower()
            filename = (
                f"models/{model_name_clean}-{embedding_model.replace('/', '_')}.joblib"
            )
            os.makedirs("models", exist_ok=True)
            joblib.dump(pipeline, filename)
            logger.info(f"Trained model '{name}' saved to '{filename}'.")

            # Optionally, upload the model to Hugging Face Hub
            # Uncomment the following lines if you want to enable uploading
            # upload_model(filename)
            # upload_to_hf = True

        # Append to the results list
        if "rmse" not in locals():
            rmse = None  # Handle cases where model was loaded from log
            mse = None
            r2 = None
        results.append(
            {
                "Model": name,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2,
                "Source": source,
            }
        )

        # Determine the best model based on lowest RMSE
        if rmse is not None and (best_metric is None or rmse < best_metric):
            best_metric = rmse
            best_model = (
                (name, pipeline) if source == "evaluated" else (name, None)
            )  # Handle if model is from log

    # Append new evaluations to the evaluation log file
    if new_evaluations:
        for entry in new_evaluations:
            append_to_log(evaluation_log_file_path, entry)

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Display the results
    logger.info("\nModel Performances:")
    logger.info(results_df.to_string(index=False))

    # Determine the best model considering both logged and newly evaluated models
    if not results_df.empty:
        best_row = results_df.loc[results_df["RMSE"].idxmin()]
        best_method = best_row["Model"]
        best_mse = best_row["MSE"]
        best_rmse = best_row["RMSE"]
        best_r2 = best_row["R2 Score"]

        logger.info(
            f"\nConclusion: The best performing model is '{best_method}' with an RMSE of {best_rmse:.4f}, MSE of {best_mse:.4f}, and an R2 Score of {best_r2:.4f}."
        )

        # Optionally, load the best model from logs or save the newly trained best model
        if best_row["Source"] == "evaluated":
            # The best_model has already been saved earlier
            logger.info(f"The best model '{best_method}' has been saved.")
            upload_to_hf = True
        else:
            logger.info(
                f"The best model '{best_method}' was retrieved from existing logs."
            )

        if upload_to_hf:
            # Upload the best model to Hugging Face Hub
            model_name_clean = best_method.replace(" ", "_").lower()
            filename = (
                f"models/{model_name_clean}-{embedding_model.replace('/', '_')}.joblib"
            )
            upload_model(filename)
    else:
        logger.warning("No model performances to display.")

    # Implement Ensemble Methods
    implement_ensemble_methods(
        X_train, X_test, y_train, y_test, results_df, base_models, dataset_hash
    )


def implement_ensemble_methods(
    X_train, X_test, y_train, y_test, results_df, base_models, dataset_hash
):
    """
    Implement ensemble methods such as VotingRegressor and StackingRegressor.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        results_df (pd.DataFrame): DataFrame containing model performances.
        base_models (list): List of base models.
        dataset_hash (str): Unique identifier for the dataset.
    """
    # Select top N models based on RMSE
    top_n = 5
    top_models = results_df.nsmallest(top_n, "RMSE")["Model"].tolist()
    logger.info(f"\nTop {top_n} models selected for ensembling: {top_models}")

    # Retrieve the trained models
    trained_models = []
    for name, _ in base_models:
        if name in top_models:
            model_filename = f"models/{name.replace(' ', '_').lower()}-{config.get('IMAGEABILITY').get('EMBEDDINGS').get('MODEL').replace('/', '_')}.joblib"
            if os.path.exists(model_filename):
                trained_model = joblib.load(model_filename)
                trained_models.append((name, trained_model))
                logger.info(f"Loaded model '{name}' for ensembling.")
            else:
                logger.warning(
                    f"Model file '{model_filename}' not found. Skipping '{name}'."
                )

    if not trained_models:
        logger.warning("No trained models available for ensembling.")
        return

    # Voting Regressor
    try:
        voting_reg = VotingRegressor(estimators=trained_models, n_jobs=-1)
        voting_reg.fit(X_train, y_train)
        voting_pred = voting_reg.predict(X_test)
        voting_mse = mean_squared_error(y_test, voting_pred)
        voting_rmse = voting_mse**0.5
        voting_r2 = r2_score(y_test, voting_pred)
        logger.info(
            f"Voting Regressor - RMSE: {voting_rmse:.4f}, MSE: {voting_mse:.4f}, R2 Score: {voting_r2:.4f}"
        )

        # Save Voting Regressor
        voting_filename = f"models/voting_regressor-{config.get('IMAGEABILITY').get('EMBEDDINGS').get('MODEL').replace('/', '_')}.joblib"
        joblib.dump(voting_reg, voting_filename)
        logger.info(f"Voting Regressor saved to '{voting_filename}'.")

    except Exception as e:
        logger.error(f"Failed to train Voting Regressor: {e}")

    # Stacking Regressor
    try:
        # Define stacking regressor with the trained models as estimators
        stacking_reg = StackingRegressor(
            estimators=trained_models,
            final_estimator=Ridge(),
            n_jobs=-1,
            passthrough=False,
        )
        stacking_reg.fit(X_train, y_train)
        stacking_pred = stacking_reg.predict(X_test)
        stacking_mse = mean_squared_error(y_test, stacking_pred)
        stacking_rmse = stacking_mse**0.5
        stacking_r2 = r2_score(y_test, stacking_pred)
        logger.info(
            f"Stacking Regressor - RMSE: {stacking_rmse:.4f}, MSE: {stacking_mse:.4f}, R2 Score: {stacking_r2:.4f}"
        )

        # Save Stacking Regressor
        stacking_filename = f"models/stacking_regressor-{config.get('IMAGEABILITY').get('EMBEDDINGS').get('MODEL').replace('/', '_')}.joblib"
        joblib.dump(stacking_reg, stacking_filename)
        logger.info(f"Stacking Regressor saved to '{stacking_filename}'.")
    except Exception as e:
        logger.error(f"Failed to train Stacking Regressor: {e}")

    # Optionally, evaluate and log ensemble models
    ensemble_results = []
    if "voting_rmse" in locals():
        ensemble_results.append(
            {
                "Model": "Voting Regressor",
                "MSE": voting_mse,
                "RMSE": voting_rmse,
                "R2 Score": voting_r2,
                "Source": "ensemble",
            }
        )
    if "stacking_rmse" in locals():
        ensemble_results.append(
            {
                "Model": "Stacking Regressor",
                "MSE": stacking_mse,
                "RMSE": stacking_rmse,
                "R2 Score": stacking_r2,
                "Source": "ensemble",
            }
        )

    # Log ensemble results
    if ensemble_results:
        logger.info("\nEnsemble Model Performances:")
        for res in ensemble_results:
            logger.info(
                f"{res['Model']} - RMSE: {res['RMSE']:.4f}, MSE: {res['MSE']:.4f}, R2 Score: {res['R2 Score']:.4f}"
            )

        # Optionally, append ensemble results to the evaluation log file
        evaluation_log_file_path = os.path.join(
            "logs", "imageability_evaluation_results.csv"
        )
        for res in ensemble_results:
            entry = {
                "dataset_hash": dataset_hash,
                "dataset_name": config.get("IMAGEABILITY")
                .get("PREDICTOR")
                .get("EVAL")
                .get("DATASET"),
                "embedding_model": config.get("IMAGEABILITY")
                .get("EMBEDDINGS")
                .get("MODEL"),
                "model_name": res["Model"],
                "MSE": res["MSE"],
                "RMSE": res["RMSE"],
                "R2 Score": res["R2 Score"],
                "timestamp": datetime.now().isoformat(),
            }
            append_to_log(evaluation_log_file_path, entry)


def upload_model(model_path: str):
    """
    Upload model to Hugging Face Hub.

    Parameters
    ----------
    model_path : str
        The path to the model file to upload.
    """
    file_name = os.path.basename(model_path)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=file_name,
        repo_id=config.get("IMAGEABILITY").get("PREDICTOR").get("REPO"),
        repo_type="model",
    )
    logger.info("Model uploaded to Hugging Face Hub")


if __name__ == "__main__":
    # Install the following extra dependencies:
    # pip install scikit-learn lightgbm xgboost catboost optuna

    # Ensure logs directory exists
    ensure_logs_directory("logs")

    # Load data
    embeddings, scores, dataset_hash = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = split_dataset(embeddings, scores)

    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_hash)

    # Ensemble methods are integrated within the train_and_evaluate_models function
