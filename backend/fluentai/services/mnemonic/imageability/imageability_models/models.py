import os
from datetime import datetime

import joblib
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.services.mnemonic.imageability.imageability_models.data import (
    append_to_log,
    ensure_logs_directory,
    load_data,
    load_existing_hyperparameters_log,
    load_existing_logs,
    split_dataset,
    upload_model,
)
from fluentai.services.mnemonic.imageability.imageability_models.ensemble import (
    implement_ensemble_methods,
)
from fluentai.services.mnemonic.imageability.imageability_models.optimization import (
    objective,
)


def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_hash):
    """
    Train multiple models with hyperparameter optimization and evaluate their performance.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        dataset_hash (str): Unique identifier for the dataset.

    Returns
    -------
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
            n_trials = 10  # Adjust based on computational resources
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
                    n_jobs=-1,
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
                    task_type="GPU",
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
