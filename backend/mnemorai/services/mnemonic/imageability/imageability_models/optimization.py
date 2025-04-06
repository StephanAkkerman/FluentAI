import hashlib
import json
import os

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from mnemorai.logger import logger
from mnemorai.services.mnemonic.imageability.imageability_models.data import (
    append_hyperparameters_log,
)


def get_optuna_search_space(trial, model_name):
    """
    Define hyperparameter search space based on the model.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        model_name (str): Name of the model.

    Returns
    -------
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

    Returns
    -------
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
