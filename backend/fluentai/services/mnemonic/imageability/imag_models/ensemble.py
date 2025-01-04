import os
from datetime import datetime

import joblib
from sklearn.ensemble import (
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.services.mnemonic.imageability.imag_models.data import (
    append_to_log,
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
