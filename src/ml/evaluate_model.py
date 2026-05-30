import logging

import mlflow

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "seriousness_prediction"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_best_run() -> dict:
    """
    Find the best run in the seriousness_prediction experiment
    by ROC-AUC score.

    Returns a dict with run_id, model_type, and all metrics.
    This is used by the Airflow DAG to log which model won.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in the experiment.")

    best = runs[0]
    result = {
        "run_id": best.info.run_id,
        "model_type": best.data.params.get("model_type"),
        **best.data.metrics,
    }

    logger.info("Best model: %s (run_id=%s)", result["model_type"], result["run_id"])
    logger.info("Metrics: %s", {k: v for k, v in result.items() if k not in ["run_id", "model_type"]})
    return result


if __name__ == "__main__":
    get_best_run()
