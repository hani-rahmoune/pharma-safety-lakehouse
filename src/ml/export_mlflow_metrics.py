import logging
from pathlib import Path

import mlflow
import pandas as pd

from src.utils.config import GOLD_LOCAL_PATH

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "seriousness_prediction"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def export_all_runs_to_parquet() -> str:
    """
    Export all MLflow experiment runs to a Parquet file in the gold layer.

    This makes model metrics queryable from BigQuery and displayable
    in Power BI on the ML Monitoring dashboard page.

    Returns the path where the file was saved.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    if not runs:
        logger.warning("No runs to export.")
        return ""

    records = []
    for run in runs:
        record = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": pd.Timestamp(run.info.start_time, unit="ms"),
            **run.data.params,
            **run.data.metrics,
        }
        records.append(record)

    df = pd.DataFrame(records)
    output_dir = Path(GOLD_LOCAL_PATH) / "mlflow_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "runs.parquet")
    df.to_parquet(output_path, index=False)

    logger.info("Exported %d runs to %s", len(records), output_path)
    return output_path


if __name__ == "__main__":
    export_all_runs_to_parquet()