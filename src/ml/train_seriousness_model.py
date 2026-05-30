import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import GOLD_LOCAL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "seriousness_prediction"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS = [
    "patient_age",
    "patient_sex",
    "country",
    "num_drugs",
    "num_reactions",
    "has_suspect_drug",
]
TARGET_COL = "is_serious"


def load_ml_features(path: str = "data/gold/ml_features") -> pd.DataFrame:
    """
    Load ML features for training.

    Priority:
    1. Local Parquet file/folder if it exists.
    2. BigQuery gold table if local data is not available.
    """
    from pathlib import Path

    local_path = Path(path)

    if local_path.exists():
        return pd.read_parquet(local_path)

    from google.cloud import bigquery
    from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET_GOLD

    client = bigquery.Client(project=GCP_PROJECT_ID)

    query = f"""
    SELECT *
    FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET_GOLD}.ml_features`
    """

    return client.query(query).to_dataframe()


def prepare_features(df: pd.DataFrame):
    """
    Prepare the feature matrix X and target vector y.

    Categorical columns (patient_sex, country) are label-encoded
    because scikit-learn models require numeric input. LabelEncoder
    converts each unique string to an integer.

    Numeric columns are filled with the column median for missing
    values. Using the median instead of the mean is more robust
    to outliers, which are common in medical data.

    Returns X (features), y (target), and the fitted encoders
    so we can apply the same encoding to new data later.
    """
    df = df[FEATURE_COLS + [TARGET_COL]].copy()

    encoders = {}
    for col in ["patient_sex", "country"]:
        le = LabelEncoder()
        df[col] = df[col].fillna("Unknown").astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    for col in ["patient_age", "num_drugs", "num_reactions", "has_suspect_drug"]:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].fillna(0).astype(int)

    logger.info(
        "Features prepared: %d rows, class balance: %d serious / %d not serious",
        len(y), y.sum(), (y == 0).sum()
    )
    return X, y, encoders


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Compute standard binary classification metrics.

    accuracy:  overall correct predictions
    precision: of all predicted serious, how many were actually serious
    recall:    of all actual serious, how many did we catch
    f1:        harmonic mean of precision and recall
    roc_auc:   area under the ROC curve, threshold-independent measure
                of how well the model separates serious from non-serious
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


def train_and_log(
    model,
    model_name: str,
    params: dict,
    X_train, X_test,
    y_train, y_test,
) -> dict:
    """
    Train a model, evaluate it, and log everything to MLflow.

    MLflow records:
    - Parameters: hyperparameters used to configure the model
    - Metrics: evaluation scores on the test set
    - Model artifact: the serialized model object

    This lets you compare runs in the MLflow UI, reproduce any
    experiment by its run ID, and load the best model for deployment.

    Returns the metrics dict so callers can inspect results.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", FEATURE_COLS)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("%s — %s", model_name, metrics)

    return metrics


def run_training() -> dict:
    """
    Train Logistic Regression and Random Forest models.
    Log both to MLflow and return their metrics.
    """
    df = load_ml_features()
    X, y, _ = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    lr_params = {"C": 1.0, "max_iter": 300, "solver": "lbfgs", "random_state": RANDOM_STATE}
    lr_metrics = train_and_log(
        LogisticRegression(**lr_params),
        "LogisticRegression",
        lr_params,
        X_train, X_test, y_train, y_test,
    )

    rf_params = {"n_estimators": 100, "max_depth": 6, "random_state": RANDOM_STATE}
    rf_metrics = train_and_log(
        RandomForestClassifier(**rf_params),
        "RandomForest",
        rf_params,
        X_train, X_test, y_train, y_test,
    )

    logger.info("Training complete. Open http://localhost:5000 to compare runs.")
    return {"LogisticRegression": lr_metrics, "RandomForest": rf_metrics}


if __name__ == "__main__":
    run_training()