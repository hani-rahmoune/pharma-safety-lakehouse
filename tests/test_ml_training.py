import numpy as np
import pandas as pd
import pytest

from src.ml.train_seriousness_model import (
    FEATURE_COLS,
    TARGET_COL,
    compute_metrics,
    prepare_features,
)


def make_sample_df(n=100):
    """
    Build a small synthetic ml_features DataFrame for testing.
    Does not call the real data or MLflow.
    """
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "patient_age":      rng.integers(0, 90, size=n).astype(float),
        "patient_sex":      rng.choice(["Male", "Female", "Unknown"], size=n),
        "country":          rng.choice(["FR", "US", "DE", "GB"], size=n),
        "num_drugs":        rng.integers(1, 5, size=n).astype(float),
        "num_reactions":    rng.integers(1, 4, size=n).astype(float),
        "has_suspect_drug": rng.integers(0, 2, size=n).astype(float),
        "is_serious":       rng.integers(0, 2, size=n),
    })


def test_prepare_features_returns_correct_shape():
    df = make_sample_df(80)
    X, y, encoders = prepare_features(df)
    assert X.shape == (80, len(FEATURE_COLS))
    assert len(y) == 80


def test_prepare_features_no_nulls_in_X():
    df = make_sample_df()
    df.loc[0, "patient_age"] = None
    df.loc[1, "patient_sex"] = None
    df.loc[2, "country"] = None
    X, y, _ = prepare_features(df)
    assert not X.isnull().any().any()


def test_prepare_features_y_is_binary():
    df = make_sample_df()
    _, y, _ = prepare_features(df)
    assert set(y.unique()).issubset({0, 1})


def test_prepare_features_categorical_columns_are_numeric():
    df = make_sample_df()
    X, _, _ = prepare_features(df)
    assert X["patient_sex"].dtype in ["int32", "int64"]
    assert X["country"].dtype in ["int32", "int64"]


def test_prepare_features_encoders_returned():
    df = make_sample_df()
    _, _, encoders = prepare_features(df)
    assert "patient_sex" in encoders
    assert "country" in encoders


def test_compute_metrics_returns_all_keys():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.3, 0.85, 0.6])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics


def test_compute_metrics_values_between_0_and_1():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.6, 0.2, 0.8, 0.55])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val} is out of range"


def test_compute_metrics_perfect_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_full_prepare_pipeline_with_missing_values():
    df = make_sample_df(50)
    df.loc[:5, "patient_age"] = None
    df.loc[10:15, "country"] = None
    df.loc[20, "is_serious"] = None
    X, y, _ = prepare_features(df)
    assert not X.isnull().any().any()
    assert set(y.unique()).issubset({0, 1})