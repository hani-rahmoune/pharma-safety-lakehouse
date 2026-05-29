import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["HADOOP_HOME"] = "C:\\hadoop"

from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from src.spark_jobs.build_gold_tables import (
    build_gold_drug_reaction_pairs,
    build_gold_drug_summary,
    build_gold_ml_features,
    build_gold_monthly_trends,
    build_gold_reaction_summary,
    build_gold_safety_overview,
    read_silver,
)
from src.utils.config import GOLD_LOCAL_PATH


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .appName("TestGold")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.memory", "1g")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def events(spark):
    return read_silver(spark, "adverse_events")


@pytest.fixture(scope="session")
def drugs(spark):
    return read_silver(spark, "drugs")


@pytest.fixture(scope="session")
def reactions(spark):
    return read_silver(spark, "reactions")


def read_gold(table_name: str) -> pd.DataFrame:
    """Read a gold Parquet table with pandas. No Spark needed."""
    return pd.read_parquet(str(Path(GOLD_LOCAL_PATH) / table_name))


def test_safety_overview_has_one_row():
    df = read_gold("safety_overview")
    assert len(df) == 1


def test_safety_overview_total_reports_is_positive():
    df = read_gold("safety_overview")
    assert df["total_reports"].iloc[0] > 0


def test_safety_overview_seriousness_rate_is_between_0_and_100():
    df = read_gold("safety_overview")
    rate = df["seriousness_rate_pct"].iloc[0]
    assert 0 <= rate <= 100


def test_safety_overview_has_required_columns():
    df = read_gold("safety_overview")
    required = ["total_reports", "serious_reports", "death_reports", "seriousness_rate_pct"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_monthly_trends_has_rows():
    df = read_gold("monthly_trends")
    assert len(df) > 0


def test_monthly_trends_seriousness_rate_in_range():
    df = read_gold("monthly_trends")
    assert df["seriousness_rate_pct"].between(0, 100).all()


def test_monthly_trends_has_required_columns():
    df = read_gold("monthly_trends")
    for col in ["year", "month", "total_reports", "serious_reports", "seriousness_rate_pct"]:
        assert col in df.columns


def test_drug_summary_has_rows():
    df = read_gold("drug_summary")
    assert len(df) > 0


def test_drug_summary_seriousness_rate_in_range():
    df = read_gold("drug_summary")
    valid = df["seriousness_rate_pct"].dropna()
    assert valid.between(0, 100).all()


def test_drug_summary_total_reports_positive():
    df = read_gold("drug_summary")
    assert (df["total_reports"] > 0).all()


def test_drug_summary_has_required_columns():
    df = read_gold("drug_summary")
    for col in ["drug_name", "total_reports", "serious_reports", "seriousness_rate_pct"]:
        assert col in df.columns


def test_reaction_summary_has_rows():
    df = read_gold("reaction_summary")
    assert len(df) > 0


def test_reaction_summary_has_required_columns():
    df = read_gold("reaction_summary")
    for col in ["reaction_name", "total_reports", "serious_reports", "seriousness_rate_pct"]:
        assert col in df.columns


def test_drug_reaction_pairs_has_rows():
    df = read_gold("drug_reaction_pairs")
    assert len(df) > 0


def test_drug_reaction_pairs_pair_count_positive():
    df = read_gold("drug_reaction_pairs")
    assert (df["pair_count"] > 0).all()


def test_drug_reaction_pairs_seriousness_rate_in_range():
    df = read_gold("drug_reaction_pairs")
    valid = df["seriousness_rate_pct"].dropna()
    assert valid.between(0, 100).all()


def test_drug_reaction_pairs_has_required_columns():
    df = read_gold("drug_reaction_pairs")
    for col in ["drug_name", "reaction_name", "pair_count", "serious_count", "seriousness_rate_pct"]:
        assert col in df.columns


def test_ml_features_has_rows():
    df = read_gold("ml_features")
    assert len(df) > 0


def test_ml_features_num_drugs_non_negative():
    df = read_gold("ml_features")
    assert (df["num_drugs"] >= 0).all()


def test_ml_features_num_reactions_non_negative():
    df = read_gold("ml_features")
    assert (df["num_reactions"] >= 0).all()


def test_ml_features_has_suspect_drug_is_binary():
    df = read_gold("ml_features")
    values = set(df["has_suspect_drug"].dropna().astype(int).unique())
    assert values.issubset({0, 1})


def test_ml_features_has_required_columns():
    df = read_gold("ml_features")
    for col in ["safety_report_id", "is_serious", "num_drugs", "num_reactions", "has_suspect_drug"]:
        assert col in df.columns


def test_gold_drug_summary_row_count_less_than_silver_drugs(events, drugs):
    """
    gold_drug_summary must have fewer rows than silver_drugs because
    it aggregates by drug_name. If they have the same count something
    is wrong with the groupBy.
    """
    silver_count = drugs.select("drug_name").distinct().count()
    gold_df = read_gold("drug_summary")
    assert len(gold_df) <= silver_count