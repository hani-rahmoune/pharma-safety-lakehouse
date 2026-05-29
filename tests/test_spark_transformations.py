import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["HADOOP_HOME"] = "C:\\hadoop"

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from src.spark_jobs.bronze_to_silver_events import (
    build_silver_adverse_events,
    build_silver_drugs,
    build_silver_reactions,
    read_bronze_json,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .appName("TestSpark")
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
def bronze_df(spark):
    """
    Read the real bronze JSON file from disk.
    This is a pure JVM operation — no Python worker needed.
    The file was written during Phase 3.
    """
    return read_bronze_json(spark, year=2024, month=1)


@pytest.fixture(scope="session")
def events_pdf(spark, bronze_df, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("events"))
    build_silver_adverse_events(bronze_df).write.mode("overwrite").parquet(path)
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def drugs_pdf(spark, bronze_df, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("drugs"))
    build_silver_drugs(bronze_df).write.mode("overwrite").parquet(path)
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def reactions_pdf(spark, bronze_df, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("reactions"))
    build_silver_reactions(bronze_df).write.mode("overwrite").parquet(path)
    return pd.read_parquet(path)


def test_adverse_events_row_count_is_positive(events_pdf):
    assert len(events_pdf) > 0


def test_report_date_column_exists(events_pdf):
    assert "report_date" in events_pdf.columns


def test_report_date_has_no_nulls(events_pdf):
    null_rate = events_pdf["report_date"].isnull().mean()
    assert null_rate < 0.05


def test_is_serious_is_binary(events_pdf):
    values = set(events_pdf["is_serious"].dropna().astype(int).unique())
    assert values.issubset({0, 1})


def test_patient_sex_mapping_contains_readable_labels(events_pdf):
    valid_labels = {"Male", "Female", "Unknown"}
    actual = set(events_pdf["patient_sex"].dropna().unique())
    assert actual.issubset(valid_labels)


def test_patient_age_is_numeric(events_pdf):
    non_null = events_pdf["patient_age"].dropna()
    assert non_null.dtype in ["int32", "int64", "float64"]


def test_no_null_safety_report_ids(events_pdf):
    assert events_pdf["safety_report_id"].isnull().sum() == 0


def test_drugs_table_has_more_rows_than_events(events_pdf, drugs_pdf):
    assert len(drugs_pdf) >= len(events_pdf)


def test_drug_names_are_uppercase(drugs_pdf):
    names = drugs_pdf["drug_name"].dropna()
    assert all(n == n.upper() for n in names)


def test_drug_role_column_exists(drugs_pdf):
    assert "drug_role" in drugs_pdf.columns


def test_drug_role_contains_valid_labels(drugs_pdf):
    valid = {"Suspect", "Concomitant", "Interacting"}
    actual = set(drugs_pdf["drug_role"].dropna().unique())
    assert actual.issubset(valid)


def test_reactions_table_has_more_rows_than_events(events_pdf, reactions_pdf):
    assert len(reactions_pdf) >= len(events_pdf)


def test_reaction_name_column_exists(reactions_pdf):
    assert "reaction_name" in reactions_pdf.columns


def test_reaction_names_are_not_empty(reactions_pdf):
    empty = (reactions_pdf["reaction_name"].fillna("").str.strip() == "").sum()
    assert empty == 0


def test_safety_report_id_links_drugs_to_events(events_pdf, drugs_pdf):
    event_ids = set(events_pdf["safety_report_id"])
    drug_ids = set(drugs_pdf["safety_report_id"])
    assert drug_ids.issubset(event_ids)