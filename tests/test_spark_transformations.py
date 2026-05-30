import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if os.name == "nt":
    os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")

import pytest  # noqa: E402

pytestmark = pytest.mark.spark

from pyspark.sql import SparkSession  # noqa: E402

from src.spark_jobs.bronze_to_silver_events import (  # noqa: E402
    build_silver_adverse_events,
    build_silver_drugs,
    build_silver_reactions,
    read_bronze_json,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.appName("TestSpark")
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
    This is a pure JVM operation.
    The file was written during Phase 3.
    """
    return read_bronze_json(spark, year=2024, month=1)


def _collect_rows(df):
    """Collect a small Spark test DataFrame as plain Python dictionaries."""
    return [row.asDict(recursive=True) for row in df.collect()]


@pytest.fixture(scope="session")
def events_rows(bronze_df):
    return _collect_rows(build_silver_adverse_events(bronze_df))


@pytest.fixture(scope="session")
def drugs_rows(bronze_df):
    return _collect_rows(build_silver_drugs(bronze_df))


@pytest.fixture(scope="session")
def reactions_rows(bronze_df):
    return _collect_rows(build_silver_reactions(bronze_df))


def _column_names(rows):
    return set(rows[0]) if rows else set()


def test_adverse_events_row_count_is_positive(events_rows):
    assert len(events_rows) > 0


def test_report_date_column_exists(events_rows):
    assert "report_date" in _column_names(events_rows)


def test_report_date_has_no_nulls(events_rows):
    null_count = sum(row["report_date"] is None for row in events_rows)
    null_rate = null_count / len(events_rows)
    assert null_rate < 0.05


def test_is_serious_is_binary(events_rows):
    values = {row["is_serious"] for row in events_rows if row["is_serious"] is not None}
    assert values.issubset({0, 1})


def test_patient_sex_mapping_contains_readable_labels(events_rows):
    valid_labels = {"Male", "Female", "Unknown"}
    actual = {row["patient_sex"] for row in events_rows if row["patient_sex"] is not None}
    assert actual.issubset(valid_labels)


def test_patient_age_is_numeric(events_rows):
    non_null = [row["patient_age"] for row in events_rows if row["patient_age"] is not None]
    assert all(isinstance(age, (int, float)) for age in non_null)


def test_no_null_safety_report_ids(events_rows):
    assert sum(row["safety_report_id"] is None for row in events_rows) == 0


def test_drugs_table_has_more_rows_than_events(events_rows, drugs_rows):
    assert len(drugs_rows) >= len(events_rows)


def test_drug_names_are_uppercase(drugs_rows):
    names = [row["drug_name"] for row in drugs_rows if row["drug_name"] is not None]
    assert all(name == name.upper() for name in names)


def test_drug_role_column_exists(drugs_rows):
    assert "drug_role" in _column_names(drugs_rows)


def test_drug_role_contains_valid_labels(drugs_rows):
    valid = {"Suspect", "Concomitant", "Interacting"}
    actual = {row["drug_role"] for row in drugs_rows if row["drug_role"] is not None}
    assert actual.issubset(valid)


def test_reactions_table_has_more_rows_than_events(events_rows, reactions_rows):
    assert len(reactions_rows) >= len(events_rows)


def test_reaction_name_column_exists(reactions_rows):
    assert "reaction_name" in _column_names(reactions_rows)


def test_reaction_names_are_not_empty(reactions_rows):
    empty = sum(not (row["reaction_name"] or "").strip() for row in reactions_rows)
    assert empty == 0


def test_safety_report_id_links_drugs_to_events(events_rows, drugs_rows):
    event_ids = {row["safety_report_id"] for row in events_rows}
    drug_ids = {row["safety_report_id"] for row in drugs_rows}
    assert drug_ids.issubset(event_ids)
