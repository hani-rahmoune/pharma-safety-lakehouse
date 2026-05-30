import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if os.name == "nt":
    os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")

import pytest  # noqa: E402

pytestmark = pytest.mark.spark

from pyspark.sql import SparkSession  # noqa: E402

from src.quality.data_quality_checks import (  # noqa: E402
    DataQualityError,
    check_drug_name_not_empty,
    check_is_serious_is_binary,
    check_no_null_safety_report_ids,
    check_patient_age_realistic,
    check_reaction_name_not_empty,
    check_required_columns_exist,
    check_row_count_minimum,
    run_adverse_events_checks,
    run_drugs_checks,
    run_reactions_checks,
)
from src.utils.config import SILVER_LOCAL_PATH  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .appName("TestQuality")
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
def events_df(spark):
    """Read silver_adverse_events from disk. Pure JVM, no Python worker."""
    path = str(Path(SILVER_LOCAL_PATH) / "adverse_events")
    return spark.read.parquet(path)


@pytest.fixture(scope="session")
def drugs_df(spark):
    path = str(Path(SILVER_LOCAL_PATH) / "drugs")
    return spark.read.parquet(path)


@pytest.fixture(scope="session")
def reactions_df(spark):
    path = str(Path(SILVER_LOCAL_PATH) / "reactions")
    return spark.read.parquet(path)


def test_events_has_no_null_safety_report_ids(events_df):
    check_no_null_safety_report_ids(events_df)


def test_events_is_serious_is_binary(events_df):
    check_is_serious_is_binary(events_df)


def test_events_report_date_null_rate_is_acceptable(events_df):
    from src.quality.data_quality_checks import check_report_date_null_rate

    check_report_date_null_rate(events_df)


def test_events_patient_age_is_realistic(events_df):
    check_patient_age_realistic(events_df)


def test_events_row_count_above_minimum(events_df):
    check_row_count_minimum(events_df, "silver_adverse_events")


def test_events_has_required_columns(events_df):
    check_required_columns_exist(
        events_df,
        ["safety_report_id", "report_date", "is_serious", "country"],
        "silver_adverse_events",
    )


def test_drugs_name_not_empty(drugs_df):
    check_drug_name_not_empty(drugs_df)


def test_drugs_row_count_above_minimum(drugs_df):
    check_row_count_minimum(drugs_df, "silver_drugs")


def test_drugs_has_required_columns(drugs_df):
    check_required_columns_exist(
        drugs_df,
        ["safety_report_id", "drug_name", "drug_role"],
        "silver_drugs",
    )


def test_reactions_name_not_empty(reactions_df):
    check_reaction_name_not_empty(reactions_df)


def test_reactions_row_count_above_minimum(reactions_df):
    check_row_count_minimum(reactions_df, "silver_reactions")


def test_reactions_has_required_columns(reactions_df):
    check_required_columns_exist(
        reactions_df,
        ["safety_report_id", "reaction_name"],
        "silver_reactions",
    )


def test_null_safety_report_id_raises(spark, tmp_path_factory):
    """
    Inject a bad row and confirm the check catches it.
    Writing a small parquet file from Python then reading it back
    avoids the Python worker crash issue.
    """
    import pandas as pd

    path = str(tmp_path_factory.mktemp("bad_events"))
    pd.DataFrame({"safety_report_id": [None, "123"]}).to_parquet(
        path + "/part.parquet"
    )
    bad_df = spark.read.parquet(path)

    with pytest.raises(DataQualityError, match="null safety_report_id"):
        check_no_null_safety_report_ids(bad_df)


def test_non_binary_is_serious_raises(spark, tmp_path_factory):
    import pandas as pd

    path = str(tmp_path_factory.mktemp("bad_serious"))
    pd.DataFrame({"is_serious": [5]}).to_parquet(path + "/part.parquet")
    bad_df = spark.read.parquet(path)

    with pytest.raises(DataQualityError, match="is_serious is not 0 or 1"):
        check_is_serious_is_binary(bad_df)


def test_unrealistic_age_raises(spark, tmp_path_factory):
    import pandas as pd

    path = str(tmp_path_factory.mktemp("bad_age"))
    pd.DataFrame({"patient_age": [200]}).to_parquet(path + "/part.parquet")
    bad_df = spark.read.parquet(path)

    with pytest.raises(DataQualityError, match="unrealistic patient_age"):
        check_patient_age_realistic(bad_df)


def test_missing_column_raises(spark, tmp_path_factory):
    import pandas as pd

    path = str(tmp_path_factory.mktemp("missing_col"))
    pd.DataFrame({"drug_name": ["HUMIRA"]}).to_parquet(path + "/part.parquet")
    bad_df = spark.read.parquet(path)

    with pytest.raises(DataQualityError, match="missing required columns"):
        check_required_columns_exist(
            bad_df,
            ["drug_name", "safety_report_id"],
            "test_table",
        )


def test_full_adverse_events_suite_passes(events_df):
    run_adverse_events_checks(events_df)


def test_full_drugs_suite_passes(drugs_df):
    run_drugs_checks(drugs_df)


def test_full_reactions_suite_passes(reactions_df):
    run_reactions_checks(reactions_df)