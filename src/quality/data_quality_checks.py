import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_NULL_RATE = 0.05
MIN_ROW_COUNT = 10


class DataQualityError(Exception):
    """
    Raised when a data quality check fails.

    Using a custom exception lets the Airflow task catch it specifically
    and fail the pipeline with a clear message, rather than a generic
    Spark or Python error.
    """
    pass


def check_no_null_safety_report_ids(df: DataFrame) -> None:
    """
    Every row must have a safety_report_id.

    A null ID means the report cannot be joined to the drugs or
    reactions tables. Any null here is a data ingestion failure.
    """
    null_count = df.filter(F.col("safety_report_id").isNull()).count()
    if null_count > 0:
        raise DataQualityError(
            f"Found {null_count} rows with null safety_report_id"
        )
    logger.info("PASS: no null safety_report_id (%d rows checked)", df.count())


def check_is_serious_is_binary(df: DataFrame) -> None:
    """
    The is_serious column must contain only 0, 1, or null.

    After the binary flag fix in Phase 4, value 2 is converted to 0.
    Any value outside {0, 1, null} means the transformation logic broke.
    """
    invalid = df.filter(
        F.col("is_serious").isNotNull() &
        ~F.col("is_serious").isin(0, 1)
    ).count()
    if invalid > 0:
        raise DataQualityError(
            f"Found {invalid} rows where is_serious is not 0 or 1"
        )
    logger.info("PASS: is_serious is binary")


def check_report_date_null_rate(df: DataFrame) -> None:
    """
    The null rate on report_date must stay below MAX_NULL_RATE (5%).

    A null report_date means the raw receivedate string failed to parse.
    A small number is acceptable (bad source data) but above 5% suggests
    the date format changed in the source API response.
    """
    total = df.count()
    null_count = df.filter(F.col("report_date").isNull()).count()
    rate = null_count / total if total > 0 else 0
    if rate > MAX_NULL_RATE:
        raise DataQualityError(
            f"report_date null rate is {rate:.1%}, exceeds threshold of {MAX_NULL_RATE:.1%} "
            f"({null_count}/{total} rows)"
        )
    logger.info("PASS: report_date null rate is %.1f%%", rate * 100)


def check_patient_age_realistic(df: DataFrame) -> None:
    """
    Patient age must be between 0 and 120 inclusive.

    Values outside this range are data entry errors in the source.
    We only check non-null values — null ages are acceptable because
    age is an optional field in FAERS reports.
    """
    invalid = df.filter(
        F.col("patient_age").isNotNull() &
        ((F.col("patient_age") < 0) | (F.col("patient_age") > 120))
    ).count()
    if invalid > 0:
        raise DataQualityError(
            f"Found {invalid} rows with unrealistic patient_age (outside 0-120)"
        )
    logger.info("PASS: patient_age is realistic")


def check_row_count_minimum(df: DataFrame, table_name: str) -> None:
    """
    The table must have at least MIN_ROW_COUNT rows.

    A count at or near zero means ingestion or transformation failed
    silently — perhaps the API returned nothing or the JSON was empty.
    """
    count = df.count()
    if count < MIN_ROW_COUNT:
        raise DataQualityError(
            f"Table {table_name} has {count} rows, expected >= {MIN_ROW_COUNT}"
        )
    logger.info("PASS: %s has %d rows", table_name, count)


def check_drug_name_not_empty(df: DataFrame) -> None:
    """
    Every row in silver_drugs must have a non-empty drug_name.

    Empty drug names are useless for aggregation and dashboards.
    The transformation filters them out, so any that remain indicate
    a bug in the filter logic.
    """
    empty = df.filter(
        F.col("drug_name").isNull() |
        (F.trim(F.col("drug_name")) == "")
    ).count()
    if empty > 0:
        raise DataQualityError(
            f"Found {empty} rows with empty or null drug_name"
        )
    logger.info("PASS: drug_name is not empty")


def check_reaction_name_not_empty(df: DataFrame) -> None:
    """
    Every row in silver_reactions must have a non-empty reaction_name.
    Same reasoning as drug_name — empty values should have been
    filtered during transformation.
    """
    empty = df.filter(
        F.col("reaction_name").isNull() |
        (F.trim(F.col("reaction_name")) == "")
    ).count()
    if empty > 0:
        raise DataQualityError(
            f"Found {empty} rows with empty or null reaction_name"
        )
    logger.info("PASS: reaction_name is not empty")


def check_required_columns_exist(df: DataFrame, required: list[str], table_name: str) -> None:
    """
    All required columns must be present in the DataFrame schema.

    This catches cases where a column was accidentally dropped or
    renamed during a refactor of the transformation code.
    """
    existing = set(df.columns)
    missing = [col for col in required if col not in existing]
    if missing:
        raise DataQualityError(
            f"Table {table_name} is missing required columns: {missing}"
        )
    logger.info("PASS: all required columns exist in %s", table_name)


def run_adverse_events_checks(df: DataFrame) -> None:
    """Run all quality checks on silver_adverse_events."""
    logger.info("Running quality checks on silver_adverse_events")
    check_required_columns_exist(
        df,
        ["safety_report_id", "report_date", "is_serious", "country", "patient_sex", "patient_age"],
        "silver_adverse_events"
    )
    check_no_null_safety_report_ids(df)
    check_is_serious_is_binary(df)
    check_report_date_null_rate(df)
    check_patient_age_realistic(df)
    check_row_count_minimum(df, "silver_adverse_events")
    logger.info("All quality checks passed for silver_adverse_events")


def run_drugs_checks(df: DataFrame) -> None:
    """Run all quality checks on silver_drugs."""
    logger.info("Running quality checks on silver_drugs")
    check_required_columns_exist(
        df,
        ["safety_report_id", "drug_name", "drug_role"],
        "silver_drugs"
    )
    check_drug_name_not_empty(df)
    check_row_count_minimum(df, "silver_drugs")
    logger.info("All quality checks passed for silver_drugs")


def run_reactions_checks(df: DataFrame) -> None:
    """Run all quality checks on silver_reactions."""
    logger.info("Running quality checks on silver_reactions")
    check_required_columns_exist(
        df,
        ["safety_report_id", "reaction_name"],
        "silver_reactions"
    )
    check_reaction_name_not_empty(df)
    check_row_count_minimum(df, "silver_reactions")
    logger.info("All quality checks passed for silver_reactions")
