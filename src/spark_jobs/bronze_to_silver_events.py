import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from src.utils.config import BRONZE_LOCAL_PATH, SILVER_LOCAL_PATH
from src.utils.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def binary_openfda_flag(column_name: str):
    """
    Convert OpenFDA coded flags to analytics-ready binary values.

    OpenFDA uses:
    - 1 = yes
    - 2 = no

    Silver layer uses:
    - 1 = yes
    - 0 = no
    """
    value = F.col(column_name).cast(IntegerType())

    return (
        F.when(value == 1, F.lit(1))
        .when(value == 2, F.lit(0))
        .otherwise(F.lit(None).cast(IntegerType()))
        .cast(IntegerType())
    )


def patient_age_in_years(age_column: str, unit_column: str):
    """
    Convert OpenFDA patient onset age to integer years.

    OpenFDA patientonsetageunit:
    - 800 = decade
    - 801 = year
    - 802 = month
    - 803 = week
    - 804 = day
    - 805 = hour

    Values outside 0-120 are treated as invalid and set to null.
    """
    age = F.col(age_column).cast("double")
    unit = F.col(unit_column).cast("string")

    age_years = (
        F.when(unit == "800", age * 10)
        .when(unit == "801", age)
        .when(unit == "802", age / 12)
        .when(unit == "803", age / 52.1429)
        .when(unit == "804", age / 365.25)
        .when(unit == "805", age / 8766)
        .otherwise(age)
    )

    return (
        F.when((age_years >= 0) & (age_years <= 120), age_years.cast(IntegerType()))
        .otherwise(F.lit(None).cast(IntegerType()))
        .cast(IntegerType())
    )


def read_bronze_json(spark: SparkSession, year: int, month: int):
    """
    Read the raw bronze JSON file into a Spark DataFrame.

    multiLine=True tells Spark the entire file is one JSON array,
    not one JSON object per line.
    """
    partition = f"year={year}/month={str(month).zfill(2)}"
    path = str(Path(BRONZE_LOCAL_PATH) / partition / "events.json")

    logger.info("Reading bronze data from %s", path)

    return spark.read.option("multiLine", "true").json(path)


def build_silver_adverse_events(df):
    """
    Create the silver_adverse_events table.

    One row per adverse-event report. Contains report-level fields:
    dates, seriousness flags, country, and patient demographics.
    """
    sex_mapping = F.create_map(
        F.lit("0"),
        F.lit("Unknown"),
        F.lit("1"),
        F.lit("Male"),
        F.lit("2"),
        F.lit("Female"),
    )

    silver = (
        df.select(
            F.col("safetyreportid").alias("safety_report_id"),
            F.col("receivedate"),
            F.col("serious"),
            F.col("seriousnessdeath"),
            F.col("seriousnesshospitalization"),
            F.col("primarysourcecountry").alias("country"),
            F.col("patient.patientsex").alias("raw_sex"),
            F.col("patient.patientonsetage").alias("raw_age"),
            F.col("patient.patientonsetageunit").alias("raw_age_unit"),
        )
        .withColumn("report_date", F.to_date(F.col("receivedate"), "yyyyMMdd"))
        .withColumn("is_serious", binary_openfda_flag("serious"))
        .withColumn("death", binary_openfda_flag("seriousnessdeath"))
        .withColumn(
            "hospitalization",
            binary_openfda_flag("seriousnesshospitalization"),
        )
        .withColumn("patient_sex", sex_mapping[F.col("raw_sex").cast("string")])
        .withColumn("patient_age", patient_age_in_years("raw_age", "raw_age_unit"))
        .drop(
            "receivedate",
            "serious",
            "seriousnessdeath",
            "seriousnesshospitalization",
            "raw_sex",
            "raw_age",
            "raw_age_unit",
        )
        .dropDuplicates(["safety_report_id"])
        .filter(F.col("safety_report_id").isNotNull())
    )

    return silver


def build_silver_drugs(df):
    """
    Create the silver_drugs table.

    Explodes the patient.drug array so each drug gets its own row,
    all linked by safety_report_id.
    """
    role_mapping = F.create_map(
        F.lit("1"),
        F.lit("Suspect"),
        F.lit("2"),
        F.lit("Concomitant"),
        F.lit("3"),
        F.lit("Interacting"),
    )

    silver = (
        df.select(
            F.col("safetyreportid").alias("safety_report_id"),
            F.explode_outer("patient.drug").alias("drug"),
        )
        .select(
            "safety_report_id",
            F.upper(F.trim(F.col("drug.medicinalproduct"))).alias("drug_name"),
            F.col("drug.drugcharacterization").alias("raw_role"),
            F.col("drug.drugindication").alias("indication"),
        )
        .withColumn("drug_role", role_mapping[F.col("raw_role").cast("string")])
        .drop("raw_role")
        .filter(F.col("drug_name").isNotNull())
        .filter(F.trim(F.col("drug_name")) != "")
    )

    return silver


def build_silver_reactions(df):
    """
    Create the silver_reactions table.

    Explodes the patient.reaction array so each reaction gets its own row.
    """
    silver = (
        df.select(
            F.col("safetyreportid").alias("safety_report_id"),
            F.explode_outer("patient.reaction").alias("reaction"),
        )
        .select(
            "safety_report_id",
            F.initcap(F.trim(F.col("reaction.reactionmeddrapt"))).alias(
                "reaction_name"
            ),
        )
        .filter(F.col("reaction_name").isNotNull())
        .filter(F.trim(F.col("reaction_name")) != "")
    )

    return silver


def write_parquet(df, table_name: str):
    """
    Write a DataFrame to Parquet in the local silver layer.

    overwrite makes the job idempotent.
    """
    output_path = str(Path(SILVER_LOCAL_PATH) / table_name)

    df.write.mode("overwrite").parquet(output_path)

    logger.info("Written %s to %s", table_name, output_path)


def run_bronze_to_silver(year: int, month: int):
    """
    Run the full bronze-to-silver transformation for a given month.
    """
    spark = get_spark_session("BronzeToSilver")
    raw_df = read_bronze_json(spark, year, month)

    events = build_silver_adverse_events(raw_df)
    drugs = build_silver_drugs(raw_df)
    reactions = build_silver_reactions(raw_df)

    logger.info("silver_adverse_events row count: %d", events.count())
    logger.info("silver_drugs row count: %d", drugs.count())
    logger.info("silver_reactions row count: %d", reactions.count())

    write_parquet(events, "adverse_events")
    write_parquet(drugs, "drugs")
    write_parquet(reactions, "reactions")

    spark.stop()

    logger.info("Bronze to silver complete for %d-%02d", year, month)


if __name__ == "__main__":
    run_bronze_to_silver(year=2024, month=1)
