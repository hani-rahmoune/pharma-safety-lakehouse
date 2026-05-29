import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from src.utils.config import BRONZE_LOCAL_PATH, SILVER_LOCAL_PATH
from src.utils.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_bronze_json(spark: SparkSession, year: int, month: int):
    """
    Read the raw bronze JSON file into a Spark DataFrame.

    multiLine=True tells Spark the entire file is one JSON array,
    not one JSON object per line. Without this, Spark would try to
    read each line as a separate record and fail on our array format.
    """
    partition = f"year={year}/month={str(month).zfill(2)}"
    path = str(Path(BRONZE_LOCAL_PATH) / partition / "events.json")
    logger.info("Reading bronze data from %s", path)
    df = spark.read.option("multiLine", "true").json(path)
    return df


def build_silver_adverse_events(df):
    """
    Create the silver_adverse_events table.

    One row per adverse-event report. Contains report-level fields:
    dates, seriousness flags, country, and patient demographics.

    Transformations:
    - receivedate string YYYYMMDD parsed into a proper date column
    - seriousness flags cast from string to integer
    - patient sex numeric codes mapped to readable labels
    - patient age cast to integer
    - duplicate safety_report_id values removed
    - rows with null safety_report_id dropped
    """
    sex_mapping = F.create_map(
        F.lit("0"), F.lit("Unknown"),
        F.lit("1"), F.lit("Male"),
        F.lit("2"), F.lit("Female"),
    )

    silver = (
        df
        .select(
            F.col("safetyreportid").alias("safety_report_id"),
            F.col("receivedate"),
            F.col("serious"),
            F.col("seriousnessdeath"),
            F.col("seriousnesshospitalization"),
            F.col("primarysourcecountry").alias("country"),
            F.col("patient.patientsex").alias("raw_sex"),
            F.col("patient.patientonsetage").alias("raw_age"),
        )
        .withColumn("report_date", F.to_date(F.col("receivedate"), "yyyyMMdd"))
        .withColumn("is_serious", F.col("serious").cast(IntegerType()))
        .withColumn("death", F.col("seriousnessdeath").cast(IntegerType()))
        .withColumn(
            "hospitalization",
            F.col("seriousnesshospitalization").cast(IntegerType())
        )
        .withColumn("patient_sex", sex_mapping[F.col("raw_sex")])
        .withColumn("patient_age", F.col("raw_age").cast(IntegerType()))
        .drop(
            "receivedate", "serious", "seriousnessdeath",
            "seriousnesshospitalization", "raw_sex", "raw_age"
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

    explode_outer keeps reports with no drugs (returns null instead
    of dropping the report).

    Drug role codes:
        1 = Suspect     (suspected of causing the reaction)
        2 = Concomitant (taken at the same time, not suspected)
        3 = Interacting (interacted with the suspect drug)
    """
    role_mapping = F.create_map(
        F.lit("1"), F.lit("Suspect"),
        F.lit("2"), F.lit("Concomitant"),
        F.lit("3"), F.lit("Interacting"),
    )

    silver = (
        df
        .select(
            F.col("safetyreportid").alias("safety_report_id"),
            F.explode_outer("patient.drug").alias("drug"),
        )
        .select(
            "safety_report_id",
            F.upper(F.trim(F.col("drug.medicinalproduct"))).alias("drug_name"),
            F.col("drug.drugcharacterization").alias("raw_role"),
            F.col("drug.drugindication").alias("indication"),
        )
        .withColumn("drug_role", role_mapping[F.col("raw_role")])
        .drop("raw_role")
        .filter(F.col("drug_name").isNotNull())
        .filter(F.trim(F.col("drug_name")) != "")
    )

    return silver


def build_silver_reactions(df):
    """
    Create the silver_reactions table.

    Explodes the patient.reaction array so each reaction gets its own row.
    reactionmeddrapt is the MedDRA preferred term, a standardized medical
    terminology used across the pharmacovigilance industry.

    initcap converts PNEUMONIA to Pneumonia for cleaner display.
    """
    silver = (
        df
        .select(
            F.col("safetyreportid").alias("safety_report_id"),
            F.explode_outer("patient.reaction").alias("reaction"),
        )
        .select(
            "safety_report_id",
            F.initcap(F.trim(F.col("reaction.reactionmeddrapt"))).alias("reaction_name"),
        )
        .filter(F.col("reaction_name").isNotNull())
        .filter(F.trim(F.col("reaction_name")) != "")
    )

    return silver


def write_parquet(df, table_name: str):
    """
    Write a DataFrame to Parquet in the local silver layer.

    Parquet is a columnar binary format — faster to read than JSON
    because you can read only the columns you need, and it stores
    type information so Spark does not have to infer it each time.

    overwrite makes the job idempotent — safe to re-run.
    """
    output_path = str(Path(SILVER_LOCAL_PATH) / table_name)
    df.write.mode("overwrite").parquet(output_path)
    logger.info("Written %s to %s", table_name, output_path)


def run_bronze_to_silver(year: int, month: int):
    """
    Run the full bronze-to-silver transformation for a given month.

    The count() calls are here and not inside the transformation
    functions because count() triggers a full Spark job. Keeping them
    here means tests can call the transformation functions without
    triggering extra Spark jobs.
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