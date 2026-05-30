import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.utils.config import GOLD_LOCAL_PATH, SILVER_LOCAL_PATH
from src.utils.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_silver(spark: SparkSession, table_name: str):
    """Read a silver Parquet table from the local silver layer."""
    path = str(Path(SILVER_LOCAL_PATH) / table_name)
    return spark.read.parquet(path)


def build_gold_safety_overview(events):
    """
    Compute global KPIs across all adverse-event reports.

    Returns a single-row DataFrame. This feeds the KPI cards at the
    top of the Power BI dashboard: total reports, serious reports,
    deaths, hospitalizations, and the overall seriousness rate.

    round(..., 2) gives a clean percentage like 37.45 instead of
    37.447382918... which is easier to display and store.
    """
    return events.agg(
        F.count("safety_report_id").alias("total_reports"),
        F.sum("is_serious").alias("serious_reports"),
        F.sum("death").alias("death_reports"),
        F.sum("hospitalization").alias("hospitalization_reports"),
        F.round(
            F.sum("is_serious") / F.count("safety_report_id") * 100, 2
        ).alias("seriousness_rate_pct"),
    )


def build_gold_monthly_trends(events):
    """
    Aggregate report counts and seriousness rate by year and month.

    Extracts year and month from report_date using Spark date functions.
    Used for the time-series line chart showing how reporting volume
    and seriousness evolve over time.

    Ordered by year then month so the chart renders in chronological
    order without needing to sort in the BI layer.
    """
    return (
        events
        .withColumn("year", F.year("report_date"))
        .withColumn("month", F.month("report_date"))
        .groupBy("year", "month")
        .agg(
            F.count("safety_report_id").alias("total_reports"),
            F.sum("is_serious").alias("serious_reports"),
            F.sum("death").alias("death_reports"),
            F.sum("hospitalization").alias("hospitalization_reports"),
        )
        .withColumn(
            "seriousness_rate_pct",
            F.round(F.col("serious_reports") / F.col("total_reports") * 100, 2)
        )
        .orderBy("year", "month")
    )


def build_gold_drug_summary(events, drugs):
    """
    Compute per-drug metrics: total reports, serious reports, seriousness rate.

    Joins silver_drugs to silver_adverse_events to get the is_serious flag
    for each drug mention. Then aggregates by drug_name.

    Why left join: every drug row must link to an event row. A left join
    here means we keep all drug rows even if the join fails (which would
    indicate a data integrity issue and would show up as null is_serious
    values in the output).

    Ordered by total_reports descending so the top drugs appear first
    in the dashboard table.
    """
    joined = drugs.join(
        events.select("safety_report_id", "is_serious"),
        on="safety_report_id",
        how="left"
    )
    return (
        joined
        .groupBy("drug_name")
        .agg(
            F.count("safety_report_id").alias("total_reports"),
            F.sum("is_serious").alias("serious_reports"),
        )
        .withColumn(
            "seriousness_rate_pct",
            F.round(F.col("serious_reports") / F.col("total_reports") * 100, 2)
        )
        .orderBy(F.col("total_reports").desc())
    )


def build_gold_reaction_summary(events, reactions):
    """
    Compute per-reaction metrics: total mentions, serious mentions, seriousness rate.

    Same pattern as drug_summary. Each reaction is counted once per
    report it appears in. A reaction with a high seriousness rate is
    a signal worth investigating.
    """
    joined = reactions.join(
        events.select("safety_report_id", "is_serious"),
        on="safety_report_id",
        how="left"
    )
    return (
        joined
        .groupBy("reaction_name")
        .agg(
            F.count("safety_report_id").alias("total_reports"),
            F.sum("is_serious").alias("serious_reports"),
        )
        .withColumn(
            "seriousness_rate_pct",
            F.round(F.col("serious_reports") / F.col("total_reports") * 100, 2)
        )
        .orderBy(F.col("total_reports").desc())
    )


def build_gold_drug_reaction_pairs(events, drugs, reactions):
    """
    Compute drug-reaction pair counts and seriousness rates.

    This is the most important table for pharmacovigilance signal detection.
    A drug-reaction pair with a high count AND high seriousness rate is a
    potential safety signal — meaning this specific combination of drug and
    reaction appears frequently in serious adverse event reports.

    How the join works:
    1. Join drugs to events to get (report_id, drug_name, is_serious).
    2. Join reactions to events to get (report_id, reaction_name).
    3. Join both on report_id. This is a many-to-many match within each
       report: if a report has 2 drugs and 3 reactions, you get 6 pairs
       (every drug paired with every reaction in the same report).
    4. Aggregate by (drug_name, reaction_name).

    This is the standard FAERS analysis method used in pharmacovigilance.
    """
    events_slim = events.select("safety_report_id", "is_serious")
    drug_events = drugs.join(events_slim, on="safety_report_id", how="left")
    reaction_slim = reactions.select("safety_report_id", "reaction_name")
    pairs = drug_events.join(reaction_slim, on="safety_report_id", how="inner")

    return (
        pairs
        .groupBy("drug_name", "reaction_name")
        .agg(
            F.count("*").alias("pair_count"),
            F.sum("is_serious").alias("serious_count"),
        )
        .withColumn(
            "seriousness_rate_pct",
            F.round(F.col("serious_count") / F.col("pair_count") * 100, 2)
        )
        .orderBy(F.col("pair_count").desc())
    )


def build_gold_ml_features(events, drugs, reactions):
    """
    Build the feature table for the seriousness prediction model in Phase 9.

    One row per adverse-event report. Contains engineered features:
    - num_drugs: how many drugs were involved
    - num_reactions: how many reactions were reported
    - has_suspect_drug: whether at least one drug is flagged as Suspect

    These aggregate features capture complexity of the report. Reports
    with more drugs and reactions tend to be more serious. The presence
    of a suspect drug (as opposed to only concomitant drugs) is a strong
    predictor of seriousness.
    """
    drug_counts = (
        drugs
        .groupBy("safety_report_id")
        .agg(
            F.count("drug_name").alias("num_drugs"),
            F.max(
                F.when(F.col("drug_role") == "Suspect", 1).otherwise(0)
            ).alias("has_suspect_drug"),
        )
    )

    reaction_counts = (
        reactions
        .groupBy("safety_report_id")
        .agg(F.count("reaction_name").alias("num_reactions"))
    )

    return (
        events
        .select(
            "safety_report_id", "report_date", "country",
            "patient_sex", "patient_age", "is_serious"
        )
        .join(drug_counts, on="safety_report_id", how="left")
        .join(reaction_counts, on="safety_report_id", how="left")
        .fillna(0, subset=["num_drugs", "num_reactions", "has_suspect_drug"])
    )


def write_gold(df, table_name: str):
    """Write a gold DataFrame to Parquet in the local gold layer."""
    output_path = str(Path(GOLD_LOCAL_PATH) / table_name)
    df.write.mode("overwrite").parquet(output_path)
    logger.info("Written gold table: %s", output_path)


def run_build_gold():
    """Run the full gold table build from silver inputs."""
    spark = get_spark_session("BuildGoldTables")

    events = read_silver(spark, "adverse_events")
    drugs = read_silver(spark, "drugs")
    reactions = read_silver(spark, "reactions")

    tables = {
        "safety_overview": build_gold_safety_overview(events),
        "monthly_trends": build_gold_monthly_trends(events),
        "drug_summary": build_gold_drug_summary(events, drugs),
        "reaction_summary": build_gold_reaction_summary(events, reactions),
        "drug_reaction_pairs": build_gold_drug_reaction_pairs(events, drugs, reactions),
        "ml_features": build_gold_ml_features(events, drugs, reactions),
    }

    for name, df in tables.items():
        write_gold(df, name)

    spark.stop()
    logger.info("Gold table build complete")


if __name__ == "__main__":
    run_build_gold()
