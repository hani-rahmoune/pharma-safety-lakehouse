from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "pharma-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="pharmacovigilance_pipeline",
    default_args=default_args,
    description="Monthly adverse-event data pipeline: ingest, transform, quality check, load",
    schedule_interval="0 3 1 * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["pharma", "safety", "lakehouse"],
) as dag:

    def _get_run_period(context):
        """
        Return the year and month to process.

        For manual Airflow runs, you can pass a config like:
        {
            "year": 2024,
            "month": 1
        }

        If no config is provided, the DAG uses a safe historical
        demo month with available openFDA data instead of using
        the current logical date, which may point to a month with
        no records yet.
        """
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run and dag_run.conf else {}

        year = int(conf.get("year", 2024))
        month = int(conf.get("month", 1))

        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {month}. Month must be between 1 and 12.")

        return year, month

    def task_ingest(**context):
        """
        Download raw adverse-event data from the openFDA API
        and save it to the local bronze layer.

        Returns the saved file path as a string for Airflow XCom.
        """
        from src.ingestion.fetch_openfda_events import ingest

        year, month = _get_run_period(context)

        output_path = ingest(year=year, month=month)

        return str(output_path)

    def task_bronze_to_silver(**context):
        """
        Run the PySpark bronze-to-silver transformation for the
        same year and month used during ingestion.
        """
        from src.spark_jobs.bronze_to_silver_events import run_bronze_to_silver

        year, month = _get_run_period(context)

        run_bronze_to_silver(year=year, month=month)

    def task_data_quality(**context):
        """
        Run data quality checks on the silver tables.

        If any check fails, the task raises an error and the pipeline
        stops before creating gold tables.
        """
        from pathlib import Path

        from pyspark.sql import SparkSession

        from src.quality.data_quality_checks import (
            run_adverse_events_checks,
            run_drugs_checks,
            run_reactions_checks,
        )
        from src.utils.config import SILVER_LOCAL_PATH

        spark = (
            SparkSession.builder
            .appName("DataQuality")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")

        try:
            events = spark.read.parquet(str(Path(SILVER_LOCAL_PATH) / "adverse_events"))
            drugs = spark.read.parquet(str(Path(SILVER_LOCAL_PATH) / "drugs"))
            reactions = spark.read.parquet(str(Path(SILVER_LOCAL_PATH) / "reactions"))

            run_adverse_events_checks(events)
            run_drugs_checks(drugs)
            run_reactions_checks(reactions)

        finally:
            spark.stop()

    def task_build_gold(**context):
        """
        Build all gold analytical tables from the silver layer.

        Produces:
        - safety_overview
        - monthly_trends
        - drug_summary
        - reaction_summary
        - drug_reaction_pairs
        - ml_features
        """
        from src.spark_jobs.build_gold_tables import run_build_gold

        run_build_gold()

    def task_load_bigquery(**context):
        """
        Upload gold Parquet files to GCS and load them into BigQuery.

        Each table is uploaded to GCS first, then loaded into BigQuery
        using WRITE_TRUNCATE so the job is safe to re-run.
        """
        from src.utils.config import BIGQUERY_DATASET_GOLD, GCS_BUCKET_NAME
        from src.utils.gcp_helpers import load_all_gold_tables_to_bigquery

        load_all_gold_tables_to_bigquery(
            gcs_bucket=GCS_BUCKET_NAME,
            dataset=BIGQUERY_DATASET_GOLD,
        )

    ingest_task = PythonOperator(
        task_id="ingest_openfda_data",
        python_callable=task_ingest,
    )

    silver_task = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=task_bronze_to_silver,
    )

    quality_task = PythonOperator(
        task_id="data_quality_checks",
        python_callable=task_data_quality,
    )

    gold_task = PythonOperator(
        task_id="build_gold_tables",
        python_callable=task_build_gold,
    )

    bq_task = PythonOperator(
        task_id="load_gold_to_bigquery",
        python_callable=task_load_bigquery,
    )

    ingest_task >> silver_task >> quality_task >> gold_task >> bq_task