import logging
from pathlib import Path

from google.cloud import bigquery, storage
from google.oauth2 import service_account

from src.utils.config import GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _get_credentials():
    """
    Load GCP credentials from the service account JSON file.

    We load credentials explicitly from the file path in config
    rather than relying on the GOOGLE_APPLICATION_CREDENTIALS
    environment variable, because the env var requires an absolute
    path while our config supports relative paths from the project root.
    """
    return service_account.Credentials.from_service_account_file(
        GOOGLE_APPLICATION_CREDENTIALS
    )


def get_storage_client():
    """Return an authenticated GCS client."""
    return storage.Client(
        credentials=_get_credentials(),
        project=GCP_PROJECT_ID
    )


def get_bigquery_client():
    """Return an authenticated BigQuery client."""
    return bigquery.Client(
        credentials=_get_credentials(),
        project=GCP_PROJECT_ID
    )


def upload_folder_to_gcs(local_path: str, gcs_bucket: str, gcs_prefix: str) -> None:
    """
    Upload all Parquet files from a local folder to GCS.

    local_path: local directory containing .parquet files
    gcs_bucket: GCS bucket name without gs://
    gcs_prefix: folder path inside the bucket

    We glob for *.parquet specifically to skip _SUCCESS and
    .crc checksum files that Spark writes alongside the data.
    """
    client = get_storage_client()
    bucket = client.bucket(gcs_bucket)

    local = Path(local_path)
    files = list(local.glob("**/*.parquet"))

    if not files:
        logger.warning("No parquet files found in %s", local_path)
        return

    for file in files:
        blob_name = f"{gcs_prefix}/{file.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(file))
        logger.info("Uploaded %s to gs://%s/%s", file.name, gcs_bucket, blob_name)

    logger.info("Uploaded %d files to gs://%s/%s", len(files), gcs_bucket, gcs_prefix)


def load_gcs_parquet_to_bigquery(
    gcs_uri_pattern: str,
    dataset: str,
    table_name: str,
) -> None:
    """
    Load Parquet files from GCS into a BigQuery table.

    gcs_uri_pattern: e.g. gs://bucket/gold/drug_summary/*.parquet
    dataset: BigQuery dataset name
    table_name: BigQuery table name

    WRITE_TRUNCATE replaces the table on each load, making the job
    idempotent — safe to re-run without creating duplicates.

    autodetect=True infers the schema from the Parquet file metadata,
    which works reliably because Parquet stores full type information.
    """
    client = get_bigquery_client()
    table_ref = f"{GCP_PROJECT_ID}.{dataset}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    logger.info("Loading %s into %s", gcs_uri_pattern, table_ref)
    load_job = client.load_table_from_uri(
        gcs_uri_pattern, table_ref, job_config=job_config
    )
    load_job.result()
    logger.info("Loaded table: %s", table_ref)


def load_all_gold_tables_to_bigquery(gcs_bucket: str, dataset: str) -> None:
    """
    Upload all gold Parquet tables to GCS then load them into BigQuery.

    This is the function the Airflow DAG will call.
    """
    from src.utils.config import GOLD_LOCAL_PATH

    gold_tables = [
        "safety_overview",
        "monthly_trends",
        "drug_summary",
        "reaction_summary",
        "drug_reaction_pairs",
        "ml_features",
    ]

    for table in gold_tables:
        local_path = str(Path(GOLD_LOCAL_PATH) / table)
        gcs_prefix = f"gold/{table}"
        gcs_uri = f"gs://{gcs_bucket}/{gcs_prefix}/*.parquet"

        upload_folder_to_gcs(local_path, gcs_bucket, gcs_prefix)
        load_gcs_parquet_to_bigquery(gcs_uri, dataset, table)

    logger.info("All gold tables loaded to BigQuery dataset: %s", dataset)
