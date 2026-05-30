from src.utils.config import BIGQUERY_DATASET_GOLD, GCS_BUCKET_NAME
from src.utils.gcp_helpers import load_all_gold_tables_to_bigquery

if __name__ == "__main__":
    load_all_gold_tables_to_bigquery(
        gcs_bucket=GCS_BUCKET_NAME,
        dataset=BIGQUERY_DATASET_GOLD,
    )
