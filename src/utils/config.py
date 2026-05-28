import os

from dotenv import load_dotenv

load_dotenv()

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "pharma-safety-lakehouse")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")

BIGQUERY_DATASET_SILVER = os.getenv(
    "BIGQUERY_DATASET_SILVER",
    "pharma_safety_silver",
)
BIGQUERY_DATASET_GOLD = os.getenv(
    "BIGQUERY_DATASET_GOLD",
    "pharma_safety_gold",
)

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")
OPENFDA_BASE_URL = "https://api.fda.gov/drug/event.json"

BRONZE_LOCAL_PATH = "data/bronze/openfda_events"
SILVER_LOCAL_PATH = "data/silver"
GOLD_LOCAL_PATH = "data/gold"
