import json
import logging
import time
from pathlib import Path

import requests

from src.utils.config import (
    BRONZE_LOCAL_PATH,
    OPENFDA_API_KEY,
    OPENFDA_BASE_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_RECORDS_PER_REQUEST = 100
MAX_TOTAL_RECORDS = 1000


def build_date_filter(year: int, month: int) -> str:
    """
    Build the openFDA date range search string for a given year and month.

    openFDA expects dates as YYYYMMDD inside a range expression.
    Example for January 2024:
        receivedate:[20240101+TO+20240131]

    The + signs are URL-encoded spaces that the API understands as
    the word TO in a range query.
    """
    month_str = str(month).zfill(2)
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
    }
    last_day = days_in_month[month]
    start = f"{year}{month_str}01"
    end = f"{year}{month_str}{last_day}"
    return f"receivedate:[{start}+TO+{end}]"


def fetch_events_page(search: str, skip: int, limit: int) -> list[dict]:
    """
    Fetch one page of adverse-event results from the openFDA API.

    We build the URL manually instead of using the requests params dict.
    This is because requests URL-encodes + as %2B, which breaks the
    Lucene range syntax that openFDA uses. The + signs in the search
    string must reach the server as literal + characters, which in a
    URL query string are interpreted as spaces — exactly what Lucene
    needs for the TO keyword in range queries.

    Example correct URL:
        .../event.json?search=receivedate:[20240101+TO+20240131]&limit=100&skip=0
    """
    url = f"{OPENFDA_BASE_URL}?search={search}&limit={limit}&skip={skip}"
    if OPENFDA_API_KEY:
        url += f"&api_key={OPENFDA_API_KEY}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        logger.info("Fetched %d records at skip=%d", len(results), skip)
        return results
    except requests.exceptions.HTTPError as error:
        if response.status_code == 404:
            logger.info("No more results at skip=%d", skip)
            return []
        logger.error("HTTP error at skip=%d: %s", skip, error)
        return []
    except requests.exceptions.RequestException as error:
        logger.error("Request failed at skip=%d: %s", skip, error)
        return []


def fetch_all_events(year: int, month: int) -> list[dict]:
    """
    Fetch all adverse-event records for a given year and month.

    Paginates through the API in steps of MAX_RECORDS_PER_REQUEST
    until MAX_TOTAL_RECORDS is reached or the API returns nothing.

    We sleep 0.3 seconds between requests to avoid hitting the
    openFDA rate limit (roughly 240 requests per minute without a key,
    1000 per minute with a key).
    """
    search = build_date_filter(year, month)
    all_records = []
    skip = 0

    logger.info("Starting ingestion for %d-%02d", year, month)

    while skip < MAX_TOTAL_RECORDS:
        page = fetch_events_page(search, skip, MAX_RECORDS_PER_REQUEST)
        if not page:
            break
        all_records.extend(page)
        skip += MAX_RECORDS_PER_REQUEST
        time.sleep(0.3)

    logger.info("Total records fetched for %d-%02d: %d", year, month, len(all_records))
    return all_records


def save_events_locally(records: list[dict], year: int, month: int) -> str:
    """
    Save raw adverse-event records to the local bronze layer.

    Creates a Hive-style partitioned directory:
        data/bronze/openfda_events/year=2024/month=01/events.json

    Hive partitioning means the folder name encodes the partition value.
    Spark can read these folders efficiently using partition pruning —
    if you only need January data, it reads only that folder.

    Returns the path where the file was saved.
    """
    partition = f"year={year}/month={str(month).zfill(2)}"
    directory = Path(BRONZE_LOCAL_PATH) / partition
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / "events.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    logger.info("Saved %d records to %s", len(records), output_path)
    return str(output_path)


def ingest(year: int, month: int) -> str:
    """
    Main ingestion entry point. Fetches data and saves it locally.
    This is what the Airflow DAG will call.
    Returns the path of the saved file.
    """
    records = fetch_all_events(year, month)
    if not records:
        raise ValueError(f"No records fetched for {year}-{month:02d}")
    return save_events_locally(records, year, month)


if __name__ == "__main__":
    ingest(year=2024, month=1)