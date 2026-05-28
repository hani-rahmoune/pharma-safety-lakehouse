import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ingestion.fetch_openfda_events import (
    build_date_filter,
    fetch_events_page,
    ingest,
    save_events_locally,
)


def test_build_date_filter_january():
    result = build_date_filter(2024, 1)
    assert result == "receivedate:[20240101+TO+20240131]"


def test_build_date_filter_pads_single_digit_month():
    result = build_date_filter(2024, 3)
    assert "20240301" in result
    assert "20240331" in result


def test_build_date_filter_december():
    result = build_date_filter(2024, 12)
    assert "20241201" in result
    assert "20241231" in result


def test_fetch_events_page_returns_results_on_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [{"safetyreportid": "111"}, {"safetyreportid": "222"}]
    }
    with patch("src.ingestion.fetch_openfda_events.requests.get", return_value=mock_response):
        result = fetch_events_page("receivedate:[20240101+TO+20240131]", 0, 10)
    assert len(result) == 2
    assert result[0]["safetyreportid"] == "111"


def test_fetch_events_page_returns_empty_on_404():
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    with patch("src.ingestion.fetch_openfda_events.requests.get", return_value=mock_response):
        result = fetch_events_page("any", 0, 10)
    assert result == []


def test_fetch_events_page_returns_empty_on_network_error():
    with patch(
        "src.ingestion.fetch_openfda_events.requests.get",
        side_effect=requests.exceptions.ConnectionError("timeout"),
    ):
        result = fetch_events_page("any", 0, 10)
    assert result == []


def test_save_events_locally_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.ingestion.fetch_openfda_events.BRONZE_LOCAL_PATH", str(tmp_path)
    )
    records = [{"safetyreportid": "abc"}, {"safetyreportid": "def"}]
    output_path = save_events_locally(records, 2024, 1)
    assert Path(output_path).exists()


def test_save_events_locally_file_content_is_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.ingestion.fetch_openfda_events.BRONZE_LOCAL_PATH", str(tmp_path)
    )
    records = [{"safetyreportid": "abc"}]
    output_path = save_events_locally(records, 2024, 1)
    with open(output_path) as f:
        saved = json.load(f)
    assert saved[0]["safetyreportid"] == "abc"


def test_save_events_locally_partition_path_is_correct(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.ingestion.fetch_openfda_events.BRONZE_LOCAL_PATH", str(tmp_path)
    )
    save_events_locally([{"id": "1"}], 2024, 3)
    expected = tmp_path / "year=2024" / "month=03" / "events.json"
    assert expected.exists()


def test_ingest_raises_when_no_records_returned(monkeypatch):
    monkeypatch.setattr(
        "src.ingestion.fetch_openfda_events.fetch_all_events",
        lambda year, month: [],
    )
    with pytest.raises(ValueError, match="No records fetched"):
        ingest(2024, 1)