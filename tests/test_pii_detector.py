"""Tests for CSV/tabular PII detection"""
import os

import pandas as pd
import pytest

from config import SAMPLE_DATA
from pii_detector import PIIDetector, RegexPIIMatch


@pytest.fixture
def detector():
    return PIIDetector(confidence_threshold=0.8)


def test_sample_data_directory_exists():
    assert os.path.isdir(SAMPLE_DATA)
    assert os.path.exists(os.path.join(SAMPLE_DATA, "customers.csv"))


def test_partial_redact_value(detector):
    assert detector.partial_redact_value("John Doe") == "Jo***e"
    assert detector.partial_redact_value("ab") == "****"


def test_confidence_threshold_is_used(monkeypatch, detector):
    captured = {}

    def fake_analyze(text, language='en', score_threshold=0.8):
        captured['score_threshold'] = score_threshold
        return []

    monkeypatch.setattr(detector.analyzer, "analyze", fake_analyze)
    detector.confidence_threshold = 0.6
    detector.detect_pii_in_text("John Doe")

    assert captured['score_threshold'] == 0.6


def test_detects_phone_column_with_regex(detector):
    df = pd.DataFrame({"phone": ["555-123-4567", "not-a-phone"]})
    results = detector.detect_pii_in_dataframe(df)

    assert "phone" in results
    assert len(results["phone"]) == 1
    assert results["phone"][0]["value"] == "555-123-4567"
    assert isinstance(results["phone"][0]["pii_detected"][0], RegexPIIMatch)


def test_smart_redaction_masks_phone_and_preserves_ids(detector):
    df = pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "phone": ["555-123-4567", "(555) 987-6543"],
        "notes": ["Email john@example.com", "No PII here"],
    })

    redacted, results = detector.smart_detect_and_redact(df)

    assert redacted.loc[0, "customer_id"] == "C001"
    assert redacted.loc[0, "phone"] == "****"
    assert "[REDACTED-EMAIL]" in redacted.loc[0, "notes"]
    assert "phone" in results


def test_redact_dataframe_partial_mode(detector):
    df = pd.DataFrame({"name": ["John Doe"]})
    pii_results = {
        "name": [{"row": 0, "value": "John Doe", "pii_detected": [RegexPIIMatch("PERSON")]}]
    }

    redacted = detector.redact_dataframe(df, pii_results, method="partial")
    assert redacted.loc[0, "name"] == "Jo***e"


def test_sample_customers_csv_loads():
    csv_path = os.path.join(SAMPLE_DATA, "customers.csv")
    df = pd.read_csv(csv_path)

    assert len(df) == 3
    assert set(df.columns) >= {"customer_id", "name", "email", "phone", "notes"}
