"""Tests for text file PII detection"""
import json
import os
import tempfile

import pytest

from config import SAMPLE_DATA
from text_pii_detector import TextPIIDetector


@pytest.fixture(scope="module")
def detector():
    return TextPIIDetector(confidence_threshold=0.7)


def test_detects_email_and_phone_in_sample_file(detector):
    sample_path = os.path.join(SAMPLE_DATA, "sample_email.txt")
    content, _ = detector.read_text_file(sample_path)
    results = detector.comprehensive_pii_detection(content)

    entity_types = {result["entity_type"] for result in results}
    texts = {result["text"].lower() for result in results}

    assert "EMAIL_ADDRESS" in entity_types or "EMAIL" in entity_types
    assert any("john.smith@company.com" in text for text in texts)
    assert any("555" in text for text in texts)


def test_save_redacted_files_does_not_mutate_analysis(detector):
    analysis = {
        "filename": "test.txt",
        "file_size": 20,
        "total_pii_found": 1,
        "pii_by_type": {"EMAIL_ADDRESS": 1},
        "pii_by_source": {"regex": 1},
        "detection_details": [],
        "redacted_versions": {
            "mask": {
                "content": "Contact **** today",
                "redaction_info": {"redactions": [{"original": "me@x.com", "replacement": "****"}]},
            }
        },
    }

    with tempfile.TemporaryDirectory() as output_dir:
        detector.save_redacted_files(analysis, output_dir)

    assert "content" in analysis["redacted_versions"]["mask"]
    assert "redaction_info" in analysis["redacted_versions"]["mask"]


def test_process_text_file_returns_redacted_versions(detector):
    sample_path = os.path.join(SAMPLE_DATA, "sample_email.txt")
    analysis = detector.process_text_file(sample_path, redaction_strategies=["mask"])

    assert analysis is not None
    assert analysis["total_pii_found"] > 0
    assert "mask" in analysis["redacted_versions"]
    assert "****" in analysis["redacted_versions"]["mask"]["content"]


def test_deduplicate_results_prefers_higher_score(detector):
    text = "john@example.com"
    results = [
        {"entity_type": "EMAIL", "start": 0, "end": 16, "score": 0.9, "text": text, "source": "presidio"},
        {"entity_type": "EMAIL", "start": 0, "end": 16, "score": 1.0, "text": text, "source": "regex"},
    ]

    unique = detector.deduplicate_results(results, text)
    assert len(unique) == 1
    assert unique[0]["score"] == 1.0
