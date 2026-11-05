"""Tests for text file PII detection"""
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


def test_regex_detects_russian_identifiers_and_sensitive_clinic(detector):
    text = (
        "Call +7 (495) 234-56-78. SNILS 142-337-890 12. "
        "Visit Bryansk City Polyclinic No. 4."
    )
    results = detector.detect_pii_regex(text)

    assert {result["entity_type"] for result in results} >= {
        "PHONE_RU", "SNILS", "SENSITIVE_ORGANIZATION"
    }
    assert any(result["text"] == "Bryansk City Polyclinic No. 4" for result in results)


def test_date_and_cross_line_entities_are_not_redacted(detector):
    text = "Movement at night.\nAlexey Sergeevich\nDate of birth: 04.11.1992."
    results = [
        {"entity_type": "DATE_TIME", "start": 12, "end": 17, "score": 0.9, "source": "presidio"},
        {"entity_type": "PERSON", "start": 19, "end": 37, "score": 0.9, "source": "spacy"},
        {"entity_type": "DATE_OF_BIRTH", "start": 52, "end": 62, "score": 1.0, "source": "regex"},
    ]

    unique = detector.deduplicate_results(results, text)

    assert unique == [
        {"entity_type": "DATE_OF_BIRTH", "start": 52, "end": 62, "score": 1.0,
         "source": "regex", "text": "04.11.1992"}
    ]


def test_date_of_birth_requires_birth_context(detector):
    text = "Date of issue: 04.11.2026. Born: 04.11.1992."
    issue_start = text.index("04.11.2026")
    birth_start = text.index("04.11.1992")
    results = [
        {"entity_type": "DATE_OF_BIRTH", "start": issue_start, "end": issue_start + 10,
         "score": 0.95, "source": "custom_ner"},
        {"entity_type": "DATE_OF_BIRTH", "start": birth_start, "end": birth_start + 10,
         "score": 1.0, "source": "regex"},
    ]

    unique = detector.deduplicate_results(results, text)

    assert [result["text"] for result in unique] == ["04.11.1992"]


def test_nested_person_candidate_yields_to_the_longer_one(detector):
    # A role-label regex span that stopped early ("Ruaridh Mac", score 1.0)
    # must not outrank the model's full name (0.95) and expose "Leod".
    text = "Student: Ruaridh MacLeod, group 7-A."
    start = text.index("Ruaridh")
    results = [
        {"entity_type": "PERSON", "start": start, "end": start + len("Ruaridh Mac"),
         "score": 1.0, "source": "regex"},
        {"entity_type": "PERSON", "start": start, "end": start + len("Ruaridh MacLeod"),
         "score": 0.95, "source": "custom_ner"},
    ]

    unique = detector.deduplicate_results(results, text)

    assert [result["text"] for result in unique] == ["Ruaridh MacLeod"]


def test_rule_layer_fixes_from_test4():
    import policy_rules

    text = ("Vehicle: Nissan Qashqai, registration AB12 CDE.\n"
            "Student: Ruaridh MacLeod, hospital number RV-88-2201.\n"
            "Call her on +33 6 12 34 56 78 or +31 20 234 5678.\n"
            "41 Rosemary Parade, Haarlem, 2011 AB\n"
            "City Hospital No. 3 confirmed.")
    found = {(r["entity_type"], r["text"]) for r in policy_rules.run_pattern_rules(text)}
    assert ("PERSON", "Ruaridh MacLeod") in found
    assert ("GENERIC_ID", "RV-88-2201") in found
    assert ("PHONE", "+33 6 12 34 56 78") in found
    assert ("PHONE", "+31 20 234 5678") in found
    assert ("ADDRESS_LINE", "41 Rosemary Parade, Haarlem, 2011 AB") in found
    assert not any(label == "GENERIC_ID" and "3" == value for label, value in found)

    vehicle = text.index("Nissan")
    ner = [{"entity_type": "PERSON", "start": vehicle, "end": vehicle + len("Nissan Qashqai"),
            "score": 0.85, "text": "Nissan Qashqai", "source": "spacy"}]
    assert policy_rules.apply_policy_filters(ner, text) == []
