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


def test_rule_layer_fixes_from_test5():
    import policy_rules

    text = ("Claimant: Fenna van der Berg\n"
            "Next of kin: Gábor Németh (brother)\n"
            "Address: 220 Lakeshore Avenue, Toronto, ON M5V 2T6\n"
            "Home address: Hauptstrasse 12, 80331 Munich\n"
            "Passenger: Oluwaseun Adeyemi, birthday 14 Feb 1979\n"
            "Staff No. 004512, payroll number P-204418, hull ID FR-JNO34912C626\n"
            "Bank account 12345678, sort code 40-11-60\n"
            "Call 020 7946 0958 or 07700 900123 or +49 (0)30 901820.\n"
            "IBAN DE89 3704 0044 0532 0130 00\n"
            "Previous practice: Lindenhof Surgery. Surgery went well.\n"
            "Referred to Riverside Speech and Language Centre.\n"
            "[14:02] Agent: Could you confirm your member number?\n"
            "[14:02] Priya Raghunathan: 88-1234-556, booked for 2026-12-01.\n"
            "[14:03] Agent: Order number 55120 is on its way.\n")
    found = {(r["entity_type"], r["text"]) for r in policy_rules.run_pattern_rules(text)}
    assert ("PERSON", "Fenna van der Berg") in found
    assert ("PERSON", "Gábor Németh") in found
    assert ("ADDRESS_LINE", "220 Lakeshore Avenue, Toronto, ON M5V 2T6") in found
    assert ("ADDRESS_LINE", "Hauptstrasse 12, 80331 Munich") in found
    assert ("DATE_OF_BIRTH", "14 Feb 1979") in found
    assert ("GENERIC_ID", "004512") in found
    assert ("GENERIC_ID", "P-204418") in found
    assert ("SERIAL_NUMBER", "FR-JNO34912C626") in found
    assert ("BANK_ACCOUNT", "40-11-60") in found
    assert ("PHONE", "020 7946 0958") in found
    assert ("PHONE", "07700 900123") in found
    assert ("PHONE", "+49 (0)30 901820") in found
    assert not any(label == "PHONE" and "0044" in value for label, value in found)
    assert ("SENSITIVE_ORGANIZATION", "Lindenhof Surgery") in found
    assert ("SENSITIVE_ORGANIZATION", "Surgery") not in found
    assert ("SENSITIVE_ORGANIZATION", "Riverside Speech and Language Centre") in found
    assert ("GENERIC_ID", "88-1234-556") in found
    assert ("GENERIC_ID", "2026-12-01") not in found
    assert ("GENERIC_ID", "55120") not in found

    def ner(label, needle, source="spacy"):
        start = text.index(needle)
        return {"entity_type": label, "start": start, "end": start + len(needle),
                "score": 0.85, "text": needle, "source": source}

    sites = ("Travels to Bergen via the Oslo hub. Engine: Yanmar 3YM30. "
             "The Bosch Serie 4 works best on the eco cycle. She works at the Hull depot.")
    text = sites
    kept = policy_rules.apply_policy_filters(
        [ner("LOCATION", "Bergen"), ner("LOCATION", "Oslo"), ner("PERSON", "Hull"),
         ner("PERSON", "Yanmar 3YM30", "custom_ner"), ner("ORGANIZATION", "Bosch Serie 4")],
        sites)
    assert [r["text"] for r in kept] == ["Bergen"]
