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


def test_rule_layer_fixes_from_test6():
    import policy_rules

    text = ("Placement: Willowbrook Care Home, 4 Orchard Rise, Kingsmere KM2 7QL\n"
            "The community nurse from Kingsmere Health Trust visits weekly.\n"
            "Tenant: María José García Lopez, DNI 48213976K\n"
            "Phone: +34 612 345 678\n"
            "Address of the property: Calle Mayor 12, 3ºB, 28013 Madrid\n"
            "SURNAME: ONYEKWERE\n"
            "GIVEN NAMES: CHIDI EMEKA\n"
            "NATIONALITY: NIGERIAN\n"
            "APPLICATION REFERENCE: VAF-2026-118822\n"
            "14:02 Agent (Sunita): Could I take your membership number?\n"
            "14:03 Rhys ap Gruffudd: It's 7719-2034-88.\n"
            "14:03 Agent (Sunita): Thanks Rhys. And the model?\n"
            "Hi Ifeoma, your appointment is on Tue 17 Nov. Reminder for Bartek: flu jab.\n"
            "Dear Customer, Hi Team, thanks again.\n"
            "Applicant 1: Mr Jan-Willem 't Hooft, born 7 March 1979\n"
            "Employer, Applicant 1: Fenland Water Services\n"
            "PCN number: EV 44021873\n"
            "Emergency contact if the parent cannot be reached: Ms Caoimhe Ní Bhriain (aunt)\n"
            "Patient: Mr Khalid bin Tariq Al-Farsi, DOB 15 Sep 1990\n"
            "Agreement: 55-20931 under EMPLOYMENT CONTRACT No. 44/26\n"
            "Driving licence: FONTA812034AS9XY\n"
            "Vehicle: Peugeot 2008, registration M-RK 4471; the EU-UK 2026 summit\n"
            "Contact: +49 171 2345678 or 0047 55 12 34 56\n"
            "IBAN DE89 3704 0044 0532 0130 00\n")
    found = {(r["entity_type"], r["text"]) for r in policy_rules.run_pattern_rules(text)}
    assert ("SENSITIVE_ORGANIZATION", "Willowbrook Care Home") in found
    assert ("SENSITIVE_ORGANIZATION", "Kingsmere Health Trust") in found
    assert ("ADDRESS_LINE", "4 Orchard Rise, Kingsmere KM2 7QL") in found
    assert ("GENERIC_ID", "48213976K") in found
    assert ("PHONE", "+34 612 345 678") in found
    assert ("ADDRESS_LINE", "Calle Mayor 12, 3ºB, 28013 Madrid") in found
    assert ("PERSON", "ONYEKWERE") in found
    assert ("PERSON", "CHIDI EMEKA") in found
    assert not any(value == "NIGERIAN" for _, value in found)
    assert ("GENERIC_ID", "VAF-2026-118822") in found
    assert ("PERSON", "Rhys ap Gruffudd") in found
    assert ("GENERIC_ID", "7719-2034-88") in found
    assert ("PERSON", "Rhys") in found
    assert ("PERSON", "Ifeoma") in found
    assert ("PERSON", "Bartek") in found
    assert not any(label == "PERSON" and value in ("Customer", "Team") for label, value in found)
    assert ("PERSON", "Mr Jan-Willem 't Hooft") in found
    assert ("PERSON", "Fenland Water Services") not in found
    assert ("GENERIC_ID", "EV 44021873") in found
    assert ("PERSON", "Ms Caoimhe Ní Bhriain") in found
    assert ("PERSON", "Mr Khalid bin Tariq Al-Farsi") in found
    assert ("GENERIC_ID", "55-20931") in found
    assert not any(label == "GENERIC_ID" and "44/26" in value for label, value in found)
    assert ("DRIVING_LICENCE", "FONTA812034AS9XY") in found
    assert ("LICENSE_PLATE", "M-RK 4471") in found
    assert not any(label == "LICENSE_PLATE" and "2026" in value for label, value in found)
    assert ("PHONE", "+49 171 2345678") in found
    assert ("PHONE", "0047 55 12 34 56") in found
    assert not any(label == "PHONE" and "0044" in value for label, value in found)

    def ner(label, needle, source="spacy", score=0.85, where=text):
        start = where.index(needle)
        return {"entity_type": label, "start": start, "end": start + len(needle),
                "score": score, "text": needle, "source": source}

    # A checksum-only identifier inside an international phone is the phone.
    phone = next(r for r in policy_rules.run_pattern_rules(text) if r["text"] == "+34 612 345 678")
    kept = policy_rules.apply_policy_filters([phone, ner("AU_ACN", "612 345 678", "presidio", 1.0)], text)
    assert [r["text"] for r in kept] == ["+34 612 345 678"]

    sites = ("Forklift models in service: Jungheinrich EFG 216.\n"
             "14:03 Rhys said hello.\nQRT-55302\nGuest\n"
             "We award the 2027 Halvorsen Scholarship. Ships to 8 Bryn Teg, Caernarfon LL55 1AB today.")
    kept = policy_rules.apply_policy_filters(
        [ner("PERSON", "Jungheinrich EFG 216", where=sites),
         ner("PERSON", "14:03 Rhys", where=sites),
         ner("PERSON", "QRT-55302\nGuest", where=sites),
         ner("ADDRESS", "2027 Halvorsen Scholarship", "custom_ner", 0.95, sites),
         ner("ADDRESS", "8 Bryn Teg, Caernarfon LL55", "custom_ner", 0.95, sites)],
        sites)
    assert [(r["entity_type"], r["text"]) for r in kept] == [
        ("PERSON", "Rhys"), ("ADDRESS", "8 Bryn Teg, Caernarfon LL55 1AB")]

    # The first name of a detected full name is the same person later on;
    # a first name that is also an ordinary word is left alone.
    pupil = ("Pupil: Sigrún Hallgrímsdóttir, Class 4M\n"
             "Medical notes: Sigrún carries an inhaler. Mark Will attends. Mark the box.")
    echoed = policy_rules.echo_person_names(
        [ner("PERSON", "Sigrún Hallgrímsdóttir", "regex", 1.0, pupil),
         ner("PERSON", "Mark Will", "regex", 1.0, pupil)], pupil)
    assert [(r["text"], r["start"]) for r in echoed] == [("Sigrún", pupil.index("Sigrún carries"))]


def test_rule_layer_fixes_from_test7():
    import policy_rules

    def rules(text, extra=()):
        results = policy_rules.run_pattern_rules(text) + list(extra)
        kept = policy_rules.apply_policy_filters(results, text)
        return {(r["entity_type"], r["text"]) for r in kept
                if r["entity_type"] != "PRONOUN"}

    def ner(label, needle, where, source="spacy", score=0.9):
        start = where.index(needle)
        return {"entity_type": label, "start": start, "end": start + len(needle),
                "score": score, "text": needle, "source": source}

    # Rules that reached too far on the blind run.
    greetings = ("Hi Peter, the lift is fixed. Dear Occupier, bins go out on Monday. "
                 "Hi Finance, please pay. Dear Committee, see below.\n"
                 "Cheers,\nOlu\n\nThanks, Team\nBest,\nProduction Manager\nKind regards,\nWanjiru")
    assert rules(greetings) == {("PERSON", "Peter"), ("PERSON", "Olu"), ("PERSON", "Wanjiru")}

    codes = ("Kennzeichen HH-AB 1234. The van B-MW 512E was towed. Room A-B 12 is free. "
             "Each unit has a USB-C 3.2 port and an AC-DC 240 V supply; region DE-HE 3 is loaded.")
    assert rules(codes) == {("LICENSE_PLATE", "HH-AB 1234"), ("LICENSE_PLATE", "B-MW 512E")}

    numbers = ("Tracking no. 0049 30 901820\nPhone: 0049 30 901820\n"
               "Invoice no. 0042 0117 3381\nbatch 0031-22-456\nSSCC: 0035 9012 3400 0001 7421\n"
               "Call the desk on 0044 114 496 0827 or call 1800 555 014.\n")
    assert rules(numbers) == {("PHONE", "0049 30 901820"), ("PHONE", "0044 114 496 0827"),
                              ("PHONE", "1800 555 014")}

    ids = ("Agreement: 55-20931\nCourtesy car: hire agreement 31/10/2026 to 03/11/2026\n"
           "Birth record No. 2014-03-09/0771\nCodice fiscale: CRBFMT85T49F205U\n"
           "Your loyalty card 6338 4410 0027 earned points. "
           "Membership renewals are due Friday, mine is AL-2291.\n"
           "Targa / Plate: FX 482 LD\nRegistration: 12 November 2026\n"
           "Date of birth (DD/MM/YYYY): 02/09/1959\n")
    assert rules(ids) == {
        ("GENERIC_ID", "55-20931"), ("GENERIC_ID", "2014-03-09/0771"),
        ("GENERIC_ID", "CRBFMT85T49F205U"), ("GENERIC_ID", "6338 4410 0027"),
        ("GENERIC_ID", "AL-2291"), ("LICENSE_PLATE", "FX 482 LD"),
        ("DATE_OF_BIRTH", "02/09/1959")}

    form = ("MIDDLE NAME: NONE\nPREVIOUS SURNAME: OKORO\nNEXT OF KIN: ADAEZE NWOSU-KERR (SISTER)\n"
            "DRIVER: COMPANY POOL VEHICLE\nPATIENT: MALE\n"
            "PREVIOUS GP: DR M. HALE, OAKFIELD DENTAL PRACTICE\n"
            "Cliente / Customer: Dott.ssa Fiammetta Carbone\n"
            "Registered keeper: Lexford Leasing Ltd\nEmergency contact: Not Applicable\n")
    assert rules(form) == {
        ("PERSON", "OKORO"), ("PERSON", "ADAEZE NWOSU-KERR"),
        ("SENSITIVE_ORGANIZATION", "OAKFIELD DENTAL PRACTICE"),
        ("PERSON", "Dott.ssa Fiammetta Carbone")}

    streets = ("Dispatch: Via Royal Mail 48, tracked\nIndirizzo: Via Garibaldi 27, 20121 Milano\n"
               "Deliver to: ul. Długa 5, 31-147 Kraków\nMorada: Rua Augusta 112, 1100-053 Lisboa\n"
               "The water is off until 13:00 while the main in Tannery Road is replaced; "
               "it is 5 minutes from the main Road.\n"
               "Address on file: The Old Forge, 6 Chapel Row, Bakewell DE45 1EW\n")
    assert rules(streets) == {
        ("ADDRESS_LINE", "Via Garibaldi 27, 20121 Milano"),
        ("ADDRESS_LINE", "ul. Długa 5, 31-147 Kraków"),
        ("ADDRESS_LINE", "Rua Augusta 112, 1100-053 Lisboa"),
        ("ADDRESS_LINE", "The Old Forge, 6 Chapel Row, Bakewell DE45 1EW")}

    places = ("She studied at Colégio São Bento and was referred from Coquet Medical Group.\n"
              "2019–2024  Data analyst, Brightwell & Sons\n2019–2021  Revenue up, costs down\n"
              "She will fly to Birmingham and we meet her at half past six in the evening.\n")
    found = rules(places, [ner("LOCATION", "Birmingham", places)])
    assert ("SENSITIVE_ORGANIZATION", "Colégio São Bento") in found
    assert ("SENSITIVE_ORGANIZATION", "Coquet Medical Group") in found
    assert ("ORGANIZATION", "Brightwell & Sons") in found
    assert ("LOCATION", "Birmingham") in found
    assert ("TIME", "half past six in the evening") in found
    assert not any("Revenue" in value or "costs" in value for _, value in found)

    # What the NER layers say is checked against the same principles.
    mixed = ("Name: Thiago Monteiro dos Santos\nItem 1 — Name: Felco 2 secateurs\n"
             "Telefono: +39 02 5550 0142\nDate of birth (DD/MM/YYYY)\n"
             "Insured with Albatros Versicherung. Officina Bellandi S.r.l. thanks you. Grazie!\n"
             "Deliver to: 9 Sráid an Mhuilinn, Dingle, Co. Kerry, V92 X7K3\n")
    found = rules(mixed, [
        ner("PERSON", "Thiago Monteiro", mixed, "custom_ner", 0.95),
        ner("PERSON", "Felco 2", mixed, "custom_ner", 0.95),
        ner("PERSON", "Telefono", mixed), ner("PERSON", "MM", mixed),
        ner("PERSON", "Albatros Versicherung", mixed, "custom_ner", 0.95),
        ner("PERSON", "Officina Bellandi", mixed, "custom_ner", 0.95),
        ner("PERSON", "Grazie", mixed),
        ner("ADDRESS", "9 Sráid an Mhuilinn, Dingle, Co. Kerry, V92", mixed, "custom_ner", 0.95)])
    assert found == {("PERSON", "Thiago Monteiro dos Santos"), ("PERSON", "Thiago Monteiro"),
                     ("PHONE", "+39 02 5550 0142"),
                     ("ADDRESS", "9 Sráid an Mhuilinn, Dingle, Co. Kerry, V92 X7K3")}
    # Without a NER person inside it, a bare "Name:" value stays a candidate.
    assert ("PERSON", "Thiago Monteiro dos Santos") not in rules(mixed)

    # The first-name echo stops at a site; a CV employer echoes, a letterhead does not.
    email = ("Brightwell & Sons\n\nFrom: Florence Abernethy\n"
             "The samples go to the Florence warehouse on Friday. Florence will sign them in.\n"
             "2019–2024  Data analyst, Brightwell & Sons\nReferee: Ada Obi, Brightwell & Sons\n")
    line = email.index("2019")
    employer = {"entity_type": "ORGANIZATION", "start": email.index("Brightwell", line),
                "end": email.index("Brightwell", line) + len("Brightwell & Sons"),
                "score": 1.0, "text": "Brightwell & Sons", "source": "regex"}
    echoed = policy_rules.echo_person_names(
        [ner("PERSON", "Florence Abernethy", email, "regex", 1.0), employer], email)
    assert [(r["entity_type"], r["text"], r["start"]) for r in echoed] == [
        ("PERSON", "Florence", email.index("Florence will")),
        ("ORGANIZATION", "Brightwell & Sons", email.index("Brightwell & Sons\n", line + 40))]
