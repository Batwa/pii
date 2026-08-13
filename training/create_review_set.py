#!/usr/bin/env python3
"""Create private, reviewable custom-NER annotations from text files or a ZIP."""
import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_pii_detector import TextPIIDetector

TRAINING_TYPES = {
    "PERSON", "DATE_OF_BIRTH", "SENSITIVE_ORGANIZATION", "ADDRESS",
    "PHONE", "PHONE_RU", "SNILS", "INN", "BANK_ACCOUNT",
    "PAYMENT_CARD", "PASSPORT_NUMBER", "EMAIL", "BIC", "VIN",
    "LICENSE_PLATE", "POLICY_NUMBER", "STUDENT_ID",
}


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def text_files(source):
    if source.is_dir():
        return sorted((path for path in source.rglob("*.txt") if "__MACOSX" not in path.parts), key=natural_key)
    if source.suffix.lower() != ".zip":
        raise ValueError("Source must be a directory or .zip archive")
    temporary = Path(tempfile.mkdtemp(prefix="pii-review-"))
    with zipfile.ZipFile(source) as archive:
        archive.extractall(temporary)
    return temporary, sorted(
        (path for path in temporary.rglob("*.txt") if "__MACOSX" not in path.parts), key=natural_key
    )


def draft_spans(detector, text):
    # Include direct identifier rules plus NER suggestions. The user must
    # proofread this draft before any private data is used for training.
    proposed = detector.detect_pii_regex(text)
    for item in detector.detect_pii_spacy(text):
        if item["entity_type"] == "ORGANIZATION":
            item = {**item, "entity_type": "SENSITIVE_ORGANIZATION"}
        proposed.append(item)
    proposed = [item for item in proposed if item["entity_type"] in TRAINING_TYPES]
    selected = detector.deduplicate_results(proposed, text)
    spans = []
    for item in selected:
        value = item["text"]
        if text.count(value) != 1:
            continue
        spans.append({"text": value, "label": item["entity_type"], "source": item["source"]})
    return spans


def regression_case(detector, source_file, text):
    """Create a reviewable exact-identifier test case for the current rules."""
    direct = [item for item in detector.detect_pii_regex(text) if item["entity_type"] in TRAINING_TYPES]
    return {
        "source_file": source_file,
        "must_redact": [
            {"text": item["text"], "label": item["entity_type"]}
            for item in detector.deduplicate_results(direct, text)
        ],
        "must_keep": [
            {"text": word, "reason": "ordinary time or duration, not PII"}
            for word in ("night", "morning", "evening")
            if re.search(rf"\b{word}\b", text, re.IGNORECASE)
        ],
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="A ZIP archive or folder containing original .txt documents")
    parser.add_argument("--output", default="training/data/private")
    parser.add_argument("--dev-count", type=int, default=4)
    parser.add_argument("--numbered-only", action="store_true", help="Use only filenames such as 1.txt through 20.txt")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    result = text_files(source)
    temporary, files = (result if isinstance(result, tuple) else (None, result))
    if args.numbered_only:
        files = [path for path in files if path.stem.isdigit()]
    if len(files) <= args.dev_count:
        raise ValueError("Not enough documents to create train and development sets")

    detector = TextPIIDetector()
    records = []
    regression_cases = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        records.append({"source_file": path.name, "text": text, "spans": draft_spans(detector, text)})
        regression_cases.append(regression_case(detector, path.name, text))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_records, dev_records = records[:-args.dev_count], records[-args.dev_count:]
    write_jsonl(output / "train.jsonl", train_records)
    write_jsonl(output / "dev.jsonl", dev_records)
    write_jsonl(output / "regression_cases.jsonl", regression_cases)
    (output / "REVIEW_CHECKLIST.md").write_text(
        "# Review checklist\n\n"
        "These annotations are drafts, not training truth. Review every record before training.\n\n"
        "- Keep only full personal names that should be redacted as `PERSON`; remove locations, street names, job titles, and generic roles incorrectly suggested as people.\n"
        "- Label only birth dates as `DATE_OF_BIRTH`; do not label document issue dates, durations, or words such as `night`.\n"
        "- Label clinics, hospitals, schools, and other facilities as `SENSITIVE_ORGANIZATION` only if your privacy policy requires them to be hidden.\n"
        "- Keep direct identifiers (phone numbers, SNILS, INN, bank accounts, payment cards, passports, BICs, VINs, and policy/student IDs) when they should be redacted.\n"
        "- Add missing spans, remove false positives, and keep every span text unique within its document.\n"
        "- Do not commit this directory; it contains private source documents.\n",
        encoding="utf-8",
    )
    print(f"Created {len(train_records)} training and {len(dev_records)} development drafts plus {len(regression_cases)} regression cases in {output}")
    if temporary:
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
