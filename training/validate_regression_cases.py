#!/usr/bin/env python3
"""Validate reviewed must-redact and must-keep cases against the detector."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_pii_detector import TextPIIDetector


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="training/data/private/train.jsonl")
    parser.add_argument("--dev", default="training/data/private/dev.jsonl")
    parser.add_argument("--cases", default="training/data/private/regression_cases.jsonl")
    args = parser.parse_args()

    documents = {
        record["source_file"]: record["text"]
        for path in (args.train, args.dev)
        for record in read_jsonl(path)
    }
    detector = TextPIIDetector()
    failures = 0
    for case in read_jsonl(args.cases):
        text = documents.get(case["source_file"])
        if text is None:
            print(f"MISSING SOURCE: {case['source_file']}")
            failures += 1
            continue
        detected = detector.comprehensive_pii_detection(text)
        found = {(item["text"], item["entity_type"]) for item in detected}
        for expected in case.get("must_redact", []):
            if (expected["text"], expected["label"]) not in found:
                print(f"MISSED {case['source_file']}: {expected['label']} {expected['text']!r}")
                failures += 1
        detected_text = {item["text"].lower() for item in detected}
        for expected in case.get("must_keep", []):
            if expected["text"].lower() in detected_text:
                print(f"FALSE POSITIVE {case['source_file']}: {expected['text']!r}")
                failures += 1
    if failures:
        raise SystemExit(f"{failures} regression checks failed")
    print("All reviewed regression checks passed")


if __name__ == "__main__":
    main()
