#!/usr/bin/env python3
"""Score the detector against the human-gold corpus.

Coverage-based: a must_redact span passes when every alphanumeric character of
every occurrence lies inside some detected span — the label does not need to
match, because the product outcome is whether the text is masked. A must_keep
span fails when any detected span overlaps any of its occurrences.

Matching is case-insensitive on word boundaries and counts ALL occurrences, so
short gold strings such as pronouns cover every instance in the document.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_pii_detector import TextPIIDetector

GOLD_DIR = Path(__file__).resolve().parent / "data" / "gold"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def occurrences(needle, haystack):
    pattern = re.escape(needle)
    if needle[0].isalnum():
        pattern = r"\b" + pattern
    if needle[-1].isalnum():
        pattern = pattern + r"\b"
    return [m.span() for m in re.finditer(pattern, haystack, re.IGNORECASE)]


def covered(span, text, detected):
    """True when every alphanumeric char of text[span] is inside a detected span."""
    start, end = span
    for i in range(start, end):
        if not text[i].isalnum():
            continue
        if not any(d_start <= i < d_end for d_start, d_end in detected):
            return False
    return True


def evaluate(detector, docs_path, cases_path, verbose=True):
    docs = {r["source_file"]: r["text"] for r in read_jsonl(docs_path)}
    redact_pass = redact_fail = keep_pass = keep_fail = 0
    by_label = defaultdict(lambda: [0, 0])  # label -> [pass, fail]
    failures = []

    for case in read_jsonl(cases_path):
        text = docs[case["source_file"]]
        detections = detector.comprehensive_pii_detection(text)
        spans = [(d["start"], d["end"]) for d in detections]

        for item in case.get("must_redact", []):
            label = item.get("label", "?")
            occ = occurrences(item["text"], text)
            bad = [o for o in occ if not covered(o, text, spans)]
            if bad:
                redact_fail += 1
                by_label[label][1] += 1
                start = bad[0][0]
                context = text[max(0, start - 30):bad[0][1] + 20].replace("\n", " | ")
                failures.append(("MISS", case["source_file"], label, item["text"],
                                 f"{len(bad)}/{len(occ)} occurrences uncovered ...{context!r}"))
            else:
                redact_pass += 1
                by_label[label][0] += 1

        for item in case.get("must_keep", []):
            occ = occurrences(item["text"], text)
            hit = None
            for o_start, o_end in occ:
                for d in detections:
                    if d["start"] < o_end and d["end"] > o_start:
                        hit = d
                        break
                if hit:
                    break
            if hit:
                keep_fail += 1
                failures.append(("OVER", case["source_file"], hit["entity_type"],
                                 item["text"],
                                 f"redacted by {hit['source']} as {hit['text']!r}"))
            else:
                keep_pass += 1

    if verbose:
        for kind, source_file, label, text, detail in failures:
            print(f"{kind}  {source_file:28} {label:16} {text!r:34} {detail}")
        print()
        for label in sorted(by_label):
            ok, bad = by_label[label]
            print(f"  {label:16} {ok}/{ok + bad}")
    total_redact = redact_pass + redact_fail
    total_keep = keep_pass + keep_fail
    recall = redact_pass / total_redact if total_redact else 1.0
    keep_rate = keep_pass / total_keep if total_keep else 1.0
    print(f"\nredact coverage: {redact_pass}/{total_redact} ({recall:.1%})   "
          f"keep intact: {keep_pass}/{total_keep} ({keep_rate:.1%})")
    return recall, keep_rate, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["dev", "test", "test2", "both", "all"],
                        default="dev")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    detector = TextPIIDetector()
    names = {"both": ["dev", "test"],
             "all": ["dev", "test", "test2"]}.get(args.set, [args.set])
    for name in names:
        print(f"\n================ GOLD {name.upper()} ================")
        evaluate(detector, GOLD_DIR / f"{name}_docs.jsonl",
                 GOLD_DIR / f"{name}_cases.jsonl", verbose=not args.quiet)


if __name__ == "__main__":
    main()
