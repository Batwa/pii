#!/usr/bin/env python3
"""Turn human-gold sets into NER training records.

Rounds 4 and 5 of the generated corpus left held-out recall where it was (see
"NER candidate 5" in README.md): templates teach the model what is not a
name, and no longer teach it to find more names. The gold documents are a
different kind of data. Each one was written by hand, no two share a layout,
and their labels have been through a blind run and a fix round. This script
converts the sets that are no longer held out into the training format, so
that train.py can add them to the generated corpus (--extra-train).

    python training/gold_to_training.py                    # dev ... test8
    python training/gold_to_training.py --sets dev,test    # another choice

A set that is converted here stops measuring the model: a candidate trained
on it has seen its documents. Never pass a set that is still held out, and
keep at least one fitted set out of the training data for --select-gold.

How a gold case becomes training spans (LABELING_GUIDE.md, "The split"):

- Only the four model entities are written. PERSON, DOB and SENSITIVE_ORG
  items become PERSON, DATE_OF_BIRTH and SENSITIVE_ORGANIZATION spans at
  every occurrence, as the gold scorer counts them.
- ADDRESS, CITY_RESIDENCE and ZIP items that stand next to each other on one
  line are joined into one ADDRESS span ("7 Tannery Lane, Flat 2,
  Otley"), the convention of the generated corpus. A town or a postcode
  on its own is left unlabelled: the residence and travel rules own those.
- A model span inside a rule-owned item is dropped (a first name inside an
  e-mail address), and so is a span that does not end on a token boundary.
- Where two items overlap, the longer one wins ("Rangi Tautahi" over
  "Rangi").
"""
import argparse
import json
import re
import sys
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_gold import GOLD_DIR, occurrences, read_jsonl  # noqa: E402

DEFAULT_SETS = "dev,test,test2,test3,test4,test5,test6,test7,test8"
OUTPUT = Path(__file__).resolve().parent / "data" / "handwritten" / "train.jsonl"

MODEL_LABELS = {"PERSON": "PERSON", "DOB": "DATE_OF_BIRTH",
                "SENSITIVE_ORG": "SENSITIVE_ORGANIZATION"}
ADDRESS_FAMILY = {"ADDRESS", "CITY_RESIDENCE", "ZIP"}
# What may stand between two parts of one address line: a comma, a space, a
# state code or a county ("Springfield, IL 62704", "Tralee, Co. Kerry, V92").
ADDRESS_GLUE = re.compile(r"[ \t]*,?[ \t]*(?:[A-Z]{2}|Co\.[ \t]\w+,?)?[ \t]*")


def address_spans(pieces, text):
    """Join neighbouring address-family pieces; keep groups with a street."""
    groups = []
    for start, end, label in sorted(pieces):
        if groups and start < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].add(label)
        elif groups and ADDRESS_GLUE.fullmatch(text[groups[-1][1]:start]):
            groups[-1][1] = end
            groups[-1][2].add(label)
        else:
            groups.append([start, end, {label}])
    return [(start, end, "ADDRESS") for start, end, labels in groups
            if "ADDRESS" in labels or len(labels) > 1]


def convert(case, text, nlp):
    """Return (spans, dropped) for one gold case."""
    candidates, pieces, rule_owned = [], [], []
    for item in case["must_redact"]:
        label = item["label"]
        for start, end in occurrences(item["text"], text):
            if label in MODEL_LABELS:
                candidates.append((start, end, MODEL_LABELS[label]))
            elif label in ADDRESS_FAMILY:
                pieces.append((start, end, label))
            else:
                rule_owned.append((start, end))
    candidates += address_spans(pieces, text)

    doc = nlp.make_doc(text)
    kept, dropped = [], []
    # longest first, so that a full name wins over the first name inside it
    for start, end, label in sorted(candidates, key=lambda s: (s[0] - s[1], s[0])):
        if any(start < k_end and end > k_start for k_start, k_end, _ in kept):
            continue
        inside_rule_item = any(r_start <= start and end <= r_end and (r_start, r_end) != (start, end)
                               for r_start, r_end in rule_owned)
        if inside_rule_item or doc.char_span(start, end) is None:
            dropped.append((label, text[start:end]))
            continue
        kept.append((start, end, label))
    return sorted(kept), dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", default=DEFAULT_SETS,
                        help="comma-separated gold sets that are no longer held out")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    nlp = spacy.blank("en")
    records, counts, dropped_all = [], {}, []
    for name in [s for s in args.sets.split(",") if s]:
        docs = {r["source_file"]: r["text"] for r in read_jsonl(GOLD_DIR / f"{name}_docs.jsonl")}
        for case in read_jsonl(GOLD_DIR / f"{name}_cases.jsonl"):
            text = docs[case["source_file"]]
            spans, dropped = convert(case, text, nlp)
            dropped_all += [(case["source_file"],) + d for d in dropped]
            for _, _, label in spans:
                counts[label] = counts.get(label, 0) + 1
            records.append({
                "source_file": case["source_file"], "text": text,
                "spans": [{"text": text[s:e], "label": label, "start": s, "end": e}
                          for s, e, label in spans]})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} documents from {args.sets} to {output}")
    print("spans: " + ", ".join(f"{label} {n}" for label, n in sorted(counts.items())))
    for source, label, value in dropped_all:
        print(f"dropped  {source}  {label}  {value!r}")


if __name__ == "__main__":
    main()
