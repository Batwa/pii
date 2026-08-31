#!/usr/bin/env python3
"""Draw the stratified adjudication sample from the generated corpus.

Two kinds of checks, per LABELING_GUIDE.md's workflow step 3:
- labeled spans  — "is this span correctly labeled per the guide?"
- decoy strings  — "was this correctly LEFT unlabeled?" (negative side)

Writes training/data/generated/review_sample.jsonl with one check per line:
id, kind, label, span text, document context, and source_file.
"""
import json
import random
from pathlib import Path

GEN_DIR = Path(__file__).resolve().parent / "data" / "generated"
SEED = 7
# checks per category
SPAN_QUOTA = {"PERSON": 30, "ADDRESS": 20, "DATE_OF_BIRTH": 18,
              "SENSITIVE_ORGANIZATION": 12}
DECOY_QUOTA = {"decoy_date": 5, "decoy_employer": 5, "decoy_city": 5,
               "decoy_thing_id": 4, "decoy_author_org": 8}
CONTEXT = 150


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def context_of(text, value):
    i = text.find(value)
    lead = text[max(0, i - CONTEXT):i].replace("\n", " ⏎ ")
    tail = text[i + len(value):i + len(value) + CONTEXT].replace("\n", " ⏎ ")
    return lead, tail


def main():
    rng = random.Random(SEED)
    docs = (read_jsonl(GEN_DIR / "train.jsonl") + read_jsonl(GEN_DIR / "dev.jsonl"))
    decoy_docs = (read_jsonl(GEN_DIR / "train_decoys.jsonl")
                  + read_jsonl(GEN_DIR / "dev_decoys.jsonl"))

    span_pool, seen_values = {}, set()
    for doc in docs:
        for span in doc["spans"]:
            # one check per distinct surface form keeps the sample varied
            if span["text"] in seen_values:
                continue
            seen_values.add(span["text"])
            span_pool.setdefault(span["label"], []).append((doc, span["text"]))

    # A decoy may share its surface form with a labeled span elsewhere — the
    # same facility is redacted as a third-party mention and kept as a
    # letterhead, and checking both sides of that boundary is deliberate.
    decoy_pool, seen_decoys = {}, set()
    for doc in decoy_docs:
        for decoy in doc["decoys"]:
            if doc["text"].count(decoy["text"]) != 1 or decoy["text"] in seen_decoys:
                continue
            seen_decoys.add(decoy["text"])
            decoy_pool.setdefault(decoy["kind"], []).append((doc, decoy["text"]))

    checks = []
    for label, quota in SPAN_QUOTA.items():
        for doc, value in rng.sample(span_pool[label], quota):
            lead, tail = context_of(doc["text"], value)
            checks.append({"kind": "span", "label": label, "text": value,
                           "lead": lead, "tail": tail,
                           "source_file": doc["source_file"]})
    for kind, quota in DECOY_QUOTA.items():
        for doc, value in rng.sample(decoy_pool[kind], quota):
            lead, tail = context_of(doc["text"], value)
            checks.append({"kind": "decoy", "label": kind, "text": value,
                           "lead": lead, "tail": tail,
                           "source_file": doc["source_file"]})

    rng.shuffle(checks)
    with open(GEN_DIR / "review_sample.jsonl", "w") as f:
        for n, check in enumerate(checks, 1):
            check["id"] = f"S{n:03d}"
            f.write(json.dumps(check, ensure_ascii=False) + "\n")
    print(f"wrote {len(checks)} checks "
          f"({sum(SPAN_QUOTA.values())} labeled spans + "
          f"{sum(DECOY_QUOTA.values())} decoys) -> review_sample.jsonl")


if __name__ == "__main__":
    main()
