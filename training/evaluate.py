#!/usr/bin/env python3
"""Report exact-span precision, recall, and F1 for a local custom NER model.

A span counts only when its character offsets and its label both match the
annotation. Besides the aggregate line, the report breaks the score down per
label and sorts every mismatch into one of four kinds:

  MISSED    no prediction touches the gold span
  SPURIOUS  a prediction touches no gold span
  BOUNDARY  a prediction with the right label but the wrong edges
  LABEL     a prediction with the wrong label on (or across) the gold span
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import load_examples
from config import CUSTOM_NER_MODEL

TARGET_LABELS = {"PERSON", "DATE_OF_BIRTH", "SENSITIVE_ORGANIZATION", "ADDRESS"}
ERROR_KINDS = ("MISSED", "SPURIOUS", "BOUNDARY", "LABEL")


def prf(correct, predicted, expected):
    precision = correct / predicted if predicted else 0.0
    recall = correct / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def overlapping(span, others):
    start, end = span[0], span[1]
    return [other for other in others if other[0] < end and other[1] > start]


def classify_errors(expected, predicted):
    """Sort the mismatches between two sets of (start, end, label) spans."""
    kinds = Counter()
    for gold in expected - predicted:
        hits = overlapping(gold, predicted)
        if not hits:
            kinds["MISSED"] += 1
        elif any(hit[2] == gold[2] for hit in hits):
            kinds["BOUNDARY"] += 1
        else:
            kinds["LABEL"] += 1
    for guess in predicted - expected:
        if not overlapping(guess, expected):
            kinds["SPURIOUS"] += 1
    return kinds


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=CUSTOM_NER_MODEL)
    parser.add_argument("--data", default="training/data/seed_dev.jsonl")
    parser.add_argument("--quiet", action="store_true",
                        help="print the tables only, not every mismatching document")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    expected_by, predicted_by, correct_by = Counter(), Counter(), Counter()
    errors = Counter()
    for text, annotations in load_examples(args.data):
        expected = set(annotations["entities"])
        predicted = {(ent.start_char, ent.end_char, ent.label_)
                     for ent in nlp(text).ents if ent.label_ in TARGET_LABELS}
        expected_by.update(label for _, _, label in expected)
        predicted_by.update(label for _, _, label in predicted)
        correct_by.update(label for _, _, label in expected & predicted)
        if expected != predicted:
            errors.update(classify_errors(expected, predicted))
            if not args.quiet:
                def show(spans):
                    return sorted((text[start:end], label) for start, end, label in spans)
                print(f"REVIEW: {text.replace(chr(10), ' | ')}")
                print(f"  expected:  {show(expected)}")
                print(f"  predicted: {show(predicted)}")

    print(f"\n  {'label':24} {'P':>7} {'R':>7} {'F1':>7}   correct/expected  predicted")
    for label in sorted(set(expected_by) | set(predicted_by)):
        p, r, f = prf(correct_by[label], predicted_by[label], expected_by[label])
        print(f"  {label:24} {p:7.1%} {r:7.1%} {f:7.1%}   "
              f"{correct_by[label]:>7}/{expected_by[label]:<9} {predicted_by[label]}")
    print("  errors: " + "  ".join(f"{kind} {errors[kind]}" for kind in ERROR_KINDS))

    correct = sum(correct_by.values())
    predicted_total = sum(predicted_by.values())
    expected_total = sum(expected_by.values())
    precision, recall, f1 = prf(correct, predicted_total, expected_total)
    print(f"\nprecision={precision:.1%} recall={recall:.1%} f1={f1:.1%} "
          f"({correct}/{expected_total} exact spans)")


if __name__ == "__main__":
    main()
