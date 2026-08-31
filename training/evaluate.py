#!/usr/bin/env python3
"""Report exact-span precision, recall, and F1 for a local custom NER model."""
import argparse
import sys
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).parent))
from train import load_examples

TARGET_LABELS = {"PERSON", "DATE_OF_BIRTH", "SENSITIVE_ORGANIZATION", "ADDRESS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/custom_ner")
    parser.add_argument("--data", default="training/data/seed_dev.jsonl")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    expected_total = predicted_total = correct = 0
    for text, annotations in load_examples(args.data):
        expected = {(text[start:end], label) for start, end, label in annotations["entities"]}
        predicted = {(ent.text, ent.label_) for ent in nlp(text).ents if ent.label_ in TARGET_LABELS}
        expected_total += len(expected)
        predicted_total += len(predicted)
        correct += len(expected & predicted)
        if expected != predicted:
            print(f"REVIEW: {text}")
            print(f"  expected: {sorted(expected)}")
            print(f"  predicted: {sorted(predicted)}")

    precision = correct / predicted_total if predicted_total else 0.0
    recall = correct / expected_total if expected_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"precision={precision:.1%} recall={recall:.1%} f1={f1:.1%} ({correct}/{expected_total} exact spans)")


if __name__ == "__main__":
    main()
