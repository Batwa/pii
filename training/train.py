#!/usr/bin/env python3
"""Train a local custom spaCy NER model from reviewed JSONL annotations."""
import argparse
import json
import random
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR  # noqa: E402  (the project root, see above)


def load_examples(path):
    examples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        text = record["text"]
        entities = []
        for span in record.get("spans", []):
            value, label = span["text"], span["label"]
            if "start" in span:
                # Round-3 records carry offsets, so a name may repeat.
                start = span["start"]
                if text[start:start + len(value)] != value:
                    raise ValueError(f"{path}:{line_number}: offset {start} is not {value!r}")
            else:
                start = text.find(value)
                if start < 0 or text.find(value, start + 1) >= 0:
                    raise ValueError(f"{path}:{line_number}: span must appear exactly once: {value!r}")
            entities.append((start, start + len(value), label))
        entities.sort()
        if any(left[1] > right[0] for left, right in zip(entities, entities[1:])):
            raise ValueError(f"{path}:{line_number}: entity spans overlap")
        examples.append((text, {"entities": entities}))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def dev_f1(nlp, dev_data, labels):
    """Exact-span F1 over the trained labels on the dev set."""
    correct = predicted = expected = 0
    for text, annotations in dev_data:
        gold = set(annotations["entities"])
        found = {(ent.start_char, ent.end_char, ent.label_)
                 for ent in nlp(text).ents if ent.label_ in labels}
        correct += len(gold & found)
        predicted += len(found)
        expected += len(gold)
    if not correct:
        return 0.0
    precision, recall = correct / predicted, correct / expected
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="training/data/seed_train.jsonl")
    parser.add_argument("--dev", default="training/data/seed_dev.jsonl")
    # Deliberately NOT models/custom_ner: the detector auto-loads that path, so
    # training straight into it would silently change production behavior.
    # Promote a candidate by moving it there after evaluation passes.
    parser.add_argument("--output", default=os.path.join(MODELS_DIR, "custom_ner_candidate"))
    parser.add_argument("--base-model", default="en_core_web_lg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--keep-best", action="store_true",
                        help="score the dev set after every epoch and save the best "
                             "epoch instead of the last one")
    args = parser.parse_args()

    random.seed(args.seed)
    # Also seed numpy/thinc: without this two runs with the same --seed give
    # different models, and candidate comparisons measure noise.
    spacy.util.fix_random_seed(args.seed)
    train_data = load_examples(args.train)
    dev_data = load_examples(args.dev)
    nlp = spacy.load(args.base_model)
    ner = nlp.get_pipe("ner") if "ner" in nlp.pipe_names else nlp.add_pipe("ner")
    labels = sorted({label for _, annotation in train_data for _, _, label in annotation["entities"]})
    for label in labels:
        ner.add_label(label)

    best_score, best_epoch, best_weights = -1.0, 0, None
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for epoch in range(args.epochs):
            random.shuffle(train_data)
            losses = {}
            for batch in minibatch(train_data, size=args.batch_size):
                examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
                nlp.update(examples, sgd=optimizer, drop=0.25, losses=losses)
            line = f"epoch {epoch + 1:02d}: ner_loss={losses.get('ner', 0):.4f}"
            if args.keep_best:
                # The loss still swings late in training, so the last epoch
                # can be a poor one; keep the weights of the best dev score.
                score = dev_f1(nlp, dev_data, labels)
                line += f" dev_f1={score:.4f}"
                if score > best_score:
                    best_score, best_epoch, best_weights = score, epoch + 1, ner.to_bytes()
            print(line)
        if args.keep_best:
            ner.from_bytes(best_weights)
            print(f"kept epoch {best_epoch:02d} (dev_f1={best_score:.4f})")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output)
    print(f"Saved local model to {output}")
    print(f"Evaluate it with: python training/evaluate.py --model {output} --data {args.dev}")


if __name__ == "__main__":
    main()
