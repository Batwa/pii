#!/usr/bin/env python3
"""Average the NER weights of several candidates trained from the same base.

Two runs of train.py that differ only in the seed end up as far apart on the
gold sets as two corpus rounds (see "NER candidate 4" in README.md). All runs
start from the same pretrained weights, so their parameters can be averaged
position by position; this script writes that average as one more candidate.
It is an experiment, not a step of the promotion checklist: score the result
like any other candidate before trusting it.

    python training/average_models.py --models A B --output C
"""
import argparse
from pathlib import Path

import spacy


def average(model_dirs, output):
    pipelines = [spacy.load(path) for path in model_dirs]
    components = [nlp.get_pipe("ner") for nlp in pipelines]
    labels = components[0].labels
    for path, ner in zip(model_dirs, components):
        if ner.labels != labels:
            raise ValueError(f"{path}: labels differ from {model_dirs[0]}")
    walks = [list(ner.model.walk()) for ner in components]
    if len({len(walk) for walk in walks}) != 1:
        raise ValueError("the models do not have the same architecture")
    averaged = 0
    for nodes in zip(*walks):
        for name in nodes[0].param_names:
            if not nodes[0].has_param(name):
                continue
            params = [node.get_param(name) for node in nodes]
            if any(param.shape != params[0].shape for param in params):
                raise ValueError(f"shape mismatch in {nodes[0].name}.{name}")
            nodes[0].set_param(name, sum(params[1:], params[0].copy()) / len(params))
            averaged += 1
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipelines[0].to_disk(output)
    print(f"averaged {averaged} parameter arrays of {len(model_dirs)} models -> {output}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("--models needs at least two model directories")
    average(args.models, args.output)


if __name__ == "__main__":
    main()
