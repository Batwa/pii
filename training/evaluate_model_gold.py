#!/usr/bin/env python3
"""Score a spaCy NER model ALONE against the human-gold corpus.

evaluate_gold.py scores the whole detector, where the rule layer carries most
entities and policy_rules.py discards most of what the custom model emits. This
script answers the training question instead: what would the model contribute
by itself? It runs the model directly on the gold documents and scores only the
four entities the model is trained for (see LABELING_GUIDE.md): PERSON,
ADDRESS, DATE_OF_BIRTH and SENSITIVE_ORGANIZATION. Rule-owned gold items
(pronouns, e-mails, formatted IDs, ...) are not the model's job and are skipped.

Three numbers come out per set:
  redact coverage  model-entity gold items whose every alphanumeric character
                   lies inside some model span (label-agnostic, exactly as
                   evaluate_gold.py scores); the per-label table also counts
                   how many were covered under the RIGHT label;
  keep intact      must_keep phrases that no model span touches;
  stray spans      model spans touching neither a gold PII item nor a
                   must_keep phrase, i.e. redactions of unlisted text that the
                   keep score alone cannot see.

Use it to compare the live model with a candidate before promoting one, and to
decide whether the keyword gate on custom_ner spans in policy_rules.py can be
relaxed for that candidate.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_gold import GOLD_DIR, covered, occurrences, read_jsonl

# gold-corpus label -> model label
MODEL_LABELS = {"PERSON": "PERSON", "ADDRESS": "ADDRESS",
                "DOB": "DATE_OF_BIRTH", "SENSITIVE_ORG": "SENSITIVE_ORGANIZATION"}


def touches(span, ranges):
    start, end = span
    return any(start < r_end and end > r_start for r_start, r_end in ranges)


def evaluate(nlp, docs_path, cases_path, verbose=True):
    docs = {r["source_file"]: r["text"] for r in read_jsonl(docs_path)}
    by_label = defaultdict(lambda: [0, 0, 0])  # gold label -> [covered, exact label, total]
    keep_pass = keep_fail = strays = 0
    failures = []

    for case in read_jsonl(cases_path):
        text = docs[case["source_file"]]
        ents = [(e.start_char, e.end_char, e.label_) for e in nlp(text).ents
                if e.label_ in MODEL_LABELS.values()]
        spans = [(start, end) for start, end, _ in ents]
        listed = []  # every occurrence of every gold item, PII or keep

        for item in case.get("must_redact", []):
            occ = occurrences(item["text"], text)
            listed.extend(occ)
            gold_label = item.get("label")
            if gold_label not in MODEL_LABELS:
                continue
            stats = by_label[gold_label]
            stats[2] += 1
            bad = [o for o in occ if not covered(o, text, spans)]
            if bad:
                failures.append(("MISS", case["source_file"], gold_label, item["text"],
                                 f"{len(bad)}/{len(occ)} occurrences uncovered"))
                continue
            stats[0] += 1
            same_label = [(s, e) for s, e, l in ents if l == MODEL_LABELS[gold_label]]
            if all(covered(o, text, same_label) for o in occ):
                stats[1] += 1

        for item in case.get("must_keep", []):
            occ = occurrences(item["text"], text)
            listed.extend(occ)
            hit = next((ent for ent in ents if touches(ent[:2], occ)), None)
            if hit:
                keep_fail += 1
                failures.append(("OVER", case["source_file"], hit[2], item["text"],
                                 f"model span {text[hit[0]:hit[1]]!r}"))
            else:
                keep_pass += 1

        for start, end, label in ents:
            if not touches((start, end), listed):
                strays += 1
                failures.append(("STRAY", case["source_file"], label, text[start:end],
                                 "touches no gold item"))

    if verbose:
        for kind, source_file, label, value, detail in failures:
            print(f"{kind:5} {source_file:28} {label:22} {value!r:34} {detail}")
        print()
        print(f"  {'gold label':16} {'covered':>9} {'exact label':>12}")
        for label in sorted(by_label):
            cov, exact, total = by_label[label]
            print(f"  {label:16} {cov:>4}/{total:<4} {exact:>12}")

    total = sum(v[2] for v in by_label.values())
    cov = sum(v[0] for v in by_label.values())
    keep_total = keep_pass + keep_fail
    recall = cov / total if total else 1.0
    keep_rate = keep_pass / keep_total if keep_total else 1.0
    print(f"\nredact coverage: {cov}/{total} ({recall:.1%})   "
          f"keep intact: {keep_pass}/{keep_total} ({keep_rate:.1%})   "
          f"stray spans: {strays}")
    return cov, total, keep_pass, keep_total, strays


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/custom_ner")
    parser.add_argument("--set", choices=["dev", "test", "test2", "test3", "both", "all"],
                        default="dev")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    names = {"both": ["dev", "test"],
             "all": ["dev", "test", "test2", "test3"]}.get(args.set, [args.set])
    for name in names:
        print(f"\n================ GOLD {name.upper()} -- model only: {args.model} ================")
        evaluate(nlp, GOLD_DIR / f"{name}_docs.jsonl",
                 GOLD_DIR / f"{name}_cases.jsonl", verbose=not args.quiet)


if __name__ == "__main__":
    main()
