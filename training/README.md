# Local custom-entity training

**Start with [LABELING_GUIDE.md](LABELING_GUIDE.md)** — it defines which
entities the model learns versus which the rule layer (`policy_rules.py`)
owns, the labeling criteria, and the LLM pre-labeling workflow. Label only the
model's entities; rules already beat the model on everything formatted.

`train.py` writes to `models/custom_ner_candidate` by default. The detector
auto-loads `models/custom_ner`, so promote a candidate there only after
`evaluate.py` and the gold suites pass.

## Human-gold evaluation corpus

`build_gold_corpus.py` authors 40 fully synthetic labeled documents (25 dev /
15 held-out test) under `training/data/gold/`, covering the V1 public-sharing
policy: pronouns, labelled IDs, addresses with apartment/city/zip tails, worded
birth dates, sensitive facilities, employer-linked organizations, and negative
documents that must survive untouched. The header of that script records every
labeling decision taken where the policy is open.

Score the detector against it with:

```bash
python training/evaluate_gold.py --set both
```

The score is coverage-based (a span passes when all its alphanumeric characters
are masked, whatever the entity label). Tune rules against **dev only**; run
**test** at milestones — if test drops well below dev, the rules are memorizing
the dev set. `--set all` adds **test2**, a fresh batch authored after the rules
phase (the original test set absorbed five fixes from its first run, so test2
is the cleaner generalization measurement — its own first blind run scored
97.2%/100%, with one fix applied since). Current state: all three sets pass
100% redact coverage / 100% keep.

## First NER candidate (2026-11-04)

`train.py` on the generated corpus (400 train / 40 dev documents, 20 epochs,
seed 42, about 2.5 minutes on CPU) produced the first candidate trained on more
than the 16 seed examples. The live model is still that seed model: it has no
`ADDRESS` label at all, and `policy_rules.py` discards everything it emits
except birth dates and keyword-matching facilities. Measured with the two
evaluators described below:

| | live `models/custom_ner` | candidate |
|---|---|---|
| generated dev, exact-span F1 (72 spans) | 51.7% | 98.6% (one ADDRESS boundary error) |
| gold, model alone: redact coverage (89 model-entity items, 50 docs) | 71 (79.8%) | 84 (94.4%) |
| gold, model alone: keep intact (267 phrases) | 209 (78.3%) | 258 (96.6%) |
| gold, model alone: stray spans | 20 | 1 |
| gold, ADDRESS covered under the right label (23) | 0 | 23 |
| gold, DOB covered (8) | 4 | 7 |
| gold, PERSON covered (56) | 52 | 52 |
| held-out only (test + test2): coverage / keep / strays | 35/41, 106/132, 9 | 39/41, 129/132, 0 |

Plugged into the full detector (`CUSTOM_NER_MODEL=models/custom_ner_candidate`)
the candidate changes nothing: all three gold sets stay at 100% / 100% and both
regression suites pass, because the gate hides the model. Promotion is
therefore safe but pointless until the gate is relaxed for it.

Read the generated-dev F1 with care: dev shares its templates with train and
the loss is near zero from epoch 5, so that number mostly measures
memorization. The model-only gold score is the honest one, and its remaining
errors all come from layouts the templates never produce: skill lists
("Python, SQL") and language lists ("Dutch, English, French") labeled
`SENSITIVE_ORGANIZATION`; a year range ("2019-2023") and a grade ("4.6 of 5.0")
labeled `DATE_OF_BIRTH`; a weekday swallowed into a name ("Monday Aisha
Bello"); a car model as `PERSON`; and five misses (four names, one birth date).

Next steps, in order: add those decoys and layouts to
`generate_training_corpus.py` and retrain; then relax the custom-model gate in
`policy_rules.py` for PERSON and ADDRESS and re-measure with
`evaluate_model_gold.py` and the full gold suite.

This folder trains an optional spaCy NER model and maintains exact-identifier regression cases. The model is useful for contextual labels; regex rules are more reliable for fixed formats such as phone numbers and bank accounts.

- `PERSON`
- `DATE_OF_BIRTH`
- `SENSITIVE_ORGANIZATION`
- `ADDRESS`

The included seed data is synthetic. It demonstrates the annotation format and includes negative examples such as `night`, ordinary document issue dates, and generic clinic references. It is only a baseline—not enough data to claim reliable production performance.

## Add reviewed annotations locally

Put real reviewed examples in `training/data/private/`. That directory is ignored by Git so private documents cannot be committed. Use one JSON object per line:

```json
{"text":"Patient Jordan Lee visited Example Clinic No. 4.","spans":[{"text":"Jordan Lee","label":"PERSON"},{"text":"Example Clinic No. 4","label":"SENSITIVE_ORGANIZATION"}]}
```

Do not add a span for text that should remain visible. For example, a sentence containing `movement at night` should have no `night` span.

Every annotated span must occur exactly once in its `text`; the training script validates this before training.

## Create a private review draft from existing documents

To prefill drafts from a ZIP of original `.txt` files, run:

```bash
python training/create_review_set.py /path/to/originals.zip --numbered-only
```

This creates `train.jsonl`, `dev.jsonl`, `regression_cases.jsonl`, and `REVIEW_CHECKLIST.md` under `training/data/private/`. The draft includes the names, facilities, addresses, and exact identifiers the current detector finds. It does not train a model and must be proofread before training.

## Train and evaluate

```bash
python training/train.py \
  --train training/data/private/train.jsonl \
  --dev training/data/private/dev.jsonl \
  --output models/custom_ner_candidate

python training/evaluate.py \
  --model models/custom_ner_candidate \
  --data training/data/private/dev.jsonl

python training/evaluate_model_gold.py \
  --model models/custom_ner_candidate --set all

python training/validate_regression_cases.py
python training/evaluate_gold.py --set all

# Promote only after every check above passes:
# rm -rf models/custom_ner && mv models/custom_ner_candidate models/custom_ner
```

`evaluate.py` reports exact-span precision, recall and F1 per label and sorts
every mismatch into MISSED, SPURIOUS, BOUNDARY or LABEL. `evaluate_model_gold.py`
runs the model **alone** on the human-gold corpus and scores only the four
model entities, plus the must-keep phrases it touches and the stray spans that
touch no gold item at all. The full-pipeline gold score cannot judge a model:
`policy_rules.py` discards every custom-model span except birth dates and
keyword-matching facilities, so a bad model and a good one score the same there.

For a safe pipeline check with synthetic data only:

```bash
python training/train.py
python training/evaluate.py
```

The detector automatically loads `models/custom_ner` when it exists. Keep it only when its held-out evaluation and manual review meet your privacy policy. You can point to a different local model directory with the `CUSTOM_NER_MODEL` environment variable.

`validate_regression_cases.py` verifies exact identifiers with the hybrid detector. Use it after every change so that a fix for one file does not reintroduce the missed phones, account numbers, or other reviewed identifiers.

## International identifier coverage

Presidio ships country ID recognizers for Australia, Canada, India, Nigeria, the Philippines, Singapore, the UK and South Africa, but leaves them out of its default registry. `config.INTERNATIONAL_RECOGNIZERS` lists the English-capable ones and `recognizers.build_analyzer_registry()` turns them on.

Several of them carry no checksum, so they score below `CONFIDENCE_THRESHOLD` even on a valid identifier sitting next to its own label. Presidio is therefore queried at `PRESIDIO_SCORE_FLOOR` and filtered per entity through `config.ENTITY_SCORE_THRESHOLDS`. Widening any of those thresholds trades false negatives for false positives — re-measure before doing it.

`training/data/international_*.jsonl` is a synthetic corpus covering 23 identifier types plus three negative documents. It is committed rather than kept under `private/` because no value in it belongs to a real person. Run it against the detector with:

```bash
python training/validate_regression_cases.py \
  --train training/data/international_docs.jsonl \
  --dev training/data/international_docs.jsonl \
  --cases training/data/international_cases.jsonl
```

Checksum-bearing values were generated by brute force until the recognizer accepted them. An identifier with a bad checksum is silently ignored, so replacing these values with arbitrary digits will look like a broken recognizer.

The suite passes fully as of the rule layer (re-verified 2026-11-04 with both the live model and the first candidate). Before the rule layer, three cases failed on a label rather than on redaction (`CA_SIN` also validated as `AU_TFN`; `IN_PAN` and `PH_UMID` were absorbed by the custom model's `SENSITIVE_ORGANIZATION`) and two documents lost `Lagos` and `DVLA` to the general spaCy model.
