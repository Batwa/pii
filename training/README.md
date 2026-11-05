# Local custom-entity training

**Start with [LABELING_GUIDE.md](LABELING_GUIDE.md)** — it defines which
entities the model learns versus which the rule layer (`policy_rules.py`)
owns, the labeling criteria, and the LLM pre-labeling workflow. Label only the
model's entities; rules already beat the model on everything formatted.

`train.py` writes to `models/custom_ner_candidate` by default. The detector
auto-loads `models/custom_ner`, so promote a candidate there only after
`evaluate.py` and the gold suites pass.

## Generated training corpus

`generate_training_corpus.py` writes `training/data/generated/{train,dev}.jsonl`
plus `*_decoys.jsonl` (the strings that must stay unlabeled): fully synthetic,
labeled with the four model entities only, every span occurring exactly once in
its document, train and dev drawn from disjoint name/street/city/facility pools,
one document in five without entities. Round 1 (seed 20260831) is 400 train /
40 dev documents across 19 positive and 4 negative genres and went through the
first adjudication round (`review_sample.jsonl`, 107 checks). Round 2
(2026-11-04, seed 20261104, appended so round 1 stays byte-identical) adds 120
train / 12 dev documents in the layouts the first candidate mislabeled on gold:
résumés with skill lists and year ranges, CVs with language lists, school
records with grades, motor claims with vehicle models, e-mail threads with a
weekday before the quoted author, chat greetings, names in running prose or by
first name only, intake forms with lowercase birth labels, and two negative
genres (job postings, vehicle listings). Round 2 has not been through an
adjudication round yet; `build_review_sample.py` samples the whole corpus and
overwrites `review_sample.jsonl`, so write a round-2 sample to a different file
before re-running it.

## Human-gold evaluation corpus

`build_gold_corpus.py` authors 76 fully synthetic labeled documents (25 dev /
15 test / 10 test2 / 13 test3 / 13 test4) under `training/data/gold/`, covering the V1 public-sharing
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
the test sets at milestones — if they drop well below dev, the rules are
memorizing the dev set. Each test batch was authored blind and scored once
before any fix: **test** (15 docs; its first run absorbed five fixes),
**test2** (10 docs; first blind run 97.2% / 100%, one fix since, and corpus
round 2 targeted its reference-letter miss, so it is no longer blind for prose
names), **test3** (13 docs, authored 2026-11-04 after candidate 2 and the gate
change), and **test4** (13 docs, authored 2026-11-04 after the test3 fixes).
`--set all` runs all five. Current state: dev, test and test2 at 100% coverage
/ 100% keep; test3 at 97.4% / 99.0% after the fixes recorded below (semi-dev);
test4's blind run is recorded below and nothing has been fixed against it.

### test3 blind run (2026-11-04)

Full pipeline (candidate 2 live, relaxed gate): redact coverage 67/76 (88.2%),
keep intact 102/103 (99.0%), 1 stray span. Model alone: 25/30 model-entity
items (83.3%), keep 101/103, 0 strays. The nine misses and two over-redactions,
grouped by cause:

- Facility names without a keyword: "Ballymore Institute of Technology",
  "Greystone Academy", "Marchmont Wellbeing Centre" (3 misses). The model found
  all three under the right label; the keyword requirement on
  `SENSITIVE_ORGANIZATION` spans dropped them. Two of the model's spans were
  sloppy ("Caldwell & Finch Surveyors (2016-2021", "Greystone Academy in
  September"), so trusting the label outright would cost keep checks; widening
  `SENSITIVE_ORG_KEYWORDS` (institute, academy, centre) is the cheaper fix.
- Formats the rules do not know: an international phone "+44 7700 900412", an
  ISO birth date "1983-07-19", a policy number with a letter prefix "POL 4471
  9920 36", and a flat number before the street ("Flat 6, 28 Harbour Lane";
  "Flat 2" after a street passed) (4 misses).
- A customer number given in a chat reply with no label beside it, "Sure,
  CU-104772." (1 miss): the label-before-number rule has nothing to anchor on.
- A first name alone in prose, "One of them, Nkechi, now leads her own crew"
  (1 miss, pipeline and model). The model alone also missed "Rhiannon" and two
  of three "Callum" mentions, which general NER caught.
- General NER over-redaction: "Halden Kitchenware" as PERSON in the negative
  recall notice (keep failure), and "Processing" as ORGANIZATION in "Processing
  takes up to 15 working days", which the employment-context filter let through
  (stray).

What passed is worth noting too: all 15 pronouns, including "their" for the
tenants with "they" and "its" kept for parcels and a building; the payslip's
employer letterhead kept while the CV's employers redacted; the UK plate, VIN,
IBAN and UK postcodes; serial and order numbers redacted with the tracking
number kept; the travel-destination city redacted with the nationality kept;
the weekday before a quoted author, chat greetings, skill and language lists,
the year range and the GPA all kept.

### Fixes applied against test3 (2026-11-04)

Applied the same day, so test3 is semi-dev like test and test2. Rule changes
in `policy_rules.py`: the facility vocabulary gains Institute, Academy, the
Wellbeing / Wellness / Counselling / Rehabilitation Centre compounds and
Counselling Service, shared by the regex and the model-span gate, with an
optional "of Technology" tail; an international phone pattern with a mandatory
leading "+", scored just under the country-specific patterns so "+7 (495) ..."
keeps its PHONE_RU label; the ISO form and the "d.o.b." / "birth date" labels
in the birth-date rule; a letter prefix in the policy-number rule; "flat" as an
apartment designator whose value must carry a digit, which also removed the
"unit price" stray on dev; Hill, Yard, Square, Quay, Parade and Heights as
street types; and "working days/hours" no longer counts as employment context.
In `text_pii_detector.py`, an address candidate nested inside a longer address
candidate now yields to it: the first version of the "flat" rule produced a
"Flat 2" span that outranked the model's full address line and exposed the
street, city and postcode around it.

Result: test3 67/76 → 74/76 (97.4%), keep 102/103 unchanged, strays 1 → 0;
dev strays 5 → 4; test and test2 unchanged at 100% / 100%; both regression
suites and the unit tests pass. The three remaining test3 failures are the
unlabeled chat-reply ID "CU-104772" (labeling-guide decision 9, no rule yet),
the first name "Nkechi" in prose (model recall), and "Halden Kitchenware"
redacted as a person by general NER.

### test4 blind run (2026-11-04)

Full pipeline (candidate 2 live, relaxed gate, test3 fixes in): redact
coverage 56/63 (88.9%), keep intact 79/83 (95.2%), 0 stray spans. Model
alone: 24/31 model-entity items (77.4%), keep 82/83, 0 strays. Nothing has
been fixed against it. The test3 fixes generalized: Technical Institute and
Rehabilitation Centre, the hyphenated policy prefix, the UK plate, the unit
before the street, the Square and Parade street types, the ISO and
dash-separated birth dates, the "+31 20 234 5678" phone and the German IBAN
all passed; so did the Lyceum (through general NER and "enrolled at") and the
Infirmary letterhead. The seven misses and four over-redactions:

- The nested-span weakness again, now for people: the role-label regex stops
  at the inner capital of "Ruaridh MacLeod" ("Ruaridh Mac", score 1.0) and
  outranks the model's full name (score 0.95), leaving "Leod" exposed. The
  address rule from the test3 fixes should apply to PERSON spans too.
- Plural pronouns fail both ways within one batch: "they"/"their" for paper
  statements are redacted because a person is within 160 characters (2 keep
  failures), while "Their forwarding address" for the tenants is missed
  because the names are further away (1 miss). Proximity is not reference.
- A product name as a person from general NER, the third time after "Skoda
  Octavia" and "Halden Kitchenware": "Nissan Qashqai" (keep failure). The
  "Vehicle:" label in front of it could decide the type, as labels do for
  numbers.
- A hub city inside a travel sentence: "via the Lyon hub" is redacted because
  the travel context is checked per sentence, not per city (keep failure).
- Formats: a French phone with a single-digit group "+33 6 12 34 56 78", a
  Dutch postcode "2011 AB", and "hospital number" missing from the ID labels
  (3 misses).
- Known gaps: an ID answered in chat without its label "4471-8820-03", and
  the first name "Thandiwe" in prose (2 misses). The model alone also missed
  "Florentyna", two of three "Kasimir", the bare name line under "CV", the
  address behind a unit number, and the two non-ISO birth dates.

## NER candidates and the custom-model gate (2026-11-04)

Two candidates were trained with `train.py` (20 epochs, seed 42, 2.5 to 3.5
minutes on CPU each) and compared with the 16-example seed model that had been
live since the first release. Candidate 1 saw the 440 round-1 documents;
candidate 2 saw those plus the 132 round-2 documents described above. Model
alone, on the gold corpus (89 model-entity items and 267 must-keep phrases over
50 documents):

| | seed model | candidate 1 | candidate 2 (promoted) |
|---|---|---|---|
| generated dev, exact-span F1 | 51.7% | 98.6% | 98.9% |
| redact coverage | 71 (79.8%) | 84 (94.4%) | 82 (92.1%) |
| keep intact | 209 (78.3%) | 258 (96.6%) | 264 (98.9%) |
| stray spans | 20 | 1 | 0 |
| ADDRESS under the right label (23) | 0 | 23 | 23 |
| held-out only (test + test2): coverage / keep / strays | 35/41, 106/132, 9 | 39/41, 129/132, 0 | 36/41, 131/132, 0 |

Candidate 1's over-redactions were all CV-style layouts: skill and language
lists as organizations, a year range and a grade as birth dates, a weekday
swallowed into a name, a car model as a person. Round 2 removed every one of
them. Candidate 2 pays with recall: it misses three held-out names candidate 1
found, all in running prose. Training was not seeded end to end for these runs
(`train.py` now calls `spacy.util.fix_random_seed`), so part of that difference
may be run-to-run noise. The generated-dev F1 is inflated either way: dev
shares its templates with train and the loss is near zero from epoch 5.

### The gate

`policy_rules.py` used to discard every custom-model span except birth dates
and keyword-matching facilities; with the seed model that was the only safe
setting. The gate now trusts the labels in `config.CUSTOM_MODEL_TRUSTED_LABELS`
(`DATE_OF_BIRTH`, `PERSON`, `ADDRESS`); `SENSITIVE_ORGANIZATION` still needs a
facility keyword, and the job-title filter now applies to custom-model PERSON
spans too. Full pipeline with the relaxed gate, all three gold sets:

| live model | coverage | keep intact | stray spans |
|---|---|---|---|
| seed model | 100% | 265/267 (two city names redacted as people) | 5 |
| candidate 1 | 100% | 266/267 (a car model redacted as a person) | 5 |
| candidate 2 | 100% | 267/267 | 5 |

The five strays are the same five as under the old gate and belong to the rule
layer, not the model: "them" and "their" far from any person, "price" caught by
the apartment pattern, and "Pallet" / "Meter" from general NER. Candidate 2 is
the only model safe behind the relaxed gate, so it was promoted to
`models/custom_ner` on 2026-11-04; the seed model is kept locally as
`models/custom_ner_seed16`. The private regression suite, the international
suite and the unit tests all pass with it.

On gold the relaxed gate changes no score, because rules and general NER already
cover every listed item there. Its purpose is real documents: on the 20
private-suite documents it newly masks text in 11 of them, 73 alphanumeric
characters under `PERSON` and 39 under `ADDRESS` that nothing else had covered,
while the suite's 70 identifiers and keep checks still pass. Those additions
have not been hand-reviewed; diff the redacted private documents before relying
on them.

### Next steps

1. Decide the test4 findings (gold section above): extend the nested-span
   rule to PERSON, let a "Vehicle:" label veto a person, add "hospital
   number" to the ID labels, the single-digit-group phone and the Dutch
   postcode. Each fix spends test4's blindness; author test5 before the next
   milestone.
2. Plural pronouns need reference, not proximity: both failure directions are
   now in gold (test4 documents 06 and 07).
3. PERSON recall in running prose and on bare name lines, measured three
   times (candidate 2 against 1, test3, test4): more prose genres or a second
   seed.
4. The remaining rule-layer strays: "them"/"their" far from a person and
   "Pallet"/"Meter" from general NER on dev; "Halden Kitchenware" on test3.
   An ID answered in chat without its label stays a known gap.

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
rules and general NER already cover the gold items, and `policy_rules.py` admits
custom-model spans only for the labels in `config.CUSTOM_MODEL_TRUSTED_LABELS`,
so a weak model hides behind the rules there. `evaluate_gold.py` also reports
stray spans (detections touching no gold item), the over-redaction signal the
keep score alone cannot see.

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
