# Local custom-entity training

**Start with [LABELING_GUIDE.md](LABELING_GUIDE.md)** — it defines which
entities the model learns versus which the rule layer (`policy_rules.py`)
owns, the labeling criteria, and the LLM pre-labeling workflow. Label only the
model's entities; rules already beat the model on everything formatted.

### Model location

The detector loads the first of: the `CUSTOM_NER_MODEL` environment variable,
`~/pii-models/custom_ner` when that directory exists, and `models/custom_ner`
inside the repository. `train.py` writes to `custom_ner_candidate` next to the
live model (`config.MODELS_DIR`), and the evaluators default to the live model.
Keep the model directories outside any synced folder: they are about 425 MB
each, and iCloud Drive evicts such files to free disk space and then stalls
every load for minutes, which cost this project hours in November 2026. The
live model (candidate 2) and the seed backup moved to `~/pii-models/` on
2026-11-07; the in-repository `models/` location remains supported and
gitignored. Promote a candidate only after `evaluate.py` and the gold suites
pass.

The move itself lost two of the 44 model files to iCloud: every download of
`custom_ner/vocab/strings.json` and `custom_ner_seed16/ner/moves` was
cancelled by the sync daemon while the rest of the folder downloaded on
demand. Both were rebuilt by replaying the vocabulary steps of `train.py` on
`en_core_web_lg` (add the sorted labels, run `make_doc` over the training
texts): spaCy writes `strings.json` sorted and `ner/moves` as the base
model's label table plus the added labels, so both files are deterministic
given the training data. The replay reproduced the seed model's
`strings.json` and the live model's `ner/moves` byte for byte, and the two
rebuilt files have the original sizes (9,720,804 and 1,252 bytes). With them
in place, the six gold sets (full pipeline and model alone), both regression
suites and the 18 unit tests reproduce every recorded number from
`~/pii-models`, and the seed model alone reproduces its 71/89, 209/267, 20
strays; the iCloud copies under `models/` were then removed.

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

`build_gold_corpus.py` authors 89 fully synthetic labeled documents (25 dev /
15 test / 10 test2 / 13 test3 / 13 test4 / 13 test5) under `training/data/gold/`, covering the V1 public-sharing
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
change), **test4** (13 docs, authored 2026-11-04 after the test3 fixes), and
**test5** (13 docs, authored 2026-11-05 after the test4 fixes). `--set all`
runs all six. Current state: dev, test and test2 at 100% coverage / 100% keep;
test3 at 98.7% / 99.0%, test4 at 96.8% / 97.6% and test5 at 98.6% / 97.8%
after the fixes recorded below. Every set has absorbed fixes, so the next
measurement needs a fresh blind batch (test6).

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

### test4 fixes (2026-11-05)

Five of the eleven findings were fixed; test4 is semi-dev from here. In
`text_pii_detector.py` the nested-span rule now covers two entity families
(`NESTED_SPAN_FAMILIES`): a PERSON candidate nested inside a longer PERSON
candidate yields to it, exactly as address candidates already did, so the
role-label span "Ruaridh Mac" no longer outranks the model's "Ruaridh MacLeod".
The name regexes (`labelled_person`, `chat_speaker`, `quoted_author`,
`greeting_name`) also share a `_NAME_WORD` fragment that allows one inner
capital ("MacLeod", "McDonald"), so the rule finds the full name itself when no
model span is there. In `policy_rules.py`: a thing label directly before a span
("Vehicle:", "Make/Model:", "Product:"; `THING_LABEL_PREFIX`) vetoes a PERSON
from the NER layers, the way an ID label decides a number's type; "hospital
number" / "hospital id" joins the `generic_id` labels (the bare word "hospital"
does not, so "City Hospital No. 3" stays with the facility pattern);
`phone_intl` accepts a single-digit group straight after the country code
("+33 6 12 34 56 78"); and a Dutch postcode ("2011 AB") is an address
continuation segment. Two unit tests cover the dedup rule and the four rule
fixes.

Result: test4 56/63 → 60/63 (95.2%), keep 79/83 → 80/83, 0 strays; dev, test,
test2 and test3 unchanged (dev strays still 4); both regression suites and the
17 unit tests pass. The six remaining test4 failures are the ones no rule fixes
cheaply: plural pronouns both ways (they/their near a person in document 06,
"Their forwarding address" far from the tenants in 07), "via the Lyon hub"
under the sentence-level travel context, the unlabeled chat-reply ID
"4471-8820-03" (guide decision 9), and the first name "Thandiwe" in prose.

### test5 blind run (2026-11-05)

Thirteen documents, 72 must-redact items and 89 must-keep phrases, written
without running the detector. They probe names with lowercase particles,
accents and inner capitals; thing labels other than vehicles; hub and branch
cities inside travel sentences; Canadian and German postcodes; UK and German
phone forms; staff, payroll and hull numbers; facilities without a keyword
(Nursery, Surgery, Language Centre); plural pronouns in both directions; and
the chat-reply ID a third time.

Full pipeline (candidate 2 live, relaxed gate, test4 fixes in): redact
coverage 54/72 (75.0%), keep intact 82/89 (92.1%), 0 stray spans. Model
alone: 22/30 model-entity items (73.3%; PERSON 15/23, ADDRESS 2/2, DOB 2/2,
SENSITIVE_ORG 3/3), keep 84/89, 0 strays. Nothing has been fixed against it.
The test4 fixes held: O'Connell, McCarthy and Oyelaran-Smith all passed
through the name regexes (the model stopped at "Bartholomew Oyelaran" and the
nested-span rule made that partial yield to the full regex span), the
accented Zsófia Németh and Tadhg Ó Briain passed through the model, bare
"Tadhg" and "Adaeze" and the particle name "Pieter de Vries" came from general
NER, they/their for two named people were redacted, both letterheads and the
appliance names stayed visible. The eighteen misses and seven over-redactions:

- The model stops at a lowercase particle or an accent and the trusted PERSON
  label then redacts the fragment and exposes the rest: "Fenna van" of "Fenna
  van der Berg", "Németh" of "Gábor Németh" (2 misses). Model recall also
  fails on the greeting "Oisín," (the greeting regex's [a-z] excludes í) and
  the prose first name "Hannelore" (2 misses).
- The model finds the facilities but the keyword gate drops them: "Lindenhof
  Surgery" and "Riverside Speech and Language Centre" (2 misses). The model
  also labels the German address "Hauptstrasse 12, 80331" as a facility, so
  the gate drops it too, and the regex knows neither the street-then-number
  order nor the postcode-before-city form (2 misses).
- A Canadian postcode: the address extension reads ", ON M5V" as a city name
  and strands "2T6" (1 miss).
- Labels the rules do not know: "Staff No.", "payroll number", "hull ID", and
  "birthday" — the model found "14 Feb 1979" as a birth date, but
  `DOB_CONTEXT` has no "birthday" (4 misses).
- Formats: UK national phones "020 7946 0958" and "07700 900123", the German
  trunk prefix "+49 (0)30 901820", the sort code "40-11-60" (4 misses).
- The chat-reply ID "88-1234-556" with its label in the previous line, a third
  time (1 miss).
- Plural pronouns for things near a person, "they" in documents 01 and 09 (2
  keep failures); the hub and branch cities "Oslo" and "Stavanger" in a travel
  sentence, as "Lyon" before (2); "Yanmar 3YM30" as a PERSON from the custom
  model itself, behind an "Engine:" label the thing-label veto does not list
  (1); the city "Hull" as a PERSON from general NER (1); "The Bosch Serie 4"
  as an employer because "works best" matches the employment context (1).

### test5 fixes (2026-11-05)

Twenty of the twenty-five findings were fixed, all in `policy_rules.py`;
test5 is semi-dev from here. Labels: "birthday" joins the birth-date labels
and `DOB_CONTEXT`, month names may be abbreviated ("14 Feb 1979"); "staff" and
"payroll" join "hospital" as ID labels that need a number word; "hull ID" is a
serial number; a labelled sort code is a bank detail. Facility words: Infirmary,
Sanatorium, GP/dental/medical practices, Surgery and Nursery (both need a name
word in front, so a sentence-initial "Surgery went well" is not a facility),
and Language/Speech/Therapy/Treatment/Dialysis/Cancer/Diagnostic/Crisis/Day
Centre; the capitalised run before a facility word may carry "and"/"of"/"for"
("Riverside Speech and Language Centre"). Formats: a UK national phone
(`phone_uk`, leading 0 plus exactly ten digits in two or three groups, the
count fixed by a lookahead so IBAN fragments never qualify); the "(0)" trunk
prefix after a country code; the Canadian province-and-postcode and the German
postcode-before-city as address continuation segments; a German-order street
line (`address_line_de`: "Hauptstrasse 12", "Berliner Allee 3"). Names: the
name-word fragment takes Latin-1 letters ("Németh", "Oisín"), lowercase
particles may sit between name words ("Fenna van der Berg", "Pieter de Vries"),
and "next of kin", "passenger", "pupil", "proposer", "guardian", "spouse",
"beneficiary" and "driver" are role labels. Context: `OPERATIONAL_SITE_SUFFIX`
("hub", "branch", "depot", "warehouse", "office", ...) after a city keeps it
visible inside a travel or residence sentence and vetoes a PERSON reading
("the Hull depot"); "Engine:", "Motor:", "Appliance:", "Instrument:" join the
thing labels; "works best/well/on/with/..." no longer counts as employment.
Chat: `chat_reply_ids` implements labeling-guide decision 9 — in a `[hh:mm]
Speaker:` transcript, a person-ID label in the previous speaker's line makes
the hyphenated or prefixed digit runs of the reply that person's identifier;
the UK phone keeps the default score so "07700 900123" beats the nested
"900123" on the tie.

Two things were tried and reverted during this round. Adding
SENSITIVE_ORGANIZATION to the nested-span families made the regex span yield
to the model's wider "Greystone Academy in September" (test3) and broke a
private-suite check ("Children's City Polyclinic No. 8" swallowed into a
longer model span); the "and/of" prefix fixes "Riverside Speech and Language
Centre" without it. And `phone_uk` at 0.99 lost to the chat-reply code
"900123" at 1.0.

Result: test5 54/72 → 71/72 (98.6%), keep 82/89 → 87/89 (97.8%), 0 strays.
Side effects on the older sets, all improvements: test3 74/76 → 75/76 (the
chat-reply ID "CU-104772"), test4 60/63 → 61/63 (the chat-reply ID
"4471-8820-03") and keep 80/83 → 81/83 ("via the Lyon hub"); dev, test and
test2 unchanged (dev strays still 4); both regression suites and the 18 unit
tests pass. The five remaining failures across test3–test5 are the two
structural ones: plural pronouns for things near a person ("they" in test5
documents 01 and 09, "they"/"their" in test4 document 06) or for people far
from their names ("Their forwarding address", test4 document 07), and first
names in prose the model does not know ("Hannelore", "Thandiwe", "Nkechi").
Plus "Halden Kitchenware" as a person from general NER on test3.

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
`models/custom_ner` on 2026-11-04 (since 2026-11-07 both live under
`~/pii-models/`, see "Model location"); the seed model is kept locally as
`custom_ner_seed16`. The private regression suite, the international
suite and the unit tests all pass with it.

On gold the relaxed gate changes no score, because rules and general NER already
cover every listed item there. Its purpose is real documents: on the 20
private-suite documents it newly masks text in 11 of them, 73 alphanumeric
characters under `PERSON` and 39 under `ADDRESS` that nothing else had covered,
while the suite's 70 identifiers and keep checks still pass. Those additions
have not been hand-reviewed; diff the redacted private documents before relying
on them.

### Next steps

1. Author a blind gold batch test6 before the next milestone: every existing
   set has absorbed fixes (test5 on 2026-11-05). Worth probing there: the
   new rules on fresh surface forms (other particles and diacritics, other
   national phone and postcode formats, other site nouns and thing labels,
   chat transcripts with several codes), and whatever test6's author thinks
   the rule layer has never seen.
2. Corpus round 3 for the model: names with lowercase particles, accents and
   hyphens (the model gave "Fenna van", "Németh", "Bartholomew Oyelaran" on
   test5, and the rules carried those), German-order addresses, facilities
   it mislabels, and more first names in running prose.
3. Plural pronouns need reference, not proximity: both failure directions are
   in gold (test4 documents 06 and 07, test5 documents 01 and 09) and they
   are now the largest remaining error class.
4. The remaining rule-layer over-redactions: "them"/"their" far from a person
   and "Pallet"/"Meter" from general NER on dev (strays); "Halden Kitchenware"
   on test3 (keep failure).

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
# Candidates land next to the live model (default: ~/pii-models/custom_ner_candidate)
python training/train.py \
  --train training/data/private/train.jsonl \
  --dev training/data/private/dev.jsonl

python training/evaluate.py \
  --model ~/pii-models/custom_ner_candidate \
  --data training/data/private/dev.jsonl

python training/evaluate_model_gold.py \
  --model ~/pii-models/custom_ner_candidate --set all

python training/validate_regression_cases.py
python training/evaluate_gold.py --set all

# Promote only after every check above passes:
# mv ~/pii-models/custom_ner ~/pii-models/custom_ner_previous
# mv ~/pii-models/custom_ner_candidate ~/pii-models/custom_ner
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

The detector automatically loads the live model (see "Model location" above: `$CUSTOM_NER_MODEL`, then `~/pii-models/custom_ner`, then `models/custom_ner`) when it exists. Keep it only when its held-out evaluation and manual review meet your privacy policy.

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
