# Labeling guide — V1 public-sharing policy

This guide is the single reference for labeling documents, whether the labels
come from a human or from LLM pre-labeling. It encodes the V1 policy (red.pdf)
plus every decision recorded in `build_gold_corpus.py`. If a case is not
covered here, flag it for review instead of guessing — and once decided, add
the decision to this file.

## The split: what the model learns vs. what rules own

The detector is layered (see `policy_rules.py`). **Only label entities from
the MODEL column for NER training.** Everything in the RULES column is handled
by deterministic patterns — labeling it for training would waste examples and
teach the model to compete with regex it can never beat.

| MODEL learns (label these) | RULES own (do not label for training) |
|---|---|
| `PERSON` — names in any layout | Phones, e-mails, IP addresses |
| `SENSITIVE_ORGANIZATION` — clinics, hospitals, schools, shelters | All formatted IDs: SSN, SNILS, INN, passports, cards, accounts, IBAN, VIN, plates, policy/student/customer/badge/case IDs |
| `ADDRESS` — street + apartment + city + postal code | Pronouns (closed word list) |
| `DATE_OF_BIRTH` — the date value near a birth label | Labelled IDs of any country (label-before-number rule) |
|  | Times with persons, class/grade designators |

## Annotation format

One JSON object per line (JSONL):

```json
{"source_file":"doc_01.txt","text":"...full document...","spans":[{"text":"Jordan Lee","label":"PERSON"}]}
```

- Every span's `text` must appear **exactly once** in the document
  (`train.py` refuses ambiguous spans). If a name repeats, the detector's
  name-echo rule covers repeats — label the first occurrence only, and make
  sure the span string is unique (extend it with surrounding words if needed).
- Do NOT add spans for text that stays visible. Negative examples (documents
  with no spans at all) are valuable — keep the seed ratio of roughly 1 in 5.

## Labeling criteria per entity

### PERSON
- Full names, single names in greetings, names after role labels
  ("Patient: X"), chat speakers, quoted e-mail authors.
- Job titles are NOT persons ("Senior Dispatcher", "Ward Nurse" stay).
- Company names are NOT persons even in "Defendant: Stroyservice LLC".

### SENSITIVE_ORGANIZATION
- Facilities whose mention alone reveals something about a person: clinics,
  hospitals, polyclinics, hospices, shelters, schools, colleges, universities,
  medical/support/health/care centers.
- Always redacted, whether or not a person is named next to them.
- Ordinary employers are NOT sensitive organizations. (Employer redaction is
  a context rule, not a model entity, in V1.)

### ADDRESS
- Label the full home/work address on one line: street, apartment, city,
  postal code ("17 Maple Crescent, Apt 3B, Springfield, IL 62704").
- Standalone city names stay visible ("the Leeds ring road", "arrival
  Lisbon", "the Katowice warehouse", "our Frankfurt branch").
- A city inside an address or a residence sentence is part of the address.

### DATE_OF_BIRTH
- Only the date value, only when a birth label is nearby: "born", "DOB",
  "date of birth". Numeric and worded forms both count ("5 May 1994").
- Issue dates, appointment dates, contract dates, hearing dates stay visible.

## Keep rules (must NOT be labeled, and must survive redaction)

- Ordinary words: "night", "morning", "evening" — except the day-part word in
  a person+clock-time phrase ("at 2 o'clock in the morning" → redacted).
- Generic job titles, laws and article numbers, document headings.
- Prices, amounts, quantities, meter readings, percentages.
- Thing-IDs: order, invoice, tracking, serial, batch, flight, voucher,
  project codes, device serials.
- Public places and unaffiliated organizations.
- Languages and nationality adjectives ("Dutch", "Indian passport") — group
  membership is a TBD policy area; V1 keeps them.
- I/me/my/you/your (pronoun policy covers only third-person forms).

## V1 policy decisions already taken (do not re-litigate per document)

1. ALL e-mail addresses redact, role inboxes (hr@, procurement@) included.
2. Company tax IDs (INN, GSTIN) redact even with no person attached.
3. he/him/his/she/her/hers always redact; they/them/their/theirs only when a
   person is detected nearby; I/me/you forms never.
4. A bare digit run is never PII; a label next to a number decides its type.
5. Travel destinations are standalone cities (keep); residences redact.

## Workflow for scaling the corpus (LLM pre-labeling)

1. Generate or collect raw documents (see the generation prompt in the
   project conversation, or write your own following `build_gold_corpus.py`).
2. Have the LLM propose spans using THIS guide as the prompt's rulebook.
   Restrict it to the four MODEL entities.
3. Review a random sample of at least 100 proposed spans by hand; record the
   agreement rate before trusting the rest.
4. Fix or discard documents with disagreements; every disagreement worth
   keeping becomes a regression case.
5. Keep the held-out gold test sets human-only — never let the LLM label or
   even see them.

## Evaluation

- `python training/evaluate_gold.py --set both` — coverage-based scoring
  against the human-gold corpus. Tune on dev; run test at milestones only.
- `python training/validate_regression_cases.py` — exact-identifier suite.
- `python training/train.py --output models/custom_ner_candidate` — never
  train directly into `models/custom_ner`; the detector auto-loads that path.
  Promote a candidate only after `evaluate.py` and the gold suites pass.
