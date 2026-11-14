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
live model and the seed backup moved to `~/pii-models/` on 2026-11-07; the
in-repository `models/` location remains supported and gitignored. Since
2026-11-13 the live model is candidate 3 and candidate 2 sits beside it as
`custom_ner_candidate2`; to go back, swap the two directory names. Promote a
candidate only after `evaluate.py` and the gold suites pass.

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
labeled with the four model entities only, train and dev drawn from disjoint
name/street/city/facility pools, one document in five without entities. In
rounds 1 and 2 every span occurs exactly once in its document; round 3 spans
carry character offsets instead, so a name may return. The corpus is 1,900
train / 190 dev documents after round 4. Round 1 (seed 20260831) is 400 train /
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

Round 3 (2026-11-13, seed 20261113, appended from `corpus_round3.py` so rounds
1 and 2 stay byte-identical) adds 900 train / 90 dev documents in 30 genres
with entities and 7 without. It follows the brief the blind runs of test5 to
test7 left: hyphenated surnames, particles, accents, ALL-CAPS form values,
initials, titled names, bare first names in greetings, sign-offs and prose, a
first name or surname that returns after the full name, chat speakers and
"Surname, First" rows; UK addresses with a flat or a house name and a
postcode, streets with no street word, and the German, Spanish, Italian,
Portuguese, Polish, Dutch and Irish orders; birth dates in ISO, dashed,
abbreviated and block-capital forms under English and foreign labels; and the
facility kinds of the labeling guide. Its negatives are what candidate 2
mislabeled: letterheads, companies under role labels, employers, products with
a digit, quantities, ordinary dates in every birth-date format and year, a
year in front of a title, cities with a site word, role and department
addressees, empty form fields, sentences and field values that open with an
ordinary capitalised word, named public places, bank codes, and a nationality
in front of an identity document. Names come from pools of about 200 first
names and 210 surnames across many languages, against 45 and 40 in round 1.
Every pool entry was checked against the gold documents and dropped when it
shared a word with one, so the gold sets keep measuring the pattern and not
the word. The pools of the official-record genres were checked against the
private regression documents in the same way. The general first-name pool
was not, and two common given names in it also occur there; the form labels
of those genres ("Place of birth:", "Registry Office of") are the generic
wording of such records.

Round 3 deliberately leaves out what only the test8 blind run showed (French
documents, table cells, products under a bare "Name:", column gaps, address
blocks with the town on its own line, surnames that are company words):
test8 was the held-out measurement for the candidate trained on this corpus.
Those belong to round 4, after a test8 fix round. Round 3 has not been
through an adjudication round either.

Round 4 (2026-11-13, seed 202611134, appended from `corpus_round4.py` so
rounds 1 to 3 stay byte-identical) adds 480 train / 48 dev documents in 16
genres with entities and 6 without. It is written from the test8 brief
above: French attestations, letters, e-mails and school certificates
("Mme", "M.", "Mlle", "né le", French months, "Fait à <town>, le"); names,
hosts and plates in pipe, tab and semicolon tables; a product and a person
under the same bare "Name:"; two fields on one line with a column gap;
address blocks with the house, the street, the town and the postcode each on
its own line (every line is its own `ADDRESS` span, as decision 21 of the
labeling guide has it); surnames that are company or brand words, behind a
title and alone; a message board with greeting and closing words of every
kind, one message signed by the company and one addressed to a role;
streets with a function word ("48 Ivybridge by Beechcroft Crescent"); house
names with no house word; block-capital values that are not names ("TO BE
CONFIRMED"); support transcripts with the speaker's name in brackets; and
more facility mentions in running prose. Its negatives are French notices,
stock tables and parts forms whose "Name" holds a product, boards with no
person on them, block-capital forms with no names, and bay lists with
plate-shaped codes. The new pools (French names, towns and streets,
company-word surnames, bare house names) passed the same freshness check as
round 3, this time against test9 too.

Round 4 deliberately leaves out what only the test9 blind run showed (a chat
export with lowercase names, a JSON record, German and Spanish forms, tables
without separators, venue-word surnames, serial columns): test9 is the
held-out measurement for candidate 4. Round 4 has not been through an
adjudication round.

## Human-gold evaluation corpus

`build_gold_corpus.py` authors 141 fully synthetic labeled documents (25 dev /
15 test / 10 test2 / 13 each for test3 to test9) under `training/data/gold/`, covering the V1 public-sharing
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
change), **test4** (13 docs, authored 2026-11-04 after the test3 fixes),
**test5** (13 docs, authored 2026-11-05 after the test4 fixes), **test6** (13
docs, authored 2026-11-07 after the test5 fixes), **test7** (13 docs,
authored 2026-11-08 after the test6 fixes), **test8** (13 docs, authored
2026-11-09 after the test7 fixes) and **test9** (13 docs, authored
2026-11-13 after the test8 fixes, with candidate 3 live). `--set all` runs
all ten.
Current state, with candidate 3 live since 2026-11-13: dev, test and test2
at 100% coverage / 100% keep; test3 at 100% / 99.0%, test4 at 96.8% / 97.6%,
test5 at 98.6% / 97.8%, test6 at 99.1% / 97.6% and test7 at 100% / 98.7%.
Test7's blind score was 80.1% / 87.9%. Test8 scored 74.8% / 93.7% blind with
candidate 2 and 80.6% / 93.3% with candidate 3 before any rule was fixed
against it; after its fix round (2026-11-13) it is at 100% / 98.7% and test3
at 100% / 100%. Test9 scored 81.0% / 90.7% blind (2026-11-13) and nothing
has been fixed against it. Candidate 4 was measured on it once (2026-11-14)
and was not promoted, so candidate 3 stays live. Every other set is
semi-dev, and test9 becomes one with its fix round.

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

### test6 blind run (2026-11-07)

Thirteen documents, 111 must-redact items and 164 must-keep phrases, written
without running the detector, after the test5 fixes and the move of the
models to `~/pii-models`. They probe more name particles ('t, ap, Ní, bin
with Al-) and Icelandic letters; ALL-CAPS form fields; initials and the Mx
title; a chat transcript without brackets round the times; ID labels the
rules may not know (NHS, PCN, DNI, driving licence, hire agreement, visa
application reference); dotted and ordinal birth dates; the 00 international
prefix and Spanish, German and Norwegian phone shapes; a German plate; a
Spanish-order address and a Spanish school; facilities without a listed word
(Care Home, Health Trust, Colegio); BIC and "card ending" as keeps; plural
pronouns in both directions including singular "they" for a named person; a
travel city that also names an institute; and a no-PII site bulletin.

Full pipeline (candidate 2 live, relaxed gate, test5 fixes in): redact
coverage 84/111 (75.7%), keep intact 158/164 (96.3%), 0 stray spans. Model
alone: 29/53 model-entity items (54.7%; PERSON 13/31, ADDRESS 4/6, DOB 4/6,
SENSITIVE_ORG 8/10), keep 143/164, 1 stray. Nothing has been fixed against
it. What held: every ALL-CAPS field except the names (address, city,
postcode, e-mail, phone, employer, passport, birth date); the dotted
"03.11.1941" and the ordinal "3rd of March 1990"; the initials "A. Quaye" and
"P. Nwosu", "Mx Taylor Reid-Okafor", "Caoimhe Ní Bhriain", "Þórður
Hallgrímsson", "Anne-Sophie de la Fontaine", "Joaquín de la Fuente", "María
José García Lopez" and the hyphenated greeting "Dear Ms Castellanos-Reyes"
(the rules carried ten names the model missed); "Colegio San Esteban"; the
travel city "Trondheim" beside the institute of the same name; both IBANs,
the sort code, the UK plate, the display-name e-mail headers, the "4M"
class; and every keep that is not a pronoun: the letterheads, the thing
labels ("Dyson V11", "Volkswagen Golf", "Peugeot 2008"), the five site
nouns, the BIC, "card ending 4421", "NHS number (ending 5919)", the
unbracketed chat timestamps. The twenty-seven misses and six over-redactions:

- The model finds the facilities but the keyword gate drops them:
  "Willowbrook Care Home" and "Kingsmere Health Trust" (2 misses, plus the
  "Kingsmere" inside the Trust).
- UK postcodes stranded after a town the address extension reached but the
  postcode segment did not: "Kingsmere KM2 7QL", "Caernarfon LL55 1AB",
  "Westbrook WB3 8HD" (3), while "Chelmsford CM1 2QX" and "DERBY DE1 3AB"
  passed.
- Names: the particles "ap" and "'t" ("Rhys ap Gruffudd", "Jan-Willem 't
  Hooft"), a four-token name mixing a particle and a hyphenated surname
  ("Khalid bin Tariq Al-Farsi"), the ALL-CAPS "ONYEKWERE" and "CHIDI EMEKA"
  (model and regexes both want Title Case), the prose first names "Ifeoma"
  and "Bartek", and the second, bare "Sigrún" that the echo rule did not
  carry (8).
- ID labels the rules do not know: DNI, PCN number, driving licence, visa
  application reference, hire agreement (5).
- Formats: the nine-digit Spanish mobile "+34 612 345 678", the seven-digit
  German subscriber block "+49 171 2345678", the 00 prefix "0047 55 12 34
  56", the German plate "M-RK 4471", the Spanish-order address "Calle Mayor
  12, 3ºB" with its postcode "28013" (6).
- The chat-reply ID "7719-2034-88" when the transcript has no brackets round
  the times (1).
- Plural pronouns: "they" for two named applicants three lines below their
  names was kept (1 miss); "they" for the staff and for the keys, "their" for
  the Centre's families and "them" for the parcels were redacted (4 keep
  failures). Singular "they" for Temitope passed, by proximity.
- "Jungheinrich EFG 216" as a PERSON from general NER, as "Halden
  Kitchenware" before (1); "2027 Halvorsen Scholarship" as an ADDRESS from
  the custom model, the first trusted-label leak since the gate was relaxed
  (1).

Model alone, the picture behind the rules: PERSON 13/31. It missed every
hyphenated surname (Adebowale-Hart, Reid-Okafor, Rousseau-Mbeki,
Castellanos-Reyes), every particle name but "Caoimhe Ní Bhriain", both
ALL-CAPS names and all five bare first names in prose and sign-offs
("Ifeoma", "Bartek", "Temitope", the sign-off "Anjali", the second
"Sigrún"); it missed the ISO birth date and the ordinal one and two of the
six addresses; and it labelled five letterheads, an employer line and two
forklift models as facilities, "Tue 17 Nov at 09:40" and "January" (twice)
as birth dates, and a scholarship name as an address. Corpus round 3 has
its brief.

### test6 fixes (2026-11-07)

Twenty-seven of the thirty-three findings were fixed, all in
`policy_rules.py`; test6 is semi-dev from here. Facility words: Care /
Nursing / Residential Home and Health / NHS / Foundation / Hospital Trust
join the list, so the two facilities the model had found now pass the gate.
Addresses: the town segment of the address extension may carry its UK
postcode ("Kingsmere KM2 7QL" instead of the stranded "Kingsmere KM2"), and
any address span, whichever layer found it, that stops after the outward half
of a UK postcode takes the inward half ("8 Bryn Teg, Caernarfon LL55" → "...
LL55 1AB"); a Romance-order street line (`address_line_es`: "Calle Mayor 12,
3ºB", then the ", 28013 Madrid" segment). Names: the particles "ap", "ab",
"'t", "ibn", "bint", "abu", "af", "av" and "zu"; a courtesy title ahead of a
labelled name no longer uses up a name slot ("Mr Khalid bin Tariq Al-Farsi");
a role label may carry a number, a bracket or an "if" clause before its colon
("Applicant 1:", "Emergency contact if the parent cannot be reached:"), and
"hirer", "lessee", "borrower", "cardholder", "keeper", "carer",
"parent/carer" and "referred by" are role labels, but a label that is the
second item of a comma list is not ("Employer, Applicant 1: Fenland Water
Services" stays an employer line); `name_field` reads the ALL-CAPS or Title
Case value after a line-initial name label ("SURNAME: ONYEKWERE", "GIVEN
NAMES: CHIDI EMEKA", "Full name: ..."; the bare "Name:" is left out on
purpose); `greeting_person` takes the bare first name a greeting or reminder
addresses ("Hi Ifeoma,", "Reminder for Bartek:", "Thanks Rhys.") with a
stoplist of collective addressees; the echo rule also lends the first name of
a detected full name to its later bare occurrences ("Sigrún"), case-sensitive,
four letters or more, never a word that opens sentences in its own right
("Mark", "Will", "Grace"). ID labels: DNI / NIE / NIF / CURP / CPF / BSN /
PESEL label a number on their own; "application", "visa" and "permit" need a
reference word; `pcn_number` ("PCN number: EV 44021873", spaced prefix) and
`agreement_number` ("Agreement: 55-20931"; the value needs a separator or six
characters, and an employment contract number stays visible); a labelled
driving licence falls back to a regex at the passport fallback's score,
because Presidio's UK recognizer scores a valid number 0.5 with or without
its label. Formats: `phone_intl` takes the 00 dialling prefix (only when no
digit group precedes it, so the "0044 0532 0130 00" inside an IBAN is still
not a phone) and subscriber blocks of up to eight digits ("+49 171 2345678");
a Presidio identifier strictly inside a "+"/"00" phone yields to it (the
Spanish mobile had lost to a checksum-valid "Australian company number"
nested in it); `license_plate_de` ("M-RK 4471", a year-shaped block excluded
so "EU-UK 2026" never qualifies). Chat: the time may be unbracketed in both
`chat_speaker` and `chat_reply_ids`. Over-redactions: a general-NER PERSON
with a digit in it is a product code ("Jungheinrich EFG 216"), unless the
digit sits in a leading same-line token ("14:03 Rhys" → "Rhys"); a
custom-model ADDRESS that is a year followed by words with no street word
("2027 Halvorsen Scholarship") is dropped.

One thing was adjusted during the round: the first version of the digit
trim stripped across line breaks and turned spaCy's cross-line
"QRT-55302 / Guest" (test2) into a single-line "Guest" stray; the trim now
stays on one line and the span dies in dedup as before.

Result: test6 84/111 → 110/111 (99.1%), keep 158/164 → 160/164 (97.6%), 0
strays. Every other set is unchanged: dev 123/123 and 135/135 (strays still
4), test and test2 100% / 100%, test3 75/76, test4 61/63 and 81/83, test5
71/72 and 87/89; both regression suites and the 19 unit tests pass. The six
failures left on test6 and the eight on test3–test5 are the two structural
classes: plural pronouns ("they" for the staff and for the keys, "their"
for the Centre's families and "them" for the parcels redacted near a person;
"they" for the two applicants three lines below their names kept) and bare
first names in prose with no full name in the document ("Nkechi",
"Thandiwe", "Hannelore"), plus "Halden Kitchenware" from general NER, which
carries no digit for the product-code rule to see.

### test7 blind run (2026-11-08)

Thirteen documents, 141 must-redact items and 223 must-keep phrases, written
without running the detector, after the test6 fixes (commit 6c3f889). The
batch aims at the assumptions those fixes made more than at new formats:
greeting addressees that are not names ("Dear Householder,", "Hi Finance,",
"Dear Committee,", "Dear Visa Officer,") beside greetings the rule does not
list ("Good morning Katarzyna,", "Hello Tevita and Mele,"); titled names
under role labels (Dr., Prof., Rev., Sra., Herr, Dott.ssa, Eng.ª) and a
company under one ("Registered keeper: Lexford Leasing Ltd"); 00-prefixed
numbers that are not phones (the Polish postcode "00-590", an SSCC, an
invoice number, a batch); ALL-CAPS fields other than names, with "MIDDLE
NAME: NONE"; Romance-order addresses without a floor (Spanish, Italian,
Portuguese) and "Via Royal Mail 48"; hyphenated codes shaped like German
plates ("USB-C 3.2", "AC-DC 12 V", "DIN-A 4", "DE-HE 3", bay "B-C 4") beside
two real plates; a date range after "hire agreement"; and the first-name
echo against a city of the same name ("the Florence warehouse"), a
stop-listed first name ("Faith") and a three-letter one ("Ana"). New ground:
a chat export with bracketed date and time, Vietnamese and Polish names, a
two-letter surname, a house-name address, an Eircode, an Italian fiscal
code, plate and chassis number, a spaced National Insurance number, a
slash-separated claim number, "half past six in the evening", and plural
pronouns in both directions.

Full pipeline (candidate 2 live, relaxed gate, test6 fixes in): redact
coverage 113/141 (80.1%), keep intact 196/223 (87.9%), 1 stray span. Model
alone: 31/63 model-entity items (49.2%; PERSON 16/41, ADDRESS 5/9, DOB 6/7,
SENSITIVE_ORG 4/6), keep 195/223, 1 stray. Nothing has been fixed against
it; every other set, both regression suites and the 19 unit tests are
unchanged. Recall is a little above the test6 blind run (75.7%). Keep is the
lowest of any batch (test6 blind: 96.3%), which is what this batch was built
to find: fifteen of the twenty-seven keep failures are test6 rules reaching
too far.

What held: the titled names under role labels except one ("Dr. Ngozi
Chukwuemeka-Bell", "Prof. Alasdair MacKinnon", "Sra. Inmaculada Ortega Ruiz"
and "Mrs Gwenllian Llewellyn-Hughes" by rule, "Rev. Tobenna Achike", "Herr
Jürgen Weißmüller" and "Eng.ª Leonor Vasconcelos" by NER); "Dear Visa
Officer," kept, the unlisted greetings carried by NER ("Good morning
Katarzyna,", "Hello Tevita and Mele,") and "Hi Ana," by rule; every ALL-CAPS
field
that is not a name except "NONE" (title, sex, marital status, occupation,
country of birth, allergies), with the caps full name, birth date, address,
town, postcode, phones and "DR M. FAIRWEATHER"; the Spanish and Italian
addresses whole ("Avenida de Andalucía 8, 41005 Sevilla", "Via Garibaldi 27,
20121 Milano") and the Portuguese street; both real German plates, the
E-suffix one included, and the UK plate; the unspaced SSCC, "DN-118204",
"INV-20417" and "2026/0931" kept, "TPS/0044/71920" as one member ID and
"0048 22 555 01 47" as a phone; every speaker of the dated chat export
("Mamadou Sy" included), its five dates kept, "Thanks Eilidh." and the bare
"Roisin"; the Vietnamese names, "Zdeněk Novotný", "Agnieszka Wróblewska",
"Maureen O'Dowd-Farrell" and every hyphenated surname; the slash claim
number, the driving licence, the spaced National Insurance number, hospital
and NHS numbers, both passports, the chassis number under "Telaio / VIN",
the IBAN, the order, policy and serial numbers; six of seven birth dates;
all sixteen pronouns that refer to people, the plural ones for the
witnesses, the couple and the two visitors included; "Alnmouth Community
Hospital" and "Hadston Day Centre", with the letterhead "Wansbeck Valley NHS
Foundation Trust", "nursing home", "sheltered housing" and "Boots" left
alone; and the sites and things ("Sheffield plant", "Kassel depot", "Leipzig
hub", "Mill Lane", "Seat Ibiza", "Fiat Panda 1.2", "Gardena Classic 30",
"Visa ending 3310", the Royal Mail tracking number, "Driving licence:
category B, held since 2010").

The twenty-seven keep failures and the stray:

- Greeting rule: "Householder", "Finance" and "Committee" as persons (3).
- German-plate rule: "USB-C 3.2", "AC-DC 12 V", "DIN-A 4", "DE-HE 3", "DE-SN
  2" and "B-C 4" as plates (6).
- The 00 prefix: the invoice number "0042 0117 3381" and the batch
  "0031-22-456" as phones (2); the spaced SSCC as a payment card from its
  first sixteen digits (1).
- Agreement rule: the first date of "hire agreement 31/10/2026 to
  03/11/2026" as an ID (1). Name-field rule: "MIDDLE NAME: NONE" (1).
  Romance address rule: "Via Royal Mail 48" (1). First-name echo: "Florence"
  in "the Florence warehouse" (1). Role label: "Registered keeper: Lexford
  Leasing Ltd" as a person (1).
- The older address-line rule read "00 while the main in Tannery Road" as an
  address, which costs "13:00" and "Tannery Road" (2).
- NER persons: "Albatros Versicherung" and "Felco 2" from the custom model
  (the digit rule covers general NER only), "Officina Bellandi S.r.l",
  "Grazie" and the "MM" of "DD/MM/YYYY" from general NER (5), and the stray
  "Telefono".
- Plural pronouns for things or an unnamed group: "they" for the cutters,
  "them" for the patterns and the samples, "Their" for the dye house (3).

The twenty-eight misses:

- Sign-off names with no full name in the document: "Thanks, Bronagh" and
  "Regards, Margaux" (2).
- ALL-CAPS: "PREVIOUS SURNAME: ADESANYA" and "NEXT OF KIN: KWABENA ADESANYA"
  (labels the name-field rule does not list) and the facility "PRIORY LANE
  SURGERY" (3).
- Names: "Dott.ssa Fiammetta Carbone" under "Cliente / Customer:"; "Name:
  Thiago Monteiro dos Santos" under the bare label the rule leaves out (the
  model stops at "Thiago Monteiro"); "Faith Oyinlola" in prose, which general
  NER calls an organization, and the two bare "Faith" the echo stop-list
  skips (4).
- Address parts: the Polish "00-590 Warszawa" after a street the model
  labelled a person (2), the Portuguese postcode "3000-180", the Eircode
  "V92 X7K3" (the model's address stops at "V92") and the house name "The
  Old Forge" (3).
- IDs: "Codice fiscale", the Italian plate "FX 482 LD", the loyalty card
  "6338 4410 0027", and the membership number "AL-2291" whose label sits
  earlier in the sentence (4).
- A format hint between label and value: "Date of birth (DD/MM/YYYY):
  02/09/1959" (1).
- Phones: "0044 114 496 0827" is masked only from "114" on, because a nested
  national match outranks the international pattern's 0.99 score; the Irish
  freephone "1800 555 014" (2).
- Facilities and employers the gate dropped: "Coquet Medical Group" (no
  listed word, and the model's span ran on into "after two"), "Instituto
  Politécnico do Mondego" and "Colégio São Bento" (Portuguese spellings; the
  model's span crossed a line break), "Granja & Filhos" on a CV line and
  "Sông Hồng Precision Co" after "at our supplier" (5).
- Context rules: the travel city in "fly to Birmingham" and the time "half
  past six in the evening" (2).

Model alone: PERSON 16/41. It again missed every hyphenated surname
(Chukwuemeka-Bell, Adesanya-Price, Llewellyn-Hughes twice, Armstrong-Lowe,
Haddad-Whitlock, O'Dowd-Farrell), three of the four ALL-CAPS names, both
sign-offs and the greeting names ("Good morning Katarzyna,", "Hi Ana,",
"Hello Tevita"); all three Romance-order streets and the house name; and it
labelled four ordinary dates, "GHANA", "Qty 1" and "Qty 2" as birth dates,
three letterheads, two employers, a leasing company, "Wi-Fi 6", "Gardena
Classic 30", "DD/MM/YYYY" and "Birmingham on 6 December" as facilities, an
insurer, a garage and "Felco 2" as persons, with three spans that cross line
breaks. The brief for corpus round 3 stands and grows by these.

### test7 fixes (2026-11-08)

Fifty of the fifty-five findings were fixed, all in `policy_rules.py`; test7
is semi-dev from here. The rules that reached too far came first:

- Greeting addressees: `_GREETING_STOP` gains residents and account holders
  (Householder, Occupier), departments and bodies (Finance, Committee) and
  roles; the new sign-off rule shares it.
- German plates: the shape counts only where the sentence speaks of a
  vehicle or a registration (`PLATE_CONTEXT`) and never straight after a
  place or type label (`PLACE_LABEL_PREFIX`: bay, room, region, type); a
  decimal continuation ("3.2") ends the match.
- The label decides (`THING_ID_LABEL_PREFIX`): a digit run straight after an
  invoice, receipt, batch, tracking, SSCC, barcode, part, voucher, flight or
  project label is that thing's number, whichever layer read it as a phone,
  a card or an address.
- An ID value is never a bare date (`_NOT_A_DATE`, agreement and generic
  IDs); a name field or role label never holds "NONE", "N/A", "Not
  applicable" (`_NOT_A_NAME`).
- "Via", "Plaza", "Largo" and "Corso" are English words too: as street
  words they need a postcode behind them or an address label in front.
- The English street line: the house number is never the minutes of a clock
  time or the decimals of an amount, and ordinary function words between
  the number and the street word mean a sentence ("5 minutes from the main
  Road").
- The first-name echo stops ahead of a site noun ("the Florence
  warehouse"), as general NER persons already did.
- A person span that ends in a company word, or is followed by one, is a
  company in every layer (`_COMPANY_TAIL`: Ltd, GmbH, S.r.l., Leasing,
  Versicherung).
- One-word NER persons that are a courtesy word ("Grazie"), a date-format
  placeholder ("MM") or a line-initial label ahead of a number
  ("Telefono: +39") are dropped; the no-digit rule for persons now covers
  the custom model ("Felco 2").

Then the misses:

- Names: a sign-off rule (closing word, comma, one name word that ends the
  line); the name-field labels previous / former / other / birth name and
  mother's maiden name; role labels in block capitals with at least two
  capital words ("NEXT OF KIN: KWABENA ADESANYA"; people-only labels, and
  not "DRIVER: COMPANY POOL VEHICLE"); the titles Dott.ssa, Sig.ra, Eng.ª,
  Rev, Sir and others; a bare "Name:" value when a NER layer saw a person
  inside it, which repairs the model's "Thiago Monteiro" to "Thiago
  Monteiro dos Santos".
- Facilities and employers: Medical / Health Group and Partnership; the
  block-capital form of every facility word ("PRIORY LANE SURGERY");
  facility words that lead the name (Colégio, Instituto, Universidad,
  Liceo, Klinik); "at our supplier X" as employer wording; a dated CV line
  ("2015–2019  Junior surveyor, Granja & Filhos"), whose employer also
  echoes to its other mentions.
- Addresses: the Polish street line ("ul. Marszałkowska 84/12") and the
  Polish, Portuguese and Irish postcode segments; an Eircode completed like
  a UK postcode when the model stops after the routing key; a house name
  ahead of the street number ("The Old Forge, 6 Chapel Row").
- IDs: the labels codice fiscale, fiscal / tax code, national ID, ID card,
  membership and loyalty; spaced card numbers after "loyalty card"; the
  possessive form ("membership ..., mine is AL-2291"); a labelled plate in
  any short capital format ("Targa / Plate: FX 482 LD", scored 0.45 like
  the passport fallback).
- Phones: a number carried by its label ("call 1800 555 014"); a shorter
  regex phone nested in a +/00 phone yields to it, so "0044 114 496 0827"
  is masked whole.
- Context: a format hint between a birth label and its date; "fly to",
  "flew to", "travelled to"; "half past six in the evening" and worded
  hours.

Three things were adjusted during the round. The company filter exposed the
model's shorter "Officina Bellandi" ahead of "S.r.l.", so it also reads the
next word. Removing the plate match on "DE-HE 3" exposed the pronoun
pattern on its "HE", so a pronoun is never glued to a hyphen. And the first
date guard also rejected registry numbers that start with a date (the shape
"record No. 2014-03-09/0771", in two private documents), found by comparing every
detection on all 161 gold, private and international documents before and
after; the guard now rejects bare dates only. That comparison shows two
other changes outside test7: "Colegio San Esteban" (test6) is now labelled
a facility rather than an organization, and in one private document a road
intersection in an accident report is no longer masked. It had been masked
only because the year of the date before it was read as a house number; an
intersection is a public place under the policy. (The examples quoted here
are invented stand-ins of the same shape, not text from those documents.)

Result: test7 113/141 → 139/141 (98.6%), keep 196/223 → 220/223 (98.7%), 1
→ 0 strays. Every other set is unchanged: dev 123/123 and 135/135 (strays
still 4), test and test2 100% / 100%, test3 75/76, test4 61/63 and 81/83,
test5 71/72 and 87/89, test6 110/111 and 160/164; both regression suites
and the 20 unit tests pass. The five failures left on test7 are the two
structural classes again: "Faith Oyinlola" in prose, which general NER
calls an organization, with the bare "Faith" the echo stop-list skips, and
three plural pronouns for things ("they" for the cutters, "them" for the
patterns, "Their" for the dye house).

### test8 blind run (2026-11-09)

Thirteen documents, 139 must-redact items and 238 must-keep phrases, written
without running the detector, after the test7 fixes (commit e4f25fc). The
batch aims at what those fixes assumed: greeting and sign-off words outside
the lists ("Hiya Sunniva,", "Morning Dafydd —", "Ta, Gethin", "All the best,
Ottilie Varga", "Hey Caretaker,") and a one-word company sign-off ("Kind
regards, Brackenridge" under the letterhead "Brackenridge Lettings"); German
plates in a parking bay list with no vehicle wording, beside a standard, a
charger type and a project code of the same shape; phones under a thing
label ("Parcel contact:", "Shipment phone:"); identifiers whose value is a
date ("Delivery note: 2026-11-06") beside a case number that starts with
one; block-capital fields whose values are not names ("NOT APPLICABLE", "TO
BE CONFIRMED", "AS ABOVE", "ONE CAT"); "Via" and "Plaza" streets with no
postcode beside "via email" and "Plaza Hotel"; a street name with a function
word ("14 Stoke by Nayland Road"); surnames that are company words ("Fr.
Annelise Bank", "Winifred Holding"); a bare "Name:" over a rare personal
name and over a product; CV lines with no year range; and house names with
no house word ("Braeside", "DUNROAMIN"). New ground: a French attestation, a
pipe table, an address block with town and postcode on their own lines, a
support transcript with "Agent (Marisol):" speakers, spaced account, serial
and frame numbers, a bracketed area code, Spanish and Dutch plates, and
plural pronouns in both directions.

Full pipeline (candidate 2 live, relaxed gate, test7 fixes in): redact
coverage 104/139 (74.8%), keep intact 223/238 (93.7%), 3 stray spans. Model
alone: 30/71 model-entity items (42.3%; PERSON 18/53, ADDRESS 6/11, DOB 4/4,
SENSITIVE_ORG 2/3), keep 225/238, 0 strays. Nothing has been fixed against
it; every other set, both regression suites and the 20 unit tests are
unchanged.

Blind recall has now been 75.0%, 75.7%, 80.1% and 74.8% on test5 to test8,
while every fix round has ended near 99%. The batches are not like for like,
since each one aims at the newest rules and adds formats, but the reading is
the same four times: a rule fixes the wording it was written for and little
beyond it. Keep is back above test7's 87.9% because fewer rules over-reach
this time (four of the fifteen keep failures, against fifteen of
twenty-seven).

Probe by probe:

- Phones under a thing label, identifiers that are dates, the date-led case
  number, the caps values that are not names and the street with a function
  word all held.
- Greeting and sign-off words: "Hiya" and "Cheers," by rule, "Morning Dafydd"
  and "Ta, Gethin" only because general NER knew the names; "All the best,"
  is not a listed closing and general NER calls "Ottilie Varga" an
  organization, so it is missed.
- The one-word company sign-off failed: the sign-off rule takes
  "Brackenridge" as a person and the echo then masks it in the letterhead.
- German plates: the context rule is inverted on this document. The three
  plates of the bay list are missed (no vehicle word on their lines), and
  "DIN-EN 1090" is redacted because its sentence goes on to say "vehicles".
- "Via" and "Plaza" without a postcode: all three addresses missed, which is
  the price of the test7 fix for "Via Royal Mail 48"; "via email", "via the
  portal" and "Plaza Hotel" are kept.
- Company-word surnames failed three times: general NER found "Annelise
  Bank" and the company-tail rule dropped it, "Frau Bank" and "Winifred
  Holding" are organizations to general NER.
- Bare "Name:": both rare personal names are redacted; the product under
  "Name:" and the part under "Name of part:" are redacted too, by the custom
  model as a person and as an address.
- CV lines with no year range: "Corbel Freight" and "Tannahill Haulage" are
  covered, but as persons from general NER and the echo, not as employers;
  "Strathmore Wholesale" after "as a picker for" is missed.
- House names: "Braeside" is missed; "DUNROAMIN" is covered only because
  general NER calls it a person.

What else held: every name in the fixture email, the occupational-health
note and the support transcript, hyphenated and accented ones included
("Thabo Nkosana-Fields", "Freya Sandberg-Moore", "Matilda Brzezińska",
"Siobhán Kettlewell"), the speakers written "Agent (Marisol):", "Sr.
Baltasar Quintero", "R. Nakamura" and the bare "Lotte" in the table, "Mr
Joost van den Heuvel" on its own line; the whole block-capital form (name,
birth date, address, emergency contact with the relation in brackets, "MRS
P. ECCLESTONE", the spaced National Insurance number, the signature); the
Italian mobile, the UK phones, both e-mail addresses of the CV; the payroll
number, the account number given in a chat reply, the dashed serial number,
the order number, the IBAN, the French ID card number; "quarter to nine in
the evening"; both facilities and the college; all fifteen pronouns that
refer to people, "their" for two named colleagues included; and the things
("ending 2816", the meter reading, the refund reference that is a date, the
tracking number, "WB-AC 22", "HV-LH 2026", "RL-77120", "Keswick store",
"Store Manager", "Customer Accounts", "Maintenance Committee", "Otterburn
Harriers Juniors", and the letterheads "Brindlemoor Occupational Health Ltd"
and "Sparrowhawk Energy"). Four other letterheads are redacted, each by a
different route; they are in the list below.

The fifteen keep failures and the three strays:

- Sign-off and greeting rules: "Brackenridge", through the echo also
  "Brackenridge Lettings", and "Hey Caretaker," (3).
- NER persons: "Ta", "Consignee", the shop "Corvid Cycles" and the provider
  "Quillfield" with its echo into "Quillfield Broadband" from general NER;
  "Hausverwaltung Lindenhof" and "Corvid Tarn" from the custom model; and
  "Payroll number", because a general-NER span ran across a column gap from
  the name into the next label (7). Strays: "Tel" and "Quillfield".
- The custom model's address "SR Suntour XCR" under "Name of part:" (1).
- German-plate rule: "DIN-EN 1090" (1).
- Plural pronouns for an unnamed group: "they" for the surgeons and for the
  referees, "their" for the engineers (3).
- Stray: the badge "V-0420" as an ID. The gold leaves visitor badges
  unlabelled, so it shows as a stray and not as a failure.

The thirty-five misses:

- The French attestation (8): "Mme" is not a listed title, so general NER
  stops at "Mme Clémence" and the model's full name carries the wrong label;
  "M. Idrissa Coulibaly"; both birth dates, which the model found and the
  context filter dropped because "née le" and "né le" are not birth wording
  to it; "passeport n°"; the phone "06 39 98 01 84" under "Tél. :"; the
  town in "Fait à Dijon"; "Signature : C. Aubertin-Roche".
- Plates (5): the three German ones in the bay list, and the Spanish "6205
  KLM" and Dutch "9-XKR-27" in a table column headed "Vehicle reg".
- Company-word surnames (3): "Annelise Bank", "Bank", "Winifred Holding".
- Other names (6): "Ottilie Varga" after "All the best,"; "Oluchi
  Nwachukwu-Tait" under "Consignee:", where the model stops at the hyphen;
  "Morag Sutherland-Baird" alone on a line under "Referees"; "Dear Mr van
  den Heuvel,", where the model's span is "Mr van"; "Chinedu Anozie" in a
  table cell, a location to general NER; and "The mechanic, Pim, says".
- Addresses (6): the three "Via" and "Plaza" streets, "Braeside", and
  "Ludlow" and "SY8 1BB" on their own lines under the street.
- Employers (2): "EMPLOYER: WESTMORLAND DAIRIES LTD" in capitals and
  "Strathmore Wholesale".
- Spaced identifiers (3): "gas account 8841 2207 16" is masked only to the
  first space, "meter serial number G4A 00417 2209" and "Frame number: WTU
  284 K 0931 M" not at all.
- "(01632) 960044", an area code in brackets (1).
- "I fly to Bologna": no NER layer returned the city as a location, so the
  travel rule had nothing to act on (1).

Model alone: PERSON 18/53. It missed nine of the ten hyphenated surnames
(the tenth is covered under a facility label), every greeting and sign-off
name of the tenants' board, three of the four ALL-CAPS names, the particle
surname, first names when they come back alone ("Freya", "Kofi", "Oscar",
"Lucinda") and three of the six table names; it found all four birth dates
but also labelled five ordinary dates as birth dates, one of them in a span
that begins with the phone number on the line above; and it labelled a
letterhead, a company sign-off, a junior sports team, "DIN-EN 1090" and
"Bologna on the 14th" as facilities, a property manager's letterhead and a
bicycle as persons, and a fork as an address. Three of its spans cross a
line break. Corpus round 3 grows by French documents, table cells, product
names under "Name:" and column-gap layouts.

### test8 fixes (2026-11-13)

The round started from the run with candidate 3 live: 112/139, keep 222/238,
3 strays, which is 46 findings. Forty-two were fixed, all in
`policy_rules.py`; test8 is semi-dev from here. The rules and spans that
reached too far came first:

- The author in the heading: when the document opens with a capitalised
  phrase that no layer reads as a person ("Brackenridge Lettings — messages
  ..."), its first word is the author's name, and a one-word "person" equal
  to it is the author signing or being named ("Kind regards, Brackenridge",
  "Thanks for calling Quillfield"). This also ends the echo into the
  letterhead.
- Greetings and sign-offs: `_GREETING_STOP` gains site and service roles
  ("Hey Caretaker,"); "Ta" and "Thx" are sign-off words; a NER person span
  loses a leading greeting or sign-off word ("Ta, Gethin", "Hello Tevita"),
  and what is left is dropped when it is a role ("Dear Occupier").
- NER person spans stop at a column gap, a spaced dash or a pipe ("Thabo
  Nkosana-Fields   Payroll"). One-word NER persons are dropped when they
  are a contact label ahead of a number ("Tel. 089 ..."), a role label
  ahead of a colon ("Consignee:") or a first-person pronoun ("Je").
- A company read as a person: `_COMPANY_TAIL` gains trade nouns (Cycles,
  Lettings, Haulage, Freight, Kitchenware, ...) and `_COMPANY_LEAD` holds
  the words that lead a company name in other languages ("Hausverwaltung
  Lindenhof"). Such a NER span is no longer dropped; it is judged as the
  organization it is (see the employer rules below). A courtesy title in
  front keeps it a person ("Fr. Annelise Bank", "Frau Bank").
- A German-shaped plate is a standard after a conformity phrase or under a
  standards body ("certified to DIN-EN 1090").

Then the misses:

- Plates: a person on the same line links a German-shaped plate to them
  (the bay list "Bay 03 — M-KT 5038 — Fr. Annelise Bank"). A pipe or tab
  table is read by column: the header cell labels the cells below it, so
  "Vehicle reg" carries "6205 KLM" and "9-XKR-27", and "Name" and "Host"
  carry "Winifred Holding" and "Lotte". A cell must still have the shape
  of a plate or a name ("on foot" has neither).
- Names: a courtesy title carries the name behind it (`titled_person`; the
  title stays outside the span); a person noun, a comma, a name and a comma
  ("The mechanic, Pim, says"; `apposition_person`).
- Employers: the tie to a person may sit at another mention of the same
  organization (`org_is_tied`): employment wording, "at X", "as a picker
  for X", or the heading of the CV block the mention stands in ("Work
  history"). An accepted employer echoes to its other mentions whichever
  layer found it. A NER organization that runs on into the next line is
  cut at the line end ("WESTMORLAND DAIRIES LTD" + "EMERGENCY"), and a
  line-initial "Employer:" label names the employer by itself.
- Addresses: a house name under an address label needs no house word
  ("Address: Braeside, 4 Station Road"); a function word in the street
  line is allowed under a label or ahead of a postcode ("14 Stoke by
  Nayland Road, Sudbury CO10 2XA"); "Via" and "Plaza" streets count
  without a postcode when a particle follows ("Plaza de Olavide 6"), a
  floor or door follows the number ("6, 2ºB") or the word stands
  capitalised mid-sentence ("at Via Saragozza 120"); town and postcode
  lines under a street line are part of the address when a postcode line
  closes the block ("14 Mill Street / Ludlow / SY8 1BB").
- IDs: a person-ID value in spaced digit groups, eight digits or more
  ("gas account 8841 2207 16"); serial numbers in spaced capital groups
  and the labels frame, chassis and IMEI ("Frame number: WTU 284 K 0931
  M"); the passport label in French, Spanish, Portuguese, Italian and
  German with "n°" ("passeport n° 18FV20417").
- Phones: a bracketed area code ("(01632) 960044"); the labels "Tél.",
  "portable", "Handy", "móvil", "cellulare".
- French documents: the birth labels "né(e) le" and "date de naissance"
  with French month names, the same labels in German, Spanish, Italian
  and Portuguese, and the signing formula "Fait à Dijon, le ...".
- Travel: the word after "fly to", "travel to" or "trip to" is a
  destination when no NER layer tagged it ("I fly to Bologna").

Four things were adjusted during the round. `TITLE_PREFIX` is
case-insensitive and read "as a picker for" as a job title, so the
role-then-employer wording is checked first. The travel rule masked a trip
venue on test6 ("trip to Bramblewood Nature Reserve"); a longer capitalised
name or a venue word now ends the match. A line-initial "Ta," or "Dear
Occupier," was promoted by the greeting-name rule when a model span
contained it; courtesy and role words are excluded there. And the value
under "Employer:" took a closing full stop.

Two comparisons bound the side effects. Every detection on all 174 gold,
private and international documents was compared before and after: outside
test8 three change. "Halden Kitchenware" (test3) is no longer masked, which
was that set's last over-redaction; "Porto" (test3) is labelled a location
instead of a person; "Hello Tevita" (test7) is trimmed to "Tevita". Nothing
changes on the private or the international documents. The old and the new
rules were also run over the 142 generated dev documents, which no rule was
written against: five documents change. An employer under an "Employer:"
label is newly masked, a Spanish birth date ("Fecha de nacimiento:") is
newly found, "Dear Occupier" is no longer masked, and no document gains a
false positive.

Result: test8 112/139 → 139/139 (100%), keep 222/238 → 235/238 (98.7%), 3
→ 1 strays; test3 keep 102/103 → 103/103. Every other set is unchanged: dev
123/123 and 135/135 (strays still 4), test and test2 100% / 100%, test4
61/63 and 81/83, test5 71/72 and 87/89, test6 110/111 and 160/164, test7
141/141 and 220/223; both regression suites and the 22 unit tests pass.
Left on test8 are three plural pronouns for things or unnamed groups
("they" for the surgeons and for the referees, "their" for the engineers'
van) and the stray badge number "V-0420", whose policy is open. Across all
sets the failures left are sixteen plural pronouns and the two prose first
names "Thandiwe" and "Hannelore".

Read the 100% as a fitted score. The blind runs are the measurement: 75.7%
for test6, 80.1% for test7 and 74.8% for test8 with candidate 2.
Several of the new rules rest on word lists (trade nouns, person nouns,
venue words, role words) that cover what the gold sets showed, not the
language.

### test9 blind run (2026-11-13)

Thirteen documents, 168 must-redact items and 301 must-keep phrases, written
without running the detector, after the test8 fixes (commit 8dbffff) and with
candidate 3 live. The batch aims at what that fix round assumed: titles
outside the list ("Dña.", "D.", "Deacon", "Frau Dr."), closings and greeting
roles outside the lists ("With every good wish,", "Safe travels,",
"Bendithion,", "Dear Bursar,"), the apposition rule with other nouns ("The
vet next door, Ines,"), employer wording other than "Employer:" and "as a
... for" ("joined X as", "seconded to X", "apprenticeship with X", "Current
employer:"), travel verbs other than fly and travel ("relocates to", "drives
up to", "lands in", "travels on to", "is in Berlin until"), signing places
in German and Spanish, phone formats and labels ("030 / 5550 0142", "+44
(0)7700 900318", "Tfno.:"), a serial column in a table, a results table with
no pipes, a "Name" column that holds products, surnames that are venue words
("Hall", "Park", "Church", "Post", "Kirk"), a personal letterhead with a
first-name sign-off, a first name used as a label in a phone list ("Maud:
07700 900 123"), and address blocks that do not end in a postcode line. New
ground: a chat export with lowercase names, a JSON record, a German and a
Spanish form, the airline name format ("NAKASHIMA/RENJIRO MR"), and a parish
newsletter in prose. Every name word was checked against the generated
corpus, the older gold sets, the rules and the unit tests, and replaced when
one of them already had it.

Full pipeline (candidate 3 live, test8 fixes in): redact coverage 136/168
(81.0%), keep intact 273/301 (90.7%), 5 stray spans. Model alone: 72/89
model-entity items (80.9%; PERSON 56/69, ADDRESS 5/8, DOB 7/8, SENSITIVE_ORG
4/4), keep 282/301, 3 strays. Nothing has been fixed against it.

Blind recall has now been 75.0%, 75.7%, 80.1%, 74.8% (80.6% once candidate 3
was live) and 81.0% on test5 to test9. Keep is the weaker side this time:
twelve of the twenty-eight keep failures are spans of the custom model, and
seven of those are in three layouts it has never seen (the German form, the
JSON record and the chat export).

By label: PERSON 62/69, PRONOUN 14/14, PHONE 20/23, EMAIL 4/4, EMPLOYER_ORG
4/4, ADDRESS 6/8, DOB 6/8, ZIP 4/6, CITY_RESIDENCE 3/7, CITY_TRAVEL 1/4,
GENERIC_ID 3/5, PASSPORT 2/3, SENSITIVE_ORG 2/4, SERIAL_NUMBER 0/4,
LICENSE_PLATE 2/2, BANK_ACCOUNT 2/2, UK_NINO 1/1.

Probe by probe:

- Titles, closings and the personal letterhead held: every titled name is
  covered, the first-name sign-off under the writer's own name is not taken
  for a company, and the names after the three unlisted closings are found.
  Two closings are themselves redacted ("Bendithion" as a person by the
  custom model, and "ta" in lowercase).
- Employer wording held: all four employers are covered and the supplier
  ("We bought our own windows from Dovecote Joinery") stays.
- Lowercase chat names, the nickname in quotes, the airline name format and
  both apposition names are covered.
- Travel verbs failed three times out of four: "relocates to Bodø", "drives
  up to Thurso" and "lands in Molde" are missed; "travels on to Ålesund" is
  covered. "he is in Berlin until the 20th" is missed too.
- Signing places: the Spanish one is covered and the German "Göttingen,
  den 3. November 2026" is not. In the Spanish address the town after the
  postcode is missed instead.
- Serial numbers: 0 of 4. No rule reads a "Serial" table column, and "S/N"
  is not a serial label.
- The first name used as a phone label is missed ("Maud: 07700 900 123"),
  which is the price of the test8 rule that drops a one-word name ahead of a
  number.
- Venue-word surnames: "Hall, Rosamund" and "Seo-yeon Park" are missed in
  the results table, and "Park and Church" in prose.
- The "Name" column of products: "Stanley FatMax" and "Rothenberger" are
  redacted as persons.

The thirty-two misses:

- The German form (4): the phone "030 / 5550 0142" with a slash, the signing
  place, the birth date after "geb. am" with a German month, and the
  passport number after "Reisepass-Nr. / passport no. (Person 1):".
- The results table (4): "Hall, Rosamund", "Seo-yeon Park", and "Park" and
  "Church" where they come back alone.
- The personal letter (4): the house name "Tŷ Gwyn" on its own line, and
  the town and the Eircode of a block that has a county line between them;
  "Glenbrook Adolescent Unit", which the custom model found as a facility
  and the gate does not admit.
- The JSON record (5): the member number under "member_id", the birth date
  under "date_of_birth", the street under "line1", the postcode, and the
  phone "06 5550 0121".
- Travel and residence cities (5): "Bodø", "Thurso", "Molde", "Berlin", and
  "Tarazona" in the address line.
- Serial numbers (4) and the frequent-flyer number "SK 550 291 4470" (1).
- Phone list (2): "Maud" and "(ask for Priti)".
- The Spanish phone under "Tfno.:" (1).
- The parish newsletter (2): "Nerys" at the start of a sentence, and the
  Welsh "Ysbyty Cwm Derwen", which the custom model also found as a
  facility.

The twenty-eight keep failures and the five strays:

- Custom-model persons (11): "Bin" of the group name, "ta", four German
  form labels and words ("Bundesmeldegesetz", "Anschrift der Wohnung",
  "Reisepass-Nr", "Zählerstand Strom"), the JSON key "date_of_birth",
  "Warfarin", "Nana" in "Nana Jo", "Rothenberger" and "Bendithion". Strays:
  "van", "Einzug" and "la siguiente".
- A custom-model address: "que en el Padrón Municipal" (1).
- General NER (7): "Bemerkungen", the clubs "Thornwick Harriers" and "Vale
  Royal", "FRCS" and "Deacon" inside a person span, "Stanley FatMax", and
  the letterhead "Larkfield Staffing" as an organization. Stray: "Dña".
- Facility keywords on words that are not a facility (3): "Hospital no.",
  "Clinic 3" and "School office".
- Other rules (3): the cellar room "B-EG 12" as a German plate on a line
  with no vehicle, "Dear Bursar," as a greeting name, and the tax code
  "1257L" as an identifier.
- Plural pronouns (3): "they" for the patients, for the scouts and for the
  windows. Stray: "them" for the meter readings.

Labels taken while authoring are decision 22 of the labeling guide.

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

## NER candidate 3 (2026-11-13)

Candidate 3 was trained on rounds 1 to 3 of the generated corpus (1,420
documents) with `train.py --keep-best --batch-size 8 --epochs 24`, seed 42,
and replaced candidate 2 as the live model on 2026-11-13. Model alone:

| | candidate 2 | candidate 3 (promoted) |
|---|---|---|
| generated dev (142 docs), exact-span F1 | 47.4% | 98.3% |
| dev to test7: redact coverage (296 items) | 213 (72.0%) | 273 (92.2%) |
| dev to test7: PERSON / ADDRESS / DOB / SENSITIVE_ORG | 125/189, 43/51, 21/28, 24/28 | 173/189, 51/51, 28/28, 21/28 |
| dev to test7: keep intact (929) / stray spans | 869 / 2 | 924 / 2 |
| test8, held out: redact coverage (71 items) | 30 (42.3%) | 62 (87.3%) |
| test8: PERSON / ADDRESS / DOB / SENSITIVE_ORG | 18/53, 6/11, 4/4, 2/3 | 49/53, 7/11, 4/4, 2/3 |
| test8: keep intact (238) / stray spans | 225 / 0 | 233 / 1 |

Full pipeline, candidate 2 against candidate 3 with the three guards below:

| set | candidate 2 | candidate 3 |
|---|---|---|
| dev, test, test2 | 100% / 100% (4 strays on dev) | unchanged |
| test3 | 75/76, 102/103 | 76/76, 102/103 |
| test4, test5, test6 | 61/63 and 81/83, 71/72 and 87/89, 110/111 and 160/164 | unchanged |
| test7 | 139/141, 220/223 | 141/141, 220/223 |
| test8 (no rule fixed against it) | 104/139, 223/238, 3 strays | 112/139, 222/238, 3 strays |

On the older sets candidate 3 covers the prose first name "Nkechi" and
"Faith Oyinlola" with both bare "Faith"; the sixteen failures left there are
thirteen plural pronouns, "Thandiwe", "Hannelore" and "Halden Kitchenware".
On test8 it covers ten items that were missed ("Ottilie Varga" after an
unlisted closing, "Oluchi Nwachukwu-Tait", "Morag Sutherland-Baird" under a
heading, "Dear Mr van den Heuvel,", "Chinedu Anozie" in a table cell, the
three French names, one French birth date and "Via Saragozza 118") and loses
"14 Stoke by Nayland Road" with its postcode, which candidate 2 had found.
One keep is gained (the part under "Name of part:") and two are lost (the
German word after a letterhead dash, and "Je" of "Je soussignée"). Blind
recall of the pipeline on a fresh batch is 80.6%, against the 75% that four
batches in a row had shown with rules alone moving.

### The four training runs

The corpus and the trainer changed between runs; the older gold sets and the
private regression documents steered the changes, test8 did not.

1. Round 3 as first written (660 documents, 20 epochs, last epoch saved).
   Model alone 271/296 on the older sets, but in the pipeline thirteen new
   keep failures: label words and sentence openers as persons ("Smoking",
   "Photographs", "Consignee", "Householder"), a bank code, a company and a
   city as persons, products and a nature reserve as addresses.
2. More negatives (800 documents): one keep failure left, but the model lost
   two addresses with no street word ("8 Bryn Teg"), which the pipeline
   then missed, and simple two-word names it had always found. The loss
   curve still swung between 30 and 90 at epoch 20, so the last epoch was a
   draw.
3. Streets with no street word, and `--keep-best`: the dev set is scored
   after every epoch and the best epoch's weights are saved (epoch 10 of
   24). Model alone 277/296. On test8 the pipeline scored 114/139 and
   223/238. On the private documents it masked 414 more characters than
   candidate 2 and 55 fewer: real streets and names gained, but also
   medical terms and city names as persons, and one surname with an initial
   lost.
4. Official records in the Russian order, written in English with invented
   names (surname first, patronymics, initials, postcode ahead of the
   street), field values that open with a capital, "City of X" and regions
   (900 documents; best epoch 11). This run is candidate 3. On the private
   documents it masks 474 more characters than candidate 2 and 38 fewer.

Test8 was therefore scored twice, once for run 3 and once for run 4. Run 4
was chosen on the private documents, and it scores two redactions and one
keep lower on test8 than run 3 did.

### Guards added to the gate

Three small filters in `policy_rules.py` came with the promotion; none of
them changes a score with candidate 2:

- a thing label vetoes a custom-model ADDRESS as it already vetoed a PERSON
  ("Engine: Kubota D902, 16 hp" is not a street);
- a PERSON of fewer than two letters is dropped ("keys A and B.");
- a PERSON straight after a place noun and "of" is a place ("City of Tver",
  "Republic of Karelia"), and the operational-site veto ("the Hull depot")
  now covers custom-model persons too.

### On the private documents

Against candidate 2 on the 20 private-suite documents, candidate 3 newly
masks full street lines with postcode and city in most of them and nine
names or name parts in the surname-first order or with initials, all under
the trusted `ADDRESS` and `PERSON` labels. Its new false positives are one
medical term and one bank name as persons, three place names, and the
street addresses of institutions, which the policy does not ask for. What it no longer masks is
three false positives of candidate 2 and one certificate number that
candidate 2 had labelled an address by accident: no rule covers certificate
numbers, and whether they redact is an open policy question. The suite's
reviewed identifiers and keep checks all pass. These differences were read
on screen and not hand-reviewed line by line.

### Limits

- SENSITIVE_ORG fell from 24 to 21 of 28 on the older sets; the keyword
  rules cover all of them in the pipeline, so no score moved.
- Training is not stable from run to run: runs 1 to 4 gave 260 to 277 of
  296 on the older sets as the corpus grew, and the loss still swings
  late. `--keep-best` picks on the generated dev set, whose templates the
  model has seen. A second seed and a selection set built from real layouts
  are the next improvements.
- Exact-span ADDRESS precision on generated dev is 86.8%: five boundary
  errors, where the span stops at "apt." before the flat number or splits
  after the Polish "ul.". The rule layer completes both in the pipeline.

## NER candidate 4 (2026-11-14): trained, not promoted

Candidate 4 was trained on rounds 1 to 4 of the generated corpus (1,900
documents) with `train.py --keep-best --batch-size 8 --epochs 24
--select-gold dev,test,...,test8`, once with seed 42 and once with seed 7.
It is kept as `~/pii-models/custom_ner_candidate4`; candidate 3 stays the
live model.

`--select-gold` is new. `--keep-best` picks the epoch with the best F1 on
the generated dev set, whose templates the model has seen. With
`--select-gold` the trainer also scores every epoch on the named gold sets,
model alone (items covered, minus keep phrases touched, minus stray spans),
and saves the best epoch to `<output>_goldsel`. Only sets that are no
longer held out may be named; test9 was not. Each run therefore gave two
candidates:

| run | epoch kept | model alone, dev to test8 (367 items, 1,167 keeps) | pipeline, dev to test8 | suites |
|---|---|---|---|---|
| candidate 3 (live) | 11 | 335, keep 1,157, 3 strays | 828/832, keep 1,153, 5 strays | pass |
| seed 42, best generated-dev F1 | 14 | 344, keep 1,153, 3 strays | 828/832, keep 1,148, 6 strays | international suite fails (a city is redacted) |
| **seed 42, best gold score** | 7 | 346, keep 1,163, 2 strays | 828/832, keep 1,153, 6 strays | pass |
| seed 7, best generated-dev F1 | 24 | 336, keep 1,157, 1 stray | 827/832, keep 1,149, 5 strays | pass |
| seed 7, best gold score | 14 | 349, keep 1,155, 2 strays | 828/832, keep 1,147, 7 strays | pass |

The seed-42 gold-selected run was chosen on this table, before anything was
scored on test9. The table is biased towards the two gold-selected runs,
which were picked on those same sets; the held-out measurement is test9:

| | candidate 3 | candidate 4 |
|---|---|---|
| generated dev (190 docs), exact-span F1 | 89.4% | 98.3% |
| test9, model alone: redact coverage (89 items) | 72 (80.9%) | 71 (79.8%) |
| test9, model alone: PERSON / ADDRESS / DOB / SENSITIVE_ORG | 56/69, 5/8, 7/8, 4/4 | 53/69, 7/8, 7/8, 4/4 |
| test9, model alone: keep intact (301) / stray spans | 282 / 3 | 286 / 1 |
| test9, pipeline: redact coverage (168 items) | 136 (81.0%) | 135 (80.4%) |
| test9, pipeline: keep intact (301) / stray spans | 273 / 5 | 276 / 2 |

On test9 the pipeline gains four items with candidate 4 (the house name and
the town of the address block in the personal letter, the town in the
Spanish address line, the Welsh hospital) and loses five (a postcode ahead of
a German town, "Ji-woo Park", "Liesel & Gernot" in the phone list, and one
of the three "Jide"). Five keep failures and three strays go ("Bin", "ta",
"Warfarin", "Padrón Municipal", a "they" beside a dropped span; "van",
"Einzug", "la siguiente") and two come ("N.º de pasaporte", "El
Secretario").

On the older sets the pipeline score does not move, because the rules
already cover them. The detections that change there are boundaries and
labels (a French address as one span, a house name inside its address, a
title outside the name), and one new stray: the one-word reply "Fine" after
"Caller:" on test8, which the round-4 transcript genre taught ("Caller:
It's <name>").

On the 20 private-suite documents, candidate 4 turns the address of ten
documents from fragments under mixed labels into one `ADDRESS` span that
includes the postcode and the house number, and no longer calls a bank a
person. It loses the value under a "Patronymic:" label in one document, the
house number of an institution's address in a second, and part of a
hyphenated city name with its house number in a third. The suite's reviewed
identifiers and keep checks all pass.
These differences were read on screen and not hand-reviewed line by line.

Why it was not promoted: the held-out set shows no gain in recall (135
against 136), four person items and a postcode are lost for four address
and facility items gained, and one name part is lost on a private document. Less
over-redaction (three keeps, three strays) does not outweigh that for a
privacy tool. Round 4 taught the layouts of test8, where the rules had
already reached 100%, and test9 probes other layouts; the model's 80% on
test9 is the same with and without round 4.

What the runs showed about training:

- The spread between seeds and epochs is as large as the gain from a corpus
  round: 336 to 349 of 367 on the older sets across the four candidates,
  and the gold score moves by up to ten points between neighbouring epochs.
- Generated-dev F1 is not a safe selector: its best epoch of seed 42 touches
  ten more keep phrases than the gold-selected epoch and fails the
  international suite.
- The gold-selected epochs are early ones (7 and 14 of 24), which fits a
  model that over-fits the templates later on.

### Next steps

1. A test9 fix round for the rules. The thirty-two misses and twenty-eight
   keep failures are listed under "test9 blind run": travel verbs, serial
   columns and "S/N", German and Spanish labels, JSON keys, names used as
   labels in a phone list, venue-word surnames, tables without separators,
   and facility keywords on "Hospital no.", "Clinic 3" and "School office".
2. Corpus round 5 and candidate 5 after that fix round: the test9 layouts
   (a chat export with lowercase names, a JSON record, German and Spanish
   forms, fixed-width tables, venue-word surnames), one-word replies after
   a speaker label ("Caller: Fine."), a "Patronymic:" field, and hyphenated
   city names. Then a blind test10 to measure it. Averaging or voting over
   seeds is worth a try, given the spread above.
3. Plural pronouns need reference, not proximity: nineteen failures across
   all sets and the stray "them" on test9. No fix round and no model has
   reduced this class.
4. Policy questions: certificate numbers (title, birth, registration) have
   no rule; the street address of an institution is masked like a home
   address; visitor badge numbers are masked under the "badge" label but
   left unlabelled in the gold. New from test9: the gate does not admit the
   model's `SENSITIVE_ORGANIZATION` spans, and the model found both
   facilities that the keyword rules missed ("... Adolescent Unit", "Ysbyty
   ..."); whether an employer redacts on its own letterhead (a payslip);
   whether a work address, a place of birth, a booking reference and a
   password redact.
5. The remaining over-redactions outside the pronoun class: "Pallet" and
   "Meter" on dev. The prose first names ("Thandiwe will send ...", "Ring
   Hannelore on ...", "Nerys ran the cake stall") are a model task.
6. Rounds 2 to 4 of the generated corpus have not been through an
   adjudication round.

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

# Candidate 3 was trained on the generated corpus like this (about 12 minutes on CPU):
python training/train.py --keep-best --batch-size 8 --epochs 24 \
  --train training/data/generated/train.jsonl \
  --dev training/data/generated/dev.jsonl

# Candidate 4 added a second selection: the epoch that scores best on gold
# sets that are no longer held out is saved to <output>_goldsel.
python training/train.py --keep-best --batch-size 8 --epochs 24 --seed 42 \
  --select-gold dev,test,test2,test3,test4,test5,test6,test7,test8 \
  --train training/data/generated/train.jsonl \
  --dev training/data/generated/dev.jsonl

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
