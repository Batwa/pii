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
| `SENSITIVE_ORGANIZATION` — clinics, hospitals, schools, shelters mentioned as third parties | All formatted IDs: SSN, SNILS, INN, passports, cards, accounts, IBAN, VIN, plates, policy/student/customer/badge/case IDs, serial and order numbers |
| `ADDRESS` — street + apartment + city + postal code | Pronouns (closed word list) |
| `DATE_OF_BIRTH` — the date value near a birth label | Labelled IDs of any country (label-before-number rule) |
|  | Times with persons, class/grade designators |
|  | Travel-destination cities (context rule on NER output) |

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
- A name word may carry an inner capital ("MacLeod", "McDonald", "DeSouza");
  label the whole word, never the prefix alone.
- Lowercase particles and accented letters are part of the name: "Fenna van
  der Berg", "Pieter de Vries", "Gábor Németh", "Oisín". A fragment ("Fenna
  van", "Németh") is a boundary error, not a partial success (test5,
  2026-11-05).
- ALL-CAPS form values after a name label are names ("SURNAME: ONYEKWERE",
  "GIVEN NAMES: CHIDI EMEKA"); label the value, not the label. A courtesy
  title ahead of a labelled name belongs to the span ("Mr Khalid bin Tariq
  Al-Farsi") but is not a name word (test6, 2026-11-07).
- A bare first name addressed by a greeting or a reminder is a person ("Hi
  Ifeoma,", "Reminder for Bartek:", "Thanks Rhys."); collective and role
  addressees are not ("Dear Customer,", "Hi Team,", "Hi all,"). The first
  name of a full name found elsewhere in the document is the same person
  ("Sigrún" after "Pupil: Sigrún Hallgrímsdóttir") — rule-owned at runtime
  (first-name echo), but label every occurrence in gold.
- Job titles are NOT persons ("Senior Dispatcher", "Ward Nurse" stay).
- Company names are NOT persons even in "Defendant: Stroyservice LLC".
- A product or vehicle name after a thing label is NOT a person ("Vehicle:
  Nissan Qashqai", "Product: Halden Kitchenware"), whatever the NER says;
  nor is anything with a digit in it ("Jungheinrich EFG 216") — a name never
  contains one (test6, 2026-11-07).
- A sign-off name is a person ("Thanks, Bronagh", "Regards,\nMargaux"), as
  is a block-capital value under a role label that takes people ("NEXT OF
  KIN: KWABENA ADESANYA"). A value under the bare label "Name:" is a person
  when it reads as one; at runtime it needs a NER layer to agree. Form
  values such as "NONE" or "N/A" are not names, nor are courtesy words
  ("Grazie"), date-format placeholders ("MM") or a span that ends in a
  company word ("Lexford Leasing Ltd", "Albatros Versicherung"); test7,
  2026-11-08.

### SENSITIVE_ORGANIZATION
- Facilities whose mention alone reveals something about a person: clinics,
  hospitals, polyclinics, hospices, infirmaries, sanatoriums, shelters,
  schools, nurseries, colleges, universities, institutes, academies, GP /
  dental / medical practices, surgeries (the UK GP sense), and medical /
  support / health / care / wellbeing / wellness / counselling /
  rehabilitation / language / speech / therapy / treatment / dialysis / cancer
  / diagnostic / crisis / day centers (either spelling), counselling services,
  care / nursing / residential homes, health / NHS / foundation / hospital
  trusts, medical / health groups and partnerships. The same words in
  block capitals count ("PRIORY LANE SURGERY"), and so do the Romance and
  German words that lead the name: Colegio / Colégio, Instituto, Escuela,
  Universidad, Liceo, Clínica, Klinik, Hochschule (test7, 2026-11-08).
  The runtime keyword list (`SENSITIVE_ORG_KEYWORDS`) mirrors this line; a
  facility type missing from both is a miss, not a policy exception (test3,
  2026-11-04: Institute, Academy, Wellbeing Centre; test5, 2026-11-05:
  Surgery, Language Centre; test6, 2026-11-07: Care Home, Health Trust).
- Label THIRD-PARTY mentions only: "referred to X", "treated at X",
  "Education: X", "attached to X".
- Do NOT label the organization that authored the document — its own
  letterhead or header line stays visible (review verdict, 2026-08-31). At
  runtime a rule exempts a facility name standing alone on one of the first
  three lines.
- Ordinary employers are NOT sensitive organizations. (Employer redaction is
  a context rule, not a model entity, in V1.)

### ADDRESS
- Label the full home/work address on one line: street, apartment, city,
  postal code ("17 Maple Crescent, Apt 3B, Springfield, IL 62704"). A flat or
  apartment designator may precede the street ("Flat 6, 28 Harbour Lane") and
  is still part of the same address.
- Standalone city names stay visible ("the Leeds ring road", "the Katowice
  warehouse", "our Frankfurt branch") — but a travel destination ("arrival
  Lisbon") is redacted by a context rule; either way it is never labeled.
- A city inside an address or a residence sentence is part of the address.
- A postal code in any national format closes the address line ("Haarlem,
  2011 AB"; "Springfield, IL 62704"; "Bristol, BS1 4ND"; "Toronto, ON M5V
  2T6"). In the German order the number follows the street and the postcode
  precedes the city ("Hauptstrasse 12, 80331 Munich"); the whole line is one
  address. In the Romance order the street word leads and a floor/door may
  follow the number ("Calle Mayor 12, 3ºB, 28013 Madrid"). A UK postcode
  after the town is part of the line ("Kingsmere KM2 7QL"), never a stranded
  half ("Kingsmere KM2").
- A Polish street line ("ul. Marszałkowska 84/12, 00-590 Warszawa"), a
  Portuguese postcode ("3000-180 Coimbra") and an Eircode ("V92 X7K3") close
  the line like the formats above; a house name ahead of the street number
  is part of it ("The Old Forge, 6 Chapel Row"). A street word that is also
  English ("Via Royal Mail 48") is an address only with a postcode behind it
  or an address label in front; a number followed by ordinary words and a
  street word is a sentence ("13:00 while the main in Tannery Road"); test7,
  2026-11-08.
- A year followed by capitalised words is a title or an event, not an address
  ("2027 Halvorsen Scholarship"), unless a street word says otherwise (test6,
  2026-11-07: the custom model's first trusted-label leak).

### DATE_OF_BIRTH
- Only the date value, only when a birth label is nearby: "born", "DOB",
  "d.o.b.", "date of birth", "birth date", "birthday". Numeric, ISO
  ("1983-07-19") and worded forms ("5 May 1994", "14 Feb 1979") all count.
- A format hint may sit between the label and the value ("Date of birth
  (DD/MM/YYYY): 02/09/1959").
- Issue dates, appointment dates, contract dates, hearing dates stay visible.

## Keep rules (must NOT be labeled, and must survive redaction)

- Ordinary words: "night", "morning", "evening" — except the day-part word in
  a person+clock-time phrase ("at 2 o'clock in the morning" → redacted).
- Generic job titles, laws and article numbers, document headings.
- Prices, amounts, quantities, meter readings, percentages.
- Thing-IDs that are NOT trackable to a purchase: invoice, tracking, batch,
  flight, voucher, project codes. (Serial and order numbers are REDACTED —
  rule-owned, so still never labeled for training.)
- Public places and unaffiliated organizations, including the document's own
  letterhead organization.
- Operational locations: warehouse, branch, hub, and port cities — even when a
  person interacts with them. (A city as a person's travel DESTINATION is
  redacted — rule-owned, never labeled for training.)
- Languages and nationality adjectives ("Dutch", "Indian passport") — group
  membership is a TBD policy area; V1 keeps them.
- I/me/my/you/your (pronoun policy covers only third-person forms).

## V1 policy decisions already taken (do not re-litigate per document)

1. ALL e-mail addresses redact, role inboxes (hr@, procurement@) included.
2. Company tax IDs (INN, GSTIN) redact even with no person attached.
3. he/him/his/she/her/hers always redact; they/them/their/theirs only when a
   person is detected nearby; I/me/you forms never.
4. A bare digit run is never PII; a label next to a number decides its type.
5. Residence cities redact; operational cities (warehouse, branch, hub) keep.
6. The organization that authored the document (letterhead) stays visible;
   third-party facility mentions redact (verdict on S001-class checks).
7. Serial numbers and order numbers redact — they can be searched up and
   tracked. Batch numbers stay visible (verdict on S003/S036 vs S046/S061).
8. A city as a person's travel destination redacts ("arrival Gdansk");
   warehouse/branch/hub cities stay (verdict on S014 vs S005/S006/S084).
9. A customer/member/account number given in a chat reply is that person's
   identifier even when its label sits in the previous message ("Could you
   give me the customer number?" / "Sure, CU-104772."). Labeled so in gold
   test3 (2026-11-04). Rule since 2026-11-05: in a `[hh:mm] Speaker:`
   transcript, a person-ID label (account, member, customer, client, patient,
   policy, case, claim) in the previous speaker's line makes every
   hyphenated or prefixed digit run in the reply that person's identifier.
   Order and reference numbers do not count (thing-IDs).
10. A label naming a thing directly before a span ("Vehicle:", "Make/Model:",
    "Product:", "Engine:", "Appliance:") vetoes the person reading of what
    follows, as an ID label decides a number's type (decision 4). "Hospital
    number", "staff number" and "payroll number" label a person's identifier;
    the bare words label a facility or a department. A boat's hull ID is a
    serial number; a sort code belongs with its account number. Rules since
    2026-11-05 (test4 and test5 findings).
11. A city or a capitalised word followed by an operational noun ("the Oslo
    hub", "Stavanger branch office", "the Hull depot") is a site: it stays
    visible even inside a travel or residence sentence, and it is never a
    person (decisions 5 and 8 made explicit at runtime, 2026-11-05).
12. ID labels (test6, 2026-11-07): the national-ID abbreviations DNI, NIE,
    NIF, CURP, CPF, BSN and PESEL label a person's number on their own;
    "application", "visa" and "permit" need a reference word ("APPLICATION
    REFERENCE: VAF-2026-118822"); a penalty charge notice number ("PCN
    number: EV 44021873") and a hire / tenancy / loan agreement number
    ("Agreement: 55-20931") belong to the person served or party to it; an
    employment contract number stays visible ("CONTRACT No. 44/26", dev
    gold). A labelled driving licence redacts whatever its format, as a
    labelled passport does.
13. A label that is the second item of a comma list names a field of the
    first item, not a person: "Employer, Applicant 1: Fenland Water Services"
    is an employer line (test6, 2026-11-07). Transcript lines count as chat
    whether or not the time is bracketed ("14:03 Rhys ap Gruffudd: ...").
14. Digits inside an international phone are the phone, not a checksum-only
    identifier ("612 345 678" inside "+34 612 345 678" is not an Australian
    company number). The 00 dialling prefix counts like "+" unless another
    digit group precedes it, so "0044 0532 0130 00" inside an IBAN is not a
    phone (test6, 2026-11-07).
15. Labels taken while authoring test7 (2026-11-08): a loyalty-card number
    is a customer identifier and redacts; a house name ahead of the street
    ("The Old Forge, 6 Chapel Row") is part of the address; a company under
    a role label ("Registered keeper: Lexford Leasing Ltd"), an insurer and
    a chain pharmacy ("Boots") stay visible; a supplier named as the place
    a person works ("a senior buyer at our supplier X") is that person's
    employer; a facility type with no name ("a nursing home", "sheltered
    housing") stays; a plural pronoun for an unnamed group ("the cutters
    ... they") stays, as for the staff in test6; "MIDDLE NAME: NONE" is
    not a name.
16. The label decides, both ways (test7 fixes, 2026-11-08): a digit run
    straight after an invoice, receipt, batch, tracking, SSCC, barcode,
    part, voucher, flight or project label is that thing's number and stays
    visible, whatever its shape ("Invoice no. 0042 0117 3381" is not a
    phone); a number straight after a phone word ("call 1800 555 014"), a
    plate word ("Targa / Plate: FX 482 LD") or a loyalty or membership card
    label is that person's and redacts. An identifier value is never a bare
    date, but a registry number that starts with one is still an
    identifier ("record No. 2014-03-09/0771").
17. A German-shaped plate ("KS-FT 2209") counts only where the sentence
    speaks of a vehicle or a registration, and never straight after a place
    or type label: "USB-C 3.2", "DIN-A 4", "region DE-HE 3" and "Bay B-C
    4" share the shape (test7 fixes, 2026-11-08).
18. A dated CV line ("2015–2019  Junior surveyor, Granja & Filhos") is
    employment wording: the organization after the role redacts, at its
    other mentions too, except on the letterhead. "At our supplier X" names
    an employer like "at X" (test7 fixes, 2026-11-08).

19. Labels taken while authoring test8 (2026-11-09): an identifier whose
    value is a bare date stays under any label ("Delivery note:
    2026-11-06", "The refund reference is 2026-11-03"), and a case number
    that starts with a date redacts, as in decision 16. A phone after a
    label that names a thing and a contact ("Parcel contact:", "Shipment
    phone:") is the person's phone. A gas account, a utility customer
    number, a payroll number, a meter serial number and a bicycle frame
    number redact; a meter reading and "ending 2816" stay. A plate in a
    list or a table column redacts with no vehicle wording around it; a
    standard ("DIN-EN 1090"), a device type and a project code of the same
    shape stay. A surname that is a company word is still a person ("Fr.
    Annelise Bank", "Frau Bank", "Winifred Holding"). A house name ahead of
    the street is part of the address with no house word in it ("Braeside,
    4 Station Road"). A "Via" or "Plaza" street with a house number is an
    address without a postcode; "via email" and "Plaza Hotel" are not. A
    sign-off that is the company or a department stays ("Kind regards,
    Brackenridge" under that letterhead; "Customer Accounts"), and so does
    a courtesy word ("Ta,") and a role addressee ("Hey Caretaker,"). The
    product under a bare "Name:" stays. An employer on a CV line redacts
    with no year range ("Present: team leader at X", "three years as a
    picker for X"), and so does one in capitals under "EMPLOYER:". French
    birth wording ("née le", "né le") makes a birth date; the date after
    "Fait à <town>, le" stays and the town, when it is the host's home
    town, redacts with the address. A plural pronoun redacts when it
    stands for named people ("Thabo and Lucinda agreed ... their next
    review") and stays for an unnamed group ("the surgeons ... they").
    Visitor badge numbers and a support ticket number are left unlabelled:
    the policy for them is open.

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
  against the human-gold corpus, plus the stray-span count (detections that
  touch no gold item). Tune on dev; run test at milestones only.
- `python training/evaluate_model_gold.py --model <path> --set all` — the same
  gold corpus, but the NER model alone, scored on the four model entities only.
  This is the number that decides whether a candidate can be promoted or the
  keyword gate on its spans relaxed; the full-pipeline score hides the model.
- `python training/evaluate.py --model <path> --data <jsonl>` — exact-span
  precision/recall/F1 per label on a labeled JSONL set, with every mismatch
  sorted into MISSED / SPURIOUS / BOUNDARY / LABEL.
- `python training/validate_regression_cases.py` — exact-identifier suite.
- `python training/train.py` — writes `custom_ner_candidate` next to the live
  model (training/README.md, "Model location"); never train directly into the
  live model directory, the detector auto-loads it.
  Promote a candidate only after `evaluate.py` and the gold suites pass.
