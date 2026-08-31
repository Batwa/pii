#!/usr/bin/env python3
"""Generate the scaled, pre-labeled training corpus (fully synthetic).

Labels follow training/LABELING_GUIDE.md and cover ONLY the four model
entities: PERSON, SENSITIVE_ORGANIZATION, ADDRESS, DATE_OF_BIRTH. Everything
else (employers, standalone cities, thing-IDs, decoy dates, job titles) is
present in the text as hard negatives but deliberately unlabeled.

Guarantees enforced at build time:
- every labeled span occurs exactly once in its document (train.py refuses
  ambiguous spans);
- spans do not overlap;
- train and dev draw from disjoint name/street/city/facility pools, so dev
  measures generalization to unseen surface forms;
- roughly 1 in 5 documents carries no entities at all.

Each document also records its decoys — strings that must stay unlabeled —
so the adjudication sampler can test the negative side of the labels too.
"""
import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data" / "generated"
SEED = 20260831

# ---------------------------------------------------------------- pools
TRAIN_FIRST = [
    "Arina", "Bohdan", "Chiamaka", "Dariusz", "Elif", "Farid", "Gwen",
    "Hiroshi", "Ilona", "Jasper", "Kateryna", "Lamine", "Maren", "Nikhil",
    "Oyin", "Pavlo", "Quique", "Renata", "Soren", "Tunde", "Ulla", "Viktor",
    "Wanda", "Ximena", "Yusuf", "Zlata", "Amara", "Boris", "Celine", "Dmitri",
    "Efe", "Freya", "Gustav", "Halyna", "Imre", "Jolanta", "Kamil", "Leyla",
    "Matteo", "Nadira", "Oskar", "Petra", "Rohan", "Sanela", "Tomasz",
]
TRAIN_LAST = [
    "Abramov", "Bakker", "Ciftci", "Dudek", "Eriksen", "Farkas", "Grabowski",
    "Hansen", "Ivanova", "Jensen", "Kowal", "Lindgren", "Moreno", "Novak",
    "Okafor", "Pavlenko", "Quintero", "Rasmussen", "Sokolova", "Tkachenko",
    "Ueda", "Vasquez", "Wojcik", "Yildiz", "Zielinski", "Andersson", "Bello",
    "Chukwu", "Dvorak", "Egede", "Fialho", "Gruber", "Horvat", "Iversen",
    "Jankowska", "Keller", "Lombardi", "Melnik", "Nowicki", "Ostrowski",
]
DEV_FIRST = ["Agnes", "Bekele", "Csilla", "Dost", "Emeka", "Fanni", "Goran",
             "Hana", "Ivo", "Jarmila", "Kenji", "Livia", "Mirek", "Noor"]
DEV_LAST = ["Almasi", "Bergstrom", "Cichy", "Dahl", "Enescu", "Fodor",
            "Gajewski", "Halvorsen", "Iliev", "Jonsson", "Kral", "Lehto",
            "Madsen", "Nagy"]

TRAIN_STREETS = [
    "Aspen Court", "Birchwood Lane", "Clover Street", "Dovetail Road",
    "Elderberry Avenue", "Fennel Walk", "Garnet Row", "Hollybush Drive",
    "Iris Crescent", "Juniper Close", "Kestrel Way", "Larchfield Grove",
    "Magnolia Terrace", "Nettle Rise", "Orchard Street", "Pimlico Road",
    "Quarry Lane", "Rowanberry Avenue", "Saffron Walk", "Thistle Court",
]
DEV_STREETS = ["Umber Street", "Violet Grove", "Wisteria Lane",
               "Yarrow Close", "Zinnia Road", "Alder Rise"]
TRAIN_CITIES = [
    "Redbrook", "Stonefield", "Marlow Bay", "Eastvale", "Northmoor",
    "Willowford", "Greyharbor", "Suttonfield", "Ravenholm", "Oakmere",
    "Bramwich", "Corwen Vale",
]
DEV_CITIES = ["Fernstead", "Lynwick", "Harrowdale", "Milbourne"]
# Standalone-city decoys: never in an address, never labeled.
TRAIN_PUBLIC_CITIES = ["Rotterdam", "Katowice", "Leeds", "Lisbon", "Warsaw",
                       "Tampere", "Gdansk", "Frankfurt"]
DEV_PUBLIC_CITIES = ["Salford", "Aberdeen", "Bristol"]

TRAIN_FACILITY = [
    ("Riverbend", "Medical Center"), ("Copperfield", "Clinic"),
    ("Whitmoor", "Hospital"), ("Gladeside", "Polyclinic"),
    ("Ashgrove", "Secondary School"), ("Ferndale", "College"),
    ("Bellcastle", "University"), ("Harborlight", "Shelter"),
    ("Mossvale", "Support Center"), ("Clearwater", "Hospice"),
    ("Pinehurst", "Health Center"), ("Silverbirch", "Care Center"),
]
DEV_FACILITY = [("Thornfield", "Hospital"), ("Lakewood", "Clinic"),
                ("Ironbridge", "School"), ("Meadowbank", "University"),
                ("Quayside", "Shelter")]

EMPLOYERS = [  # decoys — never labeled (employer redaction is a rule, not a model entity)
    "Vektor Logistics LLC", "Bluecrane Analytics", "Ostrow Machining Group",
    "Palewood Furniture ApS", "Cindergate Media", "Tallgrass Retail",
    "Norvik Marine Services", "Amberline Freight", "Quantex Instruments",
]
THING_IDS = [  # decoys — thing identifiers that stay visible.
    # Serial and order numbers are NOT here: the review verdict redacts them
    # (rule-owned, so still unlabeled for training — just never a keep-decoy).
    ("invoice", "INV-2026-{}"), ("tracking number", "1Z-{}-{}"),
    ("batch", "{}"), ("flight", "LH{}"),
    ("voucher", "HTL-{}"), ("project code", "ORION-{}"),
]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


class DocBuilder:
    """Random pieces for one document, tracking labels and decoys."""

    def __init__(self, rng, pools):
        self.rng = rng
        self.pools = pools
        self.spans = []    # (text, label)
        self.decoys = []   # (text, kind) — must stay unlabeled

    def person(self):
        name = (f"{self.rng.choice(self.pools['first'])} "
                f"{self.rng.choice(self.pools['last'])}")
        self.spans.append((name, "PERSON"))
        return name

    def facility(self):
        """A third-party facility mention — labeled for the model."""
        prefix, kind = self.rng.choice(self.pools["facility"])
        name = f"{prefix} {kind}"
        if self.rng.random() < 0.25:
            name += f" No. {self.rng.randint(2, 19)}"
        self.spans.append((name, "SENSITIVE_ORGANIZATION"))
        return name

    def author_facility(self):
        """The letterhead organization — stays visible per the review verdict."""
        prefix, kind = self.rng.choice(self.pools["facility"])
        name = f"{prefix} {kind}"
        self.decoys.append((name, "decoy_author_org"))
        return name

    def plain_city(self):
        """A travel-destination city: rule-owned at runtime, no keep assertion."""
        return self.rng.choice(self.pools["public_cities"])

    def address(self):
        parts = [f"{self.rng.randint(2, 250)} {self.rng.choice(self.pools['streets'])}"]
        if self.rng.random() < 0.6:
            unit = self.rng.choice(["Apt", "apt.", "Unit", "Suite"])
            parts.append(f"{unit} {self.rng.randint(1, 60)}")
        parts.append(self.rng.choice(self.pools["cities"]))
        if self.rng.random() < 0.3:
            parts.append(f"{self.rng.choice(['IL', 'OH', 'TX', 'WA'])} "
                         f"{self.rng.randint(10000, 99999)}")
        addr = ", ".join(parts)
        self.spans.append((addr, "ADDRESS"))
        return addr

    def dob(self, label_word=None):
        y = self.rng.randint(1955, 2008)
        m, d = self.rng.randint(1, 12), self.rng.randint(1, 28)
        style = self.rng.randrange(5)
        if style == 0:
            value = f"{m:02d}/{d:02d}/{y}"
        elif style == 1:
            value = f"{d:02d}.{m:02d}.{y}"
        elif style == 2:
            value = f"{d} {MONTH_NAMES[m - 1]} {y}"
        elif style == 3:
            value = f"{MONTH_NAMES[m - 1]} {d}, {y}"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(d if d < 4 else 0, "th")
            value = f"{d}{suffix} of {MONTH_NAMES[m - 1]}, {y}"
        if label_word is None:
            label_word = self.rng.choice(["born", "born on", "DOB:",
                                          "Date of birth:"])
        self.spans.append((value, "DATE_OF_BIRTH"))
        return f"{label_word} {value}"

    def decoy_date(self):
        value = (f"{self.rng.randint(1, 12):02d}/"
                 f"{self.rng.randint(1, 28):02d}/{self.rng.randint(2024, 2027)}")
        self.decoys.append((value, "decoy_date"))
        return value

    def decoy_employer(self):
        name = self.rng.choice(EMPLOYERS)
        self.decoys.append((name, "decoy_employer"))
        return name

    def decoy_city(self):
        name = self.rng.choice(self.pools["public_cities"])
        self.decoys.append((name, "decoy_city"))
        return name

    def decoy_thing_id(self):
        kind, template = self.rng.choice(THING_IDS)
        value = template.format(self.rng.randint(10000, 99999),
                                self.rng.randint(100, 999))
        self.decoys.append((value, "decoy_thing_id"))
        return f"{kind} {value}"

    def amount(self):
        return f"{self.rng.randint(40, 9000)}.{self.rng.choice(['00', '50', '90'])}"


# ------------------------------------------------------------ genre builders
# Each returns the document text; labels/decoys accumulate inside DocBuilder.

def g_medical_intake(b):
    return (f"NEW PATIENT INTAKE\n{b.author_facility()}\n"
            f"Patient: {b.person()}, {b.dob()}.\n"
            f"Registered address: {b.address()}.\n"
            f"Intake completed {b.decoy_date()} by the duty nurse. "
            f"The next available review slot is in three weeks.")


def g_discharge(b):
    return (f"DISCHARGE SUMMARY\n{b.author_facility()}\n"
            f"The patient {b.person()} was admitted on {b.decoy_date()} and "
            f"discharged after four days. The chart lists the patient's "
            f"{b.dob(label_word='date of birth:')}.\n"
            f"Recovery is on track and no follow-up scan is required. "
            f"A copy of this summary goes to the referring practice.")


def g_hr_letter(b):
    return (f"EMPLOYMENT VERIFICATION\n"
            f"This letter confirms that {b.person()} has worked at "
            f"{b.decoy_employer()} since {b.decoy_date()} in a full-time "
            f"role. The current monthly salary is {b.amount()}.\n"
            f"Issued at the employee's request for a rental application.")


def g_contract(b):
    return (f"EMPLOYMENT CONTRACT\nEmployer: {b.decoy_employer()}.\n"
            f"Employee: {b.person()}, {b.dob()}, residing at {b.address()}.\n"
            f"Probation lasts three months and either side may end the "
            f"contract with two weeks of notice during it.")


def g_invoice(b):
    return (f"TAX INVOICE\nIssued: {b.decoy_date()} under {b.decoy_thing_id()}.\n"
            f"Buyer: {b.person()}, {b.address()}.\n"
            f"Total {b.amount()} including VAT, payable within 30 days. "
            f"Goods remain seller property until paid in full.")


def g_lease(b):
    return (f"RESIDENTIAL LEASE\nLandlord: {b.person()}. "
            f"Tenant: {b.person()}.\n"
            f"Premises: {b.address()}. Monthly rent {b.amount()}.\n"
            f"The deposit equals one month of rent and is held in a "
            f"government-approved scheme for the whole term.")


def g_court(b):
    return (f"COURT NOTICE\nRespondent: {b.person()}, {b.dob()}, "
            f"residing at {b.address()}.\n"
            f"The hearing is set for {b.decoy_date()} under Article "
            f"{b.rng.randint(100, 400)} of the Civil Procedure Code. "
            f"Bring originals of every payment document.")


def g_school(b):
    return (f"PARENT NOTIFICATION\n{b.author_facility()}\n"
            f"Student: {b.person()}, {b.dob()}.\n"
            f"The term begins {b.decoy_date()} and consent forms are due one "
            f"week earlier. The school office answers questions on weekdays.")


def g_claim(b):
    return (f"INSURANCE CLAIM\nClaimant: {b.person()} of {b.address()}.\n"
            f"Outpatient treatment at {b.facility()} is covered by the policy. "
            f"The assessor visits on {b.decoy_date()} in the afternoon. "
            f"Estimated repair cost {b.amount()}. Repairs may start only "
            f"after written approval from the insurer.")


def g_chat(b):
    person = b.person()
    return (f"SUPPORT TRANSCRIPT\n[10:0{b.rng.randint(1, 9)}] {person}: "
            f"My {b.decoy_thing_id()} still shows no movement.\n"
            f"[10:1{b.rng.randint(0, 9)}] Support: Thanks for waiting. The "
            f"parcel left the {b.decoy_city()} warehouse on Tuesday and a new "
            f"link is on the way to you today.")


def g_email(b):
    person = b.person()
    return (f"Subject: quarterly figures\n"
            f"Team, the audit files are due Friday. {person} said the "
            f"warehouse totals will be ready tonight.\n"
            f"Compare the totals against the checklist from "
            f"{b.decoy_employer()} before anything goes out.")


def g_incident(b):
    return (f"INCIDENT REPORT\nReported by: {b.person()}.\n"
            f"A forklift clipped rack {b.rng.randint(3, 30)} during the "
            f"evening shift on {b.decoy_date()}. Nobody was hurt and the "
            f"aisle reopened after inspection the next day.")


def g_utility(b):
    return (f"ELECTRICITY BILL\nAccount holder: {b.person()}\n"
            f"Supply address: {b.address()}\n"
            f"Consumption {b.rng.randint(120, 600)} kWh, amount due "
            f"{b.amount()} by {b.decoy_date()}. Late payments add a fee.")


def g_bank(b):
    return (f"NOTICE OF ACCOUNT CHANGE\nCustomer: {b.person()}, {b.dob()}.\n"
            f"The replacement card arrives at our {b.decoy_city()} branch "
            f"within ten working days. Bring photo identification to collect "
            f"it at the counter.")


def g_travel(b):
    return (f"TRAVEL ITINERARY\nTraveler: {b.person()}.\n"
            f"Outbound {b.decoy_thing_id()} departs {b.decoy_date()}, "
            f"arrival {b.plain_city()}. Baggage allowance is one checked "
            f"bag. Check-in opens 48 hours before departure.")


def g_shelter(b):
    return (f"CONFIDENTIAL INTAKE\n{b.author_facility()}\n"
            f"Resident: {b.person()}. Caseworker: {b.person()}.\n"
            f"A safety-planning session happens within 72 hours of arrival. "
            f"Access to this sheet is restricted to assigned staff.")


def g_referral(b):
    return (f"REFERRAL LETTER\nThe patient {b.person()}, {b.dob()}, is "
            f"referred to {b.facility()} for further assessment.\n"
            f"Recent lab results travel with this letter. Book the first "
            f"appointment within two weeks of receipt.")


def g_registration(b):
    return (f"RESIDENT REGISTRATION\nApplicant: {b.person()}, {b.dob()}.\n"
            f"The applicant remains attached to {b.facility()}. "
            f"New address: {b.address()} as of {b.decoy_date()}.\n"
            f"The certificate is collected in person and remains valid "
            f"until the next change of residence.")


# Negative genres: no labeled entities at all.

def g_bulletin(b):
    return (f"FACILITY BULLETIN\nThe cafeteria switches to the summer menu "
            f"on Monday. Elevator maintenance runs overnight; use the west "
            f"staircase during the night window.\n"
            f"The annual fire drill lands on the last Friday of the month. "
            f"Questions go to the facilities desk.")


def g_press(b):
    return (f"PRESS RELEASE\n{b.decoy_employer()} opens a distribution hub "
            f"near the {b.decoy_city()} port terminal after investing "
            f"{b.amount()} thousand.\n"
            f"Recruitment for {b.rng.randint(40, 200)} warehouse roles "
            f"starts in the autumn through the public careers portal.")


def g_delivery(b):
    return (f"DELIVERY NOTE\nShipment {b.decoy_thing_id()} holds "
            f"{b.rng.randint(6, 40)} boxes of ceramic tile.\n"
            f"Dispatched from the {b.decoy_city()} warehouse on the night "
            f"route; expected transit time {b.rng.randint(12, 60)} hours. "
            f"No signature is required at drop-off.")


def g_minutes(b):
    return (f"MEETING MINUTES — {b.decoy_thing_id()}\n"
            f"The quarterly spend of {b.amount()} was ratified and a "
            f"follow-up audit of the logistics contract was requested "
            f"before {b.decoy_date()}.\n"
            f"Minutes go to the distribution list this week.")


def g_cv(b):
    return (f"CURRICULUM VITAE\nCandidate: {b.person()}, {b.dob()}.\n"
            f"Education: {b.facility()}, diploma with honors.\n"
            f"Employment: planner at {b.decoy_employer()} since {b.decoy_date()}. "
            f"Languages: two, fluent. References on request.")


POSITIVE_GENRES = [g_cv, g_medical_intake, g_discharge, g_hr_letter, g_contract,
                   g_invoice, g_lease, g_court, g_school, g_claim, g_chat,
                   g_email, g_incident, g_utility, g_bank, g_travel,
                   g_shelter, g_referral, g_registration]
NEGATIVE_GENRES = [g_bulletin, g_press, g_delivery, g_minutes]


def unique_spans_ok(text, spans):
    seen = set()
    for value, _ in spans:
        if value in seen or text.count(value) != 1:
            return False
        seen.add(value)
    return True


def build_split(rng, pools, n_docs, prefix):
    docs, made = [], 0
    while made < n_docs:
        negative = (made % 5 == 4)  # every fifth document carries no entities
        genre = rng.choice(NEGATIVE_GENRES if negative else POSITIVE_GENRES)
        builder = DocBuilder(rng, pools)
        text = genre(builder)
        if not unique_spans_ok(text, builder.spans):
            continue  # collision — reroll with fresh random pieces
        docs.append({
            "source_file": f"{prefix}_{made:03d}_{genre.__name__[2:]}.txt",
            "text": text,
            "spans": [{"text": value, "label": label}
                      for value, label in builder.spans],
            "decoys": [{"text": value, "kind": kind}
                       for value, kind in builder.decoys],
        })
        made += 1
    return docs


def main():
    rng = random.Random(SEED)
    train_pools = {"first": TRAIN_FIRST, "last": TRAIN_LAST,
                   "streets": TRAIN_STREETS, "cities": TRAIN_CITIES,
                   "facility": TRAIN_FACILITY,
                   "public_cities": TRAIN_PUBLIC_CITIES}
    dev_pools = {"first": DEV_FIRST, "last": DEV_LAST,
                 "streets": DEV_STREETS, "cities": DEV_CITIES,
                 "facility": DEV_FACILITY, "public_cities": DEV_PUBLIC_CITIES}

    train = build_split(rng, train_pools, 400, "gen_train")
    dev = build_split(rng, dev_pools, 40, "gen_dev")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, docs in (("train", train), ("dev", dev)):
        with open(OUT_DIR / f"{name}.jsonl", "w") as f:
            for doc in docs:
                record = {k: doc[k] for k in ("source_file", "text", "spans")}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(OUT_DIR / f"{name}_decoys.jsonl", "w") as f:
            for doc in docs:
                f.write(json.dumps({"source_file": doc["source_file"],
                                    "text": doc["text"],
                                    "decoys": doc["decoys"]},
                                   ensure_ascii=False) + "\n")

    n_spans = sum(len(d["spans"]) for d in train + dev)
    by_label = {}
    for d in train + dev:
        for s in d["spans"]:
            by_label[s["label"]] = by_label.get(s["label"], 0) + 1
    print(f"wrote {len(train)} train + {len(dev)} dev docs, "
          f"{n_spans} spans -> {OUT_DIR}")
    for label, count in sorted(by_label.items()):
        print(f"  {label:24} {count}")


if __name__ == "__main__":
    main()
