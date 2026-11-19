#!/usr/bin/env python3
"""Round 5 of the generated training corpus (2026-11-19).

Round 5 teaches the list that the test9 fix round left for the model
(training/README.md, "Next steps" as of commit 29563b5, written before test10
existed):

- a group-chat export with names in lowercase, as speakers and as mentions;
- a JSON record with snake_case keys, where a name, a birth date and each
  part of the address is a value, beside keys whose value is a plan or a
  product;
- German and Spanish forms, bilingual and not, with "Hr.", "Frau Dr.",
  "Dña.", "D.", "geb.", "geb. am", "nacido el", worded dates in both
  languages, a floor or door inside the address line, and form labels
  ("Bemerkungen", "Einzug am", "Domicilio") that are not names;
- fixed-width tables with no separator: a results table and a loan sheet
  whose "Name" column holds products and whose "Serial" column holds
  serial numbers;
- surnames that are venue words ("Mrs Chapel", "Woods and Marsh ran"),
  beside the same kind of word used for the venue itself;
- a phone list where a first name is the label, beside roles and kin words
  in the same place;
- first names in running prose with nothing around them: as the subject, in
  the possessive, in the plural, after "Ring" and "Speak to";
- medicines, which are capitalised and are not people;
- one-word replies after a speaker label ("Caller: Fine.");
- "Surname:", "Name:" and "Patronymic:" as separate fields;
- towns with a hyphen or a function word in them, inside addresses.

It deliberately does NOT draw on the test10 blind run of the same day (a
comma-separated export, camelCase JSON keys, a Dutch letter, an Italian
form, meeting minutes, a quoted e-mail thread, a match report, a US address
block, a spoken birth date, "they" for one person): test10 is the held-out
measurement for the candidate trained on this corpus. Where a genre below
meets a test10 document (a call transcript, a record with a patronymic, a
note with medicines), it was written from the list above, with other
wording, labels and layouts than the test10 document has.

The module reuses the round-3 and round-4 builders, markers and consistency
check.
"""
import string

from corpus_round3 import CARE_KINDS, MEDICAL_KINDS, consistent, render
from corpus_round4 import Builder4, pools4_for

SEED_ROUND5 = 202611195

# Pools: every entry was checked on 2026-11-19 against the gold documents
# (test10 included), the rule and test files and the private regression
# documents; an entry sharing a word with any of them was dropped. Train and
# dev stay disjoint. Keep the lists as they are: the generator draws by
# position.
TRAIN_DE_FIRST = ["Annegret", "Burkhard", "Dietmar", "Elfriede", "Friedhelm", "Gudrun",
                  "Hartmut", "Ilse", "Kerstin", "Lutz", "Mechthild", "Norbert", "Ortrud",
                  "Rüdiger", "Swantje", "Torsten", "Ulrike", "Volker", "Waltraud", "Cem",
                  "Aylin", "Milena", "Heike", "Sabine", "Holger", "Katrin", "Rainer", "Anke",
                  "Uwe", "Wiebke", "Malte", "Henrike", "Tjark", "Gesine"]
DEV_DE_FIRST = ["Irmgard", "Eckhard", "Reinhild", "Bärbel", "Gerlinde"]
TRAIN_DE_LAST = ["Schulte", "Brandstetter", "Hoffmeister", "Kirchhoff", "Oberländer",
                 "Pfeiffer", "Quandt", "Riedel", "Schwarzkopf", "Tiedemann", "Uhlig",
                 "Vogelsang", "Wittkowski", "Zöllner", "Yıldırım", "Demirci", "Ahrens",
                 "Bödeker", "Clausen", "Dierks", "Eggers", "Feldkamp", "Hinrichs", "Kruse",
                 "Altenburg", "Buchholz", "Ehlers", "Hagedorn", "Niemeyer"]
DEV_DE_LAST = ["Ostermann", "Rademacher", "Steinbrink", "Thiele", "Wohlgemuth"]
TRAIN_ES_FIRST = ["Íñigo", "Maite", "Josune", "Ander", "Edurne", "Jordi", "Núria", "Oriol",
                  "Remedios", "Fulgencio", "Amparo", "Covadonga", "Dolores", "Eusebio",
                  "Florentina", "Gumersindo", "Herminia", "Jacinto", "Leocadia", "Macarena",
                  "Nicanor", "Olalla", "Paloma", "Rosendo", "Soledad", "Teodoro", "Aitana",
                  "Yeray", "Nahia", "Encarnación"]
DEV_ES_FIRST = ["Lourdes", "Ramiro", "Casimiro", "Azucena", "Trinidad"]
TRAIN_ES_LAST = ["Aguirre", "Bengoetxea", "Carrasco", "Domínguez", "Echeverría", "Figueroa",
                 "Gallardo", "Herrero", "Ibáñez", "Jáuregui", "Lasa", "Mendizábal",
                 "Navarrete", "Ochoa", "Palacios", "Quiroga", "Rebollo", "Santamaría",
                 "Tejedor", "Ugarte", "Valverde", "Yáñez", "Zabaleta", "Arrizabalaga",
                 "Barrenetxea", "Cifuentes", "Larrazábal", "Olaizola", "Bustamante",
                 "Cisneros"]
DEV_ES_LAST = ["Montoya", "Peñalver", "Zamorano", "Escudero", "Benavides"]
# Surnames that are venue words. These are ordinary words, so the check was
# narrower: none is a word of a PERSON item in any gold set.
TRAIN_VENUE_SURNAMES = ["Woods", "Bridges", "Castle", "Temple", "Abbey", "Brook", "Marsh",
                        "Moor", "Chapel", "Towers", "Gates", "Meadows", "Rivers", "Grange",
                        "Barnes", "Camp", "Cross", "Dale", "Ferry", "Heath", "Lake", "Mead",
                        "Shaw", "Wells", "Tower", "Market", "Warren", "Chambers", "Glen",
                        "Bridge", "Wood", "Mill", "Dock", "Field", "Garden", "Hill", "Lodge",
                        "House", "Booth", "Green", "Grove", "Orchard"]
DEV_VENUE_SURNAMES = ["Pool", "Port", "Yard", "Court", "Street", "Lane"]
# (town, country): towns with a hyphen or a function word in the name.
TRAIN_HYPHEN_TOWNS = [("Southend-on-Sea", "uk"), ("Weston-super-Mare", "uk"),
                      ("Ross-on-Wye", "uk"), ("Hay-on-Wye", "uk"), ("Barrow-in-Furness", "uk"),
                      ("Shoreham-by-Sea", "uk"), ("Poulton-le-Fylde", "uk"),
                      ("Ashby-de-la-Zouch", "uk"), ("Bradford-on-Avon", "uk"),
                      ("Sutton-in-Ashfield", "uk"), ("Frinton-on-Sea", "uk"),
                      ("Burnham-on-Crouch", "uk"), ("Westgate-on-Sea", "uk"),
                      ("Leamington Spa", "uk"), ("Stratford-upon-Avon", "uk"),
                      ("Berwick-upon-Tweed", "uk"),
                      ("Baden-Baden", "de"), ("Castrop-Rauxel", "de"),
                      ("Garmisch-Partenkirchen", "de"), ("Villingen-Schwenningen", "de"),
                      ("Rheda-Wiedenbrück", "de"),
                      ("Aix-en-Provence", "fr"), ("Saint-Étienne", "fr"),
                      ("Bourg-en-Bresse", "fr"), ("Saint-Nazaire", "fr"),
                      ("Clermont-Ferrand", "fr"),
                      ("Petropavlovsk-Kamchatsky", "ru"), ("Yuzhno-Sakhalinsk", "ru"),
                      ("Kamensk-Uralsky", "ru"), ("Gus-Khrustalny", "ru"),
                      ("Naberezhnye Chelny", "ru"), ("Yoshkar-Ola", "ru"),
                      ("Bielsko-Biała", "pl"), ("Kędzierzyn-Koźle", "pl"),
                      ("Sint-Niklaas", "be"), ("Knokke-Heist", "be")]
DEV_HYPHEN_TOWNS = [("Bexhill-on-Sea", "uk"), ("Idar-Oberstein", "de"),
                    ("Mörfelden-Walldorf", "de"), ("Alcalá de Henares", "es"),
                    ("Jerez de la Frontera", "es")]
TRAIN_MEDICINES = ["Atorvastatin", "Lisinopril", "Omeprazole", "Levothyroxine", "Simvastatin",
                   "Lansoprazole", "Salbutamol", "Gabapentin", "Citalopram", "Furosemide",
                   "Naproxen", "Tramadol", "Prednisolone", "Amoxicillin", "Doxycycline",
                   "Paracetamol", "Ibuprofen", "Codeine", "Diazepam", "Clopidogrel",
                   "Losartan", "Mirtazapine", "Fluoxetine", "Tamsulosin", "Lactulose",
                   "Movicol", "Gaviscon", "Fybogel", "Candesartan", "Digoxin", "Rivaroxaban",
                   "Edoxaban", "Zopiclone", "Donepezil", "Memantine", "Co-dydramol",
                   "Co-amoxiclav", "Adcal", "Laxido", "Felodipine", "Indapamide"]
DEV_MEDICINES = ["Nitrofurantoin", "Trimethoprim", "Carbocisteine", "Quetiapine",
                 "Pregabalin"]

DE_MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
             "September", "Oktober", "November", "Dezember"]
ES_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]
DE_FLOORS = ["", "", ", 3. OG links", ", 1. OG rechts", ", EG", ", 2. OG", ", Hinterhaus",
             ", 4. OG Mitte"]
ES_DOORS = ["", "", ", 2º izq.", ", 3º dcha.", ", 1ºB", ", bajo A", ", 4º C", ", ático"]
ES_PROVINCES = ["Zaragoza", "Navarra", "Gipuzkoa", "Cantabria", "Asturias", "León", "Huesca",
                "Teruel", "Álava", "Lugo"]
DE_REMARKS = [
    "Kellerabteil {n} gehört zur Wohnung. Zählerstand Strom bei Einzug: {m} kWh.",
    "Keine. Schlüsselübergabe erfolgte am Einzugstag.",
    "Stellplatz Nr. {n} wird mitvermietet. Zählerstand Wasser: {m} m³.",
    "Untervermietung ist nicht gestattet. Zählerstand Gas: {m} m³.",
]
DE_NOTICES = [
    "Die Treppenhausreinigung findet ab sofort dienstags statt.",
    "Bitte stellen Sie keine Fahrräder im Hausflur ab.",
    "Der Zählerstand wird in der kommenden Woche abgelesen.",
    "Die Heizungsanlage wird am Donnerstag gewartet; das Warmwasser bleibt bis Mittag aus.",
    "Sperrmüll ist beim Entsorgungsbetrieb anzumelden.",
]
ES_NOTICES = [
    "El corte de agua previsto afectará a todo el edificio de 9 a 12 horas.",
    "Se recuerda que la basura debe depositarse a partir de las 20 horas.",
    "La junta de vecinos se celebrará en el portal el próximo martes.",
    "El ascensor permanecerá fuera de servicio por revisión durante dos días.",
    "Las bicicletas no pueden dejarse en el rellano.",
]
CHAT_GROUPS = ["Bin day gang", "No. 14 residents", "Allotment lot", "Sunday league dads",
               "Flat share", "School run", "Street WhatsApp", "Car share", "Book club",
               "Choir altos"]
CHAT_VEHICLES = ["silver van", "white transit", "blue estate", "grey pickup", "red hatchback",
                 "black people carrier"]
CHAT_PLACES = ["the drive", "the gate", "the bin store", "the side passage", "the garages",
               "my space"]
CHAT_CLOSINGS = ["warmly", "cheers", "ta muchly", "much love", "thx", "over and out"]
CHAT_KIN = ["brother", "cousin", "dad", "sister", "mate", "lodger", "nephew"]
ROLE_LABELS = ["Plumber", "Dentist", "Vet out of hours", "Taxi", "School office", "Garage",
               "Chemist", "Electrician", "Window cleaner", "Gas emergency", "Vets",
               "Doctor's", "Boiler man", "Council", "Hairdresser", "Minicab"]
KIN_WORDS = ["Nana", "Grandad", "Auntie", "Uncle", "Granny", "Gran", "Nan", "Cousin"]
LIST_TITLES = ["USEFUL NUMBERS — stick this on the fridge", "Numbers by the phone",
               "WHO TO RING", "Emergency contacts (babysitter copy)", "Phone list — kitchen",
               "Contacts for the house-sitter"]
ROUTER_PLACES = ["hall cupboard", "under the stairs", "study", "loft", "behind the telly"]
ACKS = ["Fine.", "Lovely.", "Sure.", "Great.", "Okay.", "Right.", "Perfect.", "Thanks.",
        "Understood.", "Brilliant.", "Noted.", "Wonderful.", "OK.", "Thank you.",
        "Certainly.", "Super.", "Got it.", "Smashing."]
ANSWERS = ["Yes.", "No.", "Fine.", "Sure.", "Okay.", "Correct.", "Yep.", "Great.", "Cheers.",
           "Thanks.", "Hello?", "Speaking.", "Morning.", "Of course.", "Go on.", "Bye.",
           "Absolutely.", "Lovely.", "Right.", "Nope.", "Please.", "Ta."]
SPEAKER_PAIRS = [("Agent", "Caller"), ("Operator", "Customer"), ("Adviser", "Caller"),
                 ("Q", "A"), ("Interviewer", "Witness"), ("Call handler", "Caller"),
                 ("Receptionist", "Patient"), ("Clerk", "Applicant")]
CALL_QUEUES = ["Repairs", "Billing", "Appointments", "Lost property", "Home insurance",
               "Parking permits", "Repeat prescriptions", "Deliveries"]
CLUB_TAILS = ["AC", "Harriers", "Striders", "Wheelers", "Runners", "Tri Club", "Swimming Club",
              "Chess Club", "Rowing Club", "Road Club"]
TABLE_SETS = [("Pos", "Bib", "Name", "Club", "Time"),
              ("No.", "Race no.", "Runner", "Team", "Finish"),
              ("Lane", "Heat", "Swimmer", "Club", "Seed time"),
              ("Board", "Seed", "Player", "Club", "Result"),
              ("Place", "Number", "Rider", "Club", "Time")]
EVENTS = ["Winter 10K", "Spring Open", "Club Championships", "Handicap 5", "Autumn Rapidplay",
          "Hill Climb", "Midsummer Relay", "New Year Gala"]
OFFICIALS = ["race director", "meet secretary", "chief timekeeper", "tournament controller",
             "results steward", "entries secretary"]
TOOL_NOUNS = ["drill", "level", "washer", "blowtorch", "sander", "ladder", "generator",
              "strimmer", "torque wrench", "tile cutter"]
STORE_KINDS = ["Tool store", "Equipment loan", "Kit room", "AV cupboard", "Lab store"]
PROSE_TITLES = ["PARISH NEWS", "CLUB NEWSLETTER", "STREET BULLETIN", "PTA ROUND-UP",
                "ALLOTMENT NOTES", "VILLAGE HALL NEWS", "CHOIR NOTES"]
PROSE_STALLS = ["cake stall", "tombola", "plant sale", "raffle", "tea urn", "book stall",
                "bottle stall", "barbecue"]
PROSE_CLOSINGS = ["Bendithion", "Until next month", "With thanks to all", "Yours in haste",
                  "Onwards", "Slán go fóill", "God bless"]
RU_DOC_TITLES = ["QUESTIONNAIRE", "EXTRACT FROM THE HOUSE REGISTER", "APPLICANT'S DATA SHEET",
                 "PERSONAL RECORD CARD", "TRANSLATION OF A DIPLOMA SUPPLEMENT",
                 "NOTARISED TRANSLATION — CERTIFICATE OF FAMILY COMPOSITION"]
DOSES = ["5 mg", "10 mg", "20 mg", "40 mg", "500 mg", "2.5 mg", "75 mg", "100 micrograms",
         "one sachet", "two puffs", "10 ml"]
SYSTEMS = ["membership system", "booking system", "CRM", "patient portal", "tenant database",
           "volunteer register", "library system"]
PLAN_NAMES = ["Gold Court Plus", "Family Saver", "Off-Peak Flex", "Student Lite",
              "Platinum Care", "Weekend Only", "Starter 12", "Silver Swim", "Unlimited Classes"]
MEMBERSHIPS = ["Family", "Single", "Junior", "Concession", "Corporate", "Day pass"]
JSON_NOTES = ["Prefers clay. {Pron} knee is still strapped, no singles until March.",
              "Asked for paper statements. {Pron} direct debit moves to the 3rd.",
              "Allergic to latex. {Pron} locker key was replaced in May.",
              "Left a voicemail twice. {Pron} renewal is due at the end of the quarter.",
              "Uses the step-free entrance. {Pron} carer attends on Thursdays."]


def pools5_for(split):
    """The round-4 pools of one split plus the round-5 additions."""
    pools = dict(pools4_for(split))
    for name in ("DE_FIRST", "DE_LAST", "ES_FIRST", "ES_LAST", "VENUE_SURNAMES",
                 "HYPHEN_TOWNS", "MEDICINES"):
        pools[name.lower()] = globals()[f"{split}_{name}"]
    return pools


def pad(marked, raw, width):
    """A table cell: the marked value, padded to the width of its raw text."""
    return marked + " " * max(2, width - len(raw))


class Builder5(Builder4):
    """Round-4 pieces plus the German, Spanish, chat and table pieces of round 5."""

    # ---- German and Spanish
    def de_full(self):
        first = self.pick(self.pools["de_first"])
        if self.rng.random() < 0.2:
            left, right = self.rng.sample(self.pools["de_last"], 2)
            last = f"{left}-{right}"
        else:
            last = self.pick(self.pools["de_last"])
        return first, last, f"{first} {last}"

    def es_full(self):
        first = self.pick(self.pools["es_first"])
        one, two = self.rng.sample(self.pools["es_last"], 2)
        last = f"{one} {two}" if self.rng.random() < 0.75 else one
        return first, last, f"{first} {last}"

    def _de_date(self, year, style=None):
        m, d = self.rng.randint(1, 12), self.rng.randint(1, 28)
        style = self.rng.randrange(2) if style is None else style
        if style == 0:
            return f"{d:02d}.{m:02d}.{year}"
        return f"{d}. {DE_MONTHS[m - 1]} {year}"

    def _es_date(self, year, style=None):
        m, d = self.rng.randint(1, 12), self.rng.randint(1, 28)
        style = self.rng.randrange(2) if style is None else style
        if style == 0:
            return f"{d:02d}/{m:02d}/{year}"
        return f"{d} de {ES_MONTHS[m - 1]} de {year}"

    def de_dob(self, style=None):
        return self._de_date(self.rng.randint(1942, 2016), style)

    def es_dob(self, style=None):
        return self._es_date(self.rng.randint(1942, 2016), style)

    def de_plain_date(self, style=None):
        return self.decoy(self._de_date(self.rng.randint(2019, 2027), style), "decoy_date")

    def es_plain_date(self, style=None):
        return self.decoy(self._es_date(self.rng.randint(2019, 2027), style), "decoy_date")

    def de_address(self):
        """(marked one-line address, town)."""
        town = self.pick(self.pools["de_towns"])
        line = (f"{self.pick(self.pools['de_streets'])} {self.rng.randint(1, 140)}"
                f"{self.pick(DE_FLOORS)}, {self.rng.randint(10000, 99999)} {town}")
        return self.A(line), town

    def es_address(self):
        town = self.pick(self.pools["es_towns"])
        line = (f"{self.pick(self.pools['es_streets'])} {self.rng.randint(1, 140)}"
                f"{self.pick(ES_DOORS)}, {self.rng.randint(10000, 52999)} {town}")
        return self.A(line), town

    def de_phone(self):
        r = self.rng
        return self.pick([f"0{r.randint(30, 989)} / {r.randint(1000, 9999)} {r.randint(1000, 9999)}",
                          f"0{r.randint(151, 179)} {r.randint(1000000, 9999999)}",
                          f"+49 {r.randint(30, 989)} {r.randint(100000, 9999999)}"])

    def es_phone(self):
        r = self.rng
        return f"{self.pick([6, 9])}{r.randint(10, 99)} {r.randint(100, 999)} {r.randint(100, 999)}"

    # ---- phones, serials, medicines, towns
    def mobile(self):
        r = self.rng
        a, c, d = r.randint(100, 999), r.randint(100, 999), r.randint(100, 999)
        return self.pick([f"07{a} {c} {d}", f"07{a} {c}{d}", f"+44 (0)7{a} {c}{d}",
                          f"07{a}-{c}-{d}", f"+44 7{a} {c} {d}"])

    def serial(self):
        r, hexd = self.rng, "0123456789ABCDEF"

        def run(n, alphabet=hexd):
            return "".join(self.pick(alphabet) for _ in range(n))
        return self.pick([f"{run(4)}-{run(4)}-{run(4)}",
                          f"{r.randint(1, 9)} {r.randint(100, 999)} {run(3)} {run(3)}",
                          f"{r.randint(1, 9)}.{r.randint(100, 999)}-{r.randint(100, 999)}.{r.randint(0, 9)}",
                          f"{r.randint(10, 99)}-{r.randint(1000, 9999)}",
                          f"SN{r.randint(100000, 99999999)}",
                          f"{run(2, string.ascii_uppercase)}{r.randint(10000, 999999)}{run(1, string.ascii_uppercase)}"])

    def med(self, dose=False):
        name = self.decoy(self.pick(self.pools["medicines"]), "decoy_medicine")
        return f"{name} {self.pick(DOSES)}" if dose else name

    def venue_surname(self):
        return self.pick(self.pools["venue_surnames"])

    def hyphen_address(self, block=False):
        """A marked address whose town has a hyphen or a function word in it."""
        town, country = self.pick(self.pools["hyphen_towns"])
        r = self.rng
        if country == "uk":
            street, code = self.uk_street(), self.uk_postcode()
            if block:
                return "\n".join([self.A(street), self.A(town), self.A(code)])
            return self.A(f"{street}, {town} {code}" if r.random() < 0.6
                          else f"{street}, {town}, {code}")
        if country == "de":
            street = f"{self.pick(self.pools['de_streets'])} {r.randint(1, 140)}"
            tail = f"{r.randint(10000, 99999)} {town}"
        elif country == "fr":
            street = f"{r.randint(1, 140)} {self.pick(self.pools['fr_streets'])}"
            tail = f"{r.randint(1000, 95999):05d} {town}"
        elif country == "ru":
            street = (f"{self.pick(self.pools['ru_streets'])} {r.randint(1, 160)}, "
                      f"apt. {r.randint(1, 180)}")
            if block:
                return "\n".join([self.A(town), self.A(street)])
            return self.A(f"{town}, {street}")
        elif country == "pl":
            street = f"{self.pick(self.pools['pl_streets'])} {r.randint(1, 140)}"
            tail = f"{r.randint(10, 99)}-{r.randint(100, 999)} {town}"
        elif country == "es":
            street = f"{self.pick(self.pools['es_streets'])} {r.randint(1, 140)}"
            tail = f"{r.randint(10000, 52999)} {town}"
        else:
            street = f"{self.pick(self.pools['nl_streets'])} {r.randint(1, 140)}"
            tail = f"{r.randint(1000, 9999)} {town}"
        if block:
            return "\n".join([self.A(street), self.A(tail)])
        return self.A(f"{street}, {tail}")


# ------------------------------------------------------------ positive genres

def r5_chat_lower(b):
    a, c, d, e = b.rng.sample(b.pools["first"], 4)
    _, _, full = b.full()
    _, last, _ = b.full()
    day = b.plain_date(0)

    def stamp():
        return f"{day}, {b.clock()} - "
    a_label = b.pick([b.P(a), f"{b.P(a)} (no {b.rng.randint(2, 40)})", b.P(a.lower()),
                      f"{b.P(a)} {b.pick(['🌻', '(work)', 'x'])}"])
    c_low, d_low, e_low = b.P(c.lower()), b.P(d.lower()), b.P(e.lower())
    kind = b.rng.randrange(3)
    if kind == 0:
        lines = [
            f"{a_label}: morning all, whose is the {b.pick(CHAT_VEHICLES)} blocking "
            f"{b.pick(CHAT_PLACES)}? reg is {b.plate()}",
            f"{b.P(full)}: Not ours. Try {c_low} at {b.rng.randint(2, 40)}, there was a hire "
            f"van there last week",
            f"{a_label}: ta. {c_low} can you move it pls, i need to get out by half 8",
            f"{b.P(c)}: sorry sorry, moving it now. its {d_low}'s van, my {b.pick(CHAT_KIN)}",
            f"{b.P(c)}: staying till sunday",
            f"{b.P(full)}: While we're all here: the landlord wants meter readings by Friday. "
            f"Send them to {b.title(english=True)} {b.P(last)} on {b.mobile()}, not to the agency",
            f"{a_label}: will do. {b.pick(CHAT_CLOSINGS)}, the grumpy one at no "
            f"{b.rng.randint(2, 40)}",
        ]
    elif kind == 1:
        lines = [
            f"{a_label}: has anyone seen a parcel for {b.rng.randint(2, 40)}? courier says "
            f"left with neighbour",
            f"{b.P(c)}: not me. ask {d_low}, the van stopped outside theirs",
            f"{b.P(full)}: {d_low} is away till tuesday. {e_low} has the spare key i think",
            f"{b.P(e)}: yep got it, its in the porch. {b.P(a.lower())} knock any time after "
            f"{b.rng.randint(4, 8)}",
            f"{a_label}: lifesaver, ta {e_low}",
            f"{b.P(full)}: Reminder that bins go out tonight, not tomorrow",
        ]
    else:
        lines = [
            f"{a_label}: can anyone swap saturday? ive got the dentist at {b.clock()}",
            f"{b.P(c)}: i can do the morning if {d_low} does the afternoon",
            f"{b.P(d)}: fine by me. tell {e_low} though, the rota is on the fridge",
            f"{a_label}: will do. thanks {c_low}, thanks {d_low}",
            f"{b.P(full)}: Rota updated. {b.P(e)} has the float and the keys; ring "
            f"{b.mobile()} if the shutter sticks",
            f"{b.P(c)}: {b.pick(CHAT_CLOSINGS)} all",
        ]
    head = b.pick([f"WhatsApp chat export — group \"{b.pick(CHAT_GROUPS)}\"",
                   f"Chat export: {b.pick(CHAT_GROUPS)}", "Exported messages"])
    return head + "\n\n" + "\n".join(stamp() + line for line in lines)


def r5_json_record(b):
    first, last, full = b.full()
    _, _, coach = b.full()
    _, _, kin = b.full()
    indent = b.pick(["  ", "    ", "\t"])
    prefix = b.pick(["member", "customer", "client", "tenant", "patient", "borrower"])
    first_key = b.pick(["first_name", "given_name", "forename"])
    last_key = b.pick(["last_name", "surname", "family_name"])
    if b.rng.random() < 0.6:
        street, town, code, country = (b.uk_street(), b.pick(b.pools["uk_towns"]),
                                       b.uk_postcode(), "GB")
    else:
        street = f"{b.pick(b.pools['nl_streets'])} {b.rng.randint(1, 140)}"
        town = b.pick(b.pools["nl_towns"])
        code = (f"{b.rng.randint(1000, 9999)} {b.pick('ABCDEGHJKLNPRSTXZ')}"
                f"{b.pick('ABCDEGHJKLNPRSTXZ')}")
        country = "NL"
    address = (f"{{\"{b.pick(['line1', 'street', 'address_line_1'])}\": \"{b.A(street)}\", "
               f"\"{b.pick(['postcode', 'postal_code', 'zip'])}\": \"{b.A(code)}\", "
               f"\"{b.pick(['city', 'town'])}\": \"{b.A(town)}\", \"country\": \"{country}\"}}")
    pron = b.pick(["Her", "His"])
    fields = [
        f"\"{prefix}_id\": \"{b.pick(string.ascii_uppercase)}{b.pick(string.ascii_uppercase)}"
        f"{b.pick(string.ascii_uppercase)}-{b.rng.randint(1000, 999999):06d}\"",
        f"\"{first_key}\": \"{b.P(first)}\"",
        f"\"{last_key}\": \"{b.P(last)}\"",
        f"\"{b.pick(['date_of_birth', 'dob', 'birth_date'])}\": \"{b.D(b.dob_value(1))}\"",
        f"\"{b.pick(['joined', 'created', 'registered_on'])}\": \"{b.plain_date(1)}\"",
        f"\"email\": \"{b.email_of(first, last, 'mailbox')}\"",
        f"\"{b.pick(['mobile', 'phone', 'contact_number'])}\": \"{b.mobile()}\"",
        f"\"address\": {address}",
        f"\"membership\": \"{b.decoy(b.pick(MEMBERSHIPS), 'decoy_field_value')}\"",
        f"\"{b.pick(['plan_name', 'tariff_name', 'product_name'])}\": "
        f"\"{b.decoy(b.pick(PLAN_NAMES), 'decoy_product')}\"",
        f"\"{b.pick(['coach', 'key_worker', 'account_manager', 'case_officer'])}\": "
        f"\"{b.P(coach)}\"",
        f"\"{b.pick(['emergency_contact', 'next_of_kin'])}\": {{\"name\": \"{b.P(kin)}\", "
        f"\"relation\": \"{b.pick(['husband', 'wife', 'daughter', 'son', 'sister', 'friend'])}\", "
        f"\"phone\": \"{b.mobile()}\"}}",
        f"\"notes\": \"{b.pick(JSON_NOTES).format(Pron=pron)}\"",
    ]
    if b.rng.random() < 0.3:  # the name as one value
        fields[1:3] = [f"\"{b.pick(['full_name', 'name'])}\": \"{b.P(full)}\""]
    body = ",\n".join(indent + field for field in fields)
    return (f"Export from the {b.pick(SYSTEMS)} (support ticket, pasted by the "
            f"{b.pick(['club secretary', 'front desk', 'duty manager', 'help desk'])}):\n\n"
            f"{{\n{body}\n}}")


def r5_de_form(b):
    first, last, full = b.de_full()
    _, _, second = b.de_full()
    _, _, third = b.de_full()
    address, town = b.de_address()
    kind = b.rng.randrange(3)
    if kind == 0:
        remark = b.pick(DE_REMARKS).format(n=f"B-EG {b.rng.randint(1, 40)}",
                                           m=f"{b.rng.randint(1, 99)} {b.rng.randint(100, 999)}")
        return (f"WOHNUNGSGEBERBESTÄTIGUNG / LANDLORD'S CONFIRMATION OF RESIDENCE\n"
                f"nach § 19 Bundesmeldegesetz\n\n"
                f"Wohnungsgeber / landlord: {b.pick(['Hr.', 'Fr.', 'Herr', 'Frau'])} {b.P(full)}, "
                f"Tel. {b.de_phone()}\n"
                f"Anschrift der Wohnung / address of the flat: {address}\n"
                f"Einzug am / moved in on: {b.de_plain_date(0)}\n\n"
                f"Meldepflichtige Personen / persons moving in:\n"
                f"1. {b.pick(['Frau Dr.', 'Herr', 'Frau', 'Herr Dr.'])} {b.P(second)}, geb. "
                f"{b.D(b.de_dob(0))} in {b.city()}\n"
                f"2. {b.P(third)}, geb. am {b.D(b.de_dob(1))}\n"
                f"Reisepass-Nr. / passport no. (Person 1): "
                f"C{b.rng.randint(1, 9)}{b.pick('FGHJKLMNPRTVWXYZ')}{b.pick('FGHJKLMNPRTVWXYZ')}"
                f"{b.rng.randint(10000, 99999)}\n"
                f"Kennzeichen des Umzugswagens / removal van plate: {b.plate()}\n\n"
                f"Bemerkungen / remarks: {remark}\n\n"
                f"{town}, den {b.de_plain_date(1)}\n"
                f"Unterschrift / signature: {b.P(b.initialled(first, last))}")
    if kind == 1:
        return (f"VOLLMACHT / POWER OF ATTORNEY\n\n"
                f"Vollmachtgeber / principal: {b.P(full)}, geboren am {b.D(b.de_dob())}\n"
                f"wohnhaft / residing at: {address}\n"
                f"Bevollmächtigte Person / authorised person: "
                f"{b.pick(['Frau', 'Herr', 'Fr.', 'Hr.'])} {b.P(second)}, geb. {b.D(b.de_dob(0))}\n"
                f"Gegenstand / subject: "
                f"{b.pick(['Abholung der Unterlagen / collection of documents', 'Ummeldung des Fahrzeugs / re-registration of the vehicle', 'Schlüsselübergabe / handover of keys'])}\n"
                f"Gültig bis / valid until: {b.de_plain_date(0)}\n"
                f"Bemerkungen / remarks: keine\n\n"
                f"Ort, Datum / place, date: {town}, {b.de_plain_date(0)}\n"
                f"Unterschrift / signature: {b.P(b.initialled(first, last))}")
    return (f"{b.decoy(b.pick(['Sportverein', 'Musikschule', 'Kleingartenverein', 'Volkshochschule']) + ' ' + b.pick(b.pools['company_stems']), 'decoy_author_org')}\n"
            f"ANMELDUNG\n\n"
            f"Name, Vorname: {b.P(f'{last}, {first}')}\n"
            f"Geburtsdatum: {b.D(b.de_dob())}\n"
            f"Anschrift: {address}\n"
            f"Telefon: {b.de_phone()}\n"
            f"Erziehungsberechtigte: {b.pick(['Frau', 'Herr'])} {b.P(second)}\n"
            f"Kurs: {b.decoy(b.pick(['Anfänger', 'Fortgeschrittene', 'Dienstagsgruppe', 'Jugend B']), 'decoy_field_value')}\n"
            f"Beitrag: {b.rng.randint(8, 60)},00 EUR monatlich\n"
            f"Bemerkungen: {b.pick(['keine', 'Allergie gegen Nüsse', 'bitte Rückruf nach 17 Uhr'])}\n\n"
            f"{town}, den {b.de_plain_date()}\n"
            f"Unterschrift: {b.P(b.initialled(first, last))}")


def r5_es_form(b):
    first, last, full = b.es_full()
    _, _, second = b.es_full()
    address, town = b.es_address()
    province = b.decoy(b.pick(ES_PROVINCES), "decoy_region")
    title = b.pick(["Dña.", "D.", "Doña", "Don", "Sra.", "Sr."])
    relative = b.pick([("hijo / son", "nacido el"), ("hija / daughter", "nacida el"),
                       ("cónyuge / spouse", "nacido el"), ("madre / mother", "nacida el")])
    dni = f"{b.rng.randint(10000000, 99999999)}-{b.pick('ABCDEFGHJKLMNPQRSTVWXYZ')}"
    if b.rng.random() < 0.55:
        return (f"VOLANTE DE EMPADRONAMIENTO / CERTIFICATE OF REGISTRATION\n\n"
                f"El Ayuntamiento certifica que en el Padrón Municipal figura inscrita la "
                f"siguiente persona:\n\n"
                f"Nombre y apellidos / full name: {title} {b.P(full)}\n"
                f"Fecha de nacimiento / date of birth: {b.D(b.es_dob())}\n"
                f"N.º de pasaporte / passport no.: "
                f"{b.pick('PXAB')}{b.pick('ABCDEFGHK')}{b.pick('ABCDEFGHK')}{b.rng.randint(100000, 999999)}\n"
                f"DNI: {dni}\n"
                f"Domicilio / address: {address} ({province})\n"
                f"Tfno.: {b.es_phone()}\n\n"
                f"Convive con / lives with: {b.pick(['D.', 'Dña.'])} {b.P(second)} ({relative[0]}), "
                f"{relative[1]} {b.D(b.es_dob(0))}.\n\n"
                f"Y para que conste, se expide el presente en {town}, a {b.es_plain_date(1)}.\n"
                f"{b.decoy(b.pick(['El Secretario General', 'La Secretaria', 'El Alcalde', 'La Jefa de Negociado']), 'decoy_team_signoff')}")
    return (f"SOLICITUD DE {b.pick(['PLAZA', 'TARJETA DE RESIDENTE', 'BECA DE COMEDOR', 'ABONO ANUAL'])}\n\n"
            f"Nombre y apellidos: {b.P(full)}\n"
            f"Fecha de nacimiento: {b.D(b.es_dob())}\n"
            f"DNI/NIE: {dni}\n"
            f"Domicilio: {address}\n"
            f"Teléfono: {b.es_phone()}\n"
            f"Persona de contacto: {title} {b.P(second)}\n"
            f"Observaciones: {b.pick(['ninguna', 'familia numerosa', 'solicita turno de tarde'])}\n\n"
            f"En {town}, a {b.es_plain_date(1)}\n"
            f"Fdo.: {b.P(b.initialled(first, last))}")


def r5_fixed_table(b):
    header = b.pick(TABLE_SETS)
    stem = b.pick(b.pools["facility_stems"])
    host = b.decoy(f"{stem} {b.pick(CLUB_TAILS)}", "decoy_team")
    widths = [b.rng.randint(5, 7), b.rng.randint(6, 9), b.rng.randint(24, 30),
              b.rng.randint(22, 26)]
    rows, people = [], []
    for index in range(b.rng.randint(4, 6)):
        first, last, full = b.full()
        if b.rng.random() < 0.35:
            last = b.venue_surname()
            full = f"{first} {last}"
        style = b.rng.randrange(5)
        if style == 0:
            raw, cell = f"{last}, {first}", b.P(f"{last}, {first}")
        elif style == 1:
            title = b.title(english=True)
            raw, cell = f"{title} {full}", f"{title} {b.P(full)}"
        elif style == 2:
            raw = b.initialled(first, last)
            cell = b.P(raw)
        else:
            raw, cell = full, b.P(full)
        people.append((first, last, full))
        club = (b.pick(["unattached", "unattached", "U/A"]) if b.rng.random() < 0.3
                else b.decoy(f"{b.pick(b.pools['facility_stems'])} {b.pick(CLUB_TAILS)}",
                             "decoy_team"))
        time = f"{b.rng.randint(15, 59)}:{b.rng.randint(0, 59):02d}"
        rows.append(str(index + 1).ljust(widths[0]) + str(b.rng.randint(1, 400)).ljust(widths[1])
                    + pad(cell, raw, widths[2]) + pad(club, club, widths[3]) + time)
    head = "".join(h.ljust(w) for h, w in zip(header, widths)) + header[4]
    _, _, official = b.full()
    one, two = people[0], people[1]
    notes = b.choose(
        lambda: f"Prizes: {b.P(one[2])} collects the trophy from the timing tent. "
                f"{b.P(one[1])} and {b.P(two[1])} ran for the county last year.",
        lambda: f"{b.P(two[2])} was swapped in on the day; the entry was in the name of "
                f"{b.P(b.first() + ' ' + two[1])}.",
        lambda: f"A protest by {b.P(two[1])} was not upheld. {b.filler()}",
    )
    return (f"{host} — {b.pick(EVENTS)}, provisional results\n\n{head}\n" + "\n".join(rows)
            + f"\n\n{notes}\nQueries to the {b.pick(OFFICIALS)}, {b.P(official)} — "
            f"{b.phone()}.")


def r5_loan_sheet(b):
    fixed = b.rng.random() < 0.5
    widths = [10, 30, 18, 26]
    header = ["Item no.", "Name", "Serial", b.pick(["Signed out by", "Borrower", "Issued to"]),
              "Due back"]
    rows = []
    for _ in range(b.rng.randint(3, 5)):
        item = f"{b.pick('TKEA')}-{b.rng.randint(1, 99):03d}"
        thing = f"{b.product()} {b.pick(TOOL_NOUNS)}"
        serial = b.serial() if b.rng.random() < 0.8 else "n/a"
        first, last, full = b.full()
        style = b.rng.randrange(5)
        if style == 0:
            raw, who = first, b.P(first)
        elif style == 1:
            raw = b.initialled(first, last)
            who = b.P(raw)
        elif style == 2:
            raw = f"{b.company()} (site team)"
            who = raw
        else:
            raw, who = full, b.P(full)
        due = f"{b.rng.randint(1, 28):02d}/{b.rng.randint(1, 12):02d}"
        if fixed:
            rows.append(item.ljust(widths[0]) + thing.ljust(widths[1]) + serial.ljust(widths[2])
                        + pad(who, raw, widths[3]) + due)
        else:
            rows.append(" | ".join([item, thing, serial, who, due]))
    head = ("".join(h.ljust(w) for h, w in zip(header, widths)) + header[4] if fixed
            else " | ".join(header))
    _, keeper_last, _ = b.full()
    return (f"{b.pick(STORE_KINDS)} — loan sheet, week {b.rng.randint(1, 52)}\n\n{head}\n"
            + "\n".join(rows)
            + f"\n\nStorekeeper this week: {b.P(b.first())}. Late returns are reported to "
            f"{b.title(english=True)} {b.P(keeper_last)} in the office; the store is locked at "
            f"{b.clock()}.\nLost key fob? Ring the store on {b.phone()}.")


def r5_venue_surname(b):
    first, venue = b.first(), b.venue_surname()
    other_venue = b.venue_surname()
    _, _, other = b.full()
    title = b.pick(["Mrs", "Mr", "Ms", "Dr", "Miss"])
    kind = b.rng.randrange(3)
    if kind == 0:
        return (f"{b.pick(PROSE_TITLES)}\n\n"
                f"The toddler group meets in the church hall on Tuesdays. Speak to {title} "
                f"{b.P(venue)} if you can help with the rota. {b.P(f'{first} {venue}')} also "
                f"holds the key to the car park gate. {b.filler()}\n\n"
                f"{b.P(venue)} and {b.P(other_venue)} ran the stall at the school fair last "
                f"year, and {b.P(other)} has offered the garden for the summer party.")
    if kind == 1:
        return (f"{b.pick(['COMMITTEE', 'CONTACT LIST', 'KEY HOLDERS'])}\n"
                f"Chair: {title} {b.P(f'{first} {venue}')}\n"
                f"Bookings: {b.P(f'{other_venue}, {b.first()}')}\n"
                f"Caretaker: {b.P(other)}\n"
                f"Venue: the old {b.pick(['mill', 'chapel', 'library', 'station', 'bank'])} on "
                f"the green\n"
                f"{b.filler()} Keys are with {title} {b.P(venue)} until {b.plain_date(5)}.")
    return (f"From: {b.P(f'{first} {venue}')}\nSubject: {b.pick(['hall hire', 'pitch booking', 'court times'])}\n\n"
            f"Hello {b.P(other.split(' ')[0])},\n\nThe hall is free on the dates you asked for. "
            f"{title} {b.P(other_venue)} will open up at {b.clock()} and lock the yard "
            f"afterwards. {b.filler()}\n\n{b.pick(['Regards', 'Best', 'Thanks'])},\n"
            f"{b.P(first)}")


def r5_phone_list(b):
    names = b.rng.sample(b.pools["first"], 7)
    _, landlord, _ = b.full()
    roles = b.rng.sample(ROLE_LABELS, 5)
    rows = [
        f"{b.P(names[0])}: {b.mobile()}",
        f"{b.decoy(roles[0], 'decoy_role_label')}: {b.phone()}",
        f"{b.decoy(roles[1], 'decoy_role_label')} — {b.phone()} (ask for {b.P(names[1])})",
        f"{b.decoy(b.pick(KIN_WORDS), 'decoy_kin_word')} {b.P(names[2])} (mobile) {b.mobile()}",
        f"{b.decoy(roles[2], 'decoy_role_label')}: {b.phone()}, option {b.rng.randint(1, 4)}",
        f"{b.decoy(roles[3], 'decoy_role_label')} {b.phone()}",
        f"Landlord: {b.title(english=True)} {b.P(landlord)}, {b.intl_phone()}",
        f"Next door, no. {b.rng.randint(1, 60)}: {b.P(names[3])} & {b.P(names[4])}, "
        f"{b.phone()}",
        f"{b.P(names[5])} (work) {b.phone()}",
        f"{b.P(names[6])} – {b.mobile()}",
        f"{b.decoy(roles[4], 'decoy_role_label')}: {b.phone()}",
    ]
    # Every row is built above, so a row left out below still has its decoys
    # recorded; they are dropped again where the text does not contain them.
    b.rng.shuffle(rows)
    chosen = rows[:b.rng.randint(6, 9)]
    tail = [f"Router ({b.pick(ROUTER_PLACES)}) S/N {b.serial()}",
            "Alarm code is in the blue tin, not written here."]
    text = f"{b.pick(LIST_TITLES)}\n\n" + "\n".join(chosen) + "\n\n" + "\n".join(tail)
    b.decoys = [(value, kind) for value, kind in b.decoys if value in text]
    return text


def r5_prose_first(b):
    n = b.rng.sample(b.pools["first"], 12)
    _, last, full = b.full()
    stem = b.pick(b.pools["saints"])
    sentences = [
        lambda: f"{b.P(n[0])} ran the {b.pick(PROSE_STALLS)} single-handed after "
                f"{b.P(n[1])}'s car broke down.",
        lambda: f"The two {b.P(n[2] + 's')} counted the takings until gone ten.",
        lambda: f"{b.P(n[3])} and {b.P(n[4])} set up the hall on Friday, and the van was "
                f"still outside when the caretaker locked up.",
        lambda: f"Please remember {b.P(full)} in your thoughts; visitors are welcome at "
                f"{b.O(b.facility_name(CARE_KINDS))} on Wednesday afternoons.",
        lambda: f"Speak to {b.title(english=True)} {b.P(last)} if you can help with the rota.",
        lambda: f"Lifts to the lunch club can be arranged through "
                f"{b.pick(['Deacon', 'Sister', 'Brother', 'Father', 'Canon'])} {b.P(n[5])} on "
                f"{b.phone()}.",
        lambda: f"Congratulations to {b.P(n[6])} and {b.P(n[7])} on the birth of {b.P(n[8])}, "
                f"born {b.D(b._date_value(b.rng.randint(2024, 2026), 5))}.",
        lambda: f"{b.P(n[9])} will send the minutes round by the weekend.",
        lambda: f"Ring {b.P(n[10])} on {b.mobile()} if you can spare an hour for the litter "
                f"pick.",
        lambda: f"Thanks go to {b.P(n[11])}, who mended the gate, and to the scouts, who "
                f"cleared the car park.",
        lambda: f"{b.P(n[0])}'s marrow took first prize again.",
        lambda: "The next issue goes to print on the first of the month. Copy to the editor, "
                "please, not to the vicar.",
    ]
    chosen = b.rng.sample(sentences, b.rng.randint(5, 7))
    paragraphs = [" ".join(s() for s in chosen[:3]), " ".join(s() for s in chosen[3:])]
    editor = b.first()
    return (f"{stem.upper()} {b.pick(PROSE_TITLES)} — {b.pick(['November', 'March', 'June', 'September'])}\n\n"
            + "\n\n".join(paragraphs)
            + f"\n\n{b.pick(PROSE_CLOSINGS)},\n{b.P(editor)} (editor)")


def r5_meds_note(b):
    first, last, full = b.full()
    _, doctor, _ = b.full()
    title = b.title(english=True)
    kind = b.rng.randrange(3)
    if kind == 0:
        return (f"{b.author_facility(['NHS Foundation Trust', 'Hospital', 'Community Hospital'])}\n"
                f"Department of {b.pick(['Cardiology', 'Rheumatology', 'Respiratory Medicine', 'Haematology'])} — "
                f"outpatient appointment\n\n"
                f"Private and confidential\n{title} {b.P(full)}\n{b.block_address()}\n\n"
                f"Dear {title} {b.P(last)},\n\n"
                f"An appointment has been made for you with Dr {b.P(b.initialled(None, doctor))} "
                f"on {b.plain_date(5)} at {b.clock()} in Clinic {b.rng.randint(1, 9)}.\n\n"
                f"Please bring a list of your medicines. If you take {b.med()}, have your blood "
                f"checked at {b.O(b.facility_name(MEDICAL_KINDS[:4]))} in the week before. "
                f"{b.med()} should not be taken on the morning of the test.\n\n"
                f"Yours sincerely,\n{b.decoy(b.pick(['Appointments Team', 'Outpatient Booking', 'Clinic Coordinator']), 'decoy_team_signoff')}")
    if kind == 1:
        items = "\n".join(f"{i}. {b.med(dose=True)}, {b.pick(['one daily', 'twice daily', 'at night', 'when required', 'with food'])}"
                          for i in range(1, b.rng.randint(3, 5) + 1))
        return (f"REPEAT PRESCRIPTION REQUEST\n\n"
                f"Patient: {b.P(full)}        {b.dob(b.pick(['DOB:', 'Date of birth:', 'Born:']))}\n"
                f"Items:\n{items}\n\n"
                f"Prescriber: Dr {b.P(doctor)}\n"
                f"Collect from: {b.company()}\n"
                f"Notes: {b.med()} was stopped on {b.plain_date(5)}; do not reissue.")
    return (f"MEDICATION ROUND — {b.pick(['early', 'late', 'night'])} shift\n\n"
            f"{b.P(first)} took {b.med(dose=True)} at {b.clock()} with breakfast. "
            f"{b.med()} was withheld because of the low reading, and {b.med()} is due again at "
            f"{b.clock()}. {b.title(english=True)} {b.P(last)} asked whether the "
            f"{b.med()} could be given as a liquid; the pharmacist will ring back.\n\n"
            f"Signed: {b.P(b.initialled())}, {b.pick(['staff nurse', 'senior carer', 'ward sister'])}")


def r5_speaker_replies(b):
    agent_label, caller_label = b.pick(SPEAKER_PAIRS)
    agent = b.first()
    first, last, full = b.full()
    _, _, holder = b.full()
    acks, answers = b.rng.sample(ACKS, 5), b.rng.sample(ANSWERS, 3)
    order = [answers[0], acks[0], acks[1], acks[2], acks[3], answers[1], answers[2], acks[4]]
    queue = b.pick(CALL_QUEUES)

    def reply(i):
        return b.decoy(order[i], "decoy_reply")
    lines = [
        f"{agent_label}: Good {b.pick(['morning', 'afternoon', 'evening'])}, "
        f"{queue.lower()} line, you're speaking to {b.P(agent)}.",
        f"{caller_label}: {reply(0)}",
        f"{agent_label}: Can I take your name?",
        f"{caller_label}: {b.P(full)}.",
        f"{agent_label}: {reply(1)} And the name on the account?",
        b.choose(lambda: f"{caller_label}: Same.",
                 lambda: f"{caller_label}: {b.P(holder)}. "
                         f"{b.pick(['My partner.', 'My landlord.', 'My mother.'])}"),
        f"{agent_label}: {reply(2)} Date of birth?",
        f"{caller_label}: {b.D(b.dob_value())}.",
        f"{agent_label}: {reply(3)} And the address?",
        f"{caller_label}: {b.A(b.line_address())}.",
        f"{agent_label}: {reply(4)} Is the mobile still {b.mobile()}?",
        f"{caller_label}: {reply(5)}",
        f"{agent_label}: I've booked that for {b.plain_date(5)}. Anything else, "
        f"{b.title(english=True)} {b.P(last)}?",
        f"{caller_label}: {reply(6)}",
        f"{agent_label}: {reply(7)}",
    ]
    return (f"CALL TRANSCRIPT — {queue}\n"
            f"Duration: 0{b.rng.randint(1, 9)}:{b.rng.randint(0, 59):02d}\n\n" + "\n".join(lines))


def r5_patronymic_form(b):
    last = b.pick(b.pools["ru_last"])
    first = b.pick(b.pools["ru_first"])
    patr = b.pick(b.pools["ru_patr"])
    town, _ = b.pick([t for t in b.pools["hyphen_towns"] if t[1] == "ru"]
                     or [(b.pick(b.pools["ru_cities"]), "ru")])
    street = f"{b.pick(b.pools['ru_streets'])} {b.rng.randint(1, 160)}, apt. {b.rng.randint(1, 180)}"
    residence = b.choose(lambda: b.A(f"{town}, {street}"),
                         lambda: b.A(b.ru_address()))
    names = b.choose(
        lambda: f"Surname: {b.P(last)}\nName: {b.P(first)}\nPatronymic: {b.P(patr)}",
        lambda: f"Surname: {b.P(last)}\nGiven name: {b.P(first)}\nPatronymic (if any): "
                f"{b.P(patr)}",
        lambda: f"Surname, name, patronymic: {b.P(f'{last} {first} {patr}')}",
        lambda: f"Family name: {b.P(last)}\nFirst name: {b.P(first)}\nPatronymic: —",
    )
    lines = [b.pick(RU_DOC_TITLES), b.pick(["Translation from Russian", "(translated from the Russian original)", ""]),
             "", names,
             f"Date of birth: {b.D(b.dob_value(2))}",
             f"Citizenship: {b.decoy('Russian Federation', 'decoy_field_value')}",
             f"Place of residence: {residence}",
             f"Family members registered at the address: {b.P(b.ru_name(0))} "
             f"({b.pick(['spouse', 'mother', 'son', 'daughter'])})",
             f"Document issued: {b.plain_date(2)}",
             "",
             f"Translator: {b.P(b.ru_name(4))}. The translation is complete and accurate."]
    return "\n".join(lines)


def r5_hyphen_town(b):
    first, last, full = b.full()
    title = b.title(english=True)
    _, author = b.stem_company()
    kind = b.rng.randrange(3)
    if kind == 0:
        return (f"{author}\n{b.pick(['Delivery confirmation', 'Account update', 'Service visit'])}\n\n"
                f"{title} {b.P(full)}\n{b.hyphen_address(block=True)}\n\n"
                f"Dear {title} {b.P(last)},\n\nYour order will be delivered on "
                f"{b.plain_date(5)} between {b.clock()} and {b.clock()}. {b.filler()}\n\n"
                f"Kind regards,\n{b.decoy('Customer Services', 'decoy_team_signoff')}")
    if kind == 1:
        return (f"CHANGE OF ADDRESS\n\nName: {b.P(full)}\n"
                f"Old address: {b.hyphen_address()}\n"
                f"New address: {b.hyphen_address()}\n"
                f"Effective from: {b.plain_date()}\n"
                f"Phone: {b.phone()}\n"
                f"Signed: {b.P(b.initialled(first, last))}")
    return (f"{b.P(full)} has moved and post should now go to {b.hyphen_address()}. "
            f"{b.P(first)} asked for the statements to be re-sent. The parcel that came back "
            f"was addressed to the old flat. {b.filler()}")


# ------------------------------------------------------------ negative genres

def r5n_formulary(b):
    rows = "\n".join(f"{b.med()}  {b.pick(DOSES)}  x{b.rng.randint(1, 12)}  "
                     f"{b.pick(['shelf A', 'shelf B', 'fridge', 'controlled cupboard', 'trolley 2'])}"
                     for _ in range(b.rng.randint(4, 7)))
    return (f"WARD STOCK CHECK — {b.plain_date(5)}\n\n{rows}\n\n"
            f"{b.med()} is on back order until {b.plain_date(5)}; use {b.med()} where the "
            f"prescriber agrees. {b.med()} must be stored below 25 degrees. Checked by the "
            f"pharmacy technician and countersigned by the nurse in charge.")


def r5n_de_notice(b):
    stem = b.pick(b.pools["company_stems"])
    author = b.decoy(f"{b.pick(['Hausverwaltung', 'Wohnungsgenossenschaft', 'Mieterverein'])} {stem}",
                     "decoy_author_org")
    first, second = b.rng.sample(DE_NOTICES, 2)
    return (f"{author}\nAUSHANG / NOTICE\n\nSehr geehrte Mieterinnen und Mieter,\n\n"
            f"{first} {second}\n\n"
            f"Betreff / subject: Hausordnung\nGültig ab / valid from: {b.de_plain_date(0)}\n"
            f"Bemerkungen / remarks: keine\nEinzug am / moved in on: —\n\n"
            f"{b.decoy(b.pick(b.pools['de_towns']), 'decoy_city')}, den {b.de_plain_date(1)}\n"
            f"Mit freundlichen Grüßen\n"
            f"{b.decoy(b.pick(['Die Hausverwaltung', 'Der Vorstand', 'Ihr Serviceteam']), 'decoy_team_signoff')}")


def r5n_es_notice(b):
    stem = b.pick(b.pools["company_stems"])
    author = b.decoy(f"{b.pick(['Comunidad de Propietarios', 'Administración de Fincas', 'Ayuntamiento de'])} {stem}",
                     "decoy_author_org")
    first, second = b.rng.sample(ES_NOTICES, 2)
    return (f"{author}\nAVISO\n\nEstimados vecinos:\n\n{first} {second}\n\n"
            f"Nombre y apellidos: —\nDomicilio: —\nObservaciones: ninguna\n\n"
            f"En {b.decoy(b.pick(b.pools['es_towns']), 'decoy_city')}, a {b.es_plain_date(1)}\n"
            f"{b.decoy(b.pick(['El Presidente', 'La Administradora', 'El Secretario General']), 'decoy_team_signoff')}")


def r5n_call_script(b):
    agent_label, caller_label = b.pick(SPEAKER_PAIRS)
    acks, answers = b.rng.sample(ACKS, 3), b.rng.sample(ANSWERS, 4)
    queue = b.pick(CALL_QUEUES)
    lines = [f"{agent_label}: Thank you for calling the {queue.lower()} line.",
             f"{caller_label}: {b.decoy(answers[0], 'decoy_reply')}",
             f"{agent_label}: Are you calling about an existing booking?",
             f"{caller_label}: {b.decoy(answers[1], 'decoy_reply')}",
             f"{agent_label}: {b.decoy(acks[0], 'decoy_reply')} Is the reference number to hand?",
             f"{caller_label}: {b.decoy(answers[2], 'decoy_reply')} It's {b.spaced_id()}.",
             f"{agent_label}: {b.decoy(acks[1], 'decoy_reply')} That is booked for {b.plain_date(5)}.",
             f"{caller_label}: {b.decoy(answers[3], 'decoy_reply')}",
             f"{agent_label}: {b.decoy(acks[2], 'decoy_reply')}"]
    return (f"TRAINING SCRIPT — {queue} (no customer data)\n\n"
            + "\n".join(lines))


def r5n_json_config(b):
    indent = b.pick(["  ", "    "])
    fields = [
        f"\"product_id\": \"{b.pick(string.ascii_uppercase)}{b.rng.randint(100, 9999)}\"",
        f"\"plan_name\": \"{b.decoy(b.pick(PLAN_NAMES), 'decoy_product')}\"",
        f"\"display_name\": \"{b.decoy(b.pick(['Night Shift Team', 'Front Desk', 'Duty Manager', 'Pool Reception']), 'decoy_role_addressee')}\"",
        f"\"first_available\": \"{b.plain_date(1)}\"",
        f"\"last_updated\": \"{b.plain_date(1)}\"",
        f"\"site_name\": \"{b.decoy(b.pick(b.pools['company_stems']) + ' ' + b.pick(['Leisure Centre', 'Depot', 'Studios']), 'decoy_place')}\"",
        f"\"device_name\": \"{b.product()}\"",
        f"\"owner\": \"{b.pick(['facilities', 'it-ops', 'front-of-house'])}\"",
        f"\"notes\": \"{b.pick(['Renewal reminders go out thirty days ahead.', 'Price held until the end of the quarter.', 'Not available at the weekend.'])}\"",
    ]
    return (f"Config pasted from the {b.pick(SYSTEMS)} (no member data):\n\n{{\n"
            + ",\n".join(indent + f for f in fields) + "\n}")


def r5n_fixed_stock(b):
    widths = [10, 30, 18]
    rows = "\n".join(
        f"{b.pick('TKEA')}-{b.rng.randint(1, 99):03d}".ljust(widths[0])
        + f"{b.product()} {b.pick(TOOL_NOUNS)}".ljust(widths[1])
        + b.serial().ljust(widths[2]) + b.pick(["in store", "on loan", "repair", "missing"])
        for _ in range(b.rng.randint(4, 6)))
    head = "Item no.".ljust(widths[0]) + "Name".ljust(widths[1]) + "Serial".ljust(widths[2]) + "Status"
    return (f"{b.pick(STORE_KINDS)} — stock check, {b.plain_date(5)}\n\n{head}\n{rows}\n\n"
            f"Items marked missing are written off after the next audit. {b.filler()}")


def r5n_chat_quiet(b):
    day = b.plain_date(0)
    lines = ["morning all, bins are out",
             "ta. is it recycling or general this week",
             "general. glass is next week",
             "lovely, cheers",
             "whoever left the bike in the hall pls move it, the cleaner cant get past",
             "will do, sorry sorry",
             "meter readings are due friday btw",
             "ok thx"]
    chosen = lines[:b.rng.randint(5, 8)]
    flats = [f"Flat {b.rng.randint(1, 30)}" for _ in range(3)]
    return (f"Chat export: {b.pick(CHAT_GROUPS)}\n\n"
            + "\n".join(f"{day}, {b.clock()} - {b.pick(flats)}: {line}" for line in chosen))


ROUND5_POSITIVE = [r5_chat_lower, r5_chat_lower, r5_json_record, r5_json_record, r5_de_form,
                   r5_de_form, r5_es_form, r5_es_form, r5_fixed_table, r5_fixed_table,
                   r5_loan_sheet, r5_venue_surname, r5_venue_surname, r5_phone_list,
                   r5_phone_list, r5_prose_first, r5_prose_first, r5_meds_note, r5_meds_note,
                   r5_speaker_replies, r5_speaker_replies, r5_patronymic_form,
                   r5_hyphen_town, r5_hyphen_town]
ROUND5_NEGATIVE = [r5n_formulary, r5n_de_notice, r5n_es_notice, r5n_call_script,
                   r5n_json_config, r5n_fixed_stock, r5n_chat_quiet, r5n_formulary]


def build_round5(rng, pools, n_docs, prefix):
    docs, made = [], 0
    while made < n_docs:
        negative = (made % 5 == 4)
        genre = rng.choice(ROUND5_NEGATIVE if negative else ROUND5_POSITIVE)
        builder = Builder5(rng, pools)
        text, spans = render(genre(builder))
        if not consistent(text, spans):
            continue  # reroll: a name collided with another piece of the text
        docs.append({"source_file": f"{prefix}_{made:03d}_{genre.__name__.split('_', 1)[1]}.txt",
                     "text": text, "spans": spans,
                     "decoys": [{"text": value, "kind": kind}
                                for value, kind in builder.decoys]})
        made += 1
    return docs
