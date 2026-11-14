#!/usr/bin/env python3
"""Round 4 of the generated training corpus (2026-11-13).

Round 4 teaches what the test8 blind run showed and round 3 left out on
purpose (training/README.md, "test8 blind run"):

- French documents: an attestation, a letter with an address block, an
  informal e-mail and a school certificate, with "Mme", "M." and "Mlle",
  "né le" / "née le", French month names and "Fait à <town>, le <date>";
- table cells: names, hosts and plates in pipe, tab and semicolon tables,
  beside tables whose "Name" column holds products;
- a product and a person under the same bare "Name:" label;
- two fields on one line with a column gap between them;
- address blocks with the house, the street, the town and the postcode on
  their own lines (each line is its own ADDRESS span, as in decision 21 of
  the labeling guide);
- surnames that are company or brand words, behind a title and alone;
- a board of short messages with greeting and closing words of every kind,
  where one message is signed by the company and one is addressed to a role;
- streets with a function word and house names with no house word;
- block-capital form values that are not names;
- more facility mentions in running prose.

It deliberately does NOT draw on the test9 blind run of the same day (a chat
export with lowercase names, a JSON record, a German and a Spanish form,
tables without separators, venue-word surnames, serial columns): test9 is
the held-out measurement for the candidate trained on this corpus. The
genres below were written from the test8 brief alone.

The module reuses the round-3 builder, markers and consistency check.
"""
import string

from corpus_round3 import (Builder3, CARE_KINDS, EDUCATION_KINDS, MEDICAL_KINDS, RELATIONS,
                           ROLE_ADDRESSEES, SCHOOL_KINDS, STREET_TYPES, TEAM_SIGNOFFS,
                           consistent, pools_for, render)

SEED_ROUND4 = 202611134

# Pools: every entry was checked on 2026-11-13 against the gold documents
# (test9 included), the rule and test files and the pools of rounds 1 to 3;
# an entry sharing a word with any of them was dropped. Train and dev stay
# disjoint. Keep the lists as they are: the generator draws by position.
TRAIN_FR_FIRST = ["Maëlle", "Thibault", "Océane", "Gaspard", "Solène", "Bastien", "Capucine",
                  "Corentin", "Léonie", "Timothée", "Agathe", "Rémi", "Clotilde", "Fabrice",
                  "Ghislaine", "Yannick", "Mathilde", "Aurélien", "Bérénice", "Aminata",
                  "Yasmine", "Nolwenn", "Gwenaël", "Hortense", "Maxence", "Perrine",
                  "Séverine", "Tanguy", "Eulalie", "Honorine", "Kévin", "Laurène", "Mickaël",
                  "Noémie", "Ondine", "Pascaline", "Ludivine", "Anselme", "Guénaëlle"]
DEV_FR_FIRST = ["Élodie", "Quentin", "Sidonie", "Loïc", "Fantine"]
TRAIN_FR_LAST = ["Lefèvre", "Girard", "Blanchard", "Chevalier", "Dumont", "Garnier", "Lemoine",
                 "Perrin", "Renard", "Besson", "Charpentier", "Deschamps", "Guillot", "Hamel",
                 "Jacquet", "Le Gall", "Morvan", "Benali", "N'Diaye", "Picard", "Barbier",
                 "Colas", "Delaunay", "Ferrand", "Gauthier", "Hubert", "Joubert", "Lacombe",
                 "Maillard", "Pichon", "Rivière", "Sauvage", "Tessier", "Vidal", "Bazin",
                 "Montagne", "Bertin", "Courtois", "Lebrun"]
DEV_FR_LAST = ["Collin", "Bouvier", "Texier", "Poulain", "Jourdain"]
TRAIN_FR_TOWNS = ["Angers", "Quimper", "Albi", "Niort", "Colmar", "Vannes", "Périgueux",
                  "Bayonne", "Chartres", "Arras", "Nevers", "Annecy", "Poitiers", "Mâcon",
                  "Cahors", "Épinal", "Tarbes", "Brive", "Montauban", "Roanne"]
DEV_FR_TOWNS = ["Auxerre", "Laval", "Saintes"]
TRAIN_FR_STREETS = ["rue des Acacias", "rue du Moulin", "avenue de la Gare",
                    "boulevard Carnot", "impasse des Tilleuls", "chemin du Lavoir",
                    "place du Marché", "rue Pasteur", "allée des Cerisiers", "rue Jean Jaurès",
                    "quai de la Loire", "rue de l'Église", "avenue Gambetta",
                    "rue des Remparts", "chemin des Sources", "cours Lafayette",
                    "rue de la Poste", "rue des Lavandières", "rue du Puits",
                    "avenue des Tamaris"]
DEV_FR_STREETS = ["rue des Écoles", "avenue Foch", "chemin des Vignes"]
# Surnames that are also company, trade or brand words.
TRAIN_COMPANY_SURNAMES = ["Steel", "Glass", "Mills", "Marks", "Coates", "Brewer", "Carter",
                          "Fisher", "Lloyd", "Barclay", "Ford", "Lever", "Krupp", "Benz",
                          "Renault", "Handel", "Silver", "Stock", "Farmer", "Cook", "Burton",
                          "Boyle", "Banker", "Trader", "Cash", "Penny", "Frost", "Marsden",
                          "Laing", "Wimpey", "Jewson"]
DEV_COMPANY_SURNAMES = ["Sparks", "Dealer", "Banks", "Merchant", "Bond"]
# House names with no house word in them.
TRAIN_BARE_HOUSES = ["Greystones", "Windyridge", "Tigh na Mara", "Craigard", "Belvedere",
                     "Sunnyside", "Ashgrove", "Lyndhurst", "Westholme", "Inglenook",
                     "Thornlea", "Seaview", "Woodside", "Brooklands", "Carn Brea", "Penmoor",
                     "Longacre", "Tregarth", "Avalon", "Shangri-La", "Beechwood", "Whinfield",
                     "Ardmore", "Cragside", "Fernbank", "Pendower", "Mon Repos", "Braemar"]
DEV_BARE_HOUSES = ["Kilmory", "Eastholme", "Tall Trees", "Kelvinside"]

FR_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
             "septembre", "octobre", "novembre", "décembre"]
FR_NATIONALITIES = ["française", "malienne", "sénégalaise", "belge", "marocaine",
                    "ivoirienne", "tunisienne", "portugaise", "roumaine", "libanaise"]
FR_FACILITY_LEADS = ["Lycée", "Collège", "École", "Clinique", "Hôpital", "Crèche",
                     "Institut"]
FR_SCHOOL_LEADS = ["Lycée", "Collège", "École", "Institut"]
FR_ROLES = ["Responsable de clientèle", "Gestionnaire de dossiers", "Chargée de location",
            "Service des admissions", "Conseiller principal", "Secrétariat"]
FR_BODIES = [
    "Nous accusons réception de votre dossier et vous en remercions.",
    "Votre demande a bien été enregistrée ; aucune pièce ne manque.",
    "Le relevé des compteurs aura lieu comme prévu, entre 9 h et 12 h.",
    "Votre contrat est renouvelé aux mêmes conditions pour une durée d'un an.",
]
FR_NOTICES = [
    "L'ascenseur sera à l'arrêt pour entretien pendant deux jours.",
    "Les encombrants doivent être déposés dans le local prévu à cet effet.",
    "L'eau sera coupée de 9 h à 11 h pour des travaux sur la colonne.",
    "Les vélos ne doivent pas rester dans le hall d'entrée.",
    "La réunion annuelle se tiendra dans la salle du rez-de-chaussée.",
]
FUNCTION_WORDS = ["by", "on", "in", "le", "upon", "under", "next", "cum", "on the"]
BOARD_GREETINGS = ["Hiya", "Morning", "Evening", "Afternoon", "Hello", "Hi", "Hey", "Heya",
                   "Hallo", "Yo", "Dear", "Good morning", "Greetings"]
BOARD_CLOSINGS = ["Cheers", "Ta", "All the best", "Thanks", "Many thanks", "Speak soon",
                  "Take care", "Best", "Warm regards", "See you Monday", "Thanks again",
                  "Kind regards", "Regards", "Yours", "Much appreciated", "Ta very much"]
BOARD_MESSAGES = [
    "the plumber is booked for Monday at 10", "your parcel is behind the meter cupboard",
    "the bin store will be jet-washed on Saturday morning", "the AGM papers are attached",
    "the light on the third-floor landing is out again", "the lift engineer comes on Thursday",
    "the gate code changes at the weekend", "the gutters are being cleared next week",
    "the fire alarm test is at eleven", "the window cleaner needs the side gate open",
    "the minibus leaves from the car park at eight", "the kit bags are in the store room",
    "the hall is booked for the quiz night", "the keys are with the caretaker",
]
BOARD_KINDS = ["tenants' board", "staff board", "club noticeboard", "residents' board",
               "members' board", "allotment noticeboard"]
CAPS_NON_NAMES = [("JOINT APPLICANT", "NOT APPLICABLE"), ("GUARANTOR", "TO BE CONFIRMED"),
                  ("NEXT OF KIN", "AS ABOVE"), ("PETS", "ONE CAT"), ("PETS", "TWO DOGS"),
                  ("SMOKER", "NO"), ("CURRENT LANDLORD", "HOUSING ASSOCIATION"),
                  ("CURRENT LANDLORD", "PRIVATE LANDLORD"), ("PREVIOUS NAMES", "NONE"),
                  ("SECOND REFEREE", "SEE ATTACHED"), ("GP", "NOT REGISTERED"),
                  ("VEHICLE", "NONE"), ("CO-SIGNER", "NOT REQUIRED"),
                  ("AGENT", "SELF"), ("MARITAL STATUS", "SINGLE"),
                  ("OCCUPANTS", "TWO ADULTS"), ("PARKING", "ON STREET")]
TABLE_NAME_HEADERS = ["Name", "Visitor", "Driver", "Attendee", "Member", "Tenant",
                      "Volunteer", "Guest", "Contractor"]
TABLE_HOST_HEADERS = ["Host", "Seen by", "Signed in by", "Checked by", "Contact",
                      "Escort", "Booked by"]
TABLE_SITES = ["Data Centre", "Works", "Depot", "Studios", "Business Park", "Boatyard",
               "Distribution Centre", "Sports Hall"]
NON_PERSON_CELLS = ["on foot", "n/a", "bicycle", "taxi", "none", "pool car", "—"]
MECHANIC_NOUNS = ["mechanic", "fitter", "engineer", "technician", "installer", "surveyor"]
PART_NOUNS = ["fork", "pump", "motor", "panel", "valve", "hinge", "cassette", "gearbox",
              "heat exchanger", "control board"]
FAULTS = ["the lockout lever snapped on the first ride after the service",
          "the unit trips the fuse after ten minutes",
          "the door seal split within a month",
          "the display stays blank after charging overnight",
          "the handle came loose on the second use"]
COURSES = ["HNC Supply Chain Management", "NVQ Level 3 Business Administration",
           "BTEC Engineering", "Diploma in Health and Social Care", "City & Guilds 2365",
           "Access to Nursing", "A levels in Maths and Physics"]
CV_ROLES = ["picker", "driver's mate", "kitchen porter", "night packer", "receptionist",
            "trainee fitter", "stock controller", "delivery driver"]
TEAM_NAMES = ["Harriers Juniors", "Under 12s", "Colts", "Wanderers Youth", "Rovers Girls",
              "Academy Squad"]


def pools4_for(split):
    """The round-3 pools of one split plus the round-4 additions."""
    pools = dict(pools_for(split))
    for name in ("FR_FIRST", "FR_LAST", "FR_TOWNS", "FR_STREETS", "COMPANY_SURNAMES",
                 "BARE_HOUSES"):
        pools[name.lower()] = globals()[f"{split}_{name}"]
    return pools


class Builder4(Builder3):
    """Round-3 pieces plus the French, table and block pieces of round 4."""

    def choose(self, *variants):
        """Build one of several variants; the others record no decoys."""
        return self.pick(variants)()

    # ---- French
    def fr_full(self):
        first = self.pick(self.pools["fr_first"])
        if self.rng.random() < 0.25:
            left, right = self.rng.sample(self.pools["fr_last"], 2)
            last = f"{left}-{right}"
        else:
            last = self.pick(self.pools["fr_last"])
        return first, last, f"{first} {last}"

    def _fr_date(self, year, style=None):
        m, d = self.rng.randint(1, 12), self.rng.randint(1, 28)
        style = self.rng.randrange(3) if style is None else style
        if style == 0:
            return f"{d:02d}/{m:02d}/{year}"
        day = "1er" if d == 1 else str(d)
        if style == 1:
            return f"{day} {FR_MONTHS[m - 1]} {year}"
        return f"{d:02d}.{m:02d}.{year}"

    def fr_dob(self):
        return self._fr_date(self.rng.randint(1942, 2012))

    def fr_plain_date(self, style=None):
        return self.decoy(self._fr_date(self.rng.randint(2019, 2027), style), "decoy_date")

    def fr_street_line(self):
        return (f"{self.rng.randint(1, 140)}{self.pick(['', '', '', ' bis'])} "
                f"{self.pick(self.pools['fr_streets'])}")

    def fr_town_line(self):
        return f"{self.rng.randint(1000, 95999):05d} {self.pick(self.pools['fr_towns'])}"

    def fr_phone(self):
        r = self.rng
        return (f"0{r.randint(1, 7)} " + " ".join(f"{r.randint(0, 99):02d}" for _ in range(4)))

    def fr_facility(self, leads=FR_FACILITY_LEADS):
        return f"{self.pick(leads)} {self.pick(self.pools['facility_stems'])}"

    # ---- things
    def plate(self):
        r, letters = self.rng, "ABCDEFGHJKLMNPRSTVWXYZ"

        def run(n):
            return "".join(self.pick(letters) for _ in range(n))
        return self.pick([
            f"{run(2)}{r.randint(10, 74)} {run(3)}",
            f"{run(r.randint(1, 3))}-{run(2)} {r.randint(100, 9999)}",
            f"{r.randint(1000, 9999)} {run(3)}",
            f"{r.randint(1, 9)}-{run(3)}-{r.randint(10, 99)}",
            f"{run(2)}-{r.randint(100, 999)}-{run(2)}",
        ])

    def code(self):
        """A standard, a device type or a project code shaped like a plate."""
        r, letters = self.rng, "ABCDEFGHIKLMNPRSTVWXZ"
        left = "".join(self.pick(letters) for _ in range(r.randint(2, 3)))
        right = "".join(self.pick(letters) for _ in range(2))
        return self.decoy(f"{left}-{right} {r.randint(10, 9999)}", "decoy_code")

    def spaced_id(self):
        r = self.rng
        return " ".join(str(r.randint(1000, 9999)) for _ in range(r.randint(2, 3)))

    def stem_company(self):
        """(stem, "Stem Tail") of a company that authors the document."""
        stem = self.pick(self.pools["company_stems"])
        tail = self.pick(["Lettings", "Couriers", "Energy", "Broadband", "Cycles",
                          "Property Management", "Housing", "Sports Club", "Motors",
                          "Occupational Health Ltd", "Removals", "Garden Machinery"])
        return stem, self.decoy(f"{stem} {tail}", "decoy_author_org")

    # ---- addresses
    def bare_house(self):
        return self.pick(self.pools["bare_houses"])

    def function_street(self):
        left, right = self.rng.sample(self.pools["street_stems"], 2)
        return (f"{self.rng.randint(1, 120)} {left} {self.pick(FUNCTION_WORDS)} {right} "
                f"{self.pick(STREET_TYPES)}")

    def street4(self):
        return self.function_street() if self.rng.random() < 0.35 else self.uk_street()

    def line_address(self):
        """A one-line UK address, often with a bare house name or a function word."""
        parts = []
        roll = self.rng.random()
        if roll < 0.35:
            parts.append(self.bare_house())
        elif roll < 0.5:
            parts.append(f"Flat {self.rng.randint(1, 30)}")
        parts.append(self.street4())
        parts.append(f"{self.pick(self.pools['uk_towns'])} {self.uk_postcode()}")
        return ", ".join(parts)

    def block_address(self):
        """An address block: every line is its own marked ADDRESS span."""
        lines = []
        roll = self.rng.random()
        if roll < 0.2:
            lines.append(self.A(self.bare_house()))
        elif roll < 0.4:
            lines.append(self.A(f"Flat {self.rng.randint(1, 30)}, "
                                f"{self.pick(self.pools['street_stems'])} "
                                f"{self.pick(['Court', 'House', 'Mansions', 'Lodge'])}"))
        lines.append(self.A(self.street4()))
        town = self.pick(self.pools["uk_towns"])
        if self.rng.random() < 0.7:
            lines.extend([self.A(town), self.A(self.uk_postcode())])
        else:
            lines.append(self.A(f"{town} {self.uk_postcode()}"))
        return "\n".join(lines)

    def company_surname(self):
        return self.pick(self.pools["company_surnames"])


# ------------------------------------------------------------ positive genres

def r4_fr_attestation(b):
    first, last, full = b.fr_full()
    _, _, guest = b.fr_full()
    female = b.rng.random() < 0.5
    title, born = ("Mme", "née le") if female else ("M.", "né le")
    guest_title = b.pick(["M.", "Mme", "Mlle"])
    guest_born = "né le" if guest_title == "M." else "née le"
    french, english = b.pick([("ATTESTATION D'HÉBERGEMENT", "PROOF OF ACCOMMODATION"),
                              ("ATTESTATION SUR L'HONNEUR", "SWORN STATEMENT"),
                              ("DÉCLARATION DE DOMICILE", "DECLARATION OF RESIDENCE")])
    bilingual = b.rng.random() < 0.5
    town_line = b.fr_town_line()
    town = town_line.split(" ", 1)[1]
    lead = f"Je soussigné{'e' if female else ''}" + (" / I, the undersigned" if bilingual else "")
    host = "certifie héberger à mon domicile" + (" / certify that I host at my home"
                                                 if bilingual else "")
    lines = [f"{french} / {english}" if bilingual else french, "",
             f"{lead}, {title} {b.P(full)}, {born} {b.D(b.fr_dob())}, demeurant au "
             f"{b.A(b.fr_street_line() + ', ' + town_line)}, {host} :", "",
             f"{guest_title} {b.P(guest)}, {guest_born} {b.D(b.fr_dob())}, de nationalité "
             f"{b.decoy(b.pick(FR_NATIONALITIES), 'decoy_nationality')}, depuis le "
             f"{b.fr_plain_date(1)}.", ""]
    if b.rng.random() < 0.5:
        lines.append(f"Tél. : {b.fr_phone()}")
    lines.append(f"Fait à {town}, le {b.fr_plain_date(1)}")
    lines.append(f"Signature : {b.P(b.initialled(first, last))}")
    return "\n".join(lines)


def r4_fr_letter(b):
    first, last, full = b.fr_full()
    _, _, adviser = b.fr_full()
    _, _, writer = b.fr_full()
    civil = b.pick(["Madame", "Monsieur"])
    stem = b.pick(b.pools["company_stems"])
    author = b.decoy(f"{b.pick(['Cabinet', 'Agence', 'Groupe', 'Mutuelle'])} {stem}",
                     "decoy_author_org")
    town = b.decoy(b.pick(b.pools["fr_towns"]), "decoy_city")
    return (f"{author}\n{town}, le {b.fr_plain_date(1)}\n\n"
            f"{civil} {b.P(full)}\n{b.A(b.fr_street_line())}\n{b.A(b.fr_town_line())}\n\n"
            f"{civil},\n\n{b.pick(FR_BODIES)} Votre "
            f"{b.pick(['conseillère, Mme', 'conseiller, M.', 'interlocutrice, Mlle'])} "
            f"{b.P(adviser)}, vous contactera avant le {b.fr_plain_date(1)}.\n\n"
            f"Veuillez agréer, {civil}, l'expression de nos salutations distinguées.\n\n"
            f"{b.P(writer)}\n{b.decoy(b.pick(FR_ROLES), 'decoy_job_title')}")


def r4_fr_email(b):
    to_first, other, sender = b.rng.sample(b.pools["fr_first"], 3)
    _, last, _ = b.fr_full()
    body = b.choose(
        lambda: f"Le dossier est complet. {b.P(other)} passera prendre les clés jeudi vers "
                f"{b.clock()}.",
        lambda: f"Peux-tu vérifier le planning avec {b.P(other)} avant vendredi ? Je suis "
                f"absente jusqu'au {b.fr_plain_date(1)}.",
        lambda: f"Mme {b.P(last)} a confirmé le rendez-vous du {b.fr_plain_date(1)}. "
                f"{b.P(other)} apporte les formulaires.",
    )
    close = b.pick(["Cordialement", "Bien à vous", "Amitiés", "Bonne journée", "Merci d'avance",
                    "À bientôt"])
    return (f"Objet : {b.pick(['planning', 'état des lieux', 'dossier de location', 'réunion'])}\n\n"
            f"{b.pick(['Bonjour', 'Bonsoir', 'Salut', 'Chère', 'Cher'])} {b.P(to_first)},\n\n"
            f"{body}\n\n{close},\n{b.P(sender)}")


def r4_fr_school(b):
    _, _, head = b.fr_full()
    _, _, pupil = b.fr_full()
    female = b.rng.random() < 0.5
    author = b.decoy(b.fr_facility(FR_SCHOOL_LEADS), "decoy_author_org")
    town_line = b.fr_town_line()
    lines = [author, "CERTIFICAT DE SCOLARITÉ", "",
             f"Je soussigné{'e' if female else ''}, {'Mme' if female else 'M.'} {b.P(head)}, "
             f"{b.pick(['proviseur', 'principale', 'directeur', 'directrice'])}, certifie que "
             f"l'élève {b.P(pupil)}, {b.pick(['né le', 'née le'])} "
             f"{b.D(b._fr_date(b.rng.randint(2006, 2016)))}, "
             f"domicilié au {b.A(b.fr_street_line() + ', ' + town_line)}, est inscrit en classe "
             f"de {b.pick(['seconde', 'première', 'terminale', 'troisième', 'CM2'])} pour l'année "
             f"scolaire {b.decoy(b.pick(['2025-2026', '2026-2027']), 'decoy_year_range')}."]
    if b.rng.random() < 0.6:
        lines.append(f"Établissement précédent : {b.O(b.fr_facility(FR_SCHOOL_LEADS))}.")
    lines.append(f"Fait à {town_line.split(' ', 1)[1]}, le {b.fr_plain_date(1)}")
    return "\n".join(lines)


def r4_table(b):
    sep = b.pick([" | ", " | ", "\t", " ; "])
    third = b.pick(["Vehicle reg", "Company", "Room", "Vehicle reg"])
    stem = b.pick(b.pools["company_stems"])
    site = b.decoy(f"{stem} {b.pick(TABLE_SITES)}", "decoy_place")
    hosts = [b.full(), b.full()]
    rows, returning = [], None
    for index in range(b.rng.randint(3, 5)):
        first, last, full = b.full()
        style = b.rng.randrange(6)
        if style == 0:
            cell = b.P(full)
        elif style == 1:
            cell, returning = f"{b.title()} {b.P(full)}", last
        elif style == 2:
            cell = b.P(first)
        elif style == 3:
            cell = b.P(f"{last}, {first}")
        elif style == 4:
            cell = b.P(b.initialled(first, last))
        else:
            cell, returning = b.P(full), last
        host_first, host_last, host_full = b.pick(hosts)
        host = b.P(b.initialled(host_first, host_last) if b.rng.random() < 0.5 else host_full)
        if third == "Vehicle reg":
            extra = b.plate() if b.rng.random() < 0.7 else b.pick(NON_PERSON_CELLS)
        elif third == "Company":
            extra = b.company() if b.rng.random() < 0.7 else b.pick(["self-employed", "n/a"])
        else:
            extra = f"{b.pick(['Room', 'Hall', 'Bay'])} {b.rng.randint(1, 30)}"
        rows.append(sep.join([b.clock(), cell, host, extra, f"V-{b.rng.randint(100, 999):04d}"]))
    header = sep.join(["Time in", b.pick(TABLE_NAME_HEADERS), b.pick(TABLE_HOST_HEADERS),
                       third, "Badge"])
    notes = ""
    if returning:
        notes = (f"\n\nNotes: {b.title(english=True)} {b.P(returning)} left a phone at the gate "
                 f"and collected it at {b.clock()}. The contractors' van stayed in the loading "
                 f"bay until {b.clock()}.")
    return (f"{b.pick(['Visitor log', 'Sign-in sheet', 'Gate register', 'Attendance sheet'])} — "
            f"{site}, {b.plain_date(5)}\n\n{header}\n" + "\n".join(rows) + notes)


def r4_owner_form(b):
    first, last, full = b.full()
    _, author = b.stem_company()
    lines = [f"{author} — {b.pick(['warranty claim', 'service request', 'returns form'])}", "",
             "Product",
             f"Name: {b.product()} {b.pick(PART_NOUNS)}",
             f"{b.pick(['Frame number', 'Serial number', 'Model code'])}: "
             f"{b.pick(string.ascii_uppercase)}{b.rng.randint(100, 999)} {b.rng.randint(1000, 9999)}",
             f"Purchased: {b.plain_date(5)}, from {b.site()}", "",
             b.pick(["Owner", "Customer", "Claimant"]),
             f"Name: {b.P(full)}",
             f"Email: {b.email_of(first, last, 'mailbox')}",
             f"Daytime phone: {b.phone()}", "",
             "Fault",
             f"Name of part: {b.product()} {b.pick(PART_NOUNS)}",
             f"Description: {b.pick(FAULTS)}."]
    if b.rng.random() < 0.6:
        lines[-1] += (f" The {b.pick(MECHANIC_NOUNS)}, {b.P(b.first())}, says it was fine at the "
                      f"last check.")
    lines += ["", f"Decision: replace under warranty. Approved by {b.job()}, {b.plain_date(5)}."]
    return "\n".join(lines)


def r4_column_gap(b):
    first, last, full = b.full()
    _, _, manager = b.full()
    _, _, writer = b.full()

    def gap():
        return b.pick(["   ", "    ", "      ", "\t", "        "])
    kind = b.rng.randrange(3)
    if kind == 0:
        head = (f"{b.author_facility(MEDICAL_KINDS)}\nOccupational health report — confidential\n\n"
                f"Employee: {b.P(full)}{gap()}Payroll number: {b.rng.randint(10000, 9999999):07d}\n"
                f"Manager: {b.P(manager)}{gap()}Department: "
                f"{b.decoy(b.pick(['Dispatch', 'Finance', 'Stores', 'Quality']), 'decoy_field_value')}\n"
                f"Date of birth: {b.D(b.dob_value())}{gap()}Date of assessment: {b.plain_date()}")
    elif kind == 1:
        head = (f"{b.company('decoy_author_org')}\nSTARTER DETAILS\n\n"
                f"Name: {b.P(full)}{gap()}Start date: {b.plain_date()}\n"
                f"DOB: {b.D(b.dob_value())}{gap()}Contract: "
                f"{b.decoy(b.pick(['Permanent', 'Fixed term', 'Agency']), 'decoy_field_value')}\n"
                f"Line manager: {b.P(manager)}{gap()}Site: {b.city()}")
    else:
        head = (f"{b.pick(['CLASS REGISTER', 'ENROLMENT LIST', 'PATIENT LIST'])}\n\n"
                f"Name: {b.P(full)}{gap()}Ref: {b.rng.randint(1000, 99999)}\n"
                f"Born: {b.D(b.dob_value())}{gap()}Enrolled: {b.plain_date()}\n"
                f"Contact: {b.P(manager)}{gap()}Relationship: "
                f"{b.decoy(b.pick(RELATIONS).title(), 'decoy_field_value')}")
    tail = b.choose(
        lambda: f"\n\n{b.P(first)} attended on time. {b.filler()} Report prepared by "
                f"{b.P(writer)}, {b.job()}.",
        lambda: f"\n\nChecked by {b.P(writer)}{gap()}Date: {b.plain_date()}",
        lambda: f"\n\n{b.filler()} Signed: {b.P(b.initialled(first, last))}{gap()}Date: "
                f"{b.plain_date(0)}",
    )
    return head + tail


def r4_address_block(b):
    first, last, full = b.full()
    title = b.title(english=True)
    stem, author = b.stem_company()
    body = b.choose(
        lambda: f"We have closed your account {b.spaced_id()} and refunded the credit balance "
                f"to the bank account ending {b.rng.randint(1000, 9999)}. The refund reference "
                f"is {b.plain_date(1)} and the money should reach you within five working days.",
        lambda: f"Your appointment is confirmed for {b.plain_date(5)} at {b.clock()}. Please "
                f"bring this letter with you.",
        lambda: f"Thank you for your payment. Your next statement will be issued on "
                f"{b.plain_date(5)}.",
        lambda: f"The engineer will call between {b.clock()} and {b.clock()}. {b.filler()}",
    )
    close = b.choose(
        lambda: f"{b.decoy(b.pick(TEAM_SIGNOFFS), 'decoy_team_signoff')}\n{stem}",
        lambda: b.decoy(b.pick(TEAM_SIGNOFFS), "decoy_team_signoff"),
        lambda: f"{b.P(b.full()[2])}\n{b.job()}",
    )
    return (f"{author}\n{b.pick(['Refund confirmation', 'Appointment letter', 'Statement notice', 'Visit confirmation'])}\n\n"
            f"{title} {b.P(full)}\n{b.block_address()}\n\n"
            f"Dear {title} {b.P(last)},\n\n{body}\n\n"
            f"{b.pick(['Yours sincerely', 'Yours faithfully', 'Kind regards'])},\n{close}")


def r4_company_surname(b):
    cs_first, cs_last = b.first(), b.company_surname()
    _, _, other = b.full()
    stem = b.pick(b.pools["company_stems"])
    kind = b.rng.randrange(3)
    if kind == 0:
        author = b.decoy(f"{b.pick(['Hausverwaltung', 'Residence', 'Wohnpark'])} {stem}",
                         "decoy_author_org")
        title = b.pick(["Fr.", "Hr.", "Dr.", "Mr", "Ms", "Frau", "Herr"])
        back = b.pick(["Frau", "Herr", "Mrs", "Mr", "Dr"])
        return (f"{author} — {b.pick(['Stellplatzliste / parking bay list', 'parking bay list', 'key register'])}, "
                f"{b.plain_date(5)}\n\n"
                f"Bay 03 — {b.plate()} — {title} {b.P(f'{cs_first} {cs_last}')}\n"
                f"Bay 07 — {b.plate()} — {b.title()} {b.P(other)}\n"
                f"Bay 11 — free\n\n"
                f"Notes: The ramp is certified to {b.code()}. Gate remotes are booked to project "
                f"code {b.code()}.\n"
                f"Queries: {back} {b.P(cs_last)} chairs the residents' council, Tel. {b.phone()}.")
    if kind == 1:
        return (f"{b.company('decoy_author_org')}\nBOARD NOTES\n\n"
                f"{b.P(f'{cs_first} {cs_last}')} has joined the board. "
                f"{b.title(english=True)} {b.P(cs_last)} previously ran the finance desk at "
                f"{b.decoy(stem + ' ' + b.pick(['Holdings', 'Bank', 'Group', 'Partners']), 'decoy_company')}. "
                f"{b.filler()} {b.P(other)} takes the minutes from next month.")
    return (f"{b.pick(['COMMITTEE', 'CONTACT LIST', 'OFFICERS'])}\n"
            f"Chair: {b.title(english=True)} {b.P(f'{cs_first} {cs_last}')}\n"
            f"Treasurer: {b.P(other)}\n"
            f"Auditor: {b.company()}\n"
            f"Bank: {b.decoy(stem + ' ' + b.pick(['Bank', 'Building Society', 'Credit Union']), 'decoy_company')}\n"
            f"{b.filler()} Cheques need two signatures: {b.P(cs_last)} and "
            f"{b.P(other.split(' ')[0])}.")


def r4_board(b):
    stem, author = b.stem_company()
    names = b.rng.sample(b.pools["first"], 5)
    _, _, full = b.full()
    posts = [
        f"{b.pick(BOARD_GREETINGS)} {b.P(names[0])}, {b.pick(BOARD_MESSAGES)}. "
        f"{b.pick(BOARD_CLOSINGS)}, {b.P(names[1])}",
        f"Dear {b.decoy(b.pick(ROLE_ADDRESSEES), 'decoy_role_addressee')}, "
        f"{b.pick(BOARD_MESSAGES)}. {b.pick(BOARD_CLOSINGS)}, "
        f"{b.decoy(stem, 'decoy_company_signoff')}",
        f"{b.pick(BOARD_GREETINGS)} {b.P(names[2])} — {b.pick(BOARD_MESSAGES)}. "
        f"{b.pick(BOARD_CLOSINGS)}, {b.P(names[3])}",
        f"Hello {b.decoy(b.pick(ROLE_ADDRESSEES), 'decoy_role_addressee')}, "
        f"{b.pick(BOARD_MESSAGES)}. {b.pick(BOARD_CLOSINGS)}, {b.P(full)}",
        f"Hey {b.decoy(b.pick(['Caretaker', 'Warden', 'Steward', 'Groundsman', 'Secretary']), 'decoy_role_addressee')}, "
        f"{b.pick(BOARD_MESSAGES)}. Thanks, "
        f"{b.decoy(b.pick(['Maintenance Committee', 'The Committee', 'Site Office']), 'decoy_team_signoff')}",
        f"{b.pick(BOARD_GREETINGS)} {b.P(names[4])}, {b.pick(BOARD_MESSAGES)}. "
        f"{b.pick(BOARD_CLOSINGS)},\n{b.P(names[0])}",
    ]
    b.rng.shuffle(posts)
    chosen = posts[:b.rng.randint(3, 5)]
    body = "\n\n".join(f"{i}) {post}" for i, post in enumerate(chosen, 1))
    return (f"{author} — messages pinned to the {b.pick(BOARD_KINDS)}\n\n{body}")


def r4_manifest(b):
    _, author = b.stem_company()
    stops = []
    for n in range(1, b.rng.randint(2, 3) + 1):
        _, _, full = b.full()
        title = b.title(english=True) + " " if b.rng.random() < 0.4 else ""
        lines = [f"Stop {n}",
                 f"Consignee: {title}{b.P(full)}",
                 f"Address: {b.A(b.line_address())}",
                 f"{b.pick(['Parcel contact', 'Shipment phone', 'Contact number'])}: {b.phone()}"]
        lines.append(b.choose(
            lambda: f"Tracking no.: {b.pick(string.ascii_uppercase)}"
                    f"{b.pick(string.ascii_uppercase)} {b.spaced_id()}",
            lambda: f"Delivery note: {b.plain_date(1)}",
            lambda: f"Customer reference: {b.plain_date(0)}",
            lambda: f"Consignment: CN-{b.rng.randint(10000, 99999)}, {b.rng.randint(1, 9)} "
                    f"cartons",
        ))
        stops.append("\n".join(lines))
    return (f"{author} — driver manifest, run {b.rng.randint(2, 40)}\n\n" + "\n\n".join(stops)
            + f"\n\nDriver notes: Stop 1 asked for a call ten minutes ahead. {b.filler()}")


def r4_plain_cv(b):
    first, last, full = b.full()
    _, _, referee = b.full()
    employer = b.company("decoy_employer")
    lines = ["Curriculum vitae", "",
             f"Name: {b.P(full)}",
             f"Address: {b.A(b.line_address())}",
             f"Phone: {b.phone()}",
             f"Email: {b.email_of(first, last, 'mailbox')}",
             b.dob(b.pick(["Born:", "Date of birth:", "DOB:"])), "",
             "Work history",
             f"Present: {b.job()} at {employer}",
             f"Earlier — {b.job()}, {b.company('decoy_employer')} (night shift)",
             f"Before that, {b.pick(['two', 'three', 'four'])} years as a {b.pick(CV_ROLES)} for "
             f"{b.company('decoy_employer')}",
             "Summer jobs: fruit picking, bar work", "",
             "Training",
             f"{b.pick(COURSES)}, {b.O(b.facility_name(EDUCATION_KINDS))}",
             f"Forklift licence (counterbalance), renewed {b.year()}", "",
             "Referees",
             f"{b.P(referee)}, {b.job().lower()}, {employer}",
             "Available on request: a second referee."]
    return "\n".join(lines)


def r4_transcript(b):
    agent = b.first()
    _, _, caller = b.full()
    _, _, holder = b.full()
    stem, author = b.stem_company()

    def speaker():
        return f"{b.pick(['Agent', 'Adviser', 'Operator'])} ({b.P(agent)})"
    who = speaker()
    return (f"{author} — support log, {b.plain_date(5)}\n\n"
            f"{b.clock()} {who}: Thanks for calling {stem}. Can I take the account holder's "
            f"name?\n"
            f"{b.clock()} Caller: It's {b.P(caller)}. The account is in my "
            f"{b.pick(['wife', 'husband', 'partner', 'mother', 'son'])}'s name though, "
            f"{b.P(holder)}.\n"
            f"{b.clock()} {who}: No problem. And the account number or the landline?\n"
            f"{b.clock()} Caller: {b.pick(string.ascii_uppercase)}{b.pick(string.ascii_uppercase)}-"
            f"{b.rng.randint(1000, 9999)}-{b.rng.randint(1000, 9999)}. The landline is "
            f"{b.phone()} but that's the thing that's dead.\n"
            f"{b.clock()} {who}: I'll book an engineer for {b.plain_date(5)}. Anything else?\n"
            f"{b.clock()} Caller: No, cheers.")


def r4_health_prose(b):
    first, last, full = b.full()
    _, gp_last, gp = b.full()
    _, _, rep = b.full()
    _, _, writer = b.full()
    return b.choose(
        lambda: (
            f"{b.author_facility(['Occupational Health Service', 'Occupational Health Ltd'])}\n"
            f"Occupational health report — confidential\n\n"
            f"{b.P(first)} attended with a union representative, {b.title(english=True)} "
            f"{b.P(rep)}. The absence began on {b.plain_date(5)} after surgery at "
            f"{b.O(b.facility_name(['Orthopaedic Hospital', 'Community Hospital', 'Hospital', 'Clinic']))}. "
            f"The GP, Dr {b.P(gp)} at {b.O(b.facility_name(MEDICAL_KINDS[:4]))}, has issued a "
            f"fit note to {b.plain_date(5)}.\n\nReport prepared by {b.P(writer)}, "
            f"{b.job().lower()}."),
        lambda: (
            f"TRANSFER NOTE\n{b.P(full)} moved from {b.O(b.facility_name(SCHOOL_KINDS))} to "
            f"{b.O(b.facility_name(SCHOOL_KINDS))} in {b.pick(['Year 8', 'Year 9', 'Year 10'])}. "
            f"{b.filler()} The form tutor, {b.title(english=True)} {b.P(gp_last)}, has the "
            f"reports from {b.plain_date(5)}."),
        lambda: (
            f"CARE PLAN REVIEW\n{b.P(full)} was discharged from "
            f"{b.O(b.facility_name(['Hospital', 'Community Hospital', 'Infirmary']))} to "
            f"{b.O(b.facility_name(CARE_KINDS))} on {b.plain_date()}. {b.P(first)} now attends "
            f"{b.O(b.facility_name(['Day Centre', 'Therapy Centre', 'Rehabilitation Centre']))} "
            f"twice a week. {b.filler()} Next review with {b.P(writer)} on {b.plain_date(5)}."),
        lambda: (
            f"NOTE TO FILE\nSpoke to {b.P(full)} about the place offered at "
            f"{b.O(b.facility_name(['Nursery', 'Primary School', 'Academy', 'Sixth Form College']))}. "
            f"The referral from "
            f"{b.O(b.facility_name(['Counselling Service', 'Wellbeing Centre', 'Family Support Centre']))} "
            f"arrived on {b.plain_date()}. {b.filler()}"),
    )


def r4_squad_email(b):
    coach_first, coach_last, coach = b.full()
    players = [b.full() for _ in range(4)]
    extra = b.first()
    team = b.decoy(f"{b.pick(b.pools['facility_stems'])} {b.pick(TEAM_NAMES)}", "decoy_team")
    squad = ", ".join(b.P(p[2]) for p in players) + f" and {b.P(extra)}"
    return (f"From: Coach {b.P(coach_first)} <{b.email_of(coach_first, coach_last, 'club')}>\n"
            f"Subject: Saturday's fixture\n\n"
            f"Hello {b.decoy(b.pick(['Parents and Carers', 'Parents', 'Families', 'Squad']), 'decoy_role_addressee')},\n\n"
            f"Squad for Saturday at {b.place()}, meet {b.clock()}:\n{squad}.\n\n"
            f"{b.P(players[1][0])}'s mum has offered lifts; ring {b.phone()} if you need one. "
            f"{b.P(players[0][0])} is carrying an ankle strain and will play one half only. "
            f"{b.P(extra)}'s registration is still pending.\n\n"
            f"{b.pick(['Yours in sport', 'See you Saturday', 'Thanks all'])},\n{b.P(coach)}\n"
            f"Head coach, {team}")


def r4_caps_mixed(b):
    first, last, full = b.full()
    _, _, kin = b.full()
    ref_first, ref_last, _ = b.full()
    extras = b.rng.sample(CAPS_NON_NAMES, 4)

    def non_name(pair):
        return f"{pair[0]}: {b.decoy(pair[1], 'decoy_form_value')}"
    address = ", ".join([b.bare_house(), b.uk_street(),
                         f"{b.pick(b.pools['uk_towns'])} {b.uk_postcode()}"]
                        if b.rng.random() < 0.5 else [b.uk_address()])
    lines = [b.pick(["TENANCY APPLICATION — COMPLETE IN CAPITALS", "MEMBERSHIP FORM (CAPITALS)",
                     "APPLICATION — BLOCK CAPITALS PLEASE"]),
             f"APPLICANT: {b.P(full.upper())}",
             non_name(extras[0]),
             non_name(extras[1]),
             f"DATE OF BIRTH: {b.D(b.dob_value(b.pick([0, 5])).upper())}",
             f"CURRENT ADDRESS: {b.A(address.upper())}",
             f"EMPLOYER: {b.company('decoy_employer').upper()}",
             f"EMERGENCY CONTACT: {b.P(kin.upper())} ({b.pick(RELATIONS)}) {b.phone()}",
             non_name(extras[2]),
             f"REFEREE: {b.pick(['MRS', 'MR', 'MS', 'DR'])} "
             f"{b.P(b.initialled(ref_first, ref_last).upper())}, PREVIOUS LANDLADY",
             non_name(extras[3]),
             f"SIGNED: {b.P(b.initialled(first, last).upper())}   DATE: {b.plain_date(0)}"]
    return "\n".join(lines)


# ------------------------------------------------------------ negative genres

def r4n_fr_notice(b):
    stem = b.pick(b.pools["company_stems"])
    author = b.decoy(f"{b.pick(['Cabinet', 'Syndic', 'Résidence', 'Agence'])} {stem}",
                     "decoy_author_org")
    first, second = b.rng.sample(FR_NOTICES, 2)
    return (f"{author}\nAVIS AUX {b.pick(['LOCATAIRES', 'RÉSIDENTS', 'COPROPRIÉTAIRES'])}\n\n"
            f"Madame, Monsieur,\n\n{first} {second} Les travaux commencent le "
            f"{b.fr_plain_date(1)} et se terminent le {b.fr_plain_date(0)}.\n\n"
            f"Fait à {b.decoy(b.pick(b.pools['fr_towns']), 'decoy_city')}, le {b.fr_plain_date(1)}\n"
            f"{b.pick(['Cordialement', 'Bien à vous', 'Avec nos remerciements'])},\n"
            f"{b.decoy(b.pick(['Le syndic', 'La gérance', 'Le secrétariat', 'La direction']), 'decoy_team_signoff')}")


def r4n_product_table(b):
    sep = b.pick([" | ", "\t", " ; "])
    rows = [sep.join([f"{b.rng.randint(1, 40):03d}", f"{b.product()} {b.pick(PART_NOUNS)}",
                      str(b.rng.randint(1, 12)), f"£{b.rng.randint(9, 480)}.00"])
            for _ in range(b.rng.randint(3, 5))]
    return (f"{b.company('decoy_author_org')} — stock sheet, {b.plain_date(5)}\n\n"
            + sep.join(["Item", "Name", "Qty", "Price"]) + "\n" + "\n".join(rows)
            + f"\n\nSupplier: {b.company()}. {b.filler()}")


def r4n_product_form(b):
    _, author = b.stem_company()
    return (f"{author} — parts order\n\nProduct\n"
            f"Name: {b.product()} {b.pick(PART_NOUNS)}\n"
            f"Model code: {b.pick(string.ascii_uppercase)}{b.rng.randint(100, 999)}\n\n"
            f"Spare\nName of part: {b.product()} {b.pick(PART_NOUNS)}\n"
            f"Name of supplier: {b.company()}\n"
            f"Description: {b.pick(FAULTS)}.\n\n"
            f"Decision: reorder. Approved by {b.job()}, {b.plain_date(5)}. {b.filler()}")


def r4n_company_board(b):
    stem, author = b.stem_company()
    posts = [
        f"Dear {b.decoy(b.pick(ROLE_ADDRESSEES), 'decoy_role_addressee')}, "
        f"{b.pick(BOARD_MESSAGES)}. {b.pick(BOARD_CLOSINGS)}, {b.decoy(stem, 'decoy_company_signoff')}",
        f"Hello {b.decoy(b.pick(ROLE_ADDRESSEES), 'decoy_role_addressee')}, "
        f"{b.pick(BOARD_MESSAGES)}. {b.pick(BOARD_CLOSINGS)}, "
        f"{b.decoy(b.pick(TEAM_SIGNOFFS), 'decoy_team_signoff')}",
        f"{b.pick(['Hey', 'Hiya', 'Morning'])} "
        f"{b.decoy(b.pick(['Caretaker', 'Warden', 'Steward', 'all', 'everyone', 'team']), 'decoy_role_addressee')}, "
        f"{b.pick(BOARD_MESSAGES)}. {b.pick(BOARD_CLOSINGS)},\n"
        f"{b.decoy(stem, 'decoy_company_signoff')}",
    ]
    body = "\n\n".join(f"{i}) {post}" for i, post in enumerate(posts, 1))
    return f"{author} — messages pinned to the {b.pick(BOARD_KINDS)}\n\n{body}"


def r4n_caps_blank(b):
    pairs = b.rng.sample(CAPS_NON_NAMES, 6)
    lines = [f"{label}: {b.decoy(value, 'decoy_form_value')}" for label, value in pairs]
    return (f"{b.pick(['VEHICLE PERMIT RENEWAL', 'STORAGE UNIT FORM', 'ALLOTMENT RENEWAL'])} — "
            f"COMPLETE IN CAPITALS\n" + "\n".join(lines)
            + f"\nEMPLOYER: {b.company('decoy_employer').upper()}\nDATE: {b.plain_date(0)}")


def r4n_bay_codes(b):
    stem = b.pick(b.pools["company_stems"])
    author = b.decoy(f"{b.pick(['Hausverwaltung', 'Residence', 'Wohnpark'])} {stem}",
                     "decoy_author_org")
    return (f"{author} — parking bay list, {b.plain_date(5)}\n\n"
            f"Bay 03 — free\nBay 07 — reserved for deliveries\nBay 11 — free\n"
            f"Bay 12 — charger\n\n"
            f"Notes: The ramp is certified to {b.code()}; vehicles up to 3.5 t may use it. The "
            f"charger at bay 12 is a {b.code()} wall unit. Gate remotes are booked to project "
            f"code {b.code()}. {b.filler()}")


ROUND4_POSITIVE = [r4_fr_attestation, r4_fr_letter, r4_fr_email, r4_fr_school, r4_table,
                   r4_table, r4_owner_form, r4_column_gap, r4_address_block,
                   r4_address_block, r4_company_surname, r4_company_surname, r4_board,
                   r4_board, r4_manifest, r4_plain_cv, r4_transcript, r4_health_prose,
                   r4_health_prose, r4_squad_email, r4_caps_mixed]
ROUND4_NEGATIVE = [r4n_fr_notice, r4n_product_table, r4n_product_form, r4n_company_board,
                   r4n_caps_blank, r4n_bay_codes]


def build_round4(rng, pools, n_docs, prefix):
    docs, made = [], 0
    while made < n_docs:
        negative = (made % 5 == 4)
        genre = rng.choice(ROUND4_NEGATIVE if negative else ROUND4_POSITIVE)
        builder = Builder4(rng, pools)
        text, spans = render(genre(builder))
        if not consistent(text, spans):
            continue  # reroll: a name collided with another piece of the text
        docs.append({"source_file": f"{prefix}_{made:03d}_{genre.__name__.split('_', 1)[1]}.txt",
                     "text": text, "spans": spans,
                     "decoys": [{"text": value, "kind": kind}
                                for value, kind in builder.decoys]})
        made += 1
    return docs
