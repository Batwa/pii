"""The complete rule layer of the V1 public-sharing redaction policy.

This module owns everything that is NOT a model: regex patterns, context
filters, and span post-processing. The detection pipeline in
text_pii_detector.py runs these stages in order:

    1. run_pattern_rules(text)        — regex candidates (this module)
    2. apply_policy_filters(...)      — context rules drop or relabel candidates
    3. echo_person_names(...)         — a found name marks its other occurrences
    4. (deduplication, in the detector — uses DOB_CONTEXT/dedup constants here)
    5. relabel_by_id_context(...)     — "SIN 123..." forces the CA_SIN label

Rules live here so that model training (the spaCy NER) can evolve separately:
the model learns fuzzy entities (PERSON, SENSITIVE_ORGANIZATION, ADDRESS,
DATE_OF_BIRTH); everything with a fixed format or label belongs to this file.
See training/LABELING_GUIDE.md for the full split and the labeling criteria.

Each rule cites the policy decision it implements (red.pdf V1, plus the
decisions recorded in training/build_gold_corpus.py).
"""
import re

from config import PRONOUNS_ALWAYS, PRONOUNS_NEAR_PERSON, PERSON_PROXIMITY_CHARS, CUSTOM_MODEL_TRUSTED_LABELS

# =========================================================================
# 1. Shared vocabulary
# =========================================================================

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?"
           r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")

# Facility words that mark an organization as sensitive under the policy
# (clinics, hospitals, schools, shelters). Deliberately case-sensitive so prose
# such as "the service center" or "the school office" stays untouched. Institute,
# Academy, the British "Centre" compounds and counselling services were added
# after the test3 blind run (2026-11-04), where the model had found all three
# such facilities and this list threw them away.
# Surgery and Nursery need a name word in front so a sentence-initial
# "Surgery went well" is not a facility (test5 blind run, 2026-11-05).
# Care/Nursing Home and Health/NHS Trust came from the test6 blind run
# (2026-11-07), where the model had found both and this list dropped them.
_FACILITY_WORDS = (
    r"Hospital|Clinic|Polyclinic|Hospice|Shelter|School|College|University"
    r"|Institute|Academy|Infirmary|Sanatorium"
    r"|(?:Care|Nursing|Residential|Children's)[ \t]+Home"
    r"|(?:Health|NHS|Foundation|Hospital|Healthcare)[ \t]+Trust"
    r"|(?<=\w[ \t])(?:Surgery|Nursery)"
    r"|(?:Dental|GP|Medical|General|Family)[ \t]+Practice"
    r"|(?:Medical|Support|Health|Care|Wellbeing|Wellness|Counsell?ing"
    r"|Rehabilitation|Language|Speech|Therapy|Treatment|Dialysis|Cancer"
    r"|Diagnostic|Crisis|Day)[ \t]+Cent(?:er|re)"
    r"|Counsell?ing[ \t]+Service"
)
SENSITIVE_ORG_KEYWORDS = re.compile(r"\b(?:" + _FACILITY_WORDS + r")\b")
# A detected organization is redacted only when the document ties a person to it.
EMPLOYMENT_CONTEXT = re.compile(
    r"(?i)\b(?:employ\w*|works?(?!\s+(?:best|well|fine|properly|again|only|perfectly"
    r"|on|with|by|like|when|if|until)\b)\b|worked|working(?!\s+(?:days?|hours?|weeks?|time)\b)"
    r"|hired|joined|staff|salary"
    r"|payroll|position|internship|colleague|report(?:s|ed)?\s+to|offer\s+you)"
)
# "Engineer at Helix Dynamics" — a role followed by at/joins directly before it.
ORG_AT_PREFIX = re.compile(r"(?i)\b(?:at|joins?|joining)\s+[\"'“]?\s*$")
# A detected city/region is redacted only in a residence or work-address context.
RESIDENCE_CONTEXT = re.compile(
    r"(?i)\b(?:resid\w*|lives?\s+in|living\s+in|address|registered\s+at|home"
    r"|apt\.?|apartment|unit\s+\d|flat\s+\d|premises|office\s+at|headquarters\s+at)\b"
)
# A city as a person's journey destination is redacted (review verdict S014);
# operational locations — warehouses, branches, hubs — stay visible, so the
# nouns here are itinerary words, not "arrives"/"dispatched" cargo phrasing.
TRAVEL_CONTEXT = re.compile(
    r"(?i)\b(?:arrival|departs|departure|itinerary|travels?\s+to|trip\s+to"
    r"|fl(?:ies|ying)\s+to)\b"
)
# A city or name followed by an operational noun is a site, not a residence,
# a destination or a person: "the Oslo hub", "the Hull depot" (policy
# decisions 5 and 8; test4 "Lyon hub", test5 Oslo/Stavanger/Hull).
OPERATIONAL_SITE_SUFFIX = re.compile(
    r'(?i)^[ \t]+(?:hub|branch|depot|warehouse|exchange|terminal|plant|factory'
    r'|showroom|office|site|yard|store|outlet|facility)\b')
# "as a Senior X" / "position of X" introduces a job title, not a name or an
# employer — generic titles stay visible under the policy.
TITLE_PREFIX = re.compile(
    r'(?i)\b(?:position\s+of|as\s+an?|title\s+of)\s+(?:[A-Z][a-z]+\s+){0,2}$')
# A label naming a thing directly before a span ("Vehicle: Nissan Qashqai")
# vetoes the person reading general NER gives product names, as an ID label
# decides a number's type (test4 finding, 2026-11-05).
THING_LABEL_PREFIX = re.compile(
    r'(?i)\b(?:vehicle|car|van|truck|motorcycle|vessel|boat|make|model'
    r'|make\s*/\s*model|product|item|device|equipment|machine|brand'
    r'|manufacturer|engine|motor|appliance|instrument|tool|software)\s*[:\-]\s*$')
# A date is a DOB only with an explicit birth label directly before it.
DOB_CONTEXT = re.compile(
    r'(?i)(?:born(?:\s+on)?(?:\s+the)?|date\s+of\s+birth|birth\s+date|birthday|d\.?o\.?b\.?)\s*[:,-]?\s*$')

# When an ID label sits directly before a number, the label decides the type —
# policy page 3. Fixes checksum collisions such as a Luhn-valid Canadian SIN
# also validating as an Australian TFN.
ID_LABEL_OVERRIDES = {
    "sin": "CA_SIN", "tfn": "AU_TFN", "pan": "IN_PAN", "nino": "UK_NINO",
    "umid": "PH_UMID", "aadhaar": "IN_AADHAAR", "epic": "IN_VOTER",
}
ID_LABEL_TOKEN = re.compile(r"(?i)\b(" + "|".join(ID_LABEL_OVERRIDES) + r")\b\W{0,4}$")

# =========================================================================
# 2. Pattern rules — fixed formats and labelled values
# =========================================================================

# One word of a person's name: a capital, lowercase letters, at most one more
# capital inside ("MacLeod", "O'Brien" through the apostrophe). Never "LLC".
# Latin-1 letters cover "Németh", "Oisín", "Fernández" (test5).
_NAME_WORD = r"[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ'’-]+(?:[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ'’-]+)?"
# Lowercase particles between name words: "Fenna van der Berg", "Pieter de
# Vries". Case-sensitive on purpose, so a sentence cannot continue a name.
_NAME_PARTICLES = (r"(?:[ \t](?:van|der|den|de|von|da|di|du|la|le|del|della|dos"
                   r"|das|ten|ter|op|bin|al|el|ap|ab|ibn|bint|abu|af|av|zu|'t|’t))*")
_NAME = _NAME_WORD + r"(?:" + _NAME_PARTICLES + r"[ \t]" + _NAME_WORD + r"){1,2}"
# An optional courtesy title ahead of a labelled name. It is part of the span
# but not a name word, so "Mr Khalid bin Tariq Al-Farsi" keeps its two name
# slots for the name (test6: the title had used one up).
_TITLE_OPT = r"(?:(?:Mr|Mrs|Ms|Mx|Miss|Dr|Prof|Sr|Sra|Herr|Frau|Mme|Mlle|M)\.?[ \t]+)?"
# Addressees that are collective or role words, never a first name.
_GREETING_STOP = (
    r"(?:All|Everyone|Everybody|Team|Customers?|Colleagues?|Sirs?|Madam|Friends?"
    r"|Parents?|Guests?|Members?|Residents?|Tenants?|Patients?|Students?|Applicant"
    r"|Doctor|Nurse|Both|Partners?|Valued|Folks|Guys|Mum|Dad|Mom|You|There|Readers?"
    r"|World|Diary|Again|Later|Soon|Today|Tomorrow|Yesterday|Tonight|Monday|Tuesday"
    r"|Wednesday|Thursday|Friday|Saturday|Sunday|Support|Sales|Admin|Office|Reception"
    r"|Staff|Users?|Subscribers?|Volunteers?|Neighbou?rs?|Mr|Mrs|Ms|Mx|Miss|Dr|Prof"
    r"|Lord|Lady)")
# A UK postcode; the outward half alone is what a span tends to stop after.
_UK_POSTCODE = r"[A-Z]{1,2}\d{1,2}[A-Z]?[ \t]\d[A-Z]{2}"

TEXT_PATTERNS = {
    # --- contact details (policy: personal phones and e-mails always redact;
    # V1 decision: ALL e-mail addresses redact, role inboxes included) ---
    'email_address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    # The final separator is mandatory: a bare 10-digit run (batch code,
    # reference number) is not a phone under the policy.
    'phone': re.compile(r'(?<!\w)(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]([0-9]{4})(?!\w)'),
    # International form: a leading "+" (or the 00 dialling prefix), the
    # country code, an optional single-digit group ("+33 6 12 34 56 78",
    # test4), then at least two separated digit groups of up to eight
    # digits ("+49 171 2345678", test6). The prefix is mandatory so amounts
    # and references never qualify (test3 finding, 2026-11-04); the 00 form
    # may not follow another digit group, so the "0044 0532 0130 00" inside
    # an IBAN is not a phone.
    'phone_intl': re.compile(
        r'(?<!\w)(?:\+|(?<![\d][ .-])00)\d{1,3}'
        r'(?:[ .-]?\(0\)[ .-]?\d{1,6}(?:[ .-]\d{2,6}){1,4}'  # +49 (0)30 901820
        r'|(?:[ .-]\d)?(?:[ .-]\(?\d{2,8}\)?){2,4})(?!\w)'),
    # UK national form: a leading 0 and exactly ten more digits in two or
    # three separated groups ("020 7946 0958", "07700 900123"; test5). The
    # lookahead fixes the digit count so IBAN fragments never qualify.
    'phone_uk': re.compile(
        r'(?<!\w)(?=0(?:[ -]?\d){10}(?![ -]?\d))0\d{2,4}(?:[ -]\d{3,6}){1,2}(?!\w)'),
    'phone_ru': re.compile(r'(?<!\w)(?:(?:\+7|8)[\s-]?(?:\(\d{3,4}\)|\d{3,4})[\s-]?\d{2,3}[\s-]\d{2}[\s-]\d{2})(?!\w)'),

    # --- government and financial identifiers (policy: always redact) ---
    'ssn': re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'),
    'snils': re.compile(r'\b\d{3}-\d{3}-\d{3}\s\d{2}\b'),
    # Only labelled INNs: a bare 10-digit run is not PII under the policy
    # ("a number ... needs a recognized format or label").
    'inn_labelled': re.compile(r'\b(?:INN|ИНН)\s*:?\s*(?P<pii>\d{10}(?:\d{2})?)\b', re.IGNORECASE),
    'bank_account': re.compile(r'\b[1-9]\d{19}\b'),
    'payment_card': re.compile(r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b'),
    'passport_number': re.compile(r'\b\d{2}\s\d{2}\sNo\.\s\d{6}\b', re.IGNORECASE),
    'bic': re.compile(r'\b(?:BIC|БИК)\s*:?[\s]*(?P<pii>\d{9})\b', re.IGNORECASE),
    'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
    'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
    # Fallback for passports whose format no country recognizer accepts:
    # the label alone is enough under the policy. Scored at the country
    # recognizers' level so a specific match wins the dedup tiebreak.
    'passport_labelled': re.compile(
        r'\bpassport\s*(?:no\.?|number|#)?\s*[:#]?\s*(?P<pii>(?=[A-Z0-9]*\d)[A-Z0-9]{6,10})\b', re.IGNORECASE),
    # The same fallback for a labelled driving licence: Presidio's UK
    # recognizer scores a valid number 0.5 with or without its label (test6),
    # so the label carries it here, at the passport fallback's score.
    'driving_licence': re.compile(
        r"\b(?:driving|driver['’]?s?|DVLA)\s+licen[cs]e\s*(?:no\.?|number|#)?\s*[:#]?\s*"
        r"(?P<pii>(?=[A-Z0-9]*\d)[A-Z0-9]{5,20})\b", re.IGNORECASE),

    # --- vehicles (policy: redact when linked to a person) ---
    'vin': re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
    'license_plate': re.compile(r'\b[A-Z]\s?\d{3}\s?[A-Z]{2}\s?\d{2,3}\b'),
    # German plate: a one-to-three-letter district code, a dash, one or two
    # letters, a space and up to four digits ("M-RK 4471", test6). A
    # year-shaped digit block is excluded so "EU-UK 2026" never qualifies.
    'license_plate_de': re.compile(
        r"(?<![\w-])[A-ZÄÖÜ]{1,3}-[A-Z]{1,2}[ \t](?!(?:19|20)\d{2}\b)\d{1,4}[EH]?(?![\w-])"),

    # --- person-linked labelled identifiers (policy: customer/account/member/
    # student/patient IDs redact; thing-IDs such as order, invoice, tracking,
    # serial, batch, and flight numbers stay) ---
    # Separators are space/tab/hyphen only, never \s, so the value cannot run
    # onto the next line. The letter prefix ("POL 4471 9920 36") came from
    # the test3 blind run.
    'policy_number': re.compile(
        r'\b(?:policy|insurance\s+policy)\s+(?:No\.?|number|#)\s*:?\s*'
        r'(?P<pii>(?:[A-Z]{2,4}[ \t-]?)?\d{2,4}(?:[ \t-]\d{2,4}){1,4}|[A-Z]{0,4}[ \t-]?\d{6,12})\b',
        re.IGNORECASE),
    'student_id': re.compile(r'\bstudent\s+ID\s+No\.?:?\s*(?P<pii>[A-Z]{2,10}-\d{4}-\d{4,})\b', re.IGNORECASE),
    # "hospital" alone names a place, so it needs the number word (test4);
    # "Hospital No. 3" stays with the sensitive_organization pattern.
    # National ID abbreviations (DNI, NIE, NIF, CURP, CPF, BSN, PESEL) label a
    # person's number on their own; "application", "visa" and "permit" need
    # a reference word ("APPLICATION REFERENCE: VAF-2026-118822"); both from
    # the test6 blind run.
    'generic_id': re.compile(
        r'\b(?:(?:customer|client|member|patient|employee|student|account'
        r'|badge|case|claim|record|mrn|dni|nie|nif|curp|cpf|bsn|pesel)\s*'
        r'(?:id|number|no\.?|record|reference|ref\.?|#)?'
        r'|(?:hospital|staff|payroll)\s+(?:id|number|no\.?)'
        r'|(?:application|visa|permit)\s+(?:reference|ref\.?|number|no\.?|id))\s*[:#.]?\s*'
        r'(?P<pii>(?=[A-Z0-9/-]{0,20}\d)[A-Z0-9][A-Z0-9/-]{3,19})', re.IGNORECASE),
    # A person's group/class designation ("class 9-B", "Group PMI-2601").
    'group_class': re.compile(
        r'\b(?:class|grade|group|form)\s+'
        r'(?P<pii>\d{1,2}-?[A-Z]\b|[A-Z]{2,5}-\d{2,6})', re.IGNORECASE),
    # Trackable purchase/product identifiers (review verdict: serial and order
    # numbers can be searched up and tracked). Batch numbers stay visible.
    'serial_number': re.compile(
        r'\b(?:serial|hull)\s*(?:no\.?|number|id|#)?\s*[:#]?\s*'
        r'(?P<pii>(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{4,19})\b', re.IGNORECASE),
    'order_number': re.compile(
        r'\border\s*(?:no\.?|number|#)?\s*[:#]?\s*'
        r'(?P<pii>(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{4,19})\b', re.IGNORECASE),
    # A UK sort code identifies the account together with the account number.
    'sort_code': re.compile(
        r'\bsort\s+code\s*[:#]?\s*(?P<pii>\d{2}[- ]\d{2}[- ]\d{2})\b', re.IGNORECASE),
    # A penalty charge notice is served on the registered keeper, so its
    # number is that person's ("PCN number: EV 44021873", test6); the value
    # may carry a spaced letter prefix.
    'pcn_number': re.compile(
        r'\b(?:PCN|penalty\s+charge\s+notice)\s*(?:number|no\.?|#)?\s*[:#]?\s*'
        r'(?P<pii>(?:[A-Z]{1,4}[ \t-]?)?\d{6,12})\b', re.IGNORECASE),
    # A hire, tenancy or loan agreement number identifies its party
    # ("Agreement: 55-20931", test6). Single-line separators; the value needs
    # a digit and a separator or six characters, so "agreement 2026" is not
    # one. An employment contract number stays visible (dev gold, "44/26").
    'agreement_number': re.compile(
        r'\b(?:(?:hire|rental|tenancy|loan|credit|finance|lease)[ \t]+)?agreement'
        r'[ \t]*(?:id|number|no\.?|reference|ref\.?|#)?[ \t]*[:#.]?[ \t]*'
        r'(?P<pii>(?=[A-Z0-9/-]*\d)(?:[A-Z0-9]+[-/][A-Z0-9/-]+|[A-Z0-9]{6,20}))\b', re.IGNORECASE),

    # --- dates (policy: only birth dates redact; issue dates stay) ---
    # Month-day-year, day-month-year, and worded forms ("5 May 1994",
    # "March 3, 1985"), each anchored to an explicit birth label.
    'date_of_birth': re.compile(
        r'\b(?:date\s+of\s+birth|birth\s+date|birthday|born(?:\s+on)?|d\.?o\.?b\.?)\s*[:,-]?\s*(?:the\s+)?(?P<pii>'
        r'(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])'
        r'|(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01])[-/.](?:19|20)\d{2}'
        r'|(?:0?[1-9]|[12][0-9]|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}'
        r'|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?' + _MONTHS + r',?\s+(?:19|20)\d{2}'
        r'|' + _MONTHS + r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}'
        r')\b', re.IGNORECASE),

    # --- organizations and addresses ---
    # Single-line separators only: \s+ would let the name absorb an
    # unrelated heading from the previous line.
    # The capitalised run before the facility word may carry "and", "of",
    # "for" or "&" ("Riverside Speech and Language Centre", test5).
    'sensitive_organization': re.compile(
        r"(?P<pii>(?:[A-Z][\w.'’-]*[ \t]+(?:(?:and|of|for|&)[ \t]+)?){0,5}"
        r"(?:" + _FACILITY_WORDS + r")"
        r"(?:[ \t]+of[ \t]+[A-Z][\w-]+(?:[ \t]+(?:and[ \t]+)?[A-Z][\w-]+){0,2})?"
        r"(?:[ \t]+No\.?[ \t]*\d+)?)"),
    # The word-boundary before the street-type alternation is load-bearing:
    # without it "...of the contraCT" and "...with standaRD" match.
    'address_line': re.compile(
        r"\b\d+\s+[A-Za-z][A-Za-z\s'’]{0,40}?"
        r'\b(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr'
        r'|Court|Ct|Place|Pl|Crescent|Grove|Walk|Rise|Row|Terrace|Way|Close'
        r'|Prospekt|House|Gardens|Mews|Hill|Yard|Square|Quay|Parade|Heights)\b',
        re.IGNORECASE),
    # German-order street line: the street word carries its type as a suffix
    # and the house number follows ("Hauptstrasse 12", "Berliner Allee 3").
    'address_line_de': re.compile(
        r"\b(?P<pii>[A-ZÄÖÜ][a-zäöüß-]*?[ \t-]?"
        r"(?:[Ss]tra(?:ss|ß)e|[Ss]tr\.|[Aa]llee|[Ww]eg|[Gg]asse|[Pp]latz|[Dd]amm|[Uu]fer)"
        r"[ \t]+\d{1,4}[a-z]?)\b"),
    # Romance-order street line: the street word leads, the name follows,
    # then the number and an optional floor/door ("Calle Mayor 12, 3ºB",
    # test6); the segment extension then takes ", 28013 Madrid". English
    # look-alikes (Avenue, Place, Boulevard) are left out on purpose.
    'address_line_es': re.compile(
        r"\b(?P<pii>(?:Calle|Carrer|Carrera|Avenida|Avinguda|Paseo|Passeig|Plaza|Plaça"
        r"|Camino|Ronda|Travesía|Rua|Praça|Largo|Via|Viale|Piazza|Corso|Rue|Chemin"
        r"|Allée|Quai|Impasse|C/|Avda\.|Av\.|Pza\.|Bd\.)[ \t]+"
        r"(?:(?:de|del|dels|des|du|di|da|degli|delle|la|las|los|le|les|l')[ \t]+){0,3}"
        r"[A-ZÀ-ÖØ-Þ][\w'’-]+(?:[ \t]+(?:(?:de|del|du|di|da|la|le|les|y|e)[ \t]+)?[A-ZÀ-ÖØ-Þ][\w'’-]+){0,3}"
        r",?[ \t]+(?:n[ºo°]\.?[ \t]*)?\d{1,4}(?:[ \t]?(?:bis|ter)\b)?"
        r"(?:,[ \t]*\d{1,2}[ºo°]?[ \t]?[A-Za-z]?\b|[ \t]+\d{1,2}[ºo°][ \t]?[A-Za-z]?\b)?)"),
    # Apartment/unit designators, before or after the street line; the span
    # includes the designator word so "Flat 6" ahead of its street is masked
    # whole. The \b after the keyword keeps "unit" from matching inside
    # "units"; the digit lookahead keeps "unit price" and "flat rate" from
    # yielding a designator.
    'apt_unit': re.compile(r'\b(?P<pii>(?:apt|apartment|unit|suite|flat)\b\.?\s*#?\s*(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{1,6})\b', re.IGNORECASE),

    # --- pronouns and times (policy: third-person pronouns redact; times only
    # together with a person) ---
    'pronoun': re.compile(r'\b(?:' + '|'.join(PRONOUNS_ALWAYS) + r')\b', re.IGNORECASE),
    # Plural pronouns: kept only as candidates; apply_policy_filters drops them
    # unless a person is detected nearby.
    'pronoun_group': re.compile(r'\b(?:' + '|'.join(PRONOUNS_NEAR_PERSON) + r')\b', re.IGNORECASE),
    # Clock time with optional day-part; apply_policy_filters requires a person
    # nearby, so timetables and opening hours stay untouched.
    'time_expr': re.compile(r"\b\d{1,2}\s*o['’]?clock(?:\s+in\s+the\s+(?:morning|evening|afternoon|night))?", re.IGNORECASE),

    # --- names in layouts the general NER routinely misses. A name word may
    # carry one inner capital ("MacLeod", "McDonald", "DeSouza"): the test4
    # blind run showed the role-label rule stopping at "Ruaridh Mac". ---
    # A label that is the second item of a comma list ("Employer, Applicant
    # 1: Fenland Water Services") names a field of the first item, so it does
    # not introduce a name (test6).
    'labelled_person': re.compile(
        r'(?<!,[ \t])\b(?:patient|student|employee|customer|client|resident|traveler'
        r'|traveller|applicant|claimant|policyholder|account\s+holder'
        r'|taxpayer|respondent|plaintiff|defendant|witness|landlord|tenant'
        r'|owner|renter|caseworker|insured|guard\s+on\s+duty|reported\s+by'
        r'|mother|father|candidate|(?:emergency\s+)?contact|next\s+of\s+kin'
        r'|passenger|pupil|proposer|guardian|spouse|beneficiary|driver|hirer'
        r'|lessee|borrower|cardholder|keeper|carer|parent(?:\s*/\s*(?:carer|guardian))?'
        r'|referred\s+by)'
        # "Applicant 1:", "Emergency contact (if unavailable):", "Emergency
        # contact if the parent cannot be reached:" — a number, a bracket or
        # an "if" clause may sit between the label and the colon (test6).
        r'(?:[ \t]+\d{1,2})?(?:[ \t]*\([^)\n]{0,40}\)|[ \t]+if[^:\n]{0,50})?\s*[:\-]\s*'
        # The name itself stays case-sensitive ((?-i:)) and single-line
        # ([ \t]) so it cannot absorb "My name", "LLC", or the next line.
        r"(?P<pii>(?-i:" + _TITLE_OPT + _NAME + r"))", re.IGNORECASE),
    # Name fields of a form ("SURNAME: ONYEKWERE", "GIVEN NAMES: CHIDI EMEKA",
    # "Full name: Lebedev Kirill Andreevich"): a line-initial label, then an
    # ALL-CAPS run or Title Case name words (test6: the NER layers and the
    # name regexes all wanted Title Case). The bare label "Name:" is left out;
    # it names products and companies as often as people.
    'name_field': re.compile(
        r"^[ \t]*(?:surname|family\s+name|last\s+name|given\s+names?|first\s+names?"
        r"|forenames?|middle\s+names?|maiden\s+name|full\s+name"
        r"|(?:applicant|patient|tenant|student|employee|customer|client|resident|hirer|insured)['’]?s?\s+name"
        r"|name\s+of\s+(?:the\s+)?(?:applicant|patient|tenant|student|employee|customer|client|holder|insured))"
        r"[ \t]*[:\-][ \t]*(?P<pii>(?-i:" + _TITLE_OPT + _NAME_WORD
        + r"(?:" + _NAME_PARTICLES + r"[ \t]" + _NAME_WORD + r"){0,3}"
        r"|[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’-]+(?:[ \t]+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’-]+){0,3}))",
        re.IGNORECASE | re.MULTILINE),
    # A greeting or reminder addressed to a bare first name ("Hi Ifeoma,",
    # "Reminder for Bartek:", "Thanks Rhys."): the address word labels the
    # person (test6). Collective and role addressees are not names.
    'greeting_person': re.compile(
        r"\b(?:Hi|Hello|Hey|Hiya|Dear|Thanks|Thank you|Cheers|Reminder for|Message for"
        r"|Note for|Attention)[ \t]+(?!" + _GREETING_STOP + r"\b)(?P<pii>" + _NAME_WORD
        + r")(?=[,:.!?;]|[ \t]+(?:and|&)[ \t])"),
    # The brackets round the time are optional ("14:03 Rhys ap Gruffudd:",
    # test6).
    'chat_speaker': re.compile(
        r"(?m)^\[?\d{1,2}:\d{2}\]?[ \t]+(?P<pii>" + _NAME_WORD + r"(?:" + _NAME_PARTICLES
        + r"[ \t]" + _NAME_WORD + r")+)(?=:)"),
    'quoted_author': re.compile(
        r"\b(?P<pii>(?!(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b"
        r"|(?:January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\b)"
        + _NAME + r")[ \t]+wrote\b"),
    # Line-initial greeting name; kept only when the same name also occurs
    # inside a full PERSON detection (checked in apply_policy_filters).
    'greeting_name': re.compile(r"(?m)^(?:Dear\s+)?(?P<pii>" + _NAME_WORD + r"),(?=\s)"),
}

# A pattern anchored to an explicit label outranks a bare digit-run match over
# the same span: a labelled BIC beats a generic 9-digit SSN shape, and a
# labelled INN beats Presidio's 12-digit Aadhaar recognizer.
LABELLED_PATTERNS = {"bic", "inn_labelled", "generic_id", "group_class",
                     "student_id", "policy_number", "serial_number",
                     "order_number", "sort_code", "pcn_number", "agreement_number"}
PATTERN_ENTITY_TYPES = {"inn_labelled": "INN", "apt_unit": "ADDRESS_LINE",
                        "address_line_de": "ADDRESS_LINE",
                        "address_line_es": "ADDRESS_LINE",
                        "license_plate_de": "LICENSE_PLATE",
                        "phone_intl": "PHONE", "phone_uk": "PHONE",
                        "sort_code": "BANK_ACCOUNT",
                        "pcn_number": "GENERIC_ID", "agreement_number": "GENERIC_ID",
                        "pronoun_group": "PRONOUN_GROUP",
                        "labelled_person": "PERSON", "chat_speaker": "PERSON",
                        "quoted_author": "PERSON", "name_field": "PERSON",
                        "greeting_person": "PERSON",
                        "greeting_name": "GREETING_NAME",
                        "passport_labelled": "PASSPORT_NUMBER",
                        "driving_licence": "DRIVING_LICENCE"}
# Matches the IN_PASSPORT threshold: ties go to the country recognizer. The
# international phone scores just under the country-specific phone patterns, so
# "+7 (495) ..." keeps its PHONE_RU label when both match.
# The UK form keeps the default 1.0: a chat-reply code ("900123", also 1.0)
# nested inside "07700 900123" must lose to the longer phone span on the tie.
PATTERN_SCORES = {"passport_labelled": 0.45, "driving_licence": 0.45, "phone_intl": 0.99}

# A chat reply gives an identifier whose label sits in the previous speaker's
# line ("Could you confirm your member number?" / "88-1234-556, and ...").
# Labeling-guide decision 9: the question's label decides the reply's code.
_CHAT_LINE = re.compile(r"(?m)^\[?\d{1,2}:\d{2}\]?[ \t]+(?P<speaker>[^:\n]{1,40}):[ \t]*(?P<body>.*)$")
_ID_QUESTION = re.compile(
    r"(?i)\b(?:account|member(?:ship)?|customer|client|patient|policy|case|claim"
    r"|insurance)\s+(?:number|no\.?|id|code)\b")
_REPLY_CODE = re.compile(r"\b(?!\d{4}-\d{2}-\d{2}\b)(?:[A-Z]{1,4}-)?\d[\d-]{4,}\d\b")

# Comma-separated segments that may legitimately continue a street address.
# State-zip and postcode forms come before the bare city-name form, which
# would otherwise swallow the state letters and strand the zip code.
_ADDRESS_SEGMENT = re.compile(
    r'^,[ \t]*(?:'
    r'(?i:apt|apartment|unit|suite|flat)\.?[ \t]*#?[ \t]*(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{1,6}'  # Apt 4B
    r'|[A-Z]{2}[ \t]?\d{4,5}(?:-\d{4})?'                          # IL 62704
    r'|[A-Z]{1,2}\d{1,2}[A-Z]?[ \t]\d[A-Z]{2}'                    # BS1 4ND
    r'|\d{4}[ \t]?[A-Z]{2}'                                       # 2011 AB (NL)
    r'|(?:[A-Z]{2}[ \t]+)?[A-Z]\d[A-Z][ \t]?\d[A-Z]\d'              # ON M5V 2T6 (CA)
    r'|\d{5}[ \t]+[A-Z][\w-]+(?:[ \t][A-Z][\w-]+)?'                # 80331 Munich (DE)
    # A town, optionally two words, optionally followed by its UK postcode
    # on the same segment ("Kingsmere KM2 7QL", test6): the postcode is tried
    # before a second town word so "KM2" is not read as one.
    r'|[A-Z][\w-]+(?:[ \t]+(?:' + _UK_POSTCODE + r'|[A-Z][\w-]+(?:[ \t]+' + _UK_POSTCODE + r')?))?'
    r')')
# An address span that stops after the outward half of a UK postcode
# ("8 Bryn Teg, Caernarfon LL55" from the model, test6) takes the inward half.
_OUTWARD_TAIL = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?$")
_INWARD_NEXT = re.compile(r"[ \t]\d[A-Z]{2}\b")
# The street-type words of address_line, for the custom-model address check.
_STREET_TYPE = re.compile(
    r'\b(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr'
    r'|Court|Ct|Place|Pl|Crescent|Grove|Walk|Rise|Row|Terrace|Way|Close'
    r'|House|Gardens|Mews|Hill|Yard|Square|Quay|Parade|Heights|Flat|Apt|Unit)\b', re.IGNORECASE)

_SENTENCE_BREAK = re.compile(r'[.!?\n]')


def extend_address(text, end):
    """Extend a street-line span over following apt/city/postal segments."""
    while True:
        match = _ADDRESS_SEGMENT.match(text[end:end + 40])
        if not match:
            return end
        end += match.end()


def complete_postcode(text, start, end):
    """Extend an address span over the inward half of a UK postcode it stops in."""
    if _OUTWARD_TAIL.search(text[start:end]):
        match = _INWARD_NEXT.match(text[end:end + 5])
        if match:
            return end + match.end()
    return end


def sentence_around(text, start, end):
    """The sentence (or line) containing the span — context stays local."""
    breaks = [m.start() for m in _SENTENCE_BREAK.finditer(text, 0, start)]
    s = breaks[-1] + 1 if breaks else 0
    after = _SENTENCE_BREAK.search(text, end)
    return text[s:after.start() if after else len(text)]


def is_letterhead(text, start, end, span_text):
    """True when the span alone fills one of the document's first three lines.

    Review verdict: the organization that AUTHORED a document (its letterhead)
    stays visible; only third-party facility mentions are redacted. A name
    embedded in a longer line, or deeper in the document, is not a letterhead.
    """
    if text.count("\n", 0, start) > 2:
        return False
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip() == span_text.strip()


def chat_reply_ids(text):
    """IDs answered in a chat line whose label sits in the previous line."""
    results, previous = [], None
    for line in _CHAT_LINE.finditer(text):
        if previous is not None and previous.group('speaker') != line.group('speaker') \
                and _ID_QUESTION.search(previous.group('body')):
            for code in _REPLY_CODE.finditer(line.group('body')):
                start = line.start('body') + code.start()
                results.append({'entity_type': 'GENERIC_ID', 'start': start,
                                'end': start + len(code.group()), 'score': 1.0,
                                'text': code.group(), 'source': 'regex'})
        previous = line
    return results


def run_pattern_rules(text):
    """Stage 1: every pattern rule over the text → candidate list."""
    results = chat_reply_ids(text)
    for pii_type, pattern in TEXT_PATTERNS.items():
        for match in pattern.finditer(text):
            # Context patterns retain their surrounding text for matching,
            # but only the named PII span should be redacted.
            start, end = match.span('pii') if 'pii' in match.re.groupindex else match.span()
            if pii_type == 'sensitive_organization':
                # Do not include a sentence verb such as “Visit” merely
                # because it is capitalized at the beginning of a sentence.
                leading = re.match(r'(?i)^(?:visit|at|the|from|to|for)\s+', text[start:end])
                if leading:
                    start += leading.end()
            if pii_type in ('address_line', 'address_line_de', 'address_line_es'):
                # A street line is usually followed by apartment, city, and
                # postal-code segments; the whole location identifies a home.
                end = extend_address(text, end)
            entity_override = None
            if pii_type == 'generic_id' and re.fullmatch(r'[1-9]\d{19}', text[start:end]):
                # A labelled 20-digit value is a bank account, not a generic ID.
                entity_override = 'BANK_ACCOUNT'
            results.append({
                'entity_type': entity_override or PATTERN_ENTITY_TYPES.get(pii_type, pii_type.upper()),
                'start': start,
                'end': end,
                'score': PATTERN_SCORES.get(
                    pii_type, 1.1 if pii_type in LABELLED_PATTERNS else 1.0),
                'text': text[start:end],
                'source': 'regex'
            })
    return results


def apply_policy_filters(results, text):
    """Stage 2: enforce the context rules of the V1 public-sharing policy."""
    person_spans = [(r['start'], r['end']) for r in results
                    if r['entity_type'] in ('PERSON', 'PRONOUN')]
    address_ends = [r['end'] for r in results
                    if r['entity_type'] == 'ADDRESS_LINE']
    person_texts = {r['text'] for r in results if r['entity_type'] == 'PERSON'}
    # Regex phones with an international prefix: a checksum-only identifier
    # Presidio finds strictly inside one ("612 345 678" as an Australian
    # company number inside "+34 612 345 678", test6) is part of the phone.
    intl_phones = [(r['start'], r['end']) for r in results
                   if r['entity_type'] == 'PHONE' and r.get('source') == 'regex'
                   and r['text'].startswith(('+', '00'))]

    def near_person(start, end, radius=PERSON_PROXIMITY_CHARS):
        return any(p_start - radius <= end and start <= p_end + radius
                   for p_start, p_end in person_spans)

    filtered = []
    for result in results:
        entity, source = result['entity_type'], result.get('source')
        sentence = sentence_around(text, result['start'], result['end'])
        prefix = text[max(0, result['start'] - 32):result['start']]

        # Custom-model spans are trusted for the labels the live model has
        # earned on the model-only gold run (config.CUSTOM_MODEL_TRUSTED_LABELS).
        # Anything else — in practice SENSITIVE_ORGANIZATION — still needs a
        # facility keyword in its text, the label a small model over-fires on.
        if source == 'custom_ner' and entity not in CUSTOM_MODEL_TRUSTED_LABELS \
                and not SENSITIVE_ORG_KEYWORDS.search(result['text']):
            continue
        # The document's own letterhead organization stays visible — whichever
        # layer detected it (the regex, the custom model, or general NER).
        if (entity == 'SENSITIVE_ORGANIZATION'
                or (entity == 'ORGANIZATION'
                    and SENSITIVE_ORG_KEYWORDS.search(result['text']))) \
                and is_letterhead(text, result['start'], result['end'], result['text']):
            continue
        # Group membership is a TBD policy area, and the NRP recognizer
        # over-fires on languages and nationality adjectives ("Dutch",
        # "Indian passport") — V1 keeps these.
        if entity == 'NRP':
            continue
        # An organization is personal data only when a person is tied to it
        # (employment wording) or it is a sensitive facility. "position of X"
        # or "as a X" marks a job title, not an employer.
        if entity == 'ORGANIZATION' and source in ('spacy', 'presidio'):
            if TITLE_PREFIX.search(prefix):
                continue
            if not (SENSITIVE_ORG_KEYWORDS.search(result['text'])
                    or EMPLOYMENT_CONTEXT.search(sentence)
                    or ORG_AT_PREFIX.search(prefix)):
                continue
        # The NER layers also mislabel single title words as people ("as a
        # Senior Dispatcher"); a real name never follows a title introducer.
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner') \
                and ' ' not in result['text'] and TITLE_PREFIX.search(prefix):
            continue
        # A product name after a thing label is not a person ("Vehicle: Nissan
        # Qashqai"), however the NER layers read it.
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner') \
                and THING_LABEL_PREFIX.search(prefix):
            continue
        # "the Hull depot": a site, not a person, whatever general NER says.
        operational = OPERATIONAL_SITE_SUFFIX.match(text[result['end']:result['end'] + 24])
        if entity == 'PERSON' and source in ('spacy', 'presidio') and operational:
            continue
        if source == 'presidio' and any(
                s <= result['start'] and result['end'] <= e
                and (e - s) > (result['end'] - result['start']) for s, e in intl_phones):
            continue
        # A person's name never contains a digit. General NER glues a chat
        # time onto a name ("14:03 Rhys") — strip it, on the same line only,
        # so a span that crossed a line break still dies in dedup — or reads
        # a product code as a person ("Jungheinrich EFG 216", test6) — drop it.
        if entity == 'PERSON' and source in ('spacy', 'presidio') and re.search(r'\d', result['text']):
            lead = re.match(r'(?:\S*\d\S*[ \t]+)+', result['text'])
            rest = result['text'][lead.end():] if lead else result['text']
            if not rest or re.search(r'\d', rest):
                continue
            result = {**result, 'start': result['start'] + lead.end(), 'text': rest}
        # The custom model's one trusted-label leak so far: a year followed by
        # capitalised words and no street word ("2027 Halvorsen Scholarship",
        # test6) is an event or a title, not an address.
        if entity == 'ADDRESS' and source == 'custom_ner' \
                and re.fullmatch(r'(?:19|20)\d{2}\b[^\d]*', result['text']) \
                and not _STREET_TYPE.search(result['text']):
            continue
        # An address that stops after the outward half of a UK postcode takes
        # the inward half, whichever layer found it.
        if entity in ('ADDRESS', 'ADDRESS_LINE'):
            new_end = complete_postcode(text, result['start'], result['end'])
            if new_end != result['end']:
                result = {**result, 'end': new_end, 'text': text[result['start']:new_end]}
        # A standalone city/region stays; redact only in residence context
        # or straight after a street address on the same line (commas and
        # spaces between — never across a sentence or line break).
        if entity in ('LOCATION', 'GPE', 'LOC') and source in ('spacy', 'presidio'):
            near_address = any(
                0 <= result['start'] - a_end <= 8
                and re.fullmatch(r'[ \t,]*', text[a_end:result['start']])
                for a_end in address_ends)
            # An operational site ("the Oslo hub", "Stavanger branch") stays
            # even inside a travel or residence sentence.
            if operational and not near_address:
                continue
            if not (RESIDENCE_CONTEXT.search(sentence)
                    or TRAVEL_CONTEXT.search(sentence) or near_address):
                continue
        # A line-initial greeting name counts only when the same name occurs
        # inside a fully detected person elsewhere in the document.
        if entity == 'GREETING_NAME':
            if not any(result['text'] in p and result['text'] != p for p in person_texts):
                continue
            result = {**result, 'entity_type': 'PERSON'}
        # Plural pronouns refer to things as often as to people.
        if entity == 'PRONOUN_GROUP':
            if not near_person(result['start'], result['end']):
                continue
            result = {**result, 'entity_type': 'PRONOUN'}
        # Times are redacted only next to a person ("HE left at 2 o'clock").
        if entity == 'TIME_EXPR':
            if not near_person(result['start'], result['end'], radius=120):
                continue
            result = {**result, 'entity_type': 'TIME'}
        filtered.append(result)
    return filtered


# First names that are also ordinary sentence-initial English words; the
# first-name echo leaves these alone.
_ECHO_FIRST_NAME_STOP = frozenset(
    "Will Mark Grace Hope Bill Frank Dean Earl Gene Jack Rose Dawn Faith Rich Carol "
    "Victor Wade Chase Grant Hunter Mason Lane Page Price Summer Autumn Chance "
    "Christian Major General Sandy Rusty Rocky Cliff Drew Pierce Herb Reed Lance "
    "Jade Pearl Sage June April August Sunny Angel Blue Brook Cash Clay Dale Forest "
    "Glen Jewel King Knight Marshall Miles Moss North Ocean Penny Rain Ridge Rock "
    "Sterling Stone Wells West Winter Trust Care Home".split())
_TITLE_WORD = re.compile(r"^(?:Mr|Mrs|Ms|Mx|Miss|Dr|Prof|Sr|Sra|Herr|Frau|Mme|Mlle|M)\.?$")


def echo_person_names(results, text):
    """Stage 3: repeat each detected person name over its other occurrences.

    A multi-word name also lends its first name: "Sigrún Hallgrímsdóttir" on
    the pupil line makes the later bare "Sigrún" the same person (test6).
    Case-sensitive, four letters or more, and never a word that opens
    sentences in its own right.
    """
    extra, seen = [], set()
    person_spans = [(r['start'], r['end']) for r in results if r['entity_type'] == 'PERSON']

    def echo_first_name(name):
        for match in re.finditer(r'\b' + re.escape(name) + r'\b', text):
            if any(s <= match.start() and match.end() <= e for s, e in person_spans):
                continue  # inside a detected person already
            extra.append({'entity_type': 'PERSON', 'start': match.start(),
                          'end': match.end(), 'score': 0.9,
                          'text': name, 'source': 'echo'})

    for result in results:
        name = result['text']
        if result['entity_type'] != 'PERSON' or len(name) < 4 or name in seen:
            continue
        seen.add(name)
        for match in re.finditer(r'\b' + re.escape(name) + r'\b', text):
            if match.start() != result['start']:
                extra.append({'entity_type': 'PERSON', 'start': match.start(),
                              'end': match.end(), 'score': 0.9,
                              'text': name, 'source': 'echo'})
        words = [w for w in name.split() if not _TITLE_WORD.match(w)]
        first = words[0] if len(words) > 1 else None
        if first and len(first) >= 4 and first.isalpha() and first[0].isupper() \
                and first not in _ECHO_FIRST_NAME_STOP and first not in seen:
            seen.add(first)
            echo_first_name(first)
    return extra


def relabel_by_id_context(results, text):
    """Stage 5: an explicit ID label before a span decides its entity type."""
    for result in results:
        token = ID_LABEL_TOKEN.search(text[max(0, result['start'] - 12):result['start']])
        if token:
            result['entity_type'] = ID_LABEL_OVERRIDES[token.group(1).lower()]
    return results
