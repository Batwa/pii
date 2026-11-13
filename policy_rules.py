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
    r"|(?:Medical|Health|Healthcare|Dental|Surgical)[ \t]+(?:Group|Partnership)"
    r"|(?<=\w[ \t])(?:Surgery|Nursery)"
    r"|(?:Dental|GP|Medical|General|Family)[ \t]+Practice"
    r"|(?:Medical|Support|Health|Care|Wellbeing|Wellness|Counsell?ing"
    r"|Rehabilitation|Language|Speech|Therapy|Treatment|Dialysis|Cancer"
    r"|Diagnostic|Crisis|Day)[ \t]+Cent(?:er|re)"
    r"|Counsell?ing[ \t]+Service"
)
# Facility words that lead the name in the Romance and German order ("Colégio
# São Bento", "Instituto Politécnico do Mondego", test7): schools,
# universities and hospitals only, none of them an English word.
_FACILITY_LEAD_WORDS = (
    r"Colegio|Colégio|Collège|Collegio|Instituto|Istituto|Institut|Escuela|Escola"
    r"|École|Scuola|Universidad|Universidade|Università|Université|Universität"
    r"|Liceo|Liceu|Lycée|Clínica|Clinica|Clinique|Hôpital|Ospedale|Krankenhaus"
    r"|Klinik|Klinikum|Hochschule|Grundschule|Realschule")
# The same facility words in capitals, for forms completed in block capitals
# ("PRIORY LANE SURGERY", test7). Only letters outside regex escapes change.
_FACILITY_WORDS_CAPS = re.sub(r"(?<!\\)[a-z]+", lambda m: m.group().upper(), _FACILITY_WORDS)
SENSITIVE_ORG_KEYWORDS = re.compile(
    r"\b(?:" + _FACILITY_WORDS + r"|" + _FACILITY_LEAD_WORDS + r")\b")
# A detected organization is redacted only when the document ties a person to it.
EMPLOYMENT_CONTEXT = re.compile(
    r"(?i)\b(?:employ\w*|works?(?!\s+(?:best|well|fine|properly|again|only|perfectly"
    r"|on|with|by|like|when|if|until)\b)\b|worked|working(?!\s+(?:days?|hours?|weeks?|time)\b)"
    r"|hired|joined|staff|salary"
    r"|payroll|position|internship|colleague|report(?:s|ed)?\s+to|offer\s+you)"
)
# "Engineer at Helix Dynamics" — a role followed by at/joins directly before it.
# A relationship noun may sit in between: "a senior buyer at our supplier
# Sông Hồng Precision Co." (test7).
ORG_AT_PREFIX = re.compile(
    r"(?i)\b(?:at|joins?|joining)\s+"
    r"(?:(?:our|the|their|his|her|its|a)\s+(?:supplier|client|customer|partner|subsidiary"
    r"|contractor|distributor|agency|firm|company|employer|sister\s+company"
    r"|parent\s+company)\s+)?[\"'“]?\s*$")
# "three years as a picker for Strathmore Wholesale": a role, then for / at /
# with, then the employer (test8).
ORG_ROLE_FOR_PREFIX = re.compile(
    r"(?i)\bas\s+an?\s+(?:[a-z][a-z-]*\s+){1,3}(?:for|at|with)\s+$")
# The heading of a CV block makes every organization under it an employer.
EMPLOYMENT_HEADING = re.compile(
    r"(?i)^(?:(?:work|employment|career|professional|job)\s+(?:history|experience|record)"
    r"|experience|employment|employers?|career|previous\s+(?:jobs|roles|employment))\s*:?$")
# A detected city/region is redacted only in a residence or work-address context.
RESIDENCE_CONTEXT = re.compile(
    r"(?i)\b(?:resid\w*|lives?\s+in|living\s+in|address|registered\s+at|home"
    r"|apt\.?|apartment|unit\s+\d|flat\s+\d|premises|office\s+at|headquarters\s+at)\b"
)
# A city as a person's journey destination is redacted (review verdict S014);
# operational locations — warehouses, branches, hubs — stay visible, so the
# nouns here are itinerary words, not "arrives"/"dispatched" cargo phrasing.
TRAVEL_CONTEXT = re.compile(
    r"(?i)\b(?:arrival|departs|departure|itinerary|travel(?:s|led|ed|ling|ing)?\s+to"
    r"|trip\s+to|fl(?:y|ies|ying|ew)\s+to)\b"
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
PLACE_OF_PREFIX = re.compile(
    r'(?i)\b(?:city|town|village|municipality|district|region|republic|province'
    r'|county|canton)\s+of\s+$')
# A label naming a thing-ID directly before a digit run decides its type:
# invoice, batch, flight, voucher, tracking and project identifiers stay
# visible, so the run is not a phone or a card whatever its shape ("Invoice
# no. 0042 0117 3381", "batch 0031-22-456", "SSCC: 0035 9012 ...", test7).
THING_ID_LABEL_PREFIX = re.compile(
    r'(?i)\b(?:invoice|receipt|batch|lot|tracking|sscc|gtin|ean|upc|barcode|sku|part'
    r'|voucher|flight|project|consignment|shipment|parcel|waybill|awb|delivery\s+note)'
    r'\s*(?:no\.?|number|num\.?|#|id|code|ref\.?|reference)?\s*[:#.]?\s*$')
_DIGIT_SHAPED = re.compile(r'[\d \t().+/-]+')
# A German-shaped plate counts only where the text speaks of a vehicle or a
# registration, and never straight after a place or type label: "USB-C 3.2",
# "DIN-A 4", "region DE-HE 3" and "Bay B-C 4" share the shape (test7).
PLATE_CONTEXT = re.compile(
    r'(?i)\b(?:registration|registered|reg\.?|plates?|kennzeichen|vehicles?|cars?|vans?'
    r'|trucks?|lorr(?:y|ies)|trailers?|motorcycles?|motorbikes?|fahrzeuge?|pkw|lkw)\b')
PLACE_LABEL_PREFIX = re.compile(
    r'(?i)\b(?:bay|room|zone|gate|dock|row|aisle|shelf|rack|slot|section|sector|area'
    r'|region|block|building|wing|level|floor|port|type|class|version|standard|norm'
    r'|size|format|category|code)\s*[:#]?\s*$')
# Birth labels in French, German, Spanish, Italian and Portuguese ("née le
# 27/08/1975", test8), and the French month names for the worded date.
_BIRTH_LABELS_INTL = (r"n[ée]e?\s+le|date\s+de\s+naissance|geboren\s+am|geb\.|geburtsdatum"
                      r"|fecha\s+de\s+nacimiento|nacid[oa]\s+el|data\s+di\s+nascita"
                      r"|nat[oa]\s+il|data\s+de\s+nascimento|nascid[oa]\s+(?:em|a)")
_MONTHS_FR = (r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre"
              r"|octobre|novembre|d[ée]cembre)")
# A date is a DOB only with an explicit birth label directly before it; a
# format hint may sit between the two ("Date of birth (DD/MM/YYYY): ...").
DOB_CONTEXT = re.compile(
    r'(?i)(?:born(?:\s+on)?(?:\s+the)?|date\s+of\s+birth|birth\s+date|birthday|d\.?o\.?b\.?'
    r'|' + _BIRTH_LABELS_INTL + r')'
    r'(?:\s*\([^)\n]{0,24}\))?\s*[:,-]?\s*$')

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
# Italian and Portuguese titles carry a suffix after the dot ("Dott.ssa",
# "Sig.ra", "Eng.ª"; test7).
_TITLES = (r"(?:Mr|Mrs|Ms|Mx|Miss|Dr|Dra|Prof|Rev|Fr|Sr|Sra|Srta|Herr|Frau|Mme|Mlle|M"
           r"|Dott|Sig|Ing|Eng|Avv|Cllr|Sir|Dame|Capt|Sgt)(?:\.(?:ssa|ra|na))?\.?[ªº]?")
_TITLE_OPT = r"(?:" + _TITLES + r"[ \t]+)?"
# Addressees that are collective or role words, never a first name.
_GREETING_STOP = (
    r"(?:All|Everyone|Everybody|Team|Customers?|Colleagues?|Sirs?|Madam|Friends?"
    r"|Parents?|Guests?|Members?|Residents?|Tenants?|Patients?|Students?|Applicants?"
    r"|Doctor|Nurse|Both|Partners?|Valued|Folks|Guys|Mum|Dad|Mom|You|There|Readers?"
    r"|World|Diary|Again|Later|Soon|Today|Tomorrow|Yesterday|Tonight|Monday|Tuesday"
    r"|Wednesday|Thursday|Friday|Saturday|Sunday|Support|Sales|Admin|Office|Reception"
    r"|Staff|Users?|Subscribers?|Volunteers?|Neighbou?rs?|Mr|Mrs|Ms|Mx|Miss|Dr|Prof"
    r"|Lord|Lady"
    # Residents and account holders ("Dear Householder,", test7).
    r"|Householders?|Occupiers?|Occupants?|Homeowners?|Leaseholders?|Landlords?|Owners?"
    r"|Cardholders?|Policyholders?|Shareholders?|Stakeholders?|Investors?|Donors?"
    r"|Supporters?"
    # Departments and bodies ("Hi Finance,", "Dear Committee,", test7).
    r"|Finance|Accounts|Accounting|Payroll|Marketing|Legal|Compliance|Procurement"
    r"|Purchasing|Operations|Logistics|Facilities|Maintenance|Security|Housekeeping"
    r"|Billing|Helpdesk|Services?|Management|Dispatch|Warehouse|Recruitment|Engineering"
    r"|Production|Planning|Committee|Board|Council|Trustees|Governors?|Panel|Editors?"
    r"|Crew|Family"
    # Roles.
    r"|Managers?|Supervisors?|Directors?|Officers?|Professors?|Teachers?|Drivers?"
    r"|Candidates?|Participants?|Attendees?|Delegates?|Passengers?|Visitors?|Suppliers?"
    r"|Vendors?|Contractors?|Clients?|Buyers?|Sellers?|Employees?|Employers?|Recruiters?"
    r"|Hiring|Ladies|Gentlemen|Children|Kids|Class|Pupils|Graduates|Alumni|Players?"
    r"|Coach|Captain|Councillors?|Minister|Senator|President|Mayor|Judge|Sister|Brother"
    r"|Father|Mother|Madame|Monsieur|Comrades?|Mates?|Darling|Love"
    # Site and service roles ("Hey Caretaker,", test8).
    r"|Caretakers?|Janitors?|Concierge|Wardens?|Receptionists?|Agents?|Operators?"
    r"|Engineers?|Technicians?|Cleaners?|Secretary|Treasurer|Chair|Chairman|Chairwoman"
    r"|Clerk|Organi[sz]ers?|Moderators?|Reviewers?|Hosts?|Stewards?|Administrators?)")
# Lowercase nouns that name a person by role or relation, for the apposition
# rule ("the mechanic, Pim, says").
_PERSON_NOUNS = (
    r"mechanic|engineer|technician|plumber|electrician|driver|manager|supervisor|owner"
    r"|landlord|landlady|tenant|neighbou?r|caretaker|nurse|doctor|teacher|coach|receptionist"
    r"|secretary|assistant|colleague|friend|brother|sister|son|daughter|wife|husband|partner"
    r"|mother|father|mum|mom|dad|uncle|aunt|cousin|nephew|niece|boss|agent|advis[eo]r"
    r"|solicitor|accountant|surveyor|inspector|officer|carer|cleaner|gardener|builder|foreman"
    r"|chef|instructor|tutor|student|apprentice|trainee|intern|volunteer|director|chair"
    r"|chairman|chairwoman|treasurer|representative|rep|courier|operator|dispatcher|fitter"
    r"|installer|referee|witness|guarantor|applicant|patient|client|customer|visitor|guest")
# Form values that fill a name field without being a name ("MIDDLE NAME:
# NONE", test7).
_NOT_A_NAME = (r"(?!(?:none|n/?a|nil|null|not|unknown|same|as|see|tbc|tba|tbd|blank|yes|no"
               r"|withheld|refused|prefer|optional|required|pending|ditto|void|self)\b)")
# An identifier value is never a bare date ("hire agreement 31/10/2026 to
# 03/11/2026", test7). Day and month ranges are checked so a hyphenated
# code such as "55-20931" is not mistaken for one, and the date must be the
# whole value: a registry number that starts with one ("record No.
# 2014-03-09/0771") is still an identifier.
_NOT_A_DATE = (r"(?!(?:(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}"
               r"|(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01])[/.-](?:19|20)\d{2}"
               r"|(?:19|20)\d{2}[/.-](?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01]))(?![\w/-]))")
_CLOCK_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
# Block-capital values under a role label that are roles, bodies or notes,
# not names ("DRIVER: COMPANY POOL VEHICLE", "WITNESS: POLICE OFFICER").
_CAPS_NOT_A_NAME = re.compile(
    r"\b(?:COMPANY|POOL|VEHICLE|POLICE|OFFICER|CONSTABLE|COUNCIL|DEPARTMENT|TEAM|STAFF"
    r"|MANAGER|ABOVE|ATTACHED|APPLICABLE|REQUIRED|AVAILABLE|PRESENT|PROVIDED|MALE|FEMALE"
    r"|ADULT|CHILD|SELF|LTD|LIMITED|SERVICES?|AGENCY|OFFICE|UNIT|WARD|SOCIAL|WORKER"
    r"|NURSE|DOCTOR|TEACHER|SCHOOL|HOSPITAL|CLINIC|CENTRE|CENTER|TRUST|GROUP)\b")
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
    # The same number with the area code in brackets ("(01632) 960044",
    # test8); the digit count is fixed in the same way.
    'phone_uk_bracket': re.compile(
        r'(?<!\w)\((?=0(?:[ )-]*\d){10}(?![ -]?\d))0\d{2,5}\)[ \t]?\d{3,4}(?:[ \t-]?\d{3,4})?(?!\w)'),
    # A national number in a shape no pattern above knows, carried by its
    # label ("call 1800 555 014", test7): seven to thirteen digits in
    # separated groups directly after a phone word.
    'phone_labelled': re.compile(
        r"\b(?:call(?:[ \t]+us)?(?:[ \t]+(?:on|at))?|phone|telephone|tel\.?|telefono|teléfono"
        r"|téléphone|tél\.?|portable|telefon|handy|telemóvel|móvil|cellulare|mobile|mob\.?|fax"
        r"|helpline|hotline|freephone)"
        r"[ \t]*(?:no\.?|number)?[ \t]*[:.]?[ \t]*"
        r"(?P<pii>(?=(?:[ .-]?\d){7,13}(?![ .-]?\d))\d{2,5}(?:[ .-]\d{2,5}){1,4})(?!\w)",
        re.IGNORECASE),
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
        r'\b(?:passport|passeport|pasaporte|passaporte|passaporto|reisepass)\s*'
        r'(?:no\.?|n[°ºo]\.?|nr\.?|number|numéro|número|numero|nummer|#)?\s*[:#]?\s*'
        r'(?P<pii>(?=[A-Z0-9]*\d)[A-Z0-9]{6,10})\b', re.IGNORECASE),
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
        r"(?<![\w-])[A-ZÄÖÜ]{1,3}-[A-Z]{1,2}[ \t](?!(?:19|20)\d{2}\b)\d{1,4}[EH]?(?![\w-])(?!\.\d)"),
    # A labelled plate in a format no pattern above knows ("Targa / Plate:
    # FX 482 LD", test7): two to four short capital groups holding a letter
    # and a digit. Scored like the other label fallbacks, so a country
    # recognizer keeps its label.
    'plate_labelled': re.compile(
        r"\b(?:number\s+plate|licen[cs]e\s+plate|plate|registration(?:\s+(?:no\.?|number|mark))?"
        r"|reg\.?(?:\s+no\.?)?|targa|kennzeichen|matr[ií]cula|immatriculation)(?!\w)\s*[:#]?\s*"
        r"(?P<pii>(?-i:(?=[A-Z0-9 \t-]*\d)(?=[0-9 \t-]*[A-Z])[A-Z0-9]{1,4}(?:[ \t-][A-Z0-9]{1,4}){1,3}))"
        r"(?![A-Za-z0-9])", re.IGNORECASE),

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
        r'\b(?:(?:customer|client|member(?:ship)?|loyalty|patient|employee|student|account'
        r'|badge|case|claim|record|mrn|dni|nie|nif|curp|cpf|bsn|pesel'
        r'|codice\s+fiscale|fiscal\s+code|tax\s+code|national\s+id|id(?:entity)?\s+card)\s*'
        r'(?:id|number|no\.?|record|reference|ref\.?|#)?'
        r'|(?:hospital|staff|payroll)\s+(?:id|number|no\.?)'
        r'|(?:application|visa|permit)\s+(?:reference|ref\.?|number|no\.?|id))\s*[:#.]?\s*'
        # The value may be printed in spaced digit groups, eight digits or
        # more in all ("gas account 8841 2207 16", test8).
        r'(?P<pii>' + _NOT_A_DATE + r'(?:(?=(?:\d[ \t]?){8,20}(?!\d))\d{2,6}(?:[ \t]\d{2,6}){1,4}(?![ \t]?\d)'
        r'|(?=[A-Z0-9/-]{0,20}\d)[A-Z0-9][A-Z0-9/-]{3,19}))', re.IGNORECASE),
    # A loyalty or membership card number is a customer identifier, usually
    # printed in spaced digit groups the generic value shape does not take
    # ("loyalty card 6338 4410 0027", test7).
    'card_number_labelled': re.compile(
        r'\b(?:loyalty|membership|member|rewards?|club|store|library|points)\s+card'
        r'\s*(?:no\.?|number|#)?\s*[:#]?\s*(?P<pii>\d{3,6}(?:[ \t-]\d{3,6}){1,4})\b', re.IGNORECASE),
    # "Membership renewals are due Friday, mine is AL-2291": the possessive
    # points back at a person-ID label in the same sentence (test7).
    'possessive_id': re.compile(
        r"(?i)\b(?:account|member(?:ship)?|customer|client|patient|policy|case|claim|insurance)\b"
        r"[^.!?\n]{0,60}\b(?:mine(?:\s+is|['’]s)|my\s+(?:number|one)\s+is)\s+"
        r"(?P<pii>(?-i:(?:[A-Z]{1,4}-)?\d[\d-]{2,}\d\b|[A-Z]{1,4}-\d{3,}\b))"),
    # A person's group/class designation ("class 9-B", "Group PMI-2601").
    'group_class': re.compile(
        r'\b(?:class|grade|group|form)\s+'
        r'(?P<pii>\d{1,2}-?[A-Z]\b|[A-Z]{2,5}-\d{2,6})', re.IGNORECASE),
    # Trackable purchase/product identifiers (review verdict: serial and order
    # numbers can be searched up and tracked). Batch numbers stay visible.
    'serial_number': re.compile(
        # A frame, chassis or IMEI number is a serial number under its own
        # label, and the value may be printed in spaced capital groups
        # ("Frame number: WTU 284 K 0931 M", test8).
        r'\b(?:(?:serial|hull|imei)\s*(?:no\.?|number|id|#)?|(?:frame|chassis)\s*(?:no\.?|number|#))'
        r'\s*[:#]?\s*'
        r'(?P<pii>(?-i:(?=[A-Z0-9 \t]*\d)[A-Z0-9]{1,8}(?:[ \t][A-Z0-9]{1,8}){1,5})(?![A-Za-z0-9-])'
        r'|(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{4,19}\b)', re.IGNORECASE),
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
        r'(?P<pii>' + _NOT_A_DATE + r'(?=[A-Z0-9/-]*\d)(?:[A-Z0-9]+[-/][A-Z0-9/-]+|[A-Z0-9]{6,20}))\b',
        re.IGNORECASE),

    # --- dates (policy: only birth dates redact; issue dates stay) ---
    # Month-day-year, day-month-year, and worded forms ("5 May 1994",
    # "March 3, 1985"), each anchored to an explicit birth label.
    'date_of_birth': re.compile(
        r'\b(?:date\s+of\s+birth|birth\s+date|birthday|born(?:\s+on)?|d\.?o\.?b\.?'
        r'|' + _BIRTH_LABELS_INTL + r')'
        r'(?:\s*\([^)\n]{0,24}\))?\s*[:,-]?\s*(?:the\s+)?(?P<pii>'
        r'(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])'
        r'|(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01])[-/.](?:19|20)\d{2}'
        r'|(?:0?[1-9]|[12][0-9]|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}'
        r'|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?' + _MONTHS + r',?\s+(?:19|20)\d{2}'
        r'|' + _MONTHS + r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}'
        r'|\d{1,2}(?:er)?\s+' + _MONTHS_FR + r'\s+(?:19|20)\d{2}'
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
    # The block-capital form ("PREVIOUS GP: DR M. FAIRWEATHER, PRIORY LANE
    # SURGERY", test7): at least one capital word before the facility word.
    'sensitive_organization_caps': re.compile(
        r"(?P<pii>(?:[A-Z][A-Z.'’-]+[ \t]+(?:(?:AND|OF|FOR|&)[ \t]+)?){1,4}"
        r"(?:" + _FACILITY_WORDS_CAPS + r")(?![A-Za-z]))"),
    # Romance and German order: the facility word leads ("Colégio São
    # Bento", "Instituto Politécnico do Mondego", test7).
    'sensitive_organization_lead': re.compile(
        r"\b(?P<pii>(?:" + _FACILITY_LEAD_WORDS + r")"
        r"(?:[ \t]+(?:(?:de|del|do|da|dos|das|di|della|delle|degli|des|du|la|las|los|le|les"
        r"|e|y|et|für|der|am)[ \t]+){0,3}[A-ZÀ-ÖØ-Þ][\w'’-]+){1,4})"),
    # The word-boundary before the street-type alternation is load-bearing:
    # without it "...of the contraCT" and "...with standaRD" match.
    # The house number is never the minutes of a clock time or the decimals
    # of an amount ("13:00 while the main in Tannery Road", test7).
    'address_line': re.compile(
        r"(?<!\d[:.])\b\d+\s+[A-Za-z][A-Za-z\s'’]{0,40}?"
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
    # Polish-order street line: an abbreviated street word, the name, then
    # the house number with an optional flat ("ul. Marszałkowska 84/12",
    # test7); the segment extension then takes ", 00-590 Warszawa".
    'address_line_pl': re.compile(
        r"(?<!\w)(?P<pii>(?:[Uu]l\.|[Uu]lica|[Aa]leja|(?<!et )[Aa]l\.|[Pp]lac|[Oo]s\.)[ \t]+"
        r"[^\W\d_][\w.'’-]*(?:[ \t]+[^\W\d_][\w.'’-]*){0,3}?"
        r"[ \t]+\d{1,4}[A-Za-z]?(?:[/-]\d{1,4})?)(?!\w)"),
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
    # Never glued to a hyphen: the "HE" of the region code "DE-HE 3" is not
    # a pronoun (test7).
    'pronoun': re.compile(r'(?<!-)\b(?:' + '|'.join(PRONOUNS_ALWAYS) + r')\b(?!-)', re.IGNORECASE),
    # Plural pronouns: kept only as candidates; apply_policy_filters drops them
    # unless a person is detected nearby.
    'pronoun_group': re.compile(r'(?<!-)\b(?:' + '|'.join(PRONOUNS_NEAR_PERSON) + r')\b(?!-)', re.IGNORECASE),
    # Clock time with optional day-part; apply_policy_filters requires a person
    # nearby, so timetables and opening hours stay untouched.
    # "half past six in the evening" and worded hours count like "2 o'clock"
    # (test7).
    'time_expr': re.compile(
        r"\b(?:(?:\d{1,2}|" + _CLOCK_WORDS + r")\s*o['’]?clock"
        r"|(?:half|(?:a\s+)?quarter)\s+(?:past|to)\s+(?:\d{1,2}|" + _CLOCK_WORDS + r")\b)"
        r"(?:\s+in\s+the\s+(?:morning|evening|afternoon|night)|\s+at\s+night)?", re.IGNORECASE),

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
        + _NOT_A_NAME + r"(?P<pii>(?-i:" + _TITLE_OPT + _NAME + r"))", re.IGNORECASE),
    # The same role labels on a form completed in block capitals ("NEXT OF
    # KIN: KWABENA ADESANYA (BROTHER)", test7): only labels that take people,
    # and at least two capital words, so "PATIENT: MALE" is not a name.
    'labelled_person_caps': re.compile(
        r"(?m)^[ \t]*(?:PATIENT|STUDENT|EMPLOYEE|TRAVELLER|TRAVELER|APPLICANT|CLAIMANT"
        r"|WITNESS|MOTHER|FATHER|CANDIDATE|EMERGENCY[ \t]+CONTACT|NEXT[ \t]+OF[ \t]+KIN"
        r"|PASSENGER|PUPIL|GUARDIAN|SPOUSE|PARTNER|BENEFICIARY|DRIVER|CARER"
        r"|PARENT(?:[ \t]*/[ \t]*(?:CARER|GUARDIAN))?|REFERRED[ \t]+BY)"
        r"(?:[ \t]+\d{1,2})?(?:[ \t]*\([^)\n]{0,40}\))?[ \t]*[:\-][ \t]*"
        r"(?!(?:NONE|N/?A|NIL|NOT|UNKNOWN|SAME|AS|SEE|TBC|TBA|YES|NO)\b)"
        r"(?P<pii>(?:(?:MR|MRS|MS|MX|MISS|DR|PROF|REV)\.?[ \t]+)?"
        r"[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’-]+(?:[ \t]+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’-]+){1,3})(?![a-zß-ÿ])"),
    # Name fields of a form ("SURNAME: ONYEKWERE", "GIVEN NAMES: CHIDI EMEKA",
    # "Full name: Lebedev Kirill Andreevich"): a line-initial label, then an
    # ALL-CAPS run or Title Case name words (test6: the NER layers and the
    # name regexes all wanted Title Case). The bare label "Name:" is left out;
    # it names products and companies as often as people.
    'name_field': re.compile(
        r"^[ \t]*(?:(?:previous|former|other|birth)\s+(?:sur)?names?|surname|family\s+name"
        r"|last\s+name|given\s+names?|first\s+names?"
        r"|forenames?|middle\s+names?|(?:mother['’]?s\s+)?maiden\s+name|full\s+name"
        r"|preferred\s+name|name\s+at\s+birth"
        r"|(?:applicant|patient|tenant|student|employee|customer|client|resident|hirer|insured)['’]?s?\s+name"
        r"|name\s+of\s+(?:the\s+)?(?:applicant|patient|tenant|student|employee|customer|client|holder|insured))"
        r"[ \t]*[:\-][ \t]*" + _NOT_A_NAME + r"(?P<pii>(?-i:" + _TITLE_OPT + _NAME_WORD
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
    # A sign-off: a closing word, a comma, and one name word that ends the
    # line ("Thanks, Bronagh", "Regards,\nMargaux"; test7).
    'signoff_person': re.compile(
        r"\b(?:Thanks|Thank you|Many thanks|Regards|Kind regards|Best regards|Warm regards"
        r"|Best wishes|Best|Cheers|Sincerely|Yours(?:[ \t]+(?:sincerely|faithfully|truly))?"
        r"|Love|Take care|Ta|Thx)[,!][ \t]*\n?[ \t]*(?!" + _GREETING_STOP + r"\b)(?P<pii>"
        + _NAME_WORD + r")[ \t]*(?=\n|$)"),
    # A courtesy title carries the name behind it, also where the surname
    # is a company word ("Frau Bank", test8). The title stays outside the
    # span.
    'titled_person': re.compile(
        r"(?<![\w.])(?:Mr|Mrs|Ms|Mx|Miss|Dr|Prof|Herr|Frau|Hr\.|Fr\.|Mme|Mlle|Monsieur|Madame"
        r"|Signora|Signor|Sig\.(?:ra|na)?|Dott\.(?:ssa)?|Señora?|Sra\.)\.?[ \t]+"
        r"(?!" + _GREETING_STOP + r"\b)(?P<pii>(?:(?:van|der|den|de|von|da|di|du|la|le|del|della"
        r"|dos|das|ten|ter|al|el|bin)[ \t]+){0,2}" + _NAME_WORD
        + r"(?:" + _NAME_PARTICLES + r"[ \t]" + _NAME_WORD + r"){0,2})"),
    # "The mechanic, Pim, says ...": a person noun, a comma, a name, a comma
    # (test8).
    'apposition_person': re.compile(
        r"\b(?:[Tt]he|[Mm]y|[Oo]ur|[Hh]is|[Hh]er|[Tt]heir|[Yy]our)[ \t]+(?:[a-z][a-z'’-]*[ \t]+){0,2}"
        r"(?:" + _PERSON_NOUNS + r"),[ \t]+(?!" + _GREETING_STOP + r"\b)(?P<pii>" + _TITLE_OPT
        + _NAME_WORD + r"(?:" + _NAME_PARTICLES + r"[ \t]" + _NAME_WORD + r"){0,2})(?=,)"),
    # A line-initial employer label names the employer, also in block
    # capitals, where general NER runs on into the next line ("EMPLOYER:
    # WESTMORLAND DAIRIES LTD", test8).
    'employer_field': re.compile(
        r"(?mi)^[ \t]*(?:(?:current|present|previous|former|last|most[ \t]+recent)[ \t]+)?"
        r"(?:employer(?:['’]s)?(?:[ \t]+name)?|place[ \t]+of[ \t]+work|workplace)[ \t]*[:\-][ \t]*"
        r"(?!(?:none|n/?a|nil|not|unknown|same|as|see|tbc|tba|self|unemployed|retired|student"
        r"|various|agency|housewife|homemaker)\b)"
        r"(?P<pii>(?-i:[A-ZÀ-ÖØ-Þ0-9])[^\n,;(|]{1,60}?)\.?[ \t]*(?=$|[,;(|]|[ \t]{2,})"),
    # A city named as a person's journey destination ("I fly to Bologna on
    # the 14th", test8), for the towns general NER does not tag. Sites,
    # departments and venues are not destinations: a longer capitalised name
    # ("trip to Bramblewood Nature Reserve", test6) or a venue word ends it.
    'travel_city': re.compile(
        r"\b(?i:fl(?:y|ies|ying|ew)|travel(?:s|led|ed|ling|ing)?|trip)[ \t]+(?:back[ \t]+)?to[ \t]+"
        r"(?!(?:" + _GREETING_STOP + r"|Head|Site|Gate|Terminal|Platform|Customer|Hall|Room"
        r"|Building|Block|Level|Floor|Stage|Phase|Version|Windows|January|February|March|April"
        r"|May|June|July|August|September|October|November|December)\b)"
        r"(?P<pii>" + _NAME_WORD + r"(?:[ \t]" + _NAME_WORD + r")?)(?![\w'’-])"
        r"(?![ \t]+(?:[A-ZÀ-ÖØ-Þ]|(?i:hub|branch|depot|warehouse|exchange|terminal|plant|factory"
        r"|showroom|office|site|yard|store|outlet|facility)\b))"),
    # The French signing formula names the signer's town ("Fait à Dijon, le
    # 2 novembre 2026", test8).
    'signing_place': re.compile(
        r"\bFait[ \t]+à[ \t]+(?P<pii>[A-ZÀ-ÖØ-Þ][\w'’-]+(?:[ \t-][A-ZÀ-ÖØ-Þ][\w'’-]+){0,2})"
        r"(?=,|[ \t]+le\b)"),
    # The bare label "Name:" names products and companies as often as
    # people, so its value is only a candidate: apply_policy_filters keeps it
    # when a NER layer saw a person inside it ("Name: Thiago Monteiro dos
    # Santos", test7, where the model stopped before the particle).
    'name_line': re.compile(
        r"(?m)^[ \t]*Name[ \t]*:[ \t]*(?P<pii>" + _TITLE_OPT + _NAME_WORD
        + r"(?:" + _NAME_PARTICLES + r"[ \t]" + _NAME_WORD + r"){1,3})[ \t]*$"),
    # A dated line of a CV: a year range, the role, a comma, the employer
    # ("2015–2019  Junior surveyor, Granja & Filhos", test7). The year range
    # with a role is the employment wording.
    'employment_line': re.compile(
        r"(?m)^[ \t]*(?:19|20)\d{2}[ \t]*[–—-][ \t]*"
        r"(?:(?:19|20)\d{2}|(?i:present|now|current|today|ongoing))[ \t:]+[^,\n]{2,60},[ \t]*"
        r"(?P<pii>[A-ZÀ-ÖØ-Þ][\w.'’&-]*"
        r"(?:[ \t]+(?:&|and|of|de|do|da|dos|del|e|y|[A-ZÀ-ÖØ-Þ0-9][\w.'’&-]*)){0,6})"
        r"[ \t]*(?=,|$)"),
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
                     "order_number", "sort_code", "pcn_number", "agreement_number",
                     "card_number_labelled"}
PATTERN_ENTITY_TYPES = {"inn_labelled": "INN", "apt_unit": "ADDRESS_LINE",
                        "address_line_de": "ADDRESS_LINE",
                        "address_line_es": "ADDRESS_LINE",
                        "address_line_pl": "ADDRESS_LINE",
                        "sensitive_organization_caps": "SENSITIVE_ORGANIZATION",
                        "sensitive_organization_lead": "SENSITIVE_ORGANIZATION",
                        "plate_labelled": "LICENSE_PLATE",
                        "phone_labelled": "PHONE",
                        "card_number_labelled": "GENERIC_ID",
                        "possessive_id": "GENERIC_ID",
                        "labelled_person_caps": "PERSON",
                        "signoff_person": "PERSON",
                        "titled_person": "PERSON",
                        "apposition_person": "PERSON",
                        "employer_field": "ORGANIZATION",
                        "travel_city": "LOCATION",
                        "signing_place": "LOCATION",
                        "phone_uk_bracket": "PHONE",
                        "name_line": "NAME_LINE",
                        "employment_line": "ORGANIZATION",
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
PATTERN_SCORES = {"passport_labelled": 0.45, "driving_licence": 0.45,
                  "plate_labelled": 0.45, "phone_intl": 0.99}

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
    r'|\d{2}-\d{3}[ \t]+[A-ZŁŚŻŹĆ][\w-]+(?:[ \t][A-ZŁŚŻŹĆ][\w-]+)?'  # 00-590 Warszawa (PL)
    r'|\d{4}-\d{3}[ \t]+[A-ZÀ-ÖØ-Þ][\w-]+(?:[ \t][A-ZÀ-ÖØ-Þ][\w-]+)?'  # 3000-180 Coimbra (PT)
    r'|[AC-FHKNPRTV-Y]\d{2}[ \t]?(?=[0-9A-Z]{0,3}[A-Z])[AC-FHKNPRTV-Y0-9]{4}\b'  # V92 X7K3 (IE)
    r'|Co\.[ \t]+[A-Z][\w-]+'                                      # Co. Kerry
    # A town, optionally two words, optionally followed by its UK postcode
    # on the same segment ("Kingsmere KM2 7QL", test6): the postcode is tried
    # before a second town word so "KM2" is not read as one.
    r'|[A-Z][\w-]+(?:[ \t]+(?:' + _UK_POSTCODE + r'|[A-Z][\w-]+(?:[ \t]+' + _UK_POSTCODE + r')?))?'
    r')')
# An address span that stops after the outward half of a UK postcode
# ("8 Bryn Teg, Caernarfon LL55" from the model, test6) takes the inward half.
_OUTWARD_TAIL = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?$")
_INWARD_NEXT = re.compile(r"[ \t]\d[A-Z]{2}\b")
# The same for an Irish Eircode: routing key, then four characters with at
# least one letter ("..., Co. Kerry, V92" + " X7K3", test7).
_EIRCODE_TAIL = re.compile(r"\b(?:[AC-FHKNPRTV-Y]\d{2}|D6W)$")
_EIRCODE_NEXT = re.compile(r"[ \t](?=[0-9A-Z]{0,3}[A-Z])[AC-FHKNPRTV-Y0-9]{4}\b")
# Ordinary words between a number and a street word mean a sentence, not an
# address ("5 minutes from the main Road").
_ADDRESS_GAP_STOP = re.compile(
    r"\b(?:the|a|an|in|at|to|from|for|while|and|with|by|is|are|was|were|per|each|or"
    r"|that|this|it|we|you|will|be|as|when|after|before|until|then|than|have|has|had)\b")
# Street words that are also English: "Via Royal Mail 48" is a carrier, not a
# street, unless a postcode follows or an address label precedes (test7).
_AMBIGUOUS_STREET_LEAD = re.compile(r"(?:Via|Plaza|Largo|Corso)[ \t]")
_ADDRESS_LABEL_BEFORE = re.compile(
    r"(?i)\b(?:address|indirizzo|residenza|domicilio|sede|direcci[oó]n|morada|adresse)\b[^\n]{0,12}$")
# Without a postcode or a label, three things still mark a Romance street:
# a particle after the street word ("Plaza de Olavide 6"), a floor or door
# after the number ("6, 2ºB"), and a capital mid-sentence ("lives next door
# at Via Saragozza 120"; the preposition "via" is lowercase there). "Dispatch:
# Via Royal Mail 48" has none of them (test7, test8).
_ROMANCE_PARTICLE_NEXT = re.compile(
    r"(?:Via|Plaza|Largo|Corso)[ \t]+(?:de|del|della|delle|degli|dei|di|do|da|dos|das|la|las|los)[ \t]")
_FLOOR_OR_DOOR = re.compile(r"\d[ \t]?[ºª°]")
_LOWERCASE_WORD_BEFORE = re.compile(r"[a-zß-öø-ÿ],?[ \t]$")
# A destination that ends in a venue word is a venue, not a city.
_VENUE_WORD = re.compile(
    r"\b(?:Park|Reserve|Museum|Zoo|Centre|Center|Beach|Castle|Hall|Gardens?|Airport|Station"
    r"|Stadium|Farm|Forest|Hotel|Arena|Theatre|Theater|Gallery|Abbey|Cathedral|Campus|Court"
    r"|Tower|Bridge|Pier|Market|Mall|Wharf|Quay|School|College|University|Hospital|Clinic"
    r"|Office|House|Camp|Lodge)$")
# Lines of an address block under the street line.
_BLOCK_TOWN_LINE = re.compile(
    r"[ \t]*(?P<pii>[A-Z][\w'’-]+(?:[ \t-]+[A-Za-z][\w'’-]+){0,2}),?[ \t]*")
_BLOCK_POSTCODE_LINE = re.compile(
    r"[ \t]*(?P<pii>(?:[A-Z][\w'’-]+(?:[ \t]+[A-Z][\w'’-]+){0,2},?[ \t]+)?" + _UK_POSTCODE
    + r"|\d{5}[ \t]+[A-Z][\w-]+(?:[ \t][A-Z][\w-]+)?"
    r"|\d{4}[ \t]?[A-Z]{2}[ \t]+[A-Z][\w-]+(?:[ \t][A-Z][\w-]+)?),?[ \t]*")
# Table columns that hold people and plates, and the cell shapes they take.
_TABLE_NAME_HEADER = re.compile(
    r"(?i)(?:(?:full|visitor|guest|driver|employee|staff|contact|host|tenant|member|customer"
    r"|client|patient|student|pupil)[ \t]+)?names?|visitors?|guests?|hosts?|drivers?"
    r"|employees?|attendees?|passengers?|tenants?|residents?|keepers?|participants?"
    r"|signed[ \t]+(?:in|out)[ \t]+by|escort(?:ed[ \t]+by)?|visiting|person[ \t]+visited")
_TABLE_PLATE_HEADER = re.compile(
    r"(?i)(?:vehicle[ \t]+)?(?:reg(?:istration)?\.?(?:[ \t]+(?:no\.?|number|mark))?|plate"
    r"|number[ \t]+plate|licen[cs]e[ \t]+plate|kennzeichen|vrm)")
_TABLE_NAME_CELL = re.compile(
    r"(?:" + _TITLES + r"[ \t]+)?(?!(?:None|N/?A|Nil|Unknown|Self|Staff|Contractor|Visitor"
    r"|Delivery|Courier|Reception|Security|Tbc|Tba|Various)\b)"
    r"(?P<pii>(?:[A-ZÀ-ÖØ-Þ]\.[ \t]?){0,2}" + _NAME_WORD
    + r"(?:" + _NAME_PARTICLES + r"[ \t]" + _NAME_WORD + r"){0,3})")
_TABLE_PLATE_CELL = re.compile(
    r"(?P<pii>(?=[A-Z0-9 -]{4,12}$)(?=[^\d]*\d)(?=[^A-Z]*[A-Z])[A-Z0-9]{1,4}(?:[ -][A-Z0-9]{1,4}){0,3})")
# A house name ahead of the street number, between the label and the street
# ("Address on file: The Old Forge, 6 Chapel Row", test7).
_HOUSE_NAME = re.compile(
    r"(?:^|[:;])[ \t]*(?P<house>(?:The[ \t]+)?[A-Z][\w'’-]+(?:[ \t]+[A-Z][\w'’-]+){0,3}),[ \t]*$")
_HOUSE_WORD = re.compile(
    r"^The[ \t]|\b(?:House|Cottages?|Lodge|Farm|Barn|Villa|Manor|Hall|Forge|Mill|Rectory"
    r"|Vicarage|Grange|Croft|Bungalow|View|Stables|Gables|Mansions|Buildings|Chambers"
    r"|Annexe|Court|Towers?)$")
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
    """Extend an address span over the second half of a postcode it stops in."""
    for tail, rest in ((_OUTWARD_TAIL, _INWARD_NEXT), (_EIRCODE_TAIL, _EIRCODE_NEXT)):
        if tail.search(text[start:end]):
            match = rest.match(text[end:end + 6])
            if match:
                return end + match.end()
    return end


def house_name_start(text, start):
    """Start of a house name that sits on the same line before a street number."""
    line_start = text.rfind('\n', 0, start) + 1
    match = _HOUSE_NAME.search(text[line_start:start])
    # A house word marks the name ("The Old Forge"), or an address label in
    # front of it does ("Address: Braeside, 4 Station Road", test8).
    if match and (_HOUSE_WORD.search(match.group('house'))
                  or _ADDRESS_LABEL_BEFORE.search(text[line_start:line_start + match.start('house')])):
        return line_start + match.start('house')
    return None


def address_block_lines(text, end):
    """Town and postcode lines under a street line that ends its own line.

    "14 Mill Street / Ludlow / SY8 1BB" (test8): up to two town lines count
    only when a postcode line closes the block. Each line is its own span, so
    no span crosses a line break.
    """
    if not re.match(r',?[ \t]*\n', text[end:end + 4]):
        return []
    pos = text.find('\n', end) + 1
    spans = []
    for _ in range(3):
        line_end = text.find('\n', pos)
        line_end = len(text) if line_end == -1 else line_end
        line = text[pos:line_end]
        closing = _BLOCK_POSTCODE_LINE.fullmatch(line)
        if closing:
            spans.append((pos + closing.start('pii'), pos + closing.end('pii')))
            return spans
        town = _BLOCK_TOWN_LINE.fullmatch(line)
        if not town:
            return []
        spans.append((pos + town.start('pii'), pos + town.end('pii')))
        pos = line_end + 1
    return []


def table_column_spans(text):
    """Names and plates in a pipe- or tab-separated table, read by column.

    "Time in | Name | Host | Vehicle reg | Badge" (test8): the header cell
    labels every cell below it, as a field label does on a form. A cell must
    still have the shape of a name or of a plate ("on foot" has neither).
    """
    results = []
    kinds, width = None, 0
    for line in re.finditer(r'[^\n]*(?:\n|$)', text):
        if not line.group().strip():
            kinds = None
            if line.end() >= len(text):
                break
            continue
        cells = [(m.start(1) + line.start(), m.group(1))
                 for m in re.finditer(r'[ \t]*([^|\t\n]*?)[ \t]*(?:[|\t]|\n|$)', line.group())
                 if m.end() > m.start()]
        if len(cells) < 3:
            kinds = None
            continue
        header = ['PERSON' if _TABLE_NAME_HEADER.fullmatch(c) else
                  'LICENSE_PLATE' if _TABLE_PLATE_HEADER.fullmatch(c) else None
                  for _, c in cells]
        if any(header):
            kinds, width = header, len(cells)
            continue
        if kinds is None or len(cells) != width:
            kinds = None
            continue
        for (start, cell), kind in zip(cells, kinds):
            shape = {'PERSON': _TABLE_NAME_CELL, 'LICENSE_PLATE': _TABLE_PLATE_CELL}.get(kind)
            match = shape.fullmatch(cell) if shape else None
            if match:
                results.append({'entity_type': kind, 'start': start + match.start('pii'),
                                'end': start + match.end('pii'), 'score': 1.0,
                                'text': match.group('pii'), 'source': 'regex'})
        if line.end() >= len(text):
            break
    return results


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
    results = chat_reply_ids(text) + table_column_spans(text)
    for pii_type, pattern in TEXT_PATTERNS.items():
        for match in pattern.finditer(text):
            # Context patterns retain their surrounding text for matching,
            # but only the named PII span should be redacted.
            start, end = match.span('pii') if 'pii' in match.re.groupindex else match.span()
            if pii_type in ('sensitive_organization', 'sensitive_organization_caps'):
                # Do not include a sentence verb such as “Visit” merely
                # because it is capitalized at the beginning of a sentence.
                leading = re.match(r'(?i)^(?:visit|at|the|from|to|for)\s+', text[start:end])
                if leading:
                    start += leading.end()
            if pii_type == 'sensitive_organization_caps':
                # In block capitals no lowercase word ends the run, so the
                # name starts after the last preposition ("REFERRED FROM
                # OAKFIELD DENTAL PRACTICE").
                cuts = [m.end() for m in re.finditer(
                    r'\b(?:FROM|AT|TO|BY|THE|IN|WITH|VIA)[ \t]+', text[start:end])]
                if cuts:
                    start += cuts[-1]
            if pii_type == 'labelled_person_caps' and _CAPS_NOT_A_NAME.search(text[start:end]):
                continue
            if pii_type == 'travel_city' and _VENUE_WORD.search(text[start:end]):
                continue
            if pii_type in ('address_line', 'address_line_de', 'address_line_es',
                            'address_line_pl'):
                # A street line is usually followed by apartment, city, and
                # postal-code segments; the whole location identifies a home.
                line_end = end
                end = extend_address(text, end)
                labelled = _ADDRESS_LABEL_BEFORE.search(text[max(0, start - 40):start])
                # A function word inside the street line means a sentence,
                # unless a label or a postcode says it is an address ("Address:
                # 14 Stoke by Nayland Road, Sudbury CO10 2XA", test8).
                if pii_type == 'address_line' and _ADDRESS_GAP_STOP.search(text[start:line_end]) \
                        and not labelled and not re.search(r'\d', text[line_end:end]):
                    continue
                if pii_type == 'address_line_es' and _AMBIGUOUS_STREET_LEAD.match(text, start) \
                        and not re.search(r'\d', text[line_end:end]) and not labelled \
                        and not _ROMANCE_PARTICLE_NEXT.match(text, start) \
                        and not _FLOOR_OR_DOOR.search(text[start:line_end]) \
                        and not _LOWERCASE_WORD_BEFORE.search(text[max(0, start - 3):start]):
                    continue
                for block_start, block_end in address_block_lines(text, end):
                    results.append({'entity_type': 'ADDRESS_LINE', 'start': block_start,
                                    'end': block_end, 'score': 1.0,
                                    'text': text[block_start:block_end], 'source': 'regex'})
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


# A person span never ends in a company word. The long words match in any
# case ("LTD", "Ltd"); the short legal forms only as written.
_COMPANY_TAIL = re.compile(
    r"(?:\b(?i:Ltd|Limited|Inc|Incorporated|Corp|Corporation|LLC|LLP|PLC|GmbH|Lda|Ltda|SARL|Pty"
    r"|Company|Holdings|Leasing|Insurance|Assurance|Versicherung(?:en)?|Assicurazioni|Seguros"
    r"|Bank|Logistics|Services|Solutions|Enterprises|Industries|Associates|Partners"
    # Trade nouns that end a company name ("Corvid Cycles", "Tannahill
    # Haulage", test8; "Halden Kitchenware", test3).
    r"|Lettings|Broadband|Cycles|Couriers|Freight|Haulage|Wholesale|Dairies|Energy|Motors"
    r"|Telecom|Foods|Estates|Properties|Removals|Supplies|Systems|Technologies|Kitchenware"
    r"|Hardware|Software|Furniture|Interiors|Builders|Roofing|Plumbing|Electrical"
    r"|Scaffolding|Consulting|Consultancy|Recruitment|Airways|Airlines|Railways|Brewery"
    r"|Bakery|Autos|Publishing|Pharma|Catering|Construction|Transport|Trading)"
    r"|\bS\.?r\.?l|\bS\.?p\.?A|\bS\.A|\bB\.?V|\bN\.?V|\b(?:AG|KG))\.?$")
# Company words that lead the name in German, Italian, French and Spanish
# ("Hausverwaltung Lindenhof", test8).
_COMPANY_LEAD = re.compile(
    r"^(?:Hausverwaltung|Autohaus|Firma|Spedition|Kanzlei|Bäckerei|Metzgerei|Stadtwerke"
    r"|Sparkasse|Gasthaus|Gasthof|Officina|Autofficina|Agenzia|Agencia|Agence|Grupo|Gruppo"
    r"|Groupe|Société|Cabinet)\b")
# A courtesy title in front of a span, or at its start, makes it a person
# even when the surname is a company word ("Fr. Annelise Bank", test8).
_TITLE_BEFORE = re.compile(r"(?:^|[\s(,;:—–-])" + _TITLES + r"[ \t]+$")
_TITLE_START = re.compile(_TITLES + r"[ \t]+(?=\S)")
# A NER person span never runs across a column gap, a spaced dash or a pipe
# ("Thabo Nkosana-Fields   Payroll", test8): the name is the first cell.
_COLUMN_GAP = re.compile(r"[ \t]{2,}|\t|[ \t]+[—–|/](?:[ \t]+|$)")
# Nor does it start with a greeting or a sign-off word ("Ta, Gethin").
_PERSON_LEAD_WORD = re.compile(
    r"^(?:Ta|Thx|Thanks|Cheers|Regards|Best|Love|Hi|Hiya|Hello|Hey|Dear|Morning|Evening"
    r"|Afternoon|Ciao|Sincerely|Yours)[,!]?[ \t]+(?=\S)")
# One-word NER "persons" that are a field label: a contact label ahead of a
# number wherever it stands ("Tel. 089 ..."), a role label ahead of a colon
# ("Consignee: ...").
_GREETING_STOP_WORD = re.compile(_GREETING_STOP)
_CONTACT_LABEL_WORDS = frozenset(
    "Tel Tél Fax Mob Ext Telefon Telefono Teléfono Phone Mobile Handy Cell".split())
_CONTACT_LABEL_NEXT = re.compile(r"[ \t]*[.:][ \t]*:?[ \t]*[+(\d]")
_ROLE_LABEL_WORD = re.compile(
    r"(?:" + _GREETING_STOP + r"|Consignee|Consignor|Recipient|Sender|Addressee|Shipper|Payee"
    r"|Payer|Caller|Witness|Referee|Guarantor|Signatory|Lessor|Lessee|Claimant|Insured"
    r"|Beneficiary|Proposer|Hirer|Keeper|Signed|Signature|Notes?|Subject|Attn)")
# A German-shaped plate is a standard, not a plate, after a conformity phrase
# or when it starts with a standards body ("certified to DIN-EN 1090", test8).
_STANDARD_PREFIX = re.compile(
    r"(?i)\b(?:certified|tested|rated|built|made|conforms?|complies|compliant|according"
    r"|accordance|nach|gemäß)(?:\s+(?:to|with))?\s+$")
_STANDARD_CODE = re.compile(r"^DIN-(?:EN|ISO|IEC|VDE)\b")
_COURTESY_WORDS = frozenset(
    "Grazie Merci Danke Gracias Obrigado Obrigada Thanks Regards Cheers Sincerely Saluti "
    "Cordialmente Ciao Hello Hola Bonjour Salut Prego Sorry Welcome Congratulations "
    "Greetings Ta Thx "
    # First-person and formal pronouns that open a declaration ("Je
    # soussignée", test8).
    "Je Nous Vous Ich Wir Yo Io Noi Eu".split())
_DATE_PLACEHOLDER = re.compile(r"(?:D{1,2}|M{1,4}|Y{2,4})")
_LABEL_WITH_NUMBER = re.compile(r"[ \t]*:[ \t]*[+(\d]")


_NER_SOURCES = ('spacy', 'presidio', 'custom_ner')
_BLANK_LINE = re.compile(r'\n[ \t]*\n')
_HEADING_AUTHOR = re.compile(r"(?P<word>[A-ZÀ-ÖØ-Þ][\w'’&-]{3,})[ \t]+[A-ZÀ-ÖØ-Þ&]")


def section_heading(text, pos):
    """First line of the block of lines that holds pos, when pos lies below it."""
    start = 0
    for blank in _BLANK_LINE.finditer(text, 0, pos):
        start = blank.end()
    line_end = text.find('\n', start)
    if line_end == -1 or line_end >= pos:
        return ''
    return text[start:line_end].strip()


def document_start(text):
    return len(text) - len(text.lstrip())


def heading_author_word(text, person_spans):
    """First word of the document when it opens with a capitalised phrase that
    no layer reads as a person: the author's name ("Brackenridge Lettings —
    messages ...", test8). A one-word "person" equal to that word is the
    author signing or being named ("Kind regards, Brackenridge", "Thanks for
    calling Quillfield"), and the author stays visible.
    """
    match = _HEADING_AUTHOR.match(text, document_start(text))
    if not match or any(s <= match.start('word') < e for s, e in person_spans):
        return None
    return match.group('word')


def is_company_shaped(text, start, end):
    """True when the span ends in, is followed by, or starts with a company word."""
    span = text[start:end]
    following = re.match(r'[ \t]+[^\s,;:]+', text[end:end + 20])
    return bool(_COMPANY_TAIL.search(span) or _COMPANY_LEAD.match(span)
                or (following and _COMPANY_TAIL.search(span + following.group())))


def org_is_tied(text, name):
    """True when any mention of the organization carries employment wording.

    An organization is personal data when the document ties a person to it;
    the tie may sit at another mention than the one a layer detected ("team
    leader at Corbel Freight" for the "Corbel Freight" on the referee line,
    test8). The author's own heading mention never counts.
    """
    if len(name) < 4:
        return False
    for match in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text):
        if match.start() == document_start(text) \
                or is_letterhead(text, match.start(), match.end(), name):
            continue
        prefix = text[max(0, match.start() - 48):match.start()]
        # "as a picker for X" names the employer; "as a Senior X" a title.
        if ORG_ROLE_FOR_PREFIX.search(prefix):
            return True
        if TITLE_PREFIX.search(prefix):
            continue
        if EMPLOYMENT_CONTEXT.search(sentence_around(text, match.start(), match.end())) \
                or ORG_AT_PREFIX.search(prefix) \
                or EMPLOYMENT_HEADING.search(section_heading(text, match.start())):
            return True
    return False


def apply_policy_filters(results, text):
    """Stage 2: enforce the context rules of the V1 public-sharing policy."""
    person_spans = [(r['start'], r['end']) for r in results
                    if r['entity_type'] in ('PERSON', 'PRONOUN')]
    address_ends = [r['end'] for r in results
                    if r['entity_type'] == 'ADDRESS_LINE']
    person_texts = {r['text'] for r in results if r['entity_type'] == 'PERSON'}
    ner_person_spans = [(r['start'], r['end']) for r in results
                        if r['entity_type'] == 'PERSON'
                        and r.get('source') in ('spacy', 'presidio', 'custom_ner')]
    # Regex phones with an international prefix: a checksum-only identifier
    # Presidio finds strictly inside one ("612 345 678" as an Australian
    # company number inside "+34 612 345 678", test6) is part of the phone.
    intl_phones = [(r['start'], r['end']) for r in results
                   if r['entity_type'] == 'PHONE' and r.get('source') == 'regex'
                   and r['text'].startswith(('+', '00'))]

    name_spans = [(r['start'], r['end']) for r in results if r['entity_type'] == 'PERSON']
    author_word = heading_author_word(
        text, [(s, e) for s, e in name_spans if not is_company_shaped(text, s, e)])

    def near_person(start, end, radius=PERSON_PROXIMITY_CHARS):
        return any(p_start - radius <= end and start <= p_end + radius
                   for p_start, p_end in person_spans)

    def person_on_line(start, end):
        line_start = text.rfind('\n', 0, start) + 1
        line_end = text.find('\n', end)
        line_end = len(text) if line_end == -1 else line_end
        return any(line_start <= s and e <= line_end for s, e in name_spans)

    filtered = []
    for result in results:
        entity, source = result['entity_type'], result.get('source')
        demoted = False
        # General NER runs an organization on into the next line ("...
        # DAIRIES LTD\nEMERGENCY"); the name is the part on its own line.
        if entity == 'ORGANIZATION' and source in _NER_SOURCES and '\n' in result['text']:
            first = result['text'].split('\n', 1)[0].rstrip()
            if len(first) < 2:
                continue
            result = {**result, 'end': result['start'] + len(first), 'text': first}
        if entity == 'PERSON' and source in _NER_SOURCES:
            # The name is the first cell of a span that crosses a column gap,
            # and it never starts with a greeting or sign-off word.
            gap = _COLUMN_GAP.search(result['text'])
            if gap:
                if gap.start() < 2:
                    continue
                result = {**result, 'end': result['start'] + gap.start(),
                          'text': result['text'][:gap.start()]}
            lead = _PERSON_LEAD_WORD.match(result['text'])
            if lead:
                # "Dear Occupier": what is left may be a role, not a name.
                if _GREETING_STOP_WORD.fullmatch(result['text'][lead.end():]):
                    continue
                result = {**result, 'start': result['start'] + lead.end(),
                          'text': result['text'][lead.end():]}
        sentence = sentence_around(text, result['start'], result['end'])
        prefix = text[max(0, result['start'] - 32):result['start']]

        # "The label decides": a digit run straight after a thing-ID label is
        # that thing's number, whichever layer read it as a phone, a card or
        # an address.
        if _DIGIT_SHAPED.fullmatch(result['text']) and THING_ID_LABEL_PREFIX.search(prefix):
            continue
        # A span that ends in a company word is a company, whatever layer or
        # role label called it a person ("Registered keeper: Lexford Leasing
        # Ltd", "Albatros Versicherung", "Officina Bellandi S.r.l"; test7).
        # The same when the company word follows the span: the model's
        # "Officina Bellandi" ahead of "S.r.l.".
        # A courtesy title makes it a person all the same ("Frau Bank"). A
        # NER span is then judged as the organization it is: an employer is
        # still redacted ("Dispatcher, Tannahill Haulage" under "Work
        # history", test8); a role-label value is dropped.
        if entity == 'PERSON' and is_company_shaped(text, result['start'], result['end']) \
                and not (_TITLE_BEFORE.search(prefix) or _TITLE_START.match(result['text'])):
            if source not in _NER_SOURCES:
                continue
            entity, demoted = 'ORGANIZATION', True
            result = {**result, 'entity_type': 'ORGANIZATION'}
        # The author's name from the document heading is not a person.
        if entity == 'PERSON' and author_word and result['text'] == author_word:
            continue
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
        # The tie may sit at another mention of the same organization.
        if entity == 'ORGANIZATION' and (source in ('spacy', 'presidio') or demoted):
            role_for = ORG_ROLE_FOR_PREFIX.search(text[max(0, result['start'] - 48):result['start']])
            if TITLE_PREFIX.search(prefix) and not role_for:
                continue
            if not (role_for or SENSITIVE_ORG_KEYWORDS.search(result['text'])
                    or EMPLOYMENT_CONTEXT.search(sentence)
                    or ORG_AT_PREFIX.search(prefix)
                    or (result['start'] != document_start(text)
                        and org_is_tied(text, result['text']))):
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
        # Nor is it an address: the custom model reads "Engine: Kubota D902,
        # 16 hp" as a street and a house number (candidate 3, 2026-11-13).
        if entity == 'ADDRESS' and source == 'custom_ner' \
                and THING_LABEL_PREFIX.search(prefix):
            continue
        # "City of Tver", "Republic of Karelia": the word after a place
        # noun and "of" is a place, whatever the NER layers call it.
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner') \
                and PLACE_OF_PREFIX.search(prefix):
            continue
        # A single letter is an item label, not a name ("keys A and B.").
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner') \
                and sum(ch.isalpha() for ch in result['text']) < 2:
            continue
        # "the Hull depot": a site, not a person, whatever general NER says.
        operational = OPERATIONAL_SITE_SUFFIX.match(text[result['end']:result['end'] + 24])
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner') and operational:
            continue
        # The same holds for a shorter regex phone inside one: "114 496
        # 0827" scores 1.0 and would outrank "0044 114 496 0827" at 0.99,
        # leaving the prefix visible (test7).
        if (source == 'presidio' or (source == 'regex' and entity == 'PHONE')) and any(
                s <= result['start'] and result['end'] <= e
                and (e - s) > (result['end'] - result['start']) for s, e in intl_phones):
            continue
        # A German-shaped plate needs vehicle wording and no place label.
        # A person on the same line links the plate to them, as in a list of
        # parking bays ("Bay 03 — M-KT 5038 — Fr. Annelise Bank", test8). A
        # standard is never a plate ("certified to DIN-EN 1090").
        if entity == 'LICENSE_PLATE' and source == 'regex' \
                and TEXT_PATTERNS['license_plate_de'].fullmatch(result['text']) \
                and (PLACE_LABEL_PREFIX.search(prefix)
                     or _STANDARD_PREFIX.search(prefix) or _STANDARD_CODE.match(result['text'])
                     or not (PLATE_CONTEXT.search(sentence) or PLATE_CONTEXT.search(prefix)
                             or person_on_line(result['start'], result['end']))):
            continue
        # One-word NER "persons" that are a courtesy word ("Grazie!"), a
        # date-format placeholder (the "MM" of "DD/MM/YYYY") or a field label
        # ahead of a number ("Telefono: +39 ...") are not names (test7).
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner') \
                and ' ' not in result['text']:
            line_start = text.rfind('\n', 0, result['start']) + 1
            after = text[result['end']:result['end'] + 8]
            if result['text'] in _COURTESY_WORDS \
                    or _DATE_PLACEHOLDER.fullmatch(result['text']) \
                    or (not text[line_start:result['start']].strip()
                        and _LABEL_WITH_NUMBER.match(after)) \
                    or (result['text'] in _CONTACT_LABEL_WORDS
                        and _CONTACT_LABEL_NEXT.match(after)) \
                    or (_ROLE_LABEL_WORD.fullmatch(result['text'])
                        and re.match(r'[ \t]*:', after)):
                continue
        # A person's name never contains a digit. General NER glues a chat
        # time onto a name ("14:03 Rhys") — strip it, on the same line only,
        # so a span that crossed a line break still dies in dedup — or reads
        # a product code as a person ("Jungheinrich EFG 216", test6) — drop it.
        # The custom model is held to the same rule ("Felco 2", test7).
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner') \
                and re.search(r'\d', result['text']):
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
            # A house name ahead of the street number belongs to the address,
            # unless that segment is a detected person ("Deliver to: Mele
            # Taufa, 9 ...").
            house = house_name_start(text, result['start'])
            if house is not None and result['text'][:1].isdigit() and not any(
                    s < result['start'] and house < e for s, e in person_spans):
                result = {**result, 'start': house, 'text': text[house:result['end']]}
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
            # A sign-off word is not the name it stands in front of ("Ta,"
            # inside a model span "Ta, Gethin"), nor is a role addressee
            # ("Dear Occupier,").
            if result['text'] in _COURTESY_WORDS \
                    or _GREETING_STOP_WORD.fullmatch(result['text']) \
                    or not any(result['text'] in p and result['text'] != p for p in person_texts):
                continue
            result = {**result, 'entity_type': 'PERSON'}
        # A bare "Name:" value counts only when a NER layer saw a person in it.
        if entity == 'NAME_LINE':
            if not any(s < result['end'] and result['start'] < e for s, e in ner_person_spans):
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
_TITLE_WORD = re.compile(r"^" + _TITLES + r"$")


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
            if OPERATIONAL_SITE_SUFFIX.match(text[match.end():match.end() + 24]):
                continue  # "the Florence warehouse" is a site, not the person
            extra.append({'entity_type': 'PERSON', 'start': match.start(),
                          'end': match.end(), 'score': 0.9,
                          'text': name, 'source': 'echo'})

    for result in results:
        name = result['text']
        # An employer is the same employer at its other mentions, whichever
        # layer found it; a letterhead or heading occurrence stays visible.
        if result['entity_type'] == 'ORGANIZATION' and len(name) >= 4 and name not in seen:
            seen.add(name)
            for match in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text):
                if match.start() != result['start'] \
                        and match.start() != document_start(text) \
                        and not is_letterhead(text, match.start(), match.end(), name):
                    extra.append({'entity_type': 'ORGANIZATION', 'start': match.start(),
                                  'end': match.end(), 'score': 0.9,
                                  'text': name, 'source': 'echo'})
            continue
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
