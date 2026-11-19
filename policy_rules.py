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
    # A named clinical unit ("Glenbrook Adolescent Unit", test9); the bare
    # word "Unit" is a flat or a team and never enough.
    r"|(?:Adolescent|Psychiatric|Mental[ \t]+Health|Inpatient|Maternity|Neonatal|Stroke|Renal"
    r"|Oncology|Cardiac|Coronary[ \t]+Care|Intensive[ \t]+Care|Burns|Detox|Secure"
    r"|Eating[ \t]+Disorders?|Assessment|Rehabilitation|Dialysis|Day[ \t]+Surgery"
    r"|Pupil[ \t]+Referral)[ \t]+Unit"
)
# Facility words that lead the name in the Romance and German order ("Colégio
# São Bento", "Instituto Politécnico do Mondego", test7): schools,
# universities and hospitals only, none of them an English word.
_FACILITY_LEAD_WORDS = (
    r"Colegio|Colégio|Collège|Collegio|Instituto|Istituto|Institut|Escuela|Escola"
    r"|École|Scuola|Universidad|Universidade|Università|Université|Universität"
    r"|Liceo|Liceu|Lycée|Clínica|Clinica|Clinique|Hôpital|Ospedale|Krankenhaus"
    r"|Klinik|Klinikum|Hochschule|Grundschule|Realschule"
    # Welsh and Irish: hospital, school, surgery ("Ysbyty Cwm Derwen", test9).
    r"|Ysbyty|Ysgol|Meddygfa|Scoil|Gaelscoil|Coláiste")
# The same facility words in capitals, for forms completed in block capitals
# ("PRIORY LANE SURGERY", test7). Only letters outside regex escapes change.
_FACILITY_WORDS_CAPS = re.sub(r"(?<!\\)[a-z]+", lambda m: m.group().upper(), _FACILITY_WORDS)
SENSITIVE_ORG_KEYWORDS = re.compile(
    r"\b(?:" + _FACILITY_WORDS + r"|" + _FACILITY_LEAD_WORDS + r")\b")
# A facility word with no name in front is a place in the building or a field
# label, not a named facility ("Clinic 3", "Hospital no.", "School office";
# test9).
_BARE_FACILITY = re.compile(r"(?:" + _FACILITY_WORDS + r")")
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
_BIRTH_LABELS_INTL = (r"n[ée]e?\s+le|date\s+de\s+naissance|geboren\s+am|geb\.(?:\s+am)?|geburtsdatum"
                      r"|fecha\s+de\s+nacimiento|nacid[oa]\s+el|data\s+di\s+nascita"
                      r"|nat[oa]\s+il|data\s+de\s+nascimento|nascid[oa]\s+(?:em|a)"
                      # Dutch ("geboren op 12 mei 1968", test10)
                      r"|geboren\s+op|geboortedatum")
_MONTHS_FR = (r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre"
              r"|octobre|novembre|d[ée]cembre)")
# German ("23. Juli 2015"), Spanish and Portuguese ("4 de marzo de 1984") and
# Italian ("4 marzo 1984") month names for the same worded date (test9).
_MONTHS_DE = (r"(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober"
              r"|November|Dezember)")
_MONTHS_IBERIAN = (r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|sept?iembre"
                   r"|octubre|noviembre|diciembre|janeiro|fevereiro|março|maio|junho|julho"
                   r"|setembro|outubro|novembro|dezembro)")
_MONTHS_IT = (r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre"
              r"|ottobre|novembre|dicembre)")
_MONTHS_NL = (r"(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober"
              r"|november|december)")
# A date as it is said aloud ("the fourteenth of May, nineteen forty-one" in a
# call transcript, test10): the day as an ordinal word or a number, the month,
# the year in words or digits.
_ORDINAL_WORDS = (r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh"
                  r"|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth"
                  r"|nineteenth|twentieth|twenty[ -](?:first|second|third|fourth|fifth|sixth"
                  r"|seventh|eighth|ninth)|thirtieth|thirty[ -]first)")
_NUMBER_WORDS = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
                 r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty"
                 r"|fifty|sixty|seventy|eighty|ninety|hundred|oh)")
_SPOKEN_YEAR = (r"(?:(?:nineteen|twenty|two[ \t]+thousand)(?:[ \t-]+(?:and[ \t]+)?" + _NUMBER_WORDS
                + r"){0,3}|(?:19|20)\d{2})")
_SPOKEN_DATE = (r"(?:(?:" + _ORDINAL_WORDS + r"|\d{1,2}(?:st|nd|rd|th))[ \t]+(?:of[ \t]+)?" + _MONTHS
                + r",?[ \t]+" + _SPOKEN_YEAR
                + r"|" + _MONTHS + r"[ \t]+(?:the[ \t]+)?(?:" + _ORDINAL_WORDS
                + r"|\d{1,2}(?:st|nd|rd|th)),?[ \t]+" + _SPOKEN_YEAR + r")")
# A date is a DOB only with an explicit birth label directly before it; a
# format hint may sit between the two ("Date of birth (DD/MM/YYYY): ...").
DOB_CONTEXT = re.compile(
    r'(?i)(?:born(?:\s+on)?(?:\s+the)?|date[\s_]*of[\s_]*birth|birth[\s_]*date|birthday|d\.?o\.?b\.?'
    r'|' + _BIRTH_LABELS_INTL + r')'
    # A quote may close a JSON key and open its value ('"date_of_birth": "').
    r'(?:\s*\([^)\n]{0,24}\))?["\']?\s*[:,-]?\s*["\']?$')

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
_TITLES = (r"(?:Mr|Mrs|Ms|Mx|Miss|Dr|Dra|Prof|Rev|Fr|Sr|Sra|Srta|Dña|Doña|Herr|Frau|Mme|Mlle|M"
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
    # Warehouse and transport desks ("Speak to Goods In", test10).
    r"|Goods|Returns|Despatch|Shipping|Receiving|Inbound|Outbound|Inventory"
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
    r"|Clerk|Organi[sz]ers?|Moderators?|Reviewers?|Hosts?|Stewards?|Administrators?"
    # Office holders ("Dear Bursar,", test9).
    r"|Bursar|Headteacher|Headmaster|Headmistress|Head|Principal|Registrar|Vicar|Rector"
    r"|Matron|Chaplain|Proprietor|Superintendent|Coordinator|Librarian|Postmaster"
    r"|Storekeeper|Bookkeeper|Housekeeper)")
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
# What follows a journey verb without being a town: a role or department, a
# part of a building, a month.
_NOT_A_CITY = (r"(?!(?:" + _GREETING_STOP + r"|Head|Site|Gate|Terminal|Platform|Customer|Hall|Room"
               r"|Building|Block|Level|Floor|Stage|Phase|Version|Windows|Theatre|Surgery|Court"
               r"|Class|Church|Town|January|February|March|April"
               r"|May|June|July|August|September|October|November|December)\b)")
# The town ends the name: no further capital word and no operational noun.
_CITY_END = (r"(?![\w'’-])(?![ \t]+(?:[A-ZÀ-ÖØ-Þ]|(?i:hub|branch|depot|warehouse|exchange"
             r"|terminal|plant|factory|showroom|office|site|yard|store|outlet|facility)\b))")
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
        r"|tfno\.?|tlfno\.?|tlf\.?|tel[eé]f\.?|helpline|hotline|freephone)"
        r"[ \t]*(?:no\.?|number)?[ \t]*[:.]?[ \t]*"
        # One slash may follow the area code ("Tel. 030 / 5550 0142", test9);
        # a date ("12/11/2026") has two and is not taken.
        r"(?P<pii>(?=(?:(?:[ .-]|[ \t]?/[ \t]?)?\d){7,13}(?![ .-]?\d))"
        r"\d{2,5}(?:[ \t]?/[ \t]?\d{2,5}|[ .-]\d{2,5})(?:[ .-]\d{2,5}){0,3})"
        r"(?![ \t]?/[ \t]?\d)(?!\w)",
        re.IGNORECASE),
    # A Dutch mobile: 06 and eight more digits ("06 5550 0133", test10).
    'phone_nl_mobile': re.compile(
        r'(?<![\w/.-])06[ -]?(?:\d{8}|\d{4}[ ]\d{4}|\d{2}(?:[ ]\d{2}){3})(?![\w/-]|[ .]\d)'),
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
        r'(?:no\.?|n[°ºo]\.?|nr\.?|number|numéro|número|numero|nummer|#)?\s*'
        # "passport no. (Person 1): C4FT7K2H9" (test9).
        r'(?:\([^)\n]{0,24}\)\s*)?[:#]?\s*'
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
        r'|badge|case|claim|record|mrn|dni|nie|nif|curp|cpf|bsn|pesel|nhs|frequent\s+fl[yi]er'
        r'|codice\s+fiscale|fiscal\s+code|tax\s+code|national\s+id|id(?:entity)?\s+card)\s*'
        r'(?:id|number|no\.?|record|reference|ref\.?|#)?'
        r'|(?:hospital|staff|payroll)\s+(?:id|number|no\.?)'
        r'|(?:application|visa|permit)\s+(?:reference|ref\.?|number|no\.?|id))\s*[:#.]?\s*'
        # The value may be printed in spaced digit groups, eight digits or
        # more in all ("gas account 8841 2207 16", test8).
        # and may carry a capital prefix ("Frequent flyer: SK 550 291 4470",
        # test9).
        r'(?P<pii>' + _NOT_A_DATE + r'(?:(?:(?-i:[A-Z]{2,3})[ \t])?'
        r'(?=(?:\d[ \t]?){8,20}(?!\d))\d{2,6}(?:[ \t]\d{2,6}){1,4}(?![ \t]?\d)'
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
        # "Ser. No.: 2B041JEGR09912" (test10): the abbreviation needs its dot.
        r'\b(?:(?:serial|ser\.|hull|imei)\s*(?:no\.?|number|id|#)?|s/n|(?:frame|chassis)\s*(?:no\.?|number|#))'
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
    # A police incident or crime number belongs to the person who reported
    # ("Reported to Deschutes County Sheriff, incident 26-0041873", test10).
    # With a number word any value counts; the bare word needs five digits in
    # a row, so "incident 4471" in an IT log is not one.
    'incident_number': re.compile(
        r'\b(?:(?:police|crime|incident)(?:\s+report)?\s*(?:reference|ref\.?|number|no\.?|#)\s*[:#]?\s*'
        r'|(?:incident|crime)\s+(?=[A-Z0-9/-]*\d{5}))'
        r'(?P<pii>' + _NOT_A_DATE + r'(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{4,19})\b', re.IGNORECASE),
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
        r'|\d{1,2}\.?\s+' + _MONTHS_DE + r'\s+(?:19|20)\d{2}'
        r'|\d{1,2}\s+de\s+' + _MONTHS_IBERIAN + r'\s+de\s+(?:19|20)\d{2}'
        r'|\d{1,2}º?\s+' + _MONTHS_IT + r'\s+(?:19|20)\d{2}'
        r'|\d{1,2}\s+' + _MONTHS_NL + r'\s+(?:19|20)\d{2}'
        r'|' + _SPOKEN_DATE +
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
        r"(?:[Ss]tra(?:ss|ß)e|[Ss]tr\.|[Aa]llee|[Ww]eg|[Gg]asse|[Pp]latz|[Dd]amm|[Uu]fer"
        # Dutch street words ("Lijsterbesstraat 9", test10)
        # ("blockade" and "stockade" are the English words that end alike)
        r"|[Ss]traat|[Ll]aan|[Pp]lein|[Gg]racht|(?<!c)[Kk]ade|[Ss]ingel|[Dd]ijk|[Ss]teeg)"
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
        r"(?:(?:,[ \t]*\d{1,2}[ºo°]?|[ \t]+\d{1,2}[ºo°])"
        # the door as a letter or a word ("14, 2º izq.", test9)
        r"(?:[ \t]+(?:izq(?:da|uierda)?|dcha|derecha|ctro|centro|ext(?:erior)?|int(?:erior)?)\b\.?"
        r"|[ \t]?[A-Za-z]?\b))?)"),
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
        # The other-language half of a bilingual label may come first ("Nome
        # / first name: Ottavio", test10), and a few labels that are not
        # English words label a name on their own.
        r"^[ \t]*(?:[^\W\d_][^:/\n]{0,30}/[ \t]*)?"
        r"(?:(?:previous|former|other|birth)\s+(?:sur)?names?|surname|family\s+name"
        r"|cognome|vorname|nachname|familienname|prénom|apellidos?|voornaam|voornamen|achternaam"
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
        r"|Signora|Signor|Sig\.(?:ra|na)?|Dott\.(?:ssa)?|Señora?|Sra\.|Dña|Doña)\.?[ \t]+"
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
        r"\b(?i:fl(?:y|ies|ying|ew)|travel(?:s|led|ed|ling|ing)?|trip)[ \t]+(?:(?:back|on)[ \t]+)?to[ \t]+"
        + _NOT_A_CITY + r"(?P<pii>" + _NAME_WORD + r"(?:[ \t]" + _NAME_WORD + r")?)" + _CITY_END),
    # A person as the subject of a journey or a move: "she drives up to
    # Thurso", "Ms Thorbjørnsen lands in Molde", "she relocates to Bodø"
    # (test9). The subject keeps cargo wording out ("the pallet lands in ...").
    'journey_city': re.compile(
        # A name as the subject must be a detected person: "Delivery lands in
        # Hull" and "Humber Haulage relocates to Goole" are cargo and a company
        # (test10; apply_policy_filters checks the subject group). "He is
        # moving to Tregaron" is the same wording for a move (test10).
        r"(?<![\w'’-])(?:(?i:I|we|he|she|they|you)|(?P<subject>" + _NAME_WORD + r"))[ \t]+"
        r"(?:(?i:also|then|now|still|will|is|are|was)[ \t]+)?"
        r"(?i:driv(?:es?|ing)|drove|lands?|landed|landing|relocat(?:es?|ed|ing)|mov(?:es?|ed|ing))[ \t]+"
        r"(?:(?i:back|up|down|over|out|on|across)[ \t]+)?(?i:to|in|into)[ \t]+"
        + _NOT_A_CITY + r"(?P<pii>" + _NAME_WORD + r"(?:[ \t]" + _NAME_WORD + r")?)" + _CITY_END),
    # A stay with an end: "he is in Berlin until the 20th" (test9), "Crispin
    # is away in Lanzarote until the 28th" (test10; the subject rule again).
    'stay_city': re.compile(
        r"(?<![\w'’-])(?:(?i:I|we|he|she|they)|(?P<subject>" + _NAME_WORD + r"))[ \t]+"
        r"(?i:is|am|are|was|were|will[ \t]+be)[ \t]+"
        r"(?:(?i:away|back|over|out)[ \t]+)?in[ \t]+"
        + _NOT_A_CITY + r"(?P<pii>" + _NAME_WORD + r"(?:[ \t]" + _NAME_WORD + r")?)"
        r"(?=[ \t]+(?i:until|till|through|all[ \t]+week|this[ \t]+week|next[ \t]+week"
        r"|for[ \t]+(?:the|a|two|three|four|\d)))"),
    # Work wording names the employer where no layer saw an organization:
    # "I'm on nights at Nidderdale Bakery", "I am now on the books of
    # Tideway Couriers" (test10). Roles, departments, bays and holidays are
    # not employers; a site ("works at the Hull depot") is not one either.
    'employer_phrase': re.compile(
        r"\b(?i:(?:work(?:s|ed|ing)?|employed|temping"
        r"|on[ \t]+(?:nights|days|lates|earlies|shifts?|placement|secondment)"
        r"|(?:night|day|late|early|back|weekend)[ \t]+shifts?)[ \t]+(?:at|for)"
        r"|on[ \t]+the[ \t]+(?:books|payroll)[ \t]+(?:of|at|with))[ \t]+"
        + _NOT_A_CITY + r"(?!(?:Bay|Dock|Ward|Christmas|Easter|Xmas|Home|HQ|Unit|Plot|Line|Stand"
        r"|Checkout|Till|Desk|Counter|Store)\b)"
        r"(?P<pii>[A-ZÀ-ÖØ-Þ][\w'’&-]+(?:[ \t]+(?:(?:&|and|of)[ \t]+)?[A-ZÀ-ÖØ-Þ][\w'’&-]+){0,3})"
        + _CITY_END),
    # The French signing formula names the signer's town ("Fait à Dijon, le
    # 2 novembre 2026", test8).
    'signing_place': re.compile(
        r"\bFait[ \t]+à[ \t]+(?P<pii>[A-ZÀ-ÖØ-Þ][\w'’-]+(?:[ \t-][A-ZÀ-ÖØ-Þ][\w'’-]+){0,2})"
        r"(?=,|[ \t]+le\b)"),
    # The same in German ("Göttingen, den 3. November 2026" opening a line),
    # Spanish ("en Tarazona, a 5 de noviembre de 2026") and Italian ("Bologna,
    # lì 4 novembre 2026"); test9.
    'signing_place_de': re.compile(
        r"(?m)^[ \t]*(?P<pii>[A-ZÀ-ÖØ-Þ][\w'’-]+(?:[ \t-][A-ZÀ-ÖØ-Þ][\w'’-]+){0,2})"
        r"(?=,[ \t]+(?:den|l[iì])[ \t]+\d{1,2}\b)"),
    'signing_place_es': re.compile(
        r"\b[Ee]n[ \t]+(?P<pii>[A-ZÀ-ÖØ-Þ][\w'’-]+(?:[ \t-][A-ZÀ-ÖØ-Þ][\w'’-]+){0,2})"
        r"(?=,[ \t]+a[ \t]+\d{1,2}[ \t]+de[ \t]+[a-z])"),
    # The same under a "place and date" label, in any of the form languages
    # ("Luogo e data / place and date: Cremona, 3 novembre 2026", test10).
    'signing_place_label': re.compile(
        r"(?i:luogo[ \t]+e[ \t]+data|place[ \t]+and[ \t]+date|lugar[ \t]+y[ \t]+fecha"
        r"|ort[ \t]*(?:,|und|/)[ \t]*datum|lieu[ \t]+et[ \t]+date|plaats[ \t]+en[ \t]+datum"
        r"|local[ \t]+e[ \t]+data)[ \t]*:[ \t]*"
        r"(?P<pii>[A-ZÀ-ÖØ-Þ][\w'’-]+(?:[ \t-][A-ZÀ-ÖØ-Þ][\w'’-]+){0,2})(?=,[ \t]+(?:[a-zì]{1,3}[ \t]+)?\d)"),
    # "ask for Priti", "speak to Morwenna": the request names a person
    # (test9). Roles and departments are not names.
    'ask_for_person': re.compile(
        r"(?m)\b(?i:ask(?:s|ed|ing)?[ \t]+for|speak[ \t]+(?:to|with))[ \t]+"
        r"(?!" + _GREETING_STOP + r"\b)(?P<pii>" + _NAME_WORD + r"(?:[ \t]" + _NAME_WORD + r")?)"
        r"(?![\w'’-])(?=[,:.!?;)]|[ \t]+(?:and|&|or|on|at|in|if|who|when|about|before|after)\b"
        r"|[ \t]*$)"),
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
    # A town ahead of a date is a dateline, not a greeting ("Arnhem, 9
    # november 2026", test10).
    'greeting_name': re.compile(r"(?m)^(?:Dear\s+)?(?P<pii>" + _NAME_WORD + r"),(?=\s)(?![ \t]+\d)"),
}

# A pattern anchored to an explicit label outranks a bare digit-run match over
# the same span: a labelled BIC beats a generic 9-digit SSN shape, and a
# labelled INN beats Presidio's 12-digit Aadhaar recognizer.
LABELLED_PATTERNS = {"bic", "inn_labelled", "generic_id", "group_class",
                     "student_id", "policy_number", "serial_number",
                     "order_number", "sort_code", "pcn_number", "agreement_number",
                     "card_number_labelled", "incident_number"}
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
                        "journey_city": "LOCATION",
                        "stay_city": "LOCATION",
                        "signing_place": "LOCATION",
                        "signing_place_de": "LOCATION",
                        "signing_place_es": "LOCATION",
                        "ask_for_person": "PERSON",
                        "phone_uk_bracket": "PHONE",
                        "phone_nl_mobile": "PHONE",
                        "incident_number": "GENERIC_ID",
                        "employer_phrase": "ORGANIZATION",
                        "signing_place_label": "LOCATION",
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

# A question that asks for a birth date, the line that answers it (with or
# without a speaker label), and the date in the answer.
_BIRTH_QUESTION = re.compile(
    r"(?im)\b(?:date[ \t]+of[ \t]+birth|birth[ \t]?date|d\.?o\.?b\.?|birthday"
    r"|(?:when|what[ \t]+year)[ \t]+(?:was|were)[ \t]+\w+[ \t]+born)\b[^\n?]{0,40}\?[ \t]*$")
_REPLY_LINE = re.compile(
    r"\n(?:[ \t]*\n)?[ \t]*(?:\[?\d{1,2}:\d{2}\]?[ \t]+)?(?:[^\s:][^:\n]{0,30}:[ \t]*)?"
    r"(?P<body>[^\n]{0,80})")
_REPLY_DATE = re.compile(
    r"(?i)(?<![\w/.-])(?P<pii>" + _SPOKEN_DATE
    + r"|\d{1,2}[ \t]+(?:of[ \t]+)?" + _MONTHS + r",?[ \t]+(?:19|20)\d{2}"
    r"|" + _MONTHS + r"[ \t]+\d{1,2},?[ \t]+(?:19|20)\d{2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2})(?![\w/-])")

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
    r'|(?:kv|korp|lok|m)\.[ \t]*\d{1,4}[A-Za-z]?\b'                 # kv. 61 (RU), m. 4 (PL)
    # A town, optionally two words, optionally followed by its UK postcode
    # on the same segment ("Kingsmere KM2 7QL", test6): the postcode is tried
    # before a second town word so "KM2" is not read as one.
    r'|[A-Z][\w-]+(?:[ \t]+(?:' + _UK_POSTCODE + r'|[A-Z][\w-]+(?:[ \t]+' + _UK_POSTCODE + r')?))?'
    r')')
# An address span that stops after the outward half of a UK postcode
# ("8 Bryn Teg, Caernarfon LL55" from the model, test6) takes the inward half.
_POSTCODE_SENTENCE = re.compile(r"\.[ \t]+" + _UK_POSTCODE + r"\b")
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
    r"|Office|House|Camp|Lodge|Surgery|Church|Chapel|Ward|Library|Lab|Laboratory)$")
# Lines of an address block under the street line.
_BLOCK_TOWN_LINE = re.compile(
    r"[ \t]*(?P<pii>[A-Z][\w'’-]+(?:[ \t-]+[A-Za-z][\w'’-]+){0,2}),?[ \t]*")
_BLOCK_POSTCODE_LINE = re.compile(
    r"[ \t]*(?P<pii>(?:[A-Z][\w'’-]+(?:[ \t]+[A-Z][\w'’-]+){0,2},?[ \t]+)?" + _UK_POSTCODE
    + r"|\d{5}[ \t]+[A-Z][\w-]+(?:[ \t][A-Z][\w-]+)?"
    r"|\d{4}[ \t]?[A-Z]{2}[ \t]+[A-Z][\w-]+(?:[ \t][A-Z][\w-]+)?"
    # a US town, state and ZIP ("Bend, OR 97701", test10)
    r"|[A-Z][\w'’.-]+(?:[ \t]+[A-Z][\w'’.-]+){0,2},[ \t]+[A-Z]{2}[ \t]+\d{5}(?:-\d{4})?"
    # an Irish Eircode on its own line ("A98 K7W2", test9)
    r"|[AC-FHKNPRTV-Y]\d{2}[ \t]?(?=[0-9A-Z]{0,3}[A-Z])[AC-FHKNPRTV-Y0-9]{4}),?[ \t]*")
# A county line inside the block ("Co. Wicklow") belongs to it but is not
# itself redacted (labeling-guide decision 22).
_BLOCK_COUNTY_LINE = re.compile(r"[ \t]*(?:Co\.|County)[ \t]+[A-Z][\w'’-]+,?[ \t]*")
# A house name on its own line above the street line ("Tŷ Gwyn", test9).
_BLOCK_HOUSE_LINE = re.compile(
    r"[ \t]*(?P<pii>[A-ZÀ-ÖØ-Þ][^\W\d_][\w'’-]*(?:[ \t]+[^\W\d_][\w'’-]*){0,3}),?[ \t]*")
# Table columns that hold people, plates and serial numbers, and the cell
# shapes they take.
_TABLE_NAME_HEADER = re.compile(
    r"(?i)(?:(?:full|visitor|guest|driver|employee|staff|contact|host|tenant|member|customer"
    r"|client|patient|student|pupil|runner|player|borrower"
    # the name fields of an export ("surname,forenames,dob,...", test10)
    r"|first|last|given|family|middle|maiden|preferred)[ \t]+)?names?"
    r"|(?:sur|fore|first|last)names?|visitors?|guests?"
    r"|hosts?|drivers?|employees?|attendees?|passengers?|tenants?|residents?|keepers?"
    r"|participants?|runners?|borrowers?|athletes?|signed[ \t]+(?:in|out)[ \t]+by"
    r"|(?:issued|loaned|lent)[ \t]+to|escort(?:ed[ \t]+by)?|visiting|person[ \t]+visited")
_TABLE_PLATE_HEADER = re.compile(
    r"(?i)(?:vehicle[ \t]+)?(?:reg(?:istration)?\.?(?:[ \t]+(?:no\.?|number|mark))?|plate"
    r"|number[ \t]+plate|licen[cs]e[ \t]+plate|kennzeichen|vrm)")
_TABLE_SERIAL_HEADER = re.compile(r"(?i)serial(?:[ \t]*(?:no\.?|number|#))?|s/n|imei")
# A birth-date column is personal on its own; postcode, address and phone
# columns count only beside a name column, where the row is a person's record
# (test10: an export with surname, forenames, dob, mobile and postcode).
_TABLE_DOB_HEADER = re.compile(
    r"(?i)d\.?o\.?b\.?|date[ \t]+of[ \t]+birth|birth[ \t]?date|birthday|born"
    r"|geburtsdatum|geboortedatum")
_TABLE_POSTCODE_HEADER = re.compile(
    r"(?i)post[ \t]?code|postal[ \t]+code|zip(?:[ \t]?code)?|eircode|plz")
_TABLE_ADDRESS_HEADER = re.compile(
    r"(?i)(?:(?:home|street|postal|mailing)[ \t]+)?address(?:[ \t]+line)?(?:[ \t]?[1-3])?|street")
_TABLE_PHONE_HEADER = re.compile(
    r"(?i)(?:(?:home|work|mobile|contact|emergency)[ \t]+)?(?:phone|mobile|tel\.?|telephone|cell)"
    r"(?:[ \t]+(?:no\.?|number))?")
_TABLE_PERSON_ROW_KINDS = ('POSTCODE', 'ADDRESS', 'PHONE')
# Columns that hold things or bodies: a NER "person" in one of them is a
# product, a club or a company ("Stanley FatMax", "Vale Royal Striders",
# test9).
_TABLE_THING_HEADER = re.compile(
    r"(?i)(?:item|product|tool|equipment|asset|article|model|make|description|club|team"
    r"|company|organi[sz]ation|firm|supplier|brand|category|department|dept\.?)"
    r"(?:[ \t]+(?:name|type|description))?")
# An item-number column makes a bare "Name" beside it the item's name, unless
# another column already describes the item.
_TABLE_ITEM_ID_HEADER = re.compile(
    r"(?i)(?:(?:item|asset|part|product|stock|tool|catalogue|cat\.?)[ \t]*"
    r"(?:no\.?|number|id|code|ref\.?|#)|sku)")
_TABLE_NAME_CELL = re.compile(
    r"(?:" + _TITLES + r"[ \t]+)?(?!(?:None|N/?A|Nil|Unknown|Self|Staff|Contractor|Visitor"
    r"|Delivery|Courier|Reception|Security|Tbc|Tba|Various)\b)"
    # A surname cell may open with its particle ("de Wit").
    r"(?P<pii>(?:[A-ZÀ-ÖØ-Þ]\.[ \t]?){0,2}(?:(?:van|der|den|de|von|da|di|du|la|le|del|della"
    r"|dos|das|ten|ter|al|el|bin)[ \t]+){0,2}" + _NAME_WORD
    + r"(?:" + _NAME_PARTICLES + r"[ \t]" + _NAME_WORD + r"){0,3}"
    # "Hall, Rosamund": surname first (test9)
    r"|" + _NAME_WORD + r",[ \t]+" + _NAME_WORD + r"(?:[ \t]" + _NAME_WORD + r")?)")
_TABLE_PLATE_CELL = re.compile(
    r"(?P<pii>(?=[A-Z0-9 -]{4,12}$)(?=[^\d]*\d)(?=[^A-Z]*[A-Z])[A-Z0-9]{1,4}(?:[ -][A-Z0-9]{1,4}){0,3})")
_TABLE_SERIAL_CELL = re.compile(r"(?P<pii>(?=.*\d)[A-Z0-9][A-Z0-9 ./-]{3,23})")
_TABLE_THING_CELL = re.compile(r"(?P<pii>.+)")
_TABLE_DOB_CELL = re.compile(
    r"(?i)(?P<pii>\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}(?:st|nd|rd|th)?[ \t]+" + _MONTHS + r",?[ \t]+(?:19|20)\d{2}"
    r"|" + _MONTHS + r"[ \t]+\d{1,2}(?:st|nd|rd|th)?,?[ \t]+(?:19|20)\d{2})")
_TABLE_POSTCODE_CELL = re.compile(r"(?P<pii>(?=.*\d)[A-Za-z0-9][A-Za-z0-9 -]{2,9})")
_TABLE_ADDRESS_CELL = re.compile(r"(?P<pii>(?=.*[^\W\d_]).{3,})")
_TABLE_PHONE_CELL = re.compile(r"(?P<pii>[+(]?\d[\d ()./-]{5,18}\d)")
# Column kind -> (entity type, the shape a cell must have).
_TABLE_CELL_SHAPES = {'PERSON': ('PERSON', _TABLE_NAME_CELL),
                      'LICENSE_PLATE': ('LICENSE_PLATE', _TABLE_PLATE_CELL),
                      'SERIAL_NUMBER': ('SERIAL_NUMBER', _TABLE_SERIAL_CELL),
                      'TABLE_THING_CELL': ('TABLE_THING_CELL', _TABLE_THING_CELL),
                      'DATE_OF_BIRTH': ('DATE_OF_BIRTH', _TABLE_DOB_CELL),
                      'POSTCODE': ('ADDRESS_LINE', _TABLE_POSTCODE_CELL),
                      'ADDRESS': ('ADDRESS_LINE', _TABLE_ADDRESS_CELL),
                      'PHONE': ('PHONE', _TABLE_PHONE_CELL)}
# A header line of a comma- or semicolon-separated export: bare labels, no
# space after the separator, so a sentence with commas is never one.
_CSV_LABEL = re.compile(r"(?=.{1,30}$)[A-Za-z][\w./#()'-]*(?:[ ][\w./#()'-]+){0,2}")
# One "key": "value" pair of a JSON record, and the keys that label a value
# as a field label does on a form (test9). camelCase keys are split first.
_JSON_PAIR = re.compile(r'"(?P<key>[A-Za-z][\w -]{0,40})"[ \t]*:[ \t]*"(?P<value>[^"\n]{1,120})"')
_JSON_KEY_TYPES = (
    (re.compile(r"(?:member(?:ship)?|customer|client|patient|employee|student|account|staff"
                r"|payroll|policy|loyalty|passport|national|nhs)[_ -]?(?:id|number|no|num|ref"
                r"|reference)"),
     'GENERIC_ID', re.compile(r"(?=.*\d)[A-Za-z0-9][A-Za-z0-9 /-]{3,23}")),
    (re.compile(r"date[_ -]?of[_ -]?birth|birth[_ -]?date|dob|birthday"),
     'DATE_OF_BIRTH', re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")),
    (re.compile(r"post[_ -]?code|postal[_ -]?code|zip(?:[_ -]?code)?|eircode"),
     'ADDRESS_LINE', re.compile(r"(?=.*\d)[A-Za-z0-9][A-Za-z0-9 -]{2,9}")),
    (re.compile(r"(?:address[_ -]?)?line[_ -]?[1-3]|street(?:[_ -]?address)?"
                r"|address(?:[_ -]?[1-3])?|house[_ -]?(?:number|name)"),
     'ADDRESS_LINE', re.compile(r"(?=.*[^\W\d_]).{3,}")),
    (re.compile(r"(?:(?:home|work|mobile|cell|contact|emergency)[_ -]?)?(?:phone|mobile|tel"
                r"|telephone|cell|fax)(?:[_ -]?(?:number|no))?"),
     'PHONE', re.compile(r"[+(]?\d[\d ()./-]{5,18}\d")),
    (re.compile(r"(?:first|last|sur|given|family|middle|maiden|full)[_ -]?names?|forenames?"),
     'PERSON', re.compile(r"[^\W\d_][^\d_]{1,60}")),
)
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
            break
        end += match.end()
    # A postcode said as its own sentence straight after the street line
    # ("6 Cooperage Row. DL8 2QK.", a caller's reply; test10).
    spoken = _POSTCODE_SENTENCE.match(text[end:end + 12])
    return end + spoken.end() if spoken else end


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
    no span crosses a line break. A county line may sit between the town and
    the postcode; it gets no span.
    """
    if not re.match(r',?[ \t]*\n', text[end:end + 4]):
        return []
    pos = text.find('\n', end) + 1
    spans = []
    for _ in range(4):
        line_end = text.find('\n', pos)
        line_end = len(text) if line_end == -1 else line_end
        line = text[pos:line_end]
        closing = _BLOCK_POSTCODE_LINE.fullmatch(line)
        if closing:
            spans.append((pos + closing.start('pii'), pos + closing.end('pii')))
            return spans
        if not _BLOCK_COUNTY_LINE.fullmatch(line):
            town = _BLOCK_TOWN_LINE.fullmatch(line)
            if not town or len(spans) == 2:
                return []
            spans.append((pos + town.start('pii'), pos + town.end('pii')))
        pos = line_end + 1
    return []


def block_house_line(text, start):
    """The line above a block's street line, when it has the shape of a house
    name. Only a candidate: apply_policy_filters decides (HOUSE_LINE)."""
    line_start = text.rfind('\n', 0, start) + 1
    if line_start == 0 or text[line_start:start].strip():
        return None
    above_start = text.rfind('\n', 0, line_start - 1) + 1
    match = _BLOCK_HOUSE_LINE.fullmatch(text[above_start:line_start - 1])
    if not match:
        return None
    return above_start + match.start('pii'), above_start + match.end('pii')


def _header_kinds(cells):
    """Column kinds of a header row, or None when it labels no column."""
    # "firstName" and "first_name" read as "first name".
    names = [re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', c).replace('_', ' ') for _, c in cells]
    kinds = ['PERSON' if _TABLE_NAME_HEADER.fullmatch(c) else
             'LICENSE_PLATE' if _TABLE_PLATE_HEADER.fullmatch(c) else
             'SERIAL_NUMBER' if _TABLE_SERIAL_HEADER.fullmatch(c) else
             'DATE_OF_BIRTH' if _TABLE_DOB_HEADER.fullmatch(c) else
             'POSTCODE' if _TABLE_POSTCODE_HEADER.fullmatch(c) else
             'ADDRESS' if _TABLE_ADDRESS_HEADER.fullmatch(c) else
             'PHONE' if _TABLE_PHONE_HEADER.fullmatch(c) else
             'TABLE_THING_CELL' if _TABLE_THING_HEADER.fullmatch(c) else None
             for c in names]
    if not any(k in ('PERSON', 'LICENSE_PLATE', 'SERIAL_NUMBER', 'DATE_OF_BIRTH') for k in kinds):
        return None
    if any(_TABLE_ITEM_ID_HEADER.fullmatch(c) for c in names) and 'TABLE_THING_CELL' not in kinds:
        kinds = ['TABLE_THING_CELL' if k == 'PERSON' and re.fullmatch(r'(?i)names?', c) else k
                 for k, c in zip(kinds, names)]
    if 'PERSON' not in kinds:
        kinds = [None if k in _TABLE_PERSON_ROW_KINDS else k for k in kinds]
    return kinds


def _delimited_cells(line, delim):
    """Cells of one comma- or semicolon-separated line as (offset, text)."""
    cells, start, quoted = [], 0, False
    for i, ch in enumerate(line + delim):
        if ch == '"':
            quoted = not quoted
        elif ch == delim and not quoted:
            raw = line[start:i]
            cell = raw.strip(' \t"')
            cells.append((start + (raw.find(cell) if cell else 0), cell))
            start = i + 1
    return cells


def _csv_header(line):
    """(delimiter, cells) when the line is the header of a delimited export."""
    for delim in ',;':
        if line.count(delim) >= 2 and delim + ' ' not in line:
            cells = _delimited_cells(line, delim)
            if all(_CSV_LABEL.fullmatch(cell) for _, cell in cells):
                return delim, cells
    return None


def table_column_spans(text):
    """Names, plates and serial numbers in a table, read by column.

    "Time in | Name | Host | Vehicle reg | Badge" (test8): the header cell
    labels every cell below it, as a field label does on a form. A cell must
    still have the shape of a name, a plate or a serial ("on foot" has none).
    Columns are separated by pipes or tabs, or by runs of two or more spaces
    in a fixed-width table ("Pos  Bib   Name   Club   Time", test9), where a
    cell belongs to the header it starts under, or by commas or semicolons in
    an export ("surname,forenames,dob,mobile,postcode", test10). Birth-date
    columns are read too, and postcode, address and phone columns beside a
    name column. Cells of a thing column come back as TABLE_THING_CELL
    candidates for apply_policy_filters.
    """
    results = []
    kinds, width, offsets, delim = None, 0, None, None

    def emit(line_start, row):
        for (col, cell), kind in row:
            entity, shape = _TABLE_CELL_SHAPES.get(kind, (None, None))
            match = shape.fullmatch(cell) if shape else None
            if match:
                start = line_start + col
                result = {'entity_type': entity, 'start': start + match.start('pii'),
                          'end': start + match.end('pii'), 'score': 1.0,
                          'text': match.group('pii'), 'source': 'regex'}
                if entity == 'DATE_OF_BIRTH':
                    # The birth label is the column header, not the text in
                    # front of the date.
                    result['label_elsewhere'] = True
                results.append(result)

    for line in re.finditer(r'[^\n]*(?:\n|$)', text):
        raw = line.group()
        if not raw.strip():
            kinds, delim = None, None
            if line.end() >= len(text):
                break
            continue
        piped = '|' in raw or '\t' in raw
        content = raw.rstrip('\n')
        export = None if piped else _csv_header(content)
        export_kinds = _header_kinds(export[1]) if export else None
        if export_kinds:
            kinds, width, offsets, delim = export_kinds, len(export[1]), None, export[0]
        elif delim is not None:
            cells = _delimited_cells(content, delim)
            if len(cells) == width:
                emit(line.start(), zip(cells, kinds))
            else:
                kinds, delim = None, None
        else:
            if piped:
                cells = [(m.start(1), m.group(1))
                         for m in re.finditer(r'[ \t]*([^|\t\n]*?)[ \t]*(?:[|\t]|\n|$)', raw)
                         if m.end() > m.start()]
            else:
                cells = [(m.start(), m.group()) for m in re.finditer(r'\S+(?:[ ]\S+)*', raw)]
            header = _header_kinds(cells) if len(cells) >= 3 else None
            if header:
                kinds, width = header, len(cells)
                offsets = None if piped else {col: kind for (col, _), kind in zip(cells, header)}
            elif kinds is not None and offsets is None:
                if not piped or len(cells) != width:
                    kinds = None
                else:
                    emit(line.start(), zip(cells, kinds))
            elif kinds is not None:
                aligned = [(cell, offsets[cell[0]]) for cell in cells if cell[0] in offsets]
                if piped or len(aligned) < 2:
                    kinds = None
                else:
                    emit(line.start(), aligned)
        if line.end() >= len(text):
            break
    return results


def json_field_spans(text):
    """Values of a JSON record whose key names personal data.

    '"member_id": "LTC-004417"', '"postcode": "7201 PX"' (test9): the key is
    the label. A value must still have the shape its key promises.
    """
    results = []
    for pair in _JSON_PAIR.finditer(text):
        key = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', pair.group('key')).lower()
        for keys, entity, shape in _JSON_KEY_TYPES:
            if keys.fullmatch(key):
                if shape.fullmatch(pair.group('value').strip()):
                    results.append({'entity_type': entity, 'start': pair.start('value'),
                                    'end': pair.end('value'), 'score': 1.0,
                                    'text': pair.group('value'), 'source': 'regex'})
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


def reply_birth_dates(text):
    """A birth date given in answer to a question on the line before.

    "Agent: And her date of birth?" / "Caller: The fourteenth of May, nineteen
    forty-one." (test10): the question is the label, as for the identifiers
    in chat_reply_ids. The date may be spoken, worded or numeric.
    """
    results = []
    for question in _BIRTH_QUESTION.finditer(text):
        reply = _REPLY_LINE.match(text, question.end())
        date = _REPLY_DATE.search(reply.group('body')) if reply else None
        if date:
            start = reply.start('body') + date.start('pii')
            results.append({'entity_type': 'DATE_OF_BIRTH', 'start': start,
                            'end': start + len(date.group('pii')), 'score': 1.0,
                            'text': date.group('pii'), 'source': 'regex',
                            'label_elsewhere': True})
    return results


def run_pattern_rules(text):
    """Stage 1: every pattern rule over the text → candidate list."""
    results = chat_reply_ids(text) + table_column_spans(text) + json_field_spans(text) \
        + reply_birth_dates(text)
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
            if pii_type == 'sensitive_organization' and _BARE_FACILITY.fullmatch(text, start, end):
                continue
            # A UK tax code ("Tax code: 1257L") is a rate, not an identifier;
            # the label is in the list for the Italian fiscal code (test9).
            if pii_type == 'generic_id' and end - start < 8 \
                    and re.match(r'(?i)tax\s+code', match.group()):
                continue
            if pii_type == 'labelled_person_caps' and _CAPS_NOT_A_NAME.search(text[start:end]):
                continue
            if pii_type in ('travel_city', 'journey_city', 'stay_city') \
                    and _VENUE_WORD.search(text[start:end]):
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
                block = address_block_lines(text, end)
                for block_start, block_end in block:
                    results.append({'entity_type': 'ADDRESS_LINE', 'start': block_start,
                                    'end': block_end, 'score': 1.0,
                                    'text': text[block_start:block_end], 'source': 'regex'})
                house = block_house_line(text, start) if block else None
                if house:
                    results.append({'entity_type': 'HOUSE_LINE', 'start': house[0],
                                    'end': house[1], 'score': 1.0,
                                    'text': text[house[0]:house[1]], 'source': 'regex'})
            entity_override = None
            if pii_type == 'generic_id' and re.fullmatch(r'[1-9]\d{19}', text[start:end]):
                # A labelled 20-digit value is a bank account, not a generic ID.
                entity_override = 'BANK_ACCOUNT'
            elif pii_type == 'generic_id' and re.match(r'(?i)nhs\b', match.group()):
                # The NHS label names the type, whether or not the checksum
                # satisfies the country recognizer.
                entity_override = 'UK_NHS'
            results.append({
                'entity_type': entity_override or PATTERN_ENTITY_TYPES.get(pii_type, pii_type.upper()),
                'start': start,
                'end': end,
                'score': PATTERN_SCORES.get(
                    pii_type, 1.1 if pii_type in LABELLED_PATTERNS else 1.0),
                'text': text[start:end],
                'source': 'regex'
            })
            if 'subject' in match.re.groupindex and match.group('subject'):
                results[-1]['subject'] = match.group('subject')
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
    r"|Groupe|Société|Cabinet"
    # Dutch bodies, and the word for a department in the form languages
    # ("Woonstichting De Linde", "Afdeling Verhuur"; test10).
    r"|Stichting|Woonstichting|Woningstichting|Woningcorporatie|Vereniging|Gemeente"
    r"|Afdeling|Abteilung|Dipartimento|Departamento|Département|Ufficio)\b")
# A courtesy title in front of a span, or at its start, makes it a person
# even when the surname is a company word ("Fr. Annelise Bank", test8).
_TITLE_BEFORE = re.compile(r"(?:^|[\s(,;:—–-])" + _TITLES + r"[ \t]+$")
_TITLE_START = re.compile(_TITLES + r"[ \t]+(?=\S)")
_TITLE_WORD = re.compile(r"^" + _TITLES + r"$")
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
# A kin or office word in front of a name is not part of it ("Nana Jo",
# "Deacon Ifor", test9): the name is what follows.
_ROLE_TITLE_LEAD = re.compile(
    r"^(?:Nana|Nanny|Granny|Grandma|Grandpa|Grandad|Granddad|Auntie|Aunty|Aunt|Uncle|Cousin"
    r"|Deacon|Canon|Bishop|Pastor|Vicar|Rabbi|Imam|Nurse|Coach|Councillor|Sergeant|Constable"
    r"|Inspector|Judge|Matron|Chef|Professor|Doctor|Captain|Officer|Father|Sister|Brother)"
    r"[ \t]+(?=[A-ZÀ-ÖØ-Þ])")
# Nor are the letters after it ("Mr A. K. Chakraborty FRCS").
_POST_NOMINAL_TAIL = re.compile(
    r"(?:,?[ \t]+(?:F?RC[A-Z]{1,4}|MRC[A-Z]{1,4}|MBBS|MBChB|BChir|MD|PhD|DPhil|DDS|BDS|BSc|MSc"
    r"|BEng|MEng|BA|MA|MBA|LLB|LLM|OBE|MBE|CBE|KBE|DBE|QC|KC|JP|RN|RGN|RMN|CEng|MRICS|FRICS"
    r"|ACA|FCA|ACCA|MCIPD|FCIPD|Esq\.?))+$")
# A lowercase name particle or article on its own is not a name ("the silver
# van", "la siguiente persona"; test9).
_PARTICLE_WORDS = frozenset(
    "van der den de von da di du la le del della dos das ten ter op bin al el ap ab ibn "
    "bint abu af av zu".split())
_LOWERCASE_ARTICLE_LEAD = re.compile(
    r"^(?:la|el|los|las|le|les|der|die|das|den|the|an?|il|lo|un|una|uno|une|des|du|de|que|en"
    r"|con|por|para|pour|mit|von)[ \t]")
# The title of a chat group, a folder or a list in quotes is not a person
# ('group "Bin day gang"', test9).
_LOWERCASE_LEAD = re.compile(r"^(?:[a-zß-öø-ÿ]\S*[ \t]+)+(?=\S)")
_LOWERCASE_TAIL = re.compile(r"(?:[ \t]+[a-zß-öø-ÿ]\S*)+$")
_QUOTED_TITLE_PREFIX = re.compile(
    r"(?i)\b(?:group|chat|channel|thread|folder|album|playlist|list|board|topic)\s*:?\s*[\"“‘']$")
# Nor is the name of a statute after its section ("§ 19 Bundesmeldegesetz").
_DEPARTMENT_WORD = re.compile(
    r"(?:" + _GREETING_STOP + r"|Department|Dept\.?|Division|Section|Team|Unit|Office)")
_GREETING_WORD_BEFORE = re.compile(r"\b(?:Dear|Hi|Hello|Hey|Hiya|Thanks|Cheers)[ \t]+$")
_LAW_PREFIX = re.compile(r"(?:§+|\b(?:Art\.?|Artikel|Abs\.?))[ \t]*\d+[a-z]?[ \t]+$")
# Words that label a phone number without naming a person ("Plumber: ...").
_PHONE_LABEL_WORD = re.compile(
    r"(?i)(?:" + _GREETING_STOP + r"|" + _PERSON_NOUNS + r"|tel|tél|fax|mob|ext|telefon"
    r"|telefono|teléfono|phone|mobile|handy|cell|home|work|office|emergency|taxi|vet|dentist"
    r"|doctor|surgery|school|nursery|garage|bank|pharmacy|chemist|gas|water|electric(?:ity)?"
    r"|council|police|hospital|clinic|helpline|hotline|landline|direct|switchboard|main"
    r"|out|day|night|evenings?|daytime|after|other|alt|alternative)")
_LABEL_WITH_NUMBER = re.compile(r"[ \t]*:[ \t]*[+(\d]")


# One-word NER "persons" from closed vocabularies (test10): agenda and office
# abbreviations ("4. AOB."), the word for "none" as a form value in other
# languages ("Note / notes: Nessuna."), and common medicines ("Senna only if
# needed"). A medicine no list knows is still one ahead of its dose.
_ABBREVIATION_WORDS = frozenset(
    "AOB TBC TBA TBD FAO FYI ETA ETD ASAP RSVP NB PS PTO WIP EOD COB DNA DOB NHS GP VAT "
    "PAYE HR IT QA PO HQ".split())
_NONE_WORDS = frozenset(
    "Nessuno Nessuna Niente Geen Keine Keiner Kein Ninguno Ninguna Aucun Aucune Néant "
    "Nenhum Nenhuma Brak Ingen Inget".split())
_MEDICINE_WORDS = frozenset(
    "Amlodipine Ramipril Lisinopril Losartan Candesartan Bisoprolol Atenolol Propranolol "
    "Atorvastatin Simvastatin Rosuvastatin Pravastatin Metformin Gliclazide Sitagliptin "
    "Empagliflozin Dapagliflozin Insulin Levothyroxine Omeprazole Lansoprazole Esomeprazole "
    "Pantoprazole Ranitidine Famotidine Gaviscon Senna Lactulose Laxido Movicol Docusate "
    "Loperamide Paracetamol Ibuprofen Naproxen Diclofenac Aspirin Codeine Co-codamol "
    "Co-dydramol Tramadol Morphine Oramorph Oxycodone Fentanyl Gabapentin Pregabalin "
    "Amitriptyline Nortriptyline Sertraline Citalopram Escitalopram Fluoxetine Paroxetine "
    "Mirtazapine Venlafaxine Duloxetine Trazodone Diazepam Lorazepam Temazepam Zopiclone "
    "Zolpidem Quetiapine Olanzapine Risperidone Aripiprazole Haloperidol Lithium Donepezil "
    "Memantine Rivastigmine Galantamine Warfarin Apixaban Rivaroxaban Edoxaban Dabigatran "
    "Clopidogrel Ticagrelor Heparin Enoxaparin Furosemide Bumetanide Spironolactone "
    "Bendroflumethiazide Indapamide Digoxin Amiodarone Isosorbide Salbutamol Ventolin "
    "Seretide Symbicort Fostair Tiotropium Montelukast Prednisolone Prednisone "
    "Hydrocortisone Dexamethasone Amoxicillin Flucloxacillin Penicillin Co-amoxiclav "
    "Doxycycline Clarithromycin Erythromycin Azithromycin Trimethoprim Nitrofurantoin "
    "Ciprofloxacin Metronidazole Cefalexin Fluconazole Aciclovir Alendronate Adcal "
    "Calcichew Colecalciferol Thiamine Cetirizine Loratadine Fexofenadine Chlorphenamine "
    "Piriton Tamsulosin Finasteride Solifenacin Oxybutynin Mirabegron Levodopa "
    "Co-careldopa Madopar Sinemet Ropinirole Pramipexole Lamotrigine Levetiracetam "
    "Carbamazepine Valproate Epilim Phenytoin Topiramate Sumatriptan Methotrexate "
    "Hydroxychloroquine Sulfasalazine Allopurinol Colchicine Tamoxifen Letrozole "
    "Anastrozole Semaglutide Liraglutide Latanoprost Hypromellose Cyclizine Ondansetron "
    "Metoclopramide Domperidone Buscopan Mebeverine".split())
_DOSE_NEXT = re.compile(
    r"(?i)[ \t]+\d+(?:[.,]\d+)?[ \t]?(?:mg|mcg|µg|g|ml|micrograms?|milligrams?|units?|iu)\b")
# "Trailer 14 is in Immingham until the 27th": what follows a verb of being
# and "in" is a place (test10).
_STATIVE_IN_PREFIX = re.compile(
    r"(?i)\b(?:is|are|was|were|be|been|stays?|stayed|based|located|parked|held|stuck)"
    r"[ \t]+in[ \t]+$")
# "away to Thorpe Arch": a fixture names the opponent or the ground (test10).
_FIXTURE_PREFIX = re.compile(r"(?i)\b(?:away|home)[ \t]+(?:to|at|against)[ \t]+$")
# A pet is named like a person and is not one ("Biscuit (the cat)", "her dog
# Biscuit"; labeling-guide decision 23).
_PET_NOUNS = (r"(?:cat|dog|puppy|kitten|rabbit|hamster|guinea[ \t]+pig|budgie|parrot|horse|pony"
              r"|tortoise|goldfish|ferret|gerbil|spaniel|terrier|labrador|collie|greyhound"
              r"|whippet|poodle|tabby)")
_PET_AFTER = re.compile(
    r"(?i)[ \t]*[(,][ \t]*(?:the|our|her|his|their|my|a)[ \t]+(?:[a-z]+[ \t]+)?" + _PET_NOUNS + r"\b")
_PET_BEFORE = re.compile(
    r"(?i)\b(?:the|our|her|his|their|my)[ \t]+(?:[a-z]+[ \t]+)?" + _PET_NOUNS
    + r",?[ \t]+(?:called[ \t]+|named[ \t]+)?$")
# A key of a JSON record, whatever its spelling ('"bedLabel":', test10).
_JSON_KEY_NEXT = re.compile(r'"[ \t]*:')
_CAMEL_CASE_WORD = re.compile(r"[a-z]+(?:[A-Z][a-z0-9]*)+")
# A subject line or heading in front of an address the model found: "Deposit
# — Flat 2, 18 Limekiln Bank" (test10).
_ADDRESS_LEAD_LABEL = re.compile(
    r"[A-Za-z][A-Za-z ]{0,30}?[ \t]+[—–-][ \t]+(?=\d|(?i:flat|apt|apartment|unit|suite)\b)")
# The facility that heads its own document ("Oakmere Day Centre — minibus
# rota", test10) is its author, as on a letterhead.
_HEADING_DASH_NEXT = re.compile(r"[ \t]+[—–-][ \t]+\S")

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
        if EMPLOYMENT_CONTEXT.search(
                sentence_around(text, match.start(), match.end()).replace(name, ' ')) \
                or ORG_AT_PREFIX.search(prefix) \
                or EMPLOYMENT_HEADING.search(section_heading(text, match.start())):
            return True
    return False


def apply_policy_filters(results, text):
    """Stage 2: enforce the context rules of the V1 public-sharing policy.

    Two passes: the first decides which person candidates survive, the second
    uses those survivors where a rule asks "is there a person on this line",
    so a dropped false person no longer supports a plate beside it (test9).
    """
    first = _filter_pass(results, text, None)
    survivors = [(r['start'], r['end']) for r in first if r['entity_type'] == 'PERSON']
    return _filter_pass(results, text, survivors)


def _filter_pass(results, text, surviving_persons):
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
    line_persons = name_spans if surviving_persons is None else surviving_persons
    thing_cells = [(r['start'], r['end']) for r in results
                   if r['entity_type'] == 'TABLE_THING_CELL']
    thing_texts = {r['text'] for r in results if r['entity_type'] == 'TABLE_THING_CELL'}
    # The head word of a thing cell names the same thing on its own ("Merlins"
    # for the team "Under 14 Merlins", test10).
    thing_heads = {t.split()[-1] for t in thing_texts if len(t.split()) > 1}
    phone_starts = {r['start'] for r in results
                    if r['entity_type'].startswith('PHONE') and r.get('source') == 'regex'}
    author_word = heading_author_word(
        text, [(s, e) for s, e in name_spans if not is_company_shaped(text, s, e)])
    # Words of the person candidates, for rules whose subject must be a
    # person; and the names the document gives to pets.
    subject_words = {word for s, e in line_persons
                     for word in re.findall(r"[^\W\d_][\w'’-]*", text[s:e])}
    pet_names = {r['text'] for r in results
                 if r['entity_type'] == 'PERSON' and r.get('source') in _NER_SOURCES
                 and (_PET_AFTER.match(text, r['end'])
                      or _PET_BEFORE.search(text[max(0, r['start'] - 40):r['start']]))}

    def near_person(start, end, radius=PERSON_PROXIMITY_CHARS):
        return any(p_start - radius <= end and start <= p_end + radius
                   for p_start, p_end in person_spans)

    def person_on_line(start, end):
        line_start = text.rfind('\n', 0, start) + 1
        line_end = text.find('\n', end)
        line_end = len(text) if line_end == -1 else line_end
        return any(line_start <= s and e <= line_end for s, e in line_persons)

    filtered = []
    for result in results:
        entity, source = result['entity_type'], result.get('source')
        demoted = False
        if entity == 'TABLE_THING_CELL':
            continue
        # "Delivery lands in Hull": the subject of a journey is a person.
        if result.get('subject') and result['subject'] not in subject_words:
            continue
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
            # A kin or office word in front and the letters behind are not
            # part of the name.
            role = _ROLE_TITLE_LEAD.match(result['text'])
            if role:
                result = {**result, 'start': result['start'] + role.end(),
                          'text': result['text'][role.end():]}
            tail = _POST_NOMINAL_TAIL.search(result['text'])
            if tail and tail.start() > 1:
                result = {**result, 'end': result['start'] + tail.start(),
                          'text': result['text'][:tail.start()]}
            # A cell of a thing column is a product, a club or a company,
            # and the same text is the same thing elsewhere ("Thornwick
            # Harriers" in the heading and in the Club column).
            if any(s <= result['start'] and result['end'] <= e for s, e in thing_cells) \
                    or result['text'] in thing_texts or result['text'] in thing_heads:
                continue
            # A name has no underscore (the JSON key "date_of_birth"), is not
            # a lowercase particle or an article-led lowercase phrase, and is
            # not a title on its own ("Dña").
            if '_' in result['text'] or _TITLE_WORD.match(result['text']) \
                    or (result['text'].islower()
                        and (result['text'] in _PARTICLE_WORDS
                             or _LOWERCASE_ARTICLE_LEAD.match(result['text']))):
                continue
            # Nor is it a JSON key, a camelCase identifier, a pet, or the
            # opponent in a fixture.
            if _JSON_KEY_NEXT.match(text, result['end']) \
                    or _CAMEL_CASE_WORD.fullmatch(result['text']) \
                    or result['text'] in pet_names \
                    or _FIXTURE_PREFIX.search(text[max(0, result['start'] - 16):result['start']]):
                continue
            # General NER tags lowercase phrases of other languages as people
            # ("parla inglese e spagnolo", "uw wijkbeheerder"; test10): a
            # lowercase span on a line that does use capitals is a phrase. A
            # line typed without any is chat, where a name may be lowercase;
            # the custom model, which has seen such names, is left alone.
            if source != 'custom_ner' and ' ' in result['text'] and result['text'].islower():
                line_start = text.rfind('\n', 0, result['start']) + 1
                line_end = text.find('\n', result['end'])
                if any(ch.isupper() for ch in
                       text[line_start:len(text) if line_end == -1 else line_end]):
                    continue
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
                and (is_letterhead(text, result['start'], result['end'], result['text'])
                     or (result['start'] == document_start(text)
                         and _HEADING_DASH_NEXT.match(text, result['end']))):
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
            # The employment wording must come from the context, not from the
            # name itself ("Larkfield Staffing" on its own letterhead, test9).
            if not (role_for or SENSITIVE_ORG_KEYWORDS.search(result['text'])
                    or EMPLOYMENT_CONTEXT.search(sentence.replace(result['text'], ' '))
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
                and (THING_LABEL_PREFIX.search(prefix) or _QUOTED_TITLE_PREFIX.search(prefix)
                     or _LAW_PREFIX.search(prefix)):
            continue
        # The label half of a bilingual form line is not a person
        # ("Bemerkungen / remarks: ...", "Einzug am / moved in on: ...",
        # test9): the span sits before the line's first colon, and the label
        # is short, has no comma, and carries a spaced slash and a lowercase
        # word. A sentence that merely holds a colon further on is no label
        # ("Je soussignée / I, the undersigned, Mme ...").
        if entity == 'PERSON' and source in ('spacy', 'presidio', 'custom_ner'):
            line_start = text.rfind('\n', 0, result['start']) + 1
            line_end = text.find('\n', result['end'])
            colon = text.find(':', result['end'], len(text) if line_end == -1 else line_end)
            label = text[line_start:colon] if colon != -1 else ''
            if ' / ' in label and len(label) <= 70 and ',' not in label \
                    and ':' not in text[line_start:result['start']] \
                    and re.search(r"(?:^|[ \t/(])[a-zß-öø-ÿ]", label):
                continue
            # A name holds no lowercase word but a particle. A mixed-case
            # span loses the lowercase words around the name ("que l'élève
            # Élodie Collin", "Quentin avant vendredi"); one with a lowercase
            # word inside is a phrase ("Zählerstand Strom bei Einzug").
            # All-lowercase chat names are left alone.
            if ' ' in result['text'] and not result['text'].islower():
                lead = _LOWERCASE_LEAD.match(result['text'])
                if lead and any(w.lower().strip("'’") not in _PARTICLE_WORDS
                                for w in lead.group().split()):
                    result = {**result, 'start': result['start'] + lead.end(),
                              'text': result['text'][lead.end():]}
                tail = _LOWERCASE_TAIL.search(result['text'])
                if tail:
                    result = {**result, 'end': result['start'] + tail.start(),
                              'text': result['text'][:tail.start()]}
                if any(w[:1].islower() and w.lower() not in _PARTICLE_WORDS
                       and w not in ('and', 'y', 'e', 'i', 'und', 'et', 'of', 'the', "'t", '’t')
                       for w in result['text'].split()[1:]):
                    continue
        # Nor is it an address: the custom model reads "Engine: Kubota D902,
        # 16 hp" as a street and a house number (candidate 3, 2026-11-13).
        if entity == 'ADDRESS' and source == 'custom_ner' \
                and THING_LABEL_PREFIX.search(prefix):
            continue
        # An address opens with a number or a capital, never with a lowercase
        # word ("que en el Padrón Municipal", test9).
        if entity == 'ADDRESS' and source == 'custom_ner' and result['text'][:1].islower():
            continue
        # Nor does it start in a thing column of a table ("Under 12
        # Kestrels,YM" in the team column of an export, test10).
        if entity == 'ADDRESS' and source == 'custom_ner' \
                and any(s < result['end'] and result['start'] < e for s, e in thing_cells):
            continue
        # A heading word and a dash in front of the address are not part of it.
        if entity == 'ADDRESS' and source == 'custom_ner':
            lead = _ADDRESS_LEAD_LABEL.match(result['text'])
            if lead:
                result = {**result, 'start': result['start'] + lead.end(),
                          'text': result['text'][lead.end():]}
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
            numbered = _LABEL_WITH_NUMBER.match(after) \
                if not text[line_start:result['start']].strip() else None
            # A first name may label a phone number in a contact list ("Maud:
            # 07700 900 123", test9); a role or service word never is one.
            if numbered and result['end'] + numbered.end() - 1 in phone_starts \
                    and not _PHONE_LABEL_WORD.fullmatch(result['text']):
                numbered = None
            # A closing word on its own line above the signature ("Bendithion,"
            # over "Lowri", test9) is not the signer.
            line_end = text.find('\n', result['end'])
            closing = False
            if line_end != -1 and not text[line_start:result['start']].strip() \
                    and text[result['end']:line_end].strip() == ',':
                below = line_end + 1
                below += len(text[below:]) - len(text[below:].lstrip(' \t'))
                closing = any(s == below for s, e in name_spans)
            # "Dear Bursar,": the addressee is a role.
            if _GREETING_STOP_WORD.fullmatch(result['text']) \
                    and _GREETING_WORD_BEFORE.search(prefix):
                continue
            if result['text'] in _ABBREVIATION_WORDS or result['text'] in _NONE_WORDS \
                    or result['text'] in _MEDICINE_WORDS \
                    or _DOSE_NEXT.match(text, result['end']) \
                    or _STATIVE_IN_PREFIX.search(prefix):
                continue
            if result['text'].capitalize() in _COURTESY_WORDS or closing \
                    or _DATE_PLACEHOLDER.fullmatch(result['text']) \
                    or numbered \
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
        # The line above a block's street line is a house name when the
        # line above it is a person ("Tegwen Mostyn / Tŷ Gwyn / 3 Tinnehinch
        # Lane", test9) or it carries a house word; never a company or a
        # department line. A second name there is redacted either way.
        if entity == 'HOUSE_LINE':
            above_end = text.rfind('\n', 0, result['start'])
            above_start = text.rfind('\n', 0, max(above_end, 0)) + 1
            under_person = above_end > 0 and any(
                above_start <= s and e <= above_end for s, e in name_spans)
            if is_company_shaped(text, result['start'], result['end']) \
                    or any(_DEPARTMENT_WORD.fullmatch(w) for w in result['text'].split()) \
                    or not (under_person or _HOUSE_WORD.search(result['text'])):
                continue
            result = {**result, 'entity_type': 'ADDRESS_LINE'}
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
def echo_person_names(results, text):
    """Stage 3: repeat each detected person name over its other occurrences.

    A multi-word name also lends its first name: "Sigrún Hallgrímsdóttir" on
    the pupil line makes the later bare "Sigrún" the same person (test6).
    Case-sensitive, four letters or more, and never a word that opens
    sentences in its own right.
    """
    extra, seen, surnames = [], set(), set()
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
        if len(words) > 1 and len(words[-1]) >= 4 and words[-1].isalpha() and words[-1][0].isupper():
            surnames.add(words[-1])
        first = words[0] if len(words) > 1 else None
        if first and len(first) >= 4 and first.isalpha() and first[0].isupper() \
                and first not in _ECHO_FIRST_NAME_STOP and first not in seen:
            seen.add(first)
            echo_first_name(first)
    # "The Hrabals have two spare seats": a known surname in the plural, with
    # a verb that takes people, is that family (test10).
    for surname in sorted(surnames):
        if len(surname) < 5:
            continue
        for match in re.finditer(
                r"\b[Tt]he[ \t]+(" + re.escape(surname) + r"(?:e?s|['’]s?))[ \t]+(?:have|had|are|were"
                r"|will|would|can|live|lived|said|say|offered|agreed|asked|want|wanted|kindly)\b",
                text):
            extra.append({'entity_type': 'PERSON', 'start': match.start(1), 'end': match.end(1),
                          'score': 0.9, 'text': match.group(1), 'source': 'echo'})
    # Two known surnames joined by "and" are those two people ("Park and
    # Church ran for the county", test9); a surname alone is too often an
    # ordinary word to echo.
    if len(surnames) > 1:
        either = '|'.join(re.escape(s) for s in sorted(surnames))
        for pair in re.finditer(r'\b(' + either + r')[ \t]+(?:and|&)[ \t]+(' + either + r')\b', text):
            for group in (1, 2):
                if not any(s <= pair.start(group) and pair.end(group) <= e
                           for s, e in person_spans):
                    extra.append({'entity_type': 'PERSON', 'start': pair.start(group),
                                  'end': pair.end(group), 'score': 0.9,
                                  'text': pair.group(group), 'source': 'echo'})
    return extra


def relabel_by_id_context(results, text):
    """Stage 5: an explicit ID label before a span decides its entity type."""
    for result in results:
        token = ID_LABEL_TOKEN.search(text[max(0, result['start'] - 12):result['start']])
        if token:
            result['entity_type'] = ID_LABEL_OVERRIDES[token.group(1).lower()]
    return results
