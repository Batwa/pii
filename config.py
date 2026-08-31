"""
Configuration settings for Privacy Sandbox
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_OUTPUT = os.path.join(PROJECT_ROOT, "data", "output")
SAMPLE_DATA = os.path.join(PROJECT_ROOT, "tests", "sample_data")

# PII Detection settings
CONFIDENCE_THRESHOLD = 0.8
CUSTOM_NER_MODEL = os.environ.get(
    "CUSTOM_NER_MODEL",
    os.path.join(PROJECT_ROOT, "models", "custom_ner"),
)

# Country ID recognizers that ship with Presidio but are disabled in its default
# registry. Presidio's own YAML loader cannot instantiate all of these (it always
# passes a `name` kwarg, which UsMbiRecognizer does not accept), so they are
# registered in code by recognizers.build_analyzer_registry().
INTERNATIONAL_RECOGNIZERS = [
    "AuAbnRecognizer", "AuAcnRecognizer", "AuMedicareRecognizer", "AuTfnRecognizer",
    "CaSinRecognizer",
    "InAadhaarRecognizer", "InGstinRecognizer", "InPanRecognizer",
    "InPassportRecognizer", "InVehicleRegistrationRecognizer", "InVoterRecognizer",
    "NgNinRecognizer", "NgVehicleRegistrationRecognizer",
    "PhTinRecognizer", "PhUmidRecognizer",
    "SgFinRecognizer",
    "UkDrivingLicenceRecognizer", "UkNinoRecognizer", "UkPassportRecognizer",
    "UkPostcodeRecognizer", "UkVehicleRegistrationRecognizer",
    "UsMbiRecognizer", "UsNpiRecognizer",
    "ZaIdNumberRecognizer",
]

# Presidio is queried at this floor, then filtered per entity type below. Asking
# Presidio directly for CONFIDENCE_THRESHOLD would silently drop every recognizer
# that has no checksum to validate against.
PRESIDIO_SCORE_FLOOR = 0.3

# Entity types whose best observed score is under CONFIDENCE_THRESHOLD even when a
# label such as "NINO" or "passport" sits next to the value. Each threshold is set
# just under the score measured for a valid identifier in context. Lowering the bar
# trades false negatives for false positives, so re-measure before widening these.
ENTITY_SCORE_THRESHOLDS = {
    "IN_PASSPORT": 0.40,   # observed 0.45 — letter + 7 digits, weakest pattern here
    "IN_VOTER": 0.70,      # observed 0.75
    "PH_TIN": 0.35,        # observed 0.40
    "UK_NINO": 0.45,       # observed 0.50
    "UK_POSTCODE": 0.40,   # observed 0.45
    "US_MBI": 0.60,        # observed 0.65
}

# V1 policy decisions for the public-sharing profile (red.pdf). Third-person
# singular pronouns are always redacted; the plural group is redacted only when a
# person is detected nearby, because "they/their" routinely refers to objects and
# organizations. I/me/my/you/your are kept per the policy.
PRONOUNS_ALWAYS = ("he", "him", "his", "she", "her", "hers")
PRONOUNS_NEAR_PERSON = ("they", "them", "their", "theirs")
PERSON_PROXIMITY_CHARS = 160
SUPPORTED_FILE_TYPES = {
    'tabular': ['.csv'],
    'text': ['.txt', '.json', '.md'],
    'image': ['.jpg', '.jpeg', '.png', '.bmp']
}

# Column name hints used to apply regex scanning in CSV files
PII_COLUMN_KEYWORDS = {
    'phone': 'phone_in_text',
    'mobile': 'phone_in_text',
    'tel': 'phone_in_text',
    'email': 'email_in_text',
    'e-mail': 'email_in_text',
    'ssn': 'ssn_in_text',
    'social': 'ssn_in_text',
}

# Redaction options
REDACTION_METHODS = {
    'mask': '****',
    'hash': True,
    'replace': '[REDACTED]'
}
