"""
Configuration settings for Privacy Sandbox
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_INPUT = os.path.join(PROJECT_ROOT, "data", "input")
DATA_OUTPUT = os.path.join(PROJECT_ROOT, "data", "output")
SAMPLE_DATA = os.path.join(PROJECT_ROOT, "tests", "sample_data")

# PII Detection settings
CONFIDENCE_THRESHOLD = 0.8
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
