import re
from pathlib import Path

html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

# 1. Content coverage
checks = [
    ('product', 'Privacy Sandbox'), ('value prop', 'Redact it before you share'),
    ('how it works', 'How it works'), ('compliant', 'Legal compliance'),
    ('collab', 'Safe collaboration'), ('ml', 'ML-ready data'), ('time', 'Time savings'),
    ('gdpr', 'GDPR'), ('ferpa', 'FERPA'), ('hipaa', 'HIPAA'), ('ccpa', 'CCPA'),
    ('local', 'never leave your machine'),
    ('ext text', '.txt &nbsp;.json &nbsp;.md'), ('ext img', '.png &nbsp;.bmp'),
    ('mask', 'Complete Masking'), ('smart', 'Smart Redaction'), ('pseudo', 'Pseudonymize'),
    ('partial', 'Partial mask'), ('label', 'Label'), ('pixel', 'Pixelate'),
    ('blackbox', 'Solid black box'), ('blur', 'Blur'), ('presidio', 'Presidio'),
    ('spacy', 'spaCy'), ('mediapipe', 'MediaPipe'), ('haar', 'Haar'), ('ocr', 'OCR'),
    ('report', 'privacy report'), ('engines', 'custom regex'), ('claimed', 'dual-model'),
]
missing = [k for k, v in checks if v not in html]
print('CONTENT CHECK:', 'ALL PRESENT' if not missing else 'MISSING ' + str(missing))

# 2. Tag balance
for tag in ['div', 'section', 'main', 'header', 'footer', 'aside', 'ul', 'ol', 'li',
            'table', 'thead', 'tbody', 'tr', 'figure', 'svg', 'pre', 'button', 'form']:
    o = len(re.findall(r'<%s[\s>]' % tag, html))
    c = len(re.findall(r'</%s>' % tag, html))
    if o != c:
        print('  TAG MISMATCH: %s open=%d close=%d' % (tag, o, c))
print('TAG BALANCE: checked')
