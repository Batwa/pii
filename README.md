# Privacy Sandbox (PII Identifier)

Detect and redact personally identifiable information (PII) from CSV files, text documents, and images.

## What this project does

- **CSV files**: finds names, emails, phones, and SSNs in spreadsheet columns
- **Text files** (`.txt`, `.json`, `.md`): uses NLP + regex to find and redact PII
- **Images**: uses OCR to find text PII and OpenCV to blur faces

Run the web app:

```bash
streamlit run app.py
```

Or run the **HTML design site** (dark-theme UI at `design/index.html`) with real Python
processing for all three tracks (CSV, text, images — uploads from your computer are
processed locally through `server.py`):

```bash
python3 server.py      # -> http://localhost:8000
```

Or test from the command line:

```bash
python text_pii_detector.py
python pii_detector.py
python image_detector.py
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

For image OCR on macOS, also install Tesseract:

```bash
brew install tesseract
```

Create sample files to try:

```bash
python sample_text_files.py
python sample_images.py
```

Run the test suite:

```bash
pytest
```

## Project layout

```
app.py                 Streamlit web interface
pii_detector.py        CSV / tabular PII detection
text_pii_detector.py   Text file PII detection
image_detector.py      Image PII detection
config.py              Shared settings and paths
data/input/            Put your files here
data/output/           Redacted results are saved here
tests/sample_data/     Small files used by automated tests
```

---

## What is a README?

A **README** is the instruction manual for your project. When someone (including future you) opens the repo, the README explains:

- what the project does
- how to install it
- how to run it
- where important files live

Think of it as the cover page and quick-start guide bundled together.

## What is a .gitignore?

A **`.gitignore`** tells Git which files to **leave out** of version control.

Some files shouldn't be committed because they are:

- **Generated** — like `__pycache__/` or `data/output/` (you can recreate them by running the app)
- **Personal** — like `.env` files that may contain secrets
- **Machine-specific** — like virtual environment folders

Without a `.gitignore`, your repo gets cluttered with junk files and you risk accidentally sharing passwords.

---

## Why do we need tests?

Tests are small scripts that check the code still works after you change something.

They help because:

1. **Catch regressions** — if partial redaction or phone-column detection breaks, tests fail immediately instead of you finding out during a demo
2. **Document expected behavior** — a test that says `555-123-4567` in a phone column gets redacted is a clear spec
3. **Make changes safer** — you can refactor with confidence instead of manually re-testing every file type

You don't need tests for a one-off script, but for a multi-file app like this with CSV, text, and image paths, a few focused tests save real time.

## Configuration

Key settings live in `config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONFIDENCE_THRESHOLD` | `0.8` | Minimum detection confidence (0.1 = more sensitive) |
| `DATA_INPUT` | `data/input/` | Where sample input files go |
| `DATA_OUTPUT` | `data/output/` | Where redacted output is saved |
| `SAMPLE_DATA` | `tests/sample_data/` | Small fixtures used by `pytest` |

The sidebar **Detection Confidence** slider in the Streamlit app overrides this for both CSV and text scanning.
