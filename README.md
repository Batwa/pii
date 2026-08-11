# Privacy Sandbox (PII Identifier)

Detect and redact personally identifiable information (PII) from CSV files, text documents, and images. Processing happens on your computer through a local server.

## What this project does

- **CSV files**: finds names, emails, phones, and SSNs in spreadsheet columns
- **Text files** (`.txt`, `.json`, `.md`): uses NLP + regex to find and redact PII
- **Images**: uses OCR to find text PII and OpenCV to blur faces

Run the local website (dark-theme UI at `design/index.html`) with real Python
processing for all three tracks (CSV, text, images — uploads from your computer are
processed locally through `server.py`):

```bash
python3 server.py      # -> http://localhost:8000
```

Or test from the command line:

```bash
python text_pii_detector.py
python pii_detector.py
```

## Setup

Use Python 3.13 (the version this project is tested with). Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

For image OCR, install Tesseract and ensure its executable is available on your `PATH`:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr
```

On Windows, install Tesseract with your preferred package manager or installer, then add its installation folder to `PATH`.

Create sample files to try:

```bash
python sample_text_files.py
python sample_images.py
```

Run the test suite (development dependencies only):

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## Project layout

```
server.py              Local website and processing API
design/                 Website HTML, CSS, and JavaScript
pii_detector.py        CSV / tabular PII detection
text_pii_detector.py   Text file PII detection
image_detector.py      Image PII detection
config.py              Shared settings and paths
data/input/            Put your files here
data/output/           Redacted results are saved here
tests/sample_data/     Small files used by automated tests
```

---

## Configuration

Key settings live in `config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONFIDENCE_THRESHOLD` | `0.8` | Minimum detection confidence (0.1 = more sensitive) |
| `DATA_INPUT` | `data/input/` | Where sample input files go |
| `DATA_OUTPUT` | `data/output/` | Where redacted output is saved |
| `SAMPLE_DATA` | `tests/sample_data/` | Small fixtures used by `pytest` |

The website confidence sliders override this value for CSV and text scanning.

## Limitations and responsible use

- PII detection is not guaranteed to find every identifier or avoid every false positive. Review every redacted file before sharing or relying on it.
- This project supports privacy workflows; it does not itself make data or an organization compliant with GDPR, HIPAA, FERPA, CCPA, or other laws.
- The server binds to `127.0.0.1`, so uploads stay on the computer running it. Do not expose the server to an untrusted network.
- Uploads are limited to 10 MB and are processed in memory. Use smaller files or split larger datasets.

## License

Released under the [MIT License](LICENSE).
