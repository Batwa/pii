# Design Prompt — Website for "Privacy Sandbox: PII Identifier"

**Prompt to paste into a design-making AI:**

---

Create a website for a web application called **Privacy Sandbox (PII Identifier)**. It is a privacy tool that automatically detects and redacts Personally Identifiable Information (PII) from datasets and documents so users can share and analyze data safely.

## Project Purpose & Audience

The tool serves researchers, students, developers, and business users who work with real-world data containing sensitive personal information. It helps them comply with privacy regulations (GDPR, FERPA, HIPAA, CCPA) and safely share or publish data (for AI/ML training, collaboration, publishing, and analysis) without exposing personal details.

## Core Capabilities (what the product does)

- Automatically scans files for: names, email addresses, phone numbers, credit card numbers, Social Security numbers, home addresses, dates of birth, IP addresses, and other personal identifiers.
- Applies one or more redaction strategies to hide that information:
  - Masking (replace with placeholder characters such as `****`)
  - Pseudonymization (replace real names with consistent fake names)
  - Partial masking (hide only part of the value)
  - Labeled redaction
  - For images: blur, pixelation, or solid black box covering
- Produces downloadable cleaned files plus a human-readable privacy/compliance report (summary of what was found and redacted, organized by type and by column/section).
- Handles three categories of files: CSV / tabular data, text documents, and images.
- Detection is powered by multiple engines: Microsoft Presidio (NLP-based detection), spaCy named-entity recognition, custom regex patterns, and OCR (for text inside images). Face detection in images uses a dual-model MediaPipe detector with an automatic Haar-cascade fallback.
- All processing happens locally — files are not sent to external servers and are not permanently stored.

## Website Structure & Flow (must follow this exactly)

### Page 1 — Welcome / Landing Page
- A short, clear explanation of what the tool does and why it matters (detect and redact PII before sharing or publishing data).
- A concise "how it works" explanation with a few simple steps, for example: choose your file type, upload your file, the tool scans and redacts PII, then you download the clean file and a report.
- List the key benefits prominently, for example:
  - Legal compliance with privacy regulations (GDPR, FERPA, HIPAA, CCPA).
  - Safe collaboration and data sharing without privacy risk.
  - Clean, "ML-ready" datasets for training models.
  - Time savings versus manual review.
- Mention the supported file types (CSV/tabular data, text documents, and image files).
- Include a single prominent **Start** button. Clicking it takes the user to the file-type choice (next page).

### Page 2 — File Type Selection
- Present a choice of three processing tracks: **CSV / Tabular Data**, **Text Documents**, and **Images**.
- Selecting a category reveals that category's upload and processing interface (described below).


### Page 3 — Processing & Results (per file type)

**CSV / Tabular Data track:**
- Upload a `.csv` file.
- After upload, show key file stats (row count, column count, file size) and a preview of the original data.
- Let the user choose a redaction method:
  - Smart Redaction (intelligent, preserves database/generated identifiers such as customer/employee/product IDs while redacting real PII),
  - Complete Masking (mask all detected PII),
  - Partial Redaction (hide only part of each value).
- Provide an adjustable detection sensitivity (confidence threshold) — lower means more sensitive.
- After scanning, show: which columns were affected, how many PII items were found per column, and a before/after comparison of the data.
- Provide downloads: the cleaned CSV file and a privacy/compliance report.

**Text Documents track:**
- Upload one or more `.txt`, `.json`, or `.md` files (multiple files allowed).
- Let the user pick one or more redaction strategies to apply and compare (mask, pseudonymize, partial mask, label).
- After scanning, show per-file results: file size, total PII found, PII types found, breakdown by detection source (NLP engines vs. regex vs. other), and tabs for each redaction strategy with a preview of the redacted text and a details list (what was replaced with what).
- Provide a download button for each redacted version.

**Images track:**
- Upload one or more image files (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- Let the user toggle which detection methods to run: face detection and text-PII detection (via OCR) — at least one must be enabled.
- For each method, let the user choose the redaction style: blur, pixelate, or solid black box.
- After scanning, show before/after results for each image and provide downloads of the redacted images.

### Supporting Elements (should exist somewhere in the site)
- A settings control for detection sensitivity (confidence threshold) on the CSV and text tracks (not needed for images).
- An "About" section describing: the project mission, the technology behind it, privacy/compliance benefits, the data-security guarantees (local processing, no permanent storage), and the current feature list per file type.

## Content That Must Appear (do not omit)

1. The product name and a one-to-two-sentence value proposition on the welcome page.
2. A short "how it works" explanation.
3. A list of the main benefits (compliance, safe sharing, ML-ready data, time savings).
4. A clear **Start** button on the welcome page leading to file-type selection.
5. The three file-type tracks and the full feature set described above for each.
6. Messaging that all processing is local and private (files are not uploaded to external servers).
7. The supported file extensions per track (CSV; txt/json/md; jpg/jpeg/png/bmp).

## Constraints

- Design must be **interesting, in dark tones (colors: Hex - 00072d, 051650, 0a2472, 123499, 5eaf73), and trustworthy** — the product is about privacy and security, so the overall feel should communicate reliability and boldness. Achieve this through layout, hierarchy, clarity of copy, and confidence-inspiring microcopy rather than decoration.
- Make the site **responsive** and **accessible** (readable text contrast, keyboard-navigable controls, clear focus states).
- Keep the user flow simple: welcome → choose file type → upload → scan → view results → download. Every step should be immediately understandable with minimal instruction.

## Deliverable

Produce a complete, modern web page design for this tool: layout structure, wireframe-level page composition, copy that should appear on each screen, user-flow annotations for the Start button and upload/processing/results steps, and the component list (buttons, upload areas, file-type selector, settings panel, results sections, download controls). No production code is required — design only.
