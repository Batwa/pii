# Privacy Sandbox (PII Identifier) — Website Design Document

A working, clickable high-fidelity prototype of this design ships alongside this document at
**`design/index.html`** (open it in any browser — no build step needed). Press the **"Show design
annotations"** toggle in the footer to layer user-flow notes onto the screens.

| File | Purpose |
|---|---|
| `index.html` | The entire multi-page design as a single HTML shell (4 pages, SPA-style routing) |
| `assets/styles.css` | Dark theme, palette, layout, responsive & accessibility styles |
| `assets/app.js` | View routing, stepper state, simulated upload/scan/results, tabs, validation |
| `cdp-test.js` / `cdp-audit.js` / `verify.py` | Automated smoke tests used to validate the prototype |
| `README.md` (root) | Existing project readme for the underlying Streamlit app |

---

## 1. Design goals & visual direction

**Tone:** trustworthy, bold, technical-but-approachable. The product is about privacy and security,
so the interface communicates reliability through structure, hierarchy and confident microcopy —
not decoration.

- **Dark, deep-navy canvas** built from the mandated palette:

| Role | Hex | Usage |
|---|---|---|
| `#00072d` | `--abyss` | Page background (with faint navy/green radial glows) |
| `#051650` | `--deep` | Panel / card surfaces, stepper dots |
| `#0a2472` | `--navy` | Secondary surfaces, table headers, hover fills |
| `#123499` | `--brand` | Cards under the cursor, secondary focus fills |
| `#5eaf73` | `--green` | **The single action color**: primary CTAs, success states, redaction output |

- **Hierarchy:** one page per step, a persistent 5-dot stepper during the working flow, and exactly
  **one primary button per screen** (green). Everything else is a ghost/outline button.
- **Typography:** Inter (falling back to system UI); `JetBrains Mono` for anything that *is* data —
  sample rows, PII values, file names, extension lists. Mono-in-data makes the "change" visible.
- **Trust cues repeated everywhere:** "🔒 Local processing", "Nothing leaves this machine",
  "Runs locally · nothing uploaded". Confident, repeated microcopy.

### Colour contrast (WCAG AA)

- Body text `#a9bbe8` on `#00072d` → **≈ 8.9 : 1** (AAA)
- Muted labels `#8296c9` on `#00072d` → **≈ 7.5 : 1**
- Primary CTA text `#00210c` on the green gradient `#5eaf73`→`#3f8d55` → **≥ 7 : 1**
- Green-accent type `#5eaf73` on `#051650` → **≈ 4.6 : 1** (labels/eyebrows only)

### Accessibility

- Skip-to-content link; every control is a real `<button>`/`<label>`/`<input>` (keyboard operable);
  `:focus-visible` shows a 3px green ring; focus moves to revealed panels after routing;
  tabs support arrow-key navigation; illustrative SVGs carry `role="img"` + `aria-label`;
  `prefers-reduced-motion` disables transitions; layout collapses to one column on mobile.

---

## 2. Page map & user flow

```
Welcome (Page 1)
   │  [ Start ]
   ▼
File-type selection (Page 2) ────────────────┐
   │ choose one of 3 tracks                  │ switch track (state preserved)
   ▼                                         │
Upload & processing (Page 3, per track)      │
   CSV  ·  Text  ·  Images                   │
   │  upload → (file stats + options)        │
   ▼                                         │
Scan  →  progress  →  Results & downloads ───┘
   │
   ▼
About (supporting page, from nav or footer)
```

Every working screen carries the stepper **Welcome → Choose file type → Upload → Scan →
---

## 3. Page 1 — Welcome / Landing

**Purpose:** explain the tool, prove it matters, and route the user forward with a single action.

### Layout (top to bottom)

1. **Sticky header** — brand mark (shield + underscore lock) + "Privacy Sandbox / PII Identifier",
   nav (Welcome · Choose file type · About), and a compact green **Start** pill.
2. **Hero (two columns on desktop → stacked on mobile)**
   - Left: eyebrow "Local-first privacy tooling"; H1 *"Find every piece of personal data.
     Redact it before you share."*; one-paragraph value proposition; two buttons —
     green **Start →** (the only primary CTA on the page; ghost copy "How it works" scrolls to the
     steps); microcopy under the buttons: *"No account. No upload to a server. Your files never
     leave this machine."*
   - Right: a **live redaction sample card** styled as a terminal/file preview: three rows of raw data
     (`john.doe@email.com`, `555-123-4567`… in red/mono), the divider "↓ redacting locally ↓", and the
     same rows after redaction in green/mono (`****@****.com`, `***-***-4567`), plus one preserved
     `customer_id` with the note "kept — not personal". This visually seeds the Smart-Redaction story.
3. **"What it looks for"** — one panel of category chips: names, emails, phones, credit cards, SSNs,
   addresses, dates of birth, IPs, faces, "+ other identifiers". Corner badge: 🔒 *Runs on your device*.
4. **"How it works"** — 4 numbered step cards:
   1. **Choose your file type** — "pick CSV/tabular data, text documents or images"
   2. **Upload your file** — "drag & drop, or browse; multiple files for text and images"
   3. **We scan and redact** — "Presidio, spaCy, custom regex and OCR find the PII; your strategy hides it"
   4. **Download clean results** — "redacted file plus a plain-language privacy report"
5. **Benefits** — four cards: **Legal compliance** (GDPR, FERPA, HIPAA, CCPA) · **Safe collaboration** ·
   **ML-ready data** · **Time savings**.
6. **Supported file types** — three cards listing extensions: `csv` · `txt/json/md` · `jpg/jpeg/png/bmp`.
7. **Local-processing promise + final CTA** — centered panel *"Your files never leave your machine."*
   with a closing **Start →** button and "Next step: choose your file type."

**Flow annotation (Start):** the green Start button is wired to `data-goto="select"` → activates
Page 2 and sets the stepper to "Choose file type".
Results & download**; each step highlights as the user advances, so position in the flow is never in doubt.
---

## 4. Pages 2 & 3 — File-type selection + processing (per track)

### Page 2 — selection

Progress stepper now shows *Welcome* complete and **Choose file type** current. H1: *"What are you
cleaning today?"* Three large selectable cards — **CSV / Tabular data**, **Text documents**, **Images** —
each with icon, one-line description, a 3-item feature list, a mono extension line, and a
"Open … track →" affordance. Selecting a card:

- sets `aria-pressed` on that card (green border highlight),
- advances the stepper to **Upload**,
- reveals that track's upload/processing panel immediately below,
- moves keyboard focus to the revealed panel's heading.

A green callout beneath repeats the local-processing guarantee. Copy explains "you can switch tracks
at any time without losing your place" (track state is preserved per instance).

Below the selector, three **Page 3 sections** live (only the chosen one is shown at a time). Each uses
a **two-column work layout**: a sticky **settings sidebar** (left, collapses above the panel on small
screens) and the main **upload → options → scan → results** column (right).

### CSV / Tabular data track

- **Settings sidebar:** *Detection sensitivity* slider (0.1–1.0, default 0.8, "Lower = more sensitive"),
  live value readout, scale hints "0.1 more sensitive / 1.0 stricter", and help text stating the
  threshold applies to CSV and text scanning only.
- **Upload:** single-file dropzone (`.csv`).
- **After upload:** metric strip — **Rows** (1,248) · **Columns** (5) · **File size** (86 KB) · **File**
  (`customers.csv`) — then an **Original data preview** table (raw PII tinted red), then the method choice:

  | Method | Behavior |
  |---|---|
  | **Smart Redaction** (default, "recommended") | Redacts real PII; preserves generated IDs (customer/employee/product) so records stay joinable |
  | **Complete Masking** | Every detected value becomes `****` |
  | **Partial Redaction** | Hides only one part: `***-***-4567` |

- **Scan button** row shows current threshold: "🔍 Scan CSV for PII · Threshold 0.8 · runs locally".
- **Scanning state:** animated progress bar with a column-scan caption.
- **Results:** "✓ Scan complete"; metric strip — **PII items found** (3,104) · **Affected columns** (4) ·
  **Clean columns** (1) · **Rows changed** (1,248); an amber callout summarising the findings and noting
  `customer_id` was preserved; a **Findings by column** table (column / PII type / items / action taken);
  a **Before & after comparison** of two mini-tables (raw red values left → green redacted right);
  finally the **downloads**:
  - **Download clean CSV** → `clean_customers.csv`
  - **Download privacy report** → `privacy_report_customers.txt` (totals, per-column detail, method,
    confidence threshold — plain language, compliance-file ready)
### Text documents track

- **Settings sidebar:** sensitivity slider (same as CSV) + engine chips: Presidio · spaCy NER ·
  custom regex.
- **Upload:** multi-file dropzone (`.txt` `.json` `.md`); a file list shows names + sizes.
- **Strategies:** multi-select cards (Mask · Pseudonymize · Partial mask · Label) with micro-descriptions;
  **"select at least one"** inline warning if emptied and Scan is pressed (also shown live on change).

  | Strategy | Example |
  |---|---|
  | Mask | `John Doe` → `****` |
  | Pseudonymize | `John Doe` → `Michael Bennett` (consistent mapping, name → keep readable) |
  | Partial mask | `j***@e***.com` |
  | Label | `[PERSON]` / `[EMAIL_ADDRESS]` |

- **Results (per file):** heading `📄 email.txt`; metrics — **file size · total PII · PII types ·
  redacted versions**; two breakdown cards — **by type** (PERSON 4, EMAIL_ADDRESS 3, …) and
  **by detection source** (Presidio NLP 8, spaCy NER 3, custom regex 3); a **strategy tab bar**
  (arrow-key navigable), each tab showing:
  - a mono **redacted text preview** with hit-highlighting,
  - **redaction details** ("what replaced what") as a `type: from → to` list,
  - a **per-strategy download** button (`email_mask.txt`, …).
  Results repeat as one block per uploaded file.

### Images track

- **Settings sidebar:** no confidence slider — instead, engine chips: MediaPipe (dual model) ·
  Haar cascade fallback · Tesseract OCR · Presidio + regex, with the fallback explanation.
- **Upload:** multi-file dropzone (`.jpg` `.jpeg` `.png` `.bmp`); file list with sizes.
- **Methods & styles (at least one method):**
  - checkbox **👤 Detect faces** + style select: **Blur** (default) / **Pixelate** / **Solid black box**;
  - checkbox **📝 Detect text PII (OCR)** + style select: **Solid black box** (default) / **Blur** /
    **Pixelate**.
  - Warning callout if both are switched off.
- **Results (per image):** metrics — **faces detected · text regions · PII text regions**; a
  **before/after side-by-side** image pair (in the prototype, an SVG badge illustration showing the
  face blurred via `feGaussianBlur` and name/email/phone/SSN covered by black boxes, with the employee
  ID left visible); an **OCR findings list** (region text → type → style → confidence); and one
  **download per redaction version** (`*-blur_faces.png`, `*-redact_text.png`).
---

## 5. Maintaining the flow ("user-flow annotations")

Every prototype screen includes a green, collapsible `annotation` block describing exactly what happens
at that step — for example, next to the Start button: *"routes the user to Page 2 and advances the
stepper to step 1 of 4"*. A footer checkbox toggles all annotations on/off for review or client
presentations.

---

## 6. About page

Panels in sequence: **Project mission** · **Technology behind it** (detection chips incl. the
dual-model MediaPipe + Haar fallback, Tesseract OCR, processing stack pandas/NumPy/OpenCV/Pillow) ·
**Privacy & compliance** (GDPR, FERPA, HIPAA, CCPA chips) · **Data security guarantees**
(local processing · no permanent storage · multiple strategies · audit trail) · **Current features**
(three feature cards, one per file type, listing each track's capabilities). Closing **Start →** CTA.

---

## 7. Component inventory

**Buttons** — `.btn--primary` (green gradient, dark text, pill), `.btn--ghost`, `.btn--lg`,
`.btn--sm`, `.btn--block`; `.download` cards (icon + filename); `:focus-visible` rings everywhere.
**Upload areas** — `.dropzone` (dashed border, drag-highlight state) configurable per track for
single/multi files; `.filelist` chips with name + size.
**File-type selector** — `.track` cards (buttons) with `aria-pressed` state.
**Settings panel** — sticky `.settings` aside: range input with `.settings__value` readout and scale
hints, engine chips for text/images, threshold note.
**Stepper** — `.stepper` ordered list (done · current · upcoming).
**Results sections** — `.metrics` strip, `.table-wrap` data tables (raw = red mono, redacted = green
mono), `.compare` before/after image pairs, `.tabs` strategy bar with `.kv` redaction-detail lists,
`.preview` mono text with `<mark>` hits.
**Download controls** — `.downloads` grid of cards: icon + primary label + filename/size meta.
**Callouts** — `.callout--ok` (green, local-processing guarantees) and `.callout--warn`
(amber, findings summaries / validation), plus `.annotation` design notes.

---

## 8. Validation

The prototype was verified in headless Chrome (CDP) at 1440 px and 390 px widths:

- **Interaction flow (22 checks, all passing):** Start routes to selection; each track reveals/hides
  correctly with focus moved to the revealed heading; csv/text/image upload states appear; scan shows
  progress then results; strategy/detection validation warnings appear when nothing is selected; tabs
  switch panels; confidence slider readouts stay in sync; download feedback appears; About opens.
- **Layout & rendering (18 checks, all passing)** at desktop *and* mobile: no horizontal overflow,
  dark canvas, responsive H1, 2-column → 1-column hero, green gradient primary button with dark text,
  sticky header, light-on-dark body text, no heading wrap overflow.

---

## 9. Implementation notes

- The live product is a Streamlit app (`app.py`) with the same three tracks, strategies, thresholds
  and result shapes; this design is a faithful visual blueprint for restyling or rebuilding it.
- Hit-highlighting, OCR boxes and redaction previews would be fed by the same data the backend already
  produces (`pii_results`, `redacted_versions`, `pii_by_source`, `text_regions`).