#!/usr/bin/env python3
"""Privacy Sandbox - local API server for the HTML design site (server.py)

Serves the static HTML design and exposes one API endpoint per track that the
site uses to run real local processing through the detector pipelines:

    POST /api/redact-image
        Request  : JSON
            { image: "<dataURL>", filename: "x.png",
              faces: true, ocr: true, faceStyle: "blur", textStyle: "black" }
            `image` is a dataURL produced by FileReader.readAsDataURL() in the
            browser. faceStyle / textStyle are one of: blur, pixelate, black.
        Response : JSON
            { ok: true, filename, dimensions, redacted: "<dataURL>",
              faces, textRegions, piiRegions, piiTypes }

    POST /api/process-csv
        Request  : JSON
            { content: "<dataURL>", filename: "x.csv",
              method: "smart"|"mask"|"partial", threshold: 0.8 }
        Response : JSON
            { ok: true, filename, method, threshold,
              metrics, headers, preview, before, after, findings,
              clean_csv: "<utf-8 text>", report: "<text>" }

    POST /api/process-text
        Request  : JSON
            { content: "<dataURL>", filename: "x.txt",
              strategies: ["mask","pseudonymize","partial_mask","label"],
              threshold: 0.8 }
        Response : JSON
            { ok: true, filename, file_size, total_pii_found,
              pii_by_type, pii_by_source, detection_details,
              versions: { strategy: { content, marked, redactions } } }

Run:
    python3 server.py          # -> http://localhost:8000
Then open http://localhost:8000 in your browser and use any track.

Privacy note: nothing leaves your machine. Files travel only from the browser
to this local process, which runs the detectors locally.
"""
import os
import io
import json
import base64
import tempfile
import mimetypes

import cv2
import numpy as np
import pandas as pd
import uvicorn

from image_detector import ImagePIIDetector
from pii_detector import PIIDetector
from text_pii_detector import TextPIIDetector

ROOT = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(ROOT, "design")
ASSETS = os.path.join(DESIGN, "assets")

# A single detector for the lifetime of the process (heavy to construct:
# Presidio + MediaPipe + Tesseract initialize once).
_detector = None
_csv_detector = None
_text_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = ImagePIIDetector()
    return _detector


def get_csv_detector(threshold=None):
    global _csv_detector
    if _csv_detector is None:
        _csv_detector = PIIDetector()
    if threshold is not None:
        _csv_detector.confidence_threshold = float(threshold)
    return _csv_detector


def get_text_detector(threshold=None):
    global _text_detector
    if _text_detector is None:
        _text_detector = TextPIIDetector()
    if threshold is not None:
        _text_detector.confidence_threshold = float(threshold)
    return _text_detector


FACE_METHODS = {
    "blur": "blur_faces",
    "pixelate": "pixelate_faces",
    "black": "black_box_faces",
}
TEXT_METHODS = {
    "blur": "blur_text",
    "pixelate": "pixelate_text",
    "black": "redact_text",
}

CORS = [
    (b"access-control-allow-origin", b"*"),
    (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
    (b"access-control-allow-headers", b"Content-Type"),
]

# The browser sends uploads as base64 JSON, which is larger than the original
# file. Keep both the decoded-file and request limits explicit.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 15 * 1024 * 1024


def _json_headers():
    return [(b"content-type", b"application/json"), *CORS]


async def _read_body(receive):
    body = b""
    more = True
    while more:
        msg = await receive()
        if msg["type"] == "http.request":
            chunk = msg.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                raise ValueError("Upload is too large. The maximum file size is 10 MB.")
            body += chunk
            more = msg.get("more_body", False)
    return body


def _decode_dataurl(data_url):
    if not isinstance(data_url, str):
        raise ValueError("Upload data must be a base64 string.")
    if isinstance(data_url, str) and data_url.startswith("data:"):
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    decoded = base64.b64decode(b64, validate=True)
    if len(decoded) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload is too large. The maximum file size is 10 MB.")
    return decoded


def _threshold(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Threshold must be a number between 0.1 and 1.0.") from error
    if not 0.1 <= value <= 1.0:
        raise ValueError("Threshold must be between 0.1 and 1.0.")
    return value


def _encode_png_dataurl(bgr_array):
    ok, buf = cv2.imencode(".png", bgr_array)
    if not ok or buf is None:
        raise ValueError("Failed to encode redacted image as PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


async def _redact(receive):
    try:
        raw = await _read_body(receive)
    except ValueError as e:
        return False, {"error": str(e)}, 413
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return False, {"error": "Request body must be JSON."}, 400
    if not isinstance(payload, dict) or "image" not in payload:
        return False, {"error": 'Missing "image" field.'}, 400

    try:
        img_bytes = _decode_dataurl(payload["image"])
    except Exception:
        return False, {"error": "Could not decode image data."}, 400

    face_on = bool(payload.get("faces", True))
    ocr_on = bool(payload.get("ocr", True))
    face_style = payload.get("faceStyle", "blur")
    text_style = payload.get("textStyle", "black") or "black"
    if face_style not in FACE_METHODS or text_style not in TEXT_METHODS:
        return False, {"error": "Unsupported image redaction style."}, 400
    filename = payload.get("filename", "upload.png")

    methods = []
    if face_on:
        methods.append(FACE_METHODS.get(face_style, "blur_faces"))
    if ocr_on:
        methods.append(TEXT_METHODS.get(text_style, "redact_text"))
    if not methods:
        return False, {"error": "Select face or text redaction."}, 400

    detector = get_detector()
    results = detector.process_image_bytes(img_bytes, methods, filename=filename)

    if results.get("status") != "success":
        return False, {"error": results.get("error", "Image processing failed.")}, 500

    red = results.get("redacted_image")
    if red is None:
        versions = results.get("redacted_versions") or {}
        red = list(versions.values())[-1] if versions else None
    if red is None:
        return False, {"error": "No redacted image produced."}, 500

    try:
        redacted_url = _encode_png_dataurl(red)
    except Exception as e:
        return False, {"error": str(e)}, 500

    types = sorted({t for r in results.get("pii_text_regions", []) for t in r.get("pii_types", [])})
    return True, {
        "ok": True,
        "filename": results.get("filename", filename),
        "dimensions": results.get("image_dimensions"),
        "redacted": redacted_url,
        "faces": results.get("faces_detected", 0),
        "textRegions": len(results.get("text_regions", [])),
        "piiRegions": len(results.get("pii_text_regions", [])),
        "piiTypes": types,
    }, 200


async def _send_json(send, status, obj):
    body = json.dumps(obj).encode("utf-8")
    await send({"type": "http.response.start", "status": status, "headers": _json_headers()})
    await send({"type": "http.response.body", "body": body})


async def _send_text(send, status, text, ctype=b"text/plain; charset=utf-8"):
    body = text.encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", ctype), *CORS]})
    await send({"type": "http.response.body", "body": body})


async def _serve_file(send, path, ctype=None):
    with open(path, "rb") as f:
        body = f.read()
    if ctype is None:
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", ctype.encode("utf-8")),
                            (b"cache-control", b"no-store"),
                            *CORS]})
    await send({"type": "http.response.body", "body": body})


def _entities_from(pii_detected):
    """Collect distinct PII entity types from a list of detection matches."""
    types = set()
    for m in pii_detected or []:
        t = getattr(m, "entity_type", None)
        if t:
            types.add(str(t))
    return sorted(types)


def _rows_to_lists(df, n=5):
    out = []
    for _, row in df.head(n).iterrows():
        out.append([str(v) for v in row.tolist()])
    return out


def _count_changed_rows(df_original, df_red):
    changed = 0
    for r in range(len(df_original)):
        for c in df_original.columns:
            if str(df_original.iloc[r][c]) != str(df_red.iloc[r][c]):
                changed += 1
                break
    return changed


def _build_csv_report(filename, df_original, df_red, pii_results, method, rows, cols):
    total = sum(len(i) for i in pii_results.values())
    changed = _count_changed_rows(df_original, df_red)
    lines = [
        "PRIVACY SANDBOX - CSV REDACTION REPORT",
        "=" * 42,
        "File: %s" % filename,
        "Method: %s" % method,
        "Rows: %s | Columns: %s" % (rows, cols),
        "",
        "SUMMARY",
        "-" * 42,
        "Total PII items found: %d" % total,
        "Affected columns: %d" % len(pii_results),
        "Clean columns: %d" % (cols - len(pii_results)),
        "Rows changed: %d" % changed,
        "",
        "FINDINGS BY COLUMN",
    ]
    if pii_results:
        for col, items in pii_results.items():
            types_in_col = _entities_from(
                [t for item in items for t in (item.get("pii_detected") or [])]
            )
            lines.append("  %s: %d item(s) [%s]" % (col, len(items), ", ".join(types_in_col) or "REGEX_MATCH"))
    else:
        lines.append("  No PII detected - file was already clean.")
    lines.append("")
    lines.append(
        "DETECTION STATUS: %s"
        % ("PII DETECTED — REVIEW REDACTED OUTPUT" if pii_results else "NO PII DETECTED — REVIEW OUTPUT")
    )
    return "\n".join(lines)


def mark_redacted_html(content, redactions):
    """Wrap each replaced segment of the redacted text in <mark> tags."""
    if not redactions:
        return content
    placed = []
    offset = 0
    for r in sorted(redactions, key=lambda x: x["position"][0]):
        s, e = r["position"]
        repl = r.get("replacement", "****")
        fs = s + offset
        fe = fs + len(repl)
        offset += len(repl) - (e - s)
        placed.append((fs, fe))
    out = []
    last = 0
    for s, e in placed:
        out.append(content[last:s])
        out.append("<mark>")
        out.append(content[s:e])
        out.append("</mark>")
        last = e
    out.append(content[last:])
    return "".join(out)


async def _process_csv(receive):
    try:
        raw = await _read_body(receive)
    except ValueError as e:
        return False, {"error": str(e)}, 413
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return False, {"error": "Request body must be JSON."}, 400
    if not isinstance(payload, dict) or "content" not in payload:
        return False, {"error": 'Missing "content" field.'}, 400

    filename = payload.get("filename", "upload.csv")
    method = payload.get("method", "smart")
    try:
        threshold = _threshold(payload.get("threshold"))
    except ValueError as e:
        return False, {"error": str(e)}, 400
    if method not in {"smart", "mask", "partial"}:
        return False, {"error": "Unsupported CSV redaction method."}, 400

    try:
        csv_bytes = _decode_dataurl(payload["content"])
    except Exception:
        return False, {"error": "Could not decode CSV data."}, 400
    if not csv_bytes:
        return False, {"error": "Empty file."}, 400

    try:
        df_original = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, encoding="utf-8-sig")
    except Exception as e:
        return False, {"error": "Could not parse CSV: %s" % e}, 400

    df_original = df_original.fillna("").astype(str)
    detector = get_csv_detector(threshold)

    if method == "smart":
        df_red, pii_results = detector.smart_detect_and_redact(df_original)
    else:
        pii_results = detector.detect_pii_in_dataframe(df_original)
        df_red = detector.redact_dataframe(df_original, pii_results, method=method)

    rows = int(len(df_original))
    cols = int(len(df_original.columns))
    total_pii = sum(len(items) for items in pii_results.values())
    affected = len(pii_results)

    headers = [str(h) for h in df_original.columns]
    action = {
        "smart": "Redacted - IDs preserved",
        "mask": "Masked (****)",
        "partial": "Partially masked",
    }.get(method, "Redacted")

    findings = []
    for column, items in pii_results.items():
        types_in_col = set()
        for item in items:
            for t in _entities_from(item.get("pii_detected") or []):
                types_in_col.add(t)
        findings.append({
            "column": str(column),
            "types": sorted(types_in_col) or ["REGEX_MATCH"],
            "items": len(items),
            "action": action,
        })
    findings.sort(key=lambda f: -f["items"])

    clean_csv = df_red.to_csv(index=False)
    report = _build_csv_report(filename, df_original, df_red, pii_results,
                               method, rows, cols)

    return True, {
        "ok": True,
        "filename": filename,
        "method": method,
        "threshold": detector.confidence_threshold,
        "metrics": {
            "rows": rows, "cols": cols, "total_pii": total_pii,
            "affected": affected, "clean": cols - affected,
            "changed": _count_changed_rows(df_original, df_red),
        },
        "headers": headers,
        "preview": _rows_to_lists(df_original, 5),
        "before": _rows_to_lists(df_original, 5),
        "after": _rows_to_lists(df_red, 5),
        "findings": findings,
        "clean_csv": clean_csv,
        "report": report,
    }, 200


async def _process_text(receive):
    try:
        raw = await _read_body(receive)
    except ValueError as e:
        return False, {"error": str(e)}, 413
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return False, {"error": "Request body must be JSON."}, 400
    if not isinstance(payload, dict) or "content" not in payload:
        return False, {"error": 'Missing "content" field.'}, 400

    filename = payload.get("filename", "document.txt")
    strategies = payload.get("strategies") or ["mask", "pseudonymize"]
    if not isinstance(strategies, list) or not all(
        strategy in {"mask", "pseudonymize", "partial_mask", "label"}
        for strategy in strategies
    ):
        return False, {"error": "Unsupported text redaction strategy."}, 400
    try:
        threshold = _threshold(payload.get("threshold"))
    except ValueError as e:
        return False, {"error": str(e)}, 400

    try:
        data = _decode_dataurl(payload["content"])
    except Exception:
        return False, {"error": "Could not decode text data."}, 400
    if not data:
        return False, {"error": "Empty file."}, 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".txt", ".json", ".md"):
        ext = ".txt"

    detector = get_text_detector(threshold)
    tmp = tempfile.NamedTemporaryFile("wb", suffix=ext, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        analysis = detector.process_text_file(tmp.name, redaction_strategies=strategies)
    except Exception as e:
        return False, {"error": str(e)}, 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    if not analysis:
        return False, {"error": "Text processing failed."}, 500

    # process_text_file names the report after the temp file path; override it.
    analysis["filename"] = filename

    versions = {}
    for strat, v in analysis["redacted_versions"].items():
        redactions = (v.get("redaction_info") or {}).get("redactions") or []
        versions[strat] = {
            "content": v["content"],
            "marked": mark_redacted_html(v["content"], redactions),
            "redactions": [{
                "original": r["original"],
                "replacement": r["replacement"],
                "entity_type": r["entity_type"],
                "confidence": r.get("confidence", 0.0),
                "source": r.get("source", "unknown"),
            } for r in redactions],
        }

    return True, {
        "ok": True,
        "filename": analysis["filename"],
        "file_size": analysis["file_size"],
        "total_pii_found": analysis["total_pii_found"],
        "pii_by_type": analysis["pii_by_type"],
        "pii_by_source": analysis["pii_by_source"],
        "detection_details": analysis["detection_details"],
        "versions": versions,
    }, 200


async def app(scope, receive, send):
    if scope["type"] != "http":
        if scope["type"] == "websocket":
            await send({"type": "websocket.close"})
        return

    method = scope.get("method", "GET")
    path = scope.get("path", "/")

    # CORS pre-flight
    if method == "OPTIONS":
        await send({"type": "http.response.start", "status": 204, "headers": CORS})
        await send({"type": "http.response.body", "body": b""})
        return

    if path in ("/", "/index.html") and method == "GET":
        idx = os.path.join(DESIGN, "index.html")
        if os.path.isfile(idx):
            await _serve_file(send, idx, "text/html; charset=utf-8")
        else:
            await _send_text(send, 500, "Design site missing at design/index.html")
        return

    if path.startswith("/assets/") and method == "GET":
        rel = path[len("/assets/"):].replace("\\", "/")
        rel = "/".join(p for p in rel.split("/") if p not in ("", ".", ".."))
        if not rel:
            await _send_text(send, 404, "Not found")
            return
        fp = os.path.join(ASSETS, rel)
        if os.path.isfile(fp) and os.path.realpath(fp).startswith(os.path.realpath(ASSETS)):
            await _serve_file(send, fp)
        else:
            await _send_text(send, 404, "Asset not found")
        return

    if path == "/api/redact-image" and method == "POST":
        _, payload, status = await _redact(receive)
        await _send_json(send, status, payload)
        return

    if path == "/api/process-csv" and method == "POST":
        _, payload, status = await _process_csv(receive)
        await _send_json(send, status, payload)
        return

    if path == "/api/process-text" and method == "POST":
        _, payload, status = await _process_text(receive)
        await _send_json(send, status, payload)
        return

    await _send_text(send, 404, "Not found")


if __name__ == "__main__":
    print("Privacy Sandbox local server  ->  http://localhost:8000")
    print("Serves the design site + real Python processing for all tracks:")
    print("  /api/redact-image  (OpenCV/Tesseract)  /api/process-csv  /api/process-text")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
