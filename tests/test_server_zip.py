"""Tests for downloading all text redaction outputs as one ZIP archive."""
import asyncio
import base64
import io
import json
import zipfile

from server import _create_zip


def test_create_zip_contains_each_requested_text_file():
    payload = {
        "filename": "results.zip",
        "files": [
            {"name": "note_mask.txt", "content": "data:text/plain;base64,SGVsbG8="},
            {"name": "note_label.txt", "content": "data:text/plain;charset=utf-8,Hello%20%E2%9C%93"},
        ],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    ok, response, status = asyncio.run(_create_zip(receive))

    assert ok is True
    assert status == 200
    archive = base64.b64decode(response["archive"].split(",", 1)[1])
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        assert zipped.namelist() == ["note_mask.txt", "note_label.txt"]
        assert zipped.read("note_mask.txt") == b"Hello"
        assert zipped.read("note_label.txt") == "Hello ✓".encode("utf-8")


def _call(handler, payload):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    return asyncio.run(handler(receive))


def test_apply_redactions_uses_exactly_the_reviewed_list():
    from server import _apply_redactions

    # Offsets count code points: the emoji is one character for the server.
    text = "🏡 Rose Hartley grows a rose. Ring Rose on 01632 960 111.\n"
    first, second = text.index("Rose Hartley"), text.index("Rose on")
    phone = text.index("01632")
    payload = {
        "text": text,
        "strategies": ["mask", "label"],
        "redactions": [
            {"start": phone, "end": phone + 13, "entity_type": "PHONE", "score": 1.0, "source": "regex"},
            {"start": first, "end": first + 12, "entity_type": "PERSON", "score": 0.9, "source": "spacy"},
            # added by hand, with a type the server does not know
            {"start": second, "end": second + 4, "entity_type": "<b>", "source": "somewhere"},
        ],
    }

    ok, response, status = _call(_apply_redactions, payload)

    assert ok is True and status == 200
    assert response["total_redactions"] == 3
    assert response["versions"]["mask"]["content"] == "🏡 **** grows a rose. Ring **** on ****.\n"
    assert response["versions"]["label"]["content"] == \
        "🏡 [PERSON] grows a rose. Ring [REDACTED] on [PHONE].\n"
    assert [r["original"] for r in response["versions"]["mask"]["redactions"]] == \
        ["Rose Hartley", "Rose", "01632 960 111"]
    assert response["versions"]["mask"]["redactions"][1]["source"] == "manual"


def test_apply_redactions_merges_overlaps_and_rejects_bad_offsets():
    from server import _apply_redactions, clean_review_redactions

    text = "Call Ann Marie Poole today."
    merged = clean_review_redactions(text, [
        {"start": 9, "end": 20, "entity_type": "PERSON"},
        {"start": 5, "end": 14, "entity_type": "PERSON"},
    ])
    assert [(r["start"], r["end"], r["text"]) for r in merged] == [(5, 20, "Ann Marie Poole")]

    for bad in ([{"start": 5, "end": 99}], [{"start": -1, "end": 3}], [{"start": 4, "end": 4}],
                [{"start": "5", "end": 8}], [{"start": True, "end": 8}], "nothing", [7]):
        ok, response, status = _call(_apply_redactions, {"text": text, "redactions": bad})
        assert ok is False and status == 400, bad

    ok, response, status = _call(_apply_redactions, {"text": text, "redactions": []})
    assert ok is True and response["versions"]["mask"]["content"] == text

    ok, response, status = _call(_apply_redactions, {"redactions": []})
    assert ok is False and status == 400


def test_preview_escapes_markup_in_the_document():
    from server import mark_redacted_html

    content = "<img src=x onerror=alert(1)> **** & co"
    marked = mark_redacted_html(content, [{"position": (29, 37), "replacement": "****"}])
    assert marked == "&lt;img src=x onerror=alert(1)&gt; <mark>****</mark> &amp; co"
    assert mark_redacted_html("<b>plain</b>", []) == "&lt;b&gt;plain&lt;/b&gt;"
