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
