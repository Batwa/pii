"""Tests for reading PDF and Word files into plain text for the text track."""
import asyncio
import base64
import json
import zipfile

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from document_text import DocumentTextError, docx_text, extract_document_text, pdf_text

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def make_pdf(path, pages):
    """Write a PDF whose pages hold the given (x, y, text) lines in Helvetica."""
    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=595, height=842)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        ops = ["BT", "/F1 11 Tf"]
        for x, y, text in lines:
            safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"1 0 0 1 {x} {y} Tm ({safe}) Tj")
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("cp1252"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with open(path, "wb") as handle:
        writer.write(handle)


def p(*runs):
    return "<w:p>" + "".join(runs) + "</w:p>"


def r(text):
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


def table(rows):
    return "<w:tbl>" + "".join(
        "<w:tr>" + "".join(f"<w:tc>{p(r(cell))}</w:tc>" for cell in row) + "</w:tr>"
        for row in rows) + "</w:tbl>"


def part(root, body, prolog=""):
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>{prolog}'
            f'<w:{root} xmlns:w="{W}" xmlns:mc="{MC}">{body}</w:{root}>')


def make_docx(path, body, header=None, footer=None, footnotes=None, prolog=""):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", part("document", f"<w:body>{body}</w:body>", prolog))
        if header:
            archive.writestr("word/header1.xml", part("hdr", header))
        if footer:
            archive.writestr("word/footer1.xml", part("ftr", footer))
        if footnotes:
            archive.writestr("word/footnotes.xml", part("footnotes", footnotes))


LETTER = [[(72, 770, "Fernbank Lettings"), (72, 740, "Tenant: Odalys Pemberton"),
           (72, 724, "Date of birth: 4 March 1988"),
           (72, 690, "Name"), (220, 690, "Phone"), (360, 690, "Postcode"),
           (72, 674, "Ilse Varga"), (220, 674, "07700 900 444"), (360, 674, "HG4 9ZQ"),
           (72, 640, "She lives at 14 Quarry Lane, Ripon (ground floor).")],
          [(72, 770, "Page two: ring Odalys on 01632 960 222.")]]


def test_pdf_text_keeps_lines_pages_and_column_gaps(tmp_path):
    path = tmp_path / "letter.pdf"
    make_pdf(path, LETTER)

    lines = pdf_text(str(path)).split("\n")

    assert lines[0] == "Fernbank Lettings"
    assert "Tenant: Odalys Pemberton" in lines
    assert "She lives at 14 Quarry Lane, Ripon (ground floor)." in lines
    # the cells of a table row stay apart, and the second page follows the first
    row = next(line for line in lines if line.startswith("Ilse Varga"))
    assert row.split("  ")[0] == "Ilse Varga" and "07700 900 444" in row and row.endswith("HG4 9ZQ")
    assert lines[-2] == "Page two: ring Odalys on 01632 960 222." and lines[-1] == ""


def test_pdf_without_a_text_layer_and_broken_files_give_a_clear_message(tmp_path):
    scan = tmp_path / "scan.pdf"
    make_pdf(scan, [[]])
    with pytest.raises(DocumentTextError, match="scanned images"):
        pdf_text(str(scan))

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentTextError, match="could not be read as a PDF"):
        pdf_text(str(broken))

    old = tmp_path / "letter.doc"
    old.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(DocumentTextError, match="Save the document as .docx"):
        extract_document_text(str(old))


def test_docx_text_reads_what_is_on_the_page(tmp_path):
    textbox = "<w:txbxContent>" + p(r("Box: ring Ilse")) + "</w:txbxContent>"
    body = (
        p(r("Tenancy reference for "), r("Odalys Pemberton"))
        + p(r("Born"), "<w:r><w:tab/></w:r>", r("4 March 1988"), "<w:r><w:br/></w:r>", r("in Ripon"))
        # a tracked deletion is not on the page; an insertion is
        + p("<w:del><w:r><w:delText>Old name: Odalys Crane</w:delText></w:r></w:del>",
            "<w:ins><w:r><w:t>Phone 07700 900 444</w:t></w:r></w:ins>")
        + table([["Name", "Phone", "Postcode"], ["Ilse Varga", "07700 900 555", "HG4 9ZQ"]])
        # a text box is stored twice, for new and for old readers
        + p("<w:r><mc:AlternateContent><mc:Choice><w:drawing>" + textbox + "</w:drawing></mc:Choice>"
            "<mc:Fallback><w:pict>" + textbox + "</w:pict></mc:Fallback></mc:AlternateContent></w:r>")
        + p("<w:r><w:instrText> MERGEFIELD secret </w:instrText></w:r>", r("Yours, Fernbank Lettings")))
    path = tmp_path / "reference.docx"
    make_docx(path, body, header=p(r("Fernbank Lettings · 2 Mill Yard, Ripon")), footer=p(r("Page 1")),
              footnotes="<w:footnote>" + p(r("Guarantor: Bram Okafor")) + "</w:footnote>")

    text = docx_text(str(path))

    assert text == (
        "Fernbank Lettings · 2 Mill Yard, Ripon\n\n"
        "Tenancy reference for Odalys Pemberton\nBorn\t4 March 1988\nin Ripon\n"
        "Phone 07700 900 444\n"
        "Name\tPhone\tPostcode\nIlse Varga\t07700 900 555\tHG4 9ZQ\n\n\n"
        "Box: ring Ilse\n\nYours, Fernbank Lettings\n\n"
        "Guarantor: Bram Okafor\n\nPage 1\n")


def test_docx_that_is_not_a_document_is_refused(tmp_path):
    not_word = tmp_path / "sheet.docx"
    with zipfile.ZipFile(not_word, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
    with pytest.raises(DocumentTextError, match="not a Word document"):
        docx_text(str(not_word))

    not_zip = tmp_path / "plain.docx"
    not_zip.write_bytes(b"just text")
    with pytest.raises(DocumentTextError, match="could not be read as a Word document"):
        docx_text(str(not_zip))

    # an XML part that declares entities is not parsed at all
    entities = tmp_path / "entities.docx"
    make_docx(entities, p(r("&a;")), prolog='<!DOCTYPE d [<!ENTITY a "aaaaaaaaaa">]>')
    with pytest.raises(DocumentTextError, match="could not be read"):
        docx_text(str(entities))

    empty = tmp_path / "empty.docx"
    make_docx(empty, p())
    with pytest.raises(DocumentTextError, match="holds no text"):
        docx_text(str(empty))


def _post(handler, payload):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    return asyncio.run(handler(receive))


def test_server_scans_a_pdf_and_returns_its_text_for_review(tmp_path):
    from server import _process_text

    path = tmp_path / "letter.pdf"
    make_pdf(path, LETTER)
    content = "data:application/pdf;base64," + base64.b64encode(path.read_bytes()).decode()

    ok, response, status = _post(_process_text, {
        "filename": "letter.pdf", "content": content, "strategies": ["mask"]})

    assert ok is True and status == 200
    assert response["converted_from"] == ".pdf"
    original = response["original"]
    assert "Tenant: Odalys Pemberton" in original
    # the offsets of every detection count in the text the page receives
    found = {original[d["start"]:d["end"]] for d in response["detection_details"]}
    assert all(original[d["start"]:d["end"]] == d["text"] for d in response["detection_details"])
    assert {"07700 900 444", "01632 960 222", "4 March 1988"} <= found
    # the table row is read by position, though its columns do not line up
    # exactly with the header in the extracted text
    assert {"Ilse Varga", "HG4 9ZQ"} <= found
    assert "07700 900 444" not in response["versions"]["mask"]["content"]

    ok, response, status = _post(_process_text, {
        "filename": "old.doc", "content": content, "strategies": ["mask"]})
    assert ok is False and status == 400 and ".docx" in response["error"]

    scan = tmp_path / "scan.pdf"
    make_pdf(scan, [[]])
    ok, response, status = _post(_process_text, {
        "filename": "scan.pdf", "strategies": ["mask"],
        "content": "data:application/pdf;base64," + base64.b64encode(scan.read_bytes()).decode()})
    assert ok is False and status == 400 and "scanned images" in response["error"]
