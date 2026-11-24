"""Plain text from PDF and Word files, for the text track.

The text detector works on plain text. A PDF or a .docx is read here into
the text a person would see on the page, and the redacted versions are
written as text files: the layout of the original is not kept.

Word files are read with the standard library alone (a .docx is a ZIP
archive of XML parts). PDF files need pypdf.
"""
import os
import re
import zipfile
from xml.etree import ElementTree

MAX_PDF_PAGES = 2000
# The XML parts of a .docx are compressed; this bounds what is unpacked.
MAX_DOCX_XML_BYTES = 100 * 1024 * 1024
# Fewer letters or digits than this in a whole PDF means there is no text
# layer worth reading: the pages are pictures of text.
MIN_PDF_TEXT_CHARS = 20


class DocumentTextError(ValueError):
    """The file cannot be read as text; the message is meant for the user."""


def extract_document_text(path):
    """Text of a .pdf or .docx file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return pdf_text(path)
    if ext == ".docx":
        return docx_text(path)
    if ext == ".doc":
        raise DocumentTextError(
            "Old Word files (.doc) cannot be read. Save the document as .docx or PDF and try again.")
    raise DocumentTextError(f"Unsupported document type: {ext}")


def _tidy(text):
    """Drop control characters and trailing spaces; keep the line structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{4,}", "\n\n\n", text).strip("\n") + "\n"


# --------------------------------------------------------------------- PDF

def pdf_text(path):
    """Text layer of a PDF, page after page.

    The layout mode of pypdf keeps the gaps between columns and the blank
    lines between blocks, which the table and address rules rely on. A page
    it cannot lay out is read in the plain mode instead.
    """
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as error:  # pragma: no cover - depends on the install
        raise DocumentTextError("Reading PDF files needs the pypdf package "
                                "(pip install pypdf).") from error

    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentTextError(
                "This PDF is password-protected. Remove the password and try again.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentTextError(f"This PDF has more than {MAX_PDF_PAGES} pages.")
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                text = ""
            if not text.strip():
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
            pages.append(text.strip("\n"))
    except DocumentTextError:
        raise
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as error:
        raise DocumentTextError("This file could not be read as a PDF.") from error

    text = "\n\n".join(page for page in pages if page.strip())
    if sum(ch.isalnum() for ch in text) < MIN_PDF_TEXT_CHARS:
        raise DocumentTextError(
            "This PDF has no text that can be read: its pages are probably scanned images. "
            "Export the pages as images and use the image track instead.")
    return _tidy(text)


# -------------------------------------------------------------------- DOCX

_DOCX_PARTS = (r"word/header\d*\.xml", r"word/document\.xml", r"word/footnotes\.xml",
               r"word/endnotes\.xml", r"word/footer\d*\.xml", r"word/comments\.xml")
# Containers that only wrap blocks or runs.
_PASS_THROUGH = {"sdt", "sdtContent", "customXml", "smartTag", "ins", "moveTo", "hyperlink",
                 "r", "fldSimple", "AlternateContent", "Choice", "drawing", "inline", "anchor",
                 "graphic", "graphicData", "wsp", "txbx", "pict", "shape", "textbox", "group",
                 "wgp", "footnote", "endnote", "comment"}
# Text that is not on the page: tracked deletions, field codes, properties,
# and the legacy copy of a drawing.
_SKIPPED = {"delText", "del", "moveFrom", "instrText", "pPr", "rPr", "tblPr", "trPr", "tcPr",
            "sectPr", "Fallback"}


def _local(element):
    return element.tag.rsplit("}", 1)[-1]


def _paragraph_text(paragraph):
    pieces = []

    def walk(element):
        for child in element:
            tag = _local(child)
            if tag == "t":
                pieces.append(child.text or "")
            elif tag == "tab":
                pieces.append("\t")
            elif tag in ("br", "cr"):
                pieces.append("\n")
            elif tag == "noBreakHyphen":
                pieces.append("-")
            elif tag == "txbxContent":
                # A text box anchored in this paragraph: its own lines.
                pieces.append("\n" + "\n".join(_block_lines(child)) + "\n")
            elif tag in _SKIPPED:
                continue
            else:
                walk(child)

    walk(paragraph)
    return "".join(pieces)


def _direct(element, wanted):
    """Children with the wanted tag, looking through content-control wrappers."""
    for child in element:
        tag = _local(child)
        if tag == wanted:
            yield child
        elif tag in ("sdt", "sdtContent", "customXml", "ins"):
            yield from _direct(child, wanted)


def _table_lines(table):
    """One line per row, cells separated by tabs, as a pasted table reads."""
    lines = []
    for row in _direct(table, "tr"):
        cells = [" ".join(line.strip() for line in _block_lines(cell) if line.strip())
                 for cell in _direct(row, "tc")]
        if any(cells):
            lines.append("\t".join(cells))
    return lines


def _block_lines(container):
    lines = []
    for child in container:
        tag = _local(child)
        if tag == "p":
            lines.extend(_paragraph_text(child).split("\n"))
        elif tag == "tbl":
            lines.extend(_table_lines(child))
            lines.append("")
        elif tag in _PASS_THROUGH or tag in ("body", "hdr", "ftr", "footnotes", "endnotes",
                                              "comments"):
            lines.extend(_block_lines(child))
    return lines


def docx_text(path):
    """Text of a Word document: headers, body, notes, footers and comments.

    Headers and footers come along because a letterhead holds names and
    addresses; text removed with tracked changes does not.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "word/document.xml" not in names:
                raise DocumentTextError("This file is not a Word document (.docx).")
            parts = [name for pattern in _DOCX_PARTS
                     for name in sorted(names) if re.fullmatch(pattern, name)]
            if sum(archive.getinfo(name).file_size for name in parts) > MAX_DOCX_XML_BYTES:
                raise DocumentTextError("This Word document is too large to read.")
            blocks = []
            for name in parts:
                data = archive.read(name)
                if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
                    raise DocumentTextError("This Word document could not be read.")
                root = ElementTree.fromstring(data)
                lines = _block_lines(root)
                block = "\n".join(lines).strip("\n")
                # The same header or footer is stored once per section.
                if block.strip() and block not in blocks:
                    blocks.append(block)
    except DocumentTextError:
        raise
    except (zipfile.BadZipFile, ElementTree.ParseError, OSError, KeyError) as error:
        raise DocumentTextError("This file could not be read as a Word document (.docx).") from error

    text = "\n\n".join(blocks)
    if not text.strip():
        raise DocumentTextError("This Word document holds no text.")
    return _tidy(text)
