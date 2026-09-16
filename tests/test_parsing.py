import fitz
from PIL import Image

from rag_workbench.models import Block, ParsedDocument
from rag_workbench.parsing import Parser, split_document


def test_pdf_preserves_page_and_bbox(settings, tmp_path):
    path = tmp_path / "native.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), "Atlas A1 Warranty: 24 months. Power: 65W.")
        pdf.save(path)
    parsed = Parser(settings).parse(path)
    chunks = split_document(parsed, "doc", "v1", "native.pdf", settings)
    assert chunks[0].page == 1
    assert chunks[0].bbox is not None
    assert "24 months" in chunks[0].text


def test_markdown_heading_inside_code_is_not_section(settings, tmp_path):
    path = tmp_path / "code.md"
    path.write_text("# Main\n```python\n# comment\nvalue = 1\n```\n", encoding="utf-8")
    parsed = Parser(settings).parse(path)
    assert len(parsed.blocks) == 1
    assert parsed.blocks[0].heading == "Main"


def test_markdown_rejects_external_image_path(settings, tmp_path):
    path = tmp_path / "unsafe.md"
    path.write_text("# Notes\nContent\n![x](../secret.png)", encoding="utf-8")
    parsed = Parser(settings).parse(path)
    assert parsed.statuses["../secret.png"] == "image_missing_or_outside_document_directory"


def test_unconfigured_image_backends_are_explicit(settings, tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (80, 80), "white").save(path)
    result = Parser(settings).parse(path)
    assert result.blocks == []
    state = next(iter(result.statuses.values()))
    assert state == {"ocr": "disabled", "vlm": "not_configured"}


def test_linked_image_change_updates_document_version(workbench, tmp_path):
    image = tmp_path / "figure.png"
    Image.new("RGB", (80, 80), "white").save(image)
    note = tmp_path / "linked.md"
    note.write_text("# Figure\nAtlas A1 diagram\n![figure](figure.png)", encoding="utf-8")
    first = workbench.ingest(note)
    Image.new("RGB", (80, 80), "black").save(image)
    second = workbench.ingest(note)
    assert second["status"] == "updated"
    assert first["document"]["version"] != second["document"]["version"]


def test_pdf_image_placeholder_does_not_count_as_native_text(settings, tmp_path, monkeypatch):
    image = tmp_path / "scan.png"
    Image.new("RGB", (500, 300), "white").save(image)
    path = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_image(page.rect, filename=str(image))
        pdf.save(path)
    parser = Parser(settings)
    calls = []

    def image_blocks(data, page=None):
        calls.append(page)
        return ParsedDocument(blocks=[Block(text="scanned text", kind="ocr", page=page)])

    monkeypatch.setattr(parser, "_image_blocks", image_blocks)
    parsed = parser.parse(path)
    assert calls == [1]
    assert all(block.kind == "ocr" for block in parsed.blocks)
