from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_workbench.config import Settings
from rag_workbench.models import Block, Chunk, ParsedDocument
from rag_workbench.providers import VisionProvider


class Parser:
    VERSION = "parser-v2"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.assets = settings.data_dir / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self._ocr = None

    def _image_blocks(self, data: bytes, page: int | None = None) -> ParsedDocument:
        image_id = hashlib.sha256(data).hexdigest()[:24]
        image_path = self.assets / f"{image_id}.png"
        image_path.write_bytes(data)
        blocks = []
        statuses = {}
        if self.settings.ocr_enabled:
            try:
                if self._ocr is None:
                    from rapidocr import RapidOCR

                    self._ocr = RapidOCR()
                result = self._ocr(str(image_path))
                text = "\n".join(result.txts if result.txts is not None else [])
                statuses["ocr"] = "executed" if text else "executed_no_text"
                if text:
                    blocks.append(Block(text=text, kind="ocr", page=page, image_id=image_id))
            except Exception as exc:
                statuses["ocr"] = f"failed:{type(exc).__name__}"
        else:
            statuses["ocr"] = "disabled"
        if self.settings.vlm_model:
            try:
                text = VisionProvider(self.settings).describe(data)
                blocks.append(Block(text=text, kind="vision", page=page, image_id=image_id))
                statuses["vlm"] = "executed"
            except Exception as exc:
                statuses["vlm"] = f"failed:{type(exc).__name__}"
        else:
            statuses["vlm"] = "not_configured"
        return ParsedDocument(blocks=blocks, statuses={image_id: statuses})

    def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf(path)
        if suffix == ".md":
            return self._markdown(path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            from io import BytesIO

            from PIL import Image

            stream = BytesIO()
            with Image.open(path) as image:
                image.convert("RGB").save(stream, format="PNG")
            return self._image_blocks(stream.getvalue())
        raise ValueError("只支持 PDF、Markdown、PNG、JPG 和 JPEG")

    def _pdf(self, path: Path) -> ParsedDocument:
        blocks = []
        statuses = {}
        with fitz.open(path) as pdf:
            if len(pdf) > 100:
                raise ValueError("单份 PDF 最多 100 页")
            for number, page in enumerate(pdf, 1):
                native = [
                    block
                    for block in page.get_text("blocks", sort=True)
                    if len(block) >= 7 and block[6] == 0
                ]
                page_text = "".join(str(block[4]) for block in native)
                if not page_text.strip():
                    result = self._image_blocks(
                        page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes(), page=number
                    )
                    blocks.extend(result.blocks)
                    statuses.update(result.statuses)
                    continue
                for block in native:
                    text = str(block[4]).strip()
                    if text:
                        blocks.append(Block(text=text, page=number, bbox=list(block[:4])))
                seen = set()
                for image in page.get_images():
                    if image[0] in seen:
                        continue
                    seen.add(image[0])
                    pixmap = fitz.Pixmap(pdf, image[0])
                    if pixmap.n > 3:
                        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                    result = self._image_blocks(pixmap.tobytes("png"), page=number)
                    blocks.extend(result.blocks)
                    statuses.update(result.statuses)
        return ParsedDocument(blocks=blocks, statuses=statuses)

    def _markdown(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        blocks = []
        statuses = {}
        heading = []
        current = []
        start = 1
        in_code = False
        for number, line in enumerate(lines, 1):
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match and not in_code:
                if current:
                    blocks.append(
                        Block(
                            text="\n".join(current),
                            heading=" / ".join(title for _, title in heading),
                            line_start=start,
                            line_end=number - 1,
                        )
                    )
                level = len(match[1])
                while heading and heading[-1][0] >= level:
                    heading.pop()
                heading.append((level, match[2]))
                current, start = [], number
            current.append(line)
            if line.strip().startswith("```"):
                in_code = not in_code
        if current:
            blocks.append(
                Block(
                    text="\n".join(current),
                    heading=" / ".join(title for _, title in heading),
                    line_start=start,
                    line_end=len(lines),
                )
            )
        for reference in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            asset = (path.parent / reference).resolve()
            if asset.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                statuses[reference] = "unsupported_image_reference"
                continue
            if not asset.is_relative_to(path.parent.resolve()) or not asset.is_file():
                statuses[reference] = "image_missing_or_outside_document_directory"
                continue
            result = self.parse(asset)
            blocks.extend(result.blocks)
            statuses.update(result.statuses)
        return ParsedDocument(blocks=blocks, statuses=statuses)


def split_document(
    parsed: ParsedDocument, document_id: str, version: str, source: str, settings: Settings
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = []
    seen = set()
    for block in parsed.blocks:
        document = Document(page_content=block.text, metadata={"kind": block.kind})
        for segment in splitter.split_documents([document]):
            text = segment.page_content.strip()
            fingerprint = (text, block.page, block.heading, block.image_id)
            if not text or fingerprint in seen:
                continue
            seen.add(fingerprint)
            offset = segment.metadata.get("start_index", 0)
            line_start = None
            if block.line_start:
                line_start = block.line_start + block.text[: max(offset, 0)].count("\n")
            chunk_id = hashlib.sha256(
                f"{document_id}:{version}:{len(chunks)}:{text}".encode()
            ).hexdigest()[:24]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version=version,
                    source=source,
                    text=text,
                    kind=block.kind,
                    page=block.page,
                    heading=block.heading,
                    line_start=line_start,
                    line_end=line_start + text.count("\n") if line_start else None,
                    image_id=block.image_id,
                    bbox=block.bbox,
                )
            )
    return chunks
