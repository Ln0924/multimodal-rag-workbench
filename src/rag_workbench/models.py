from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Block(BaseModel):
    text: str
    kind: str = "text"
    page: int | None = None
    heading: str = ""
    line_start: int | None = None
    line_end: int | None = None
    image_id: str | None = None
    bbox: list[float] | None = None


class ParsedDocument(BaseModel):
    blocks: list[Block]
    statuses: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    document_id: str
    version: str
    source: str
    text: str
    kind: str = "text"
    page: int | None = None
    heading: str = ""
    line_start: int | None = None
    line_end: int | None = None
    image_id: str | None = None
    bbox: list[float] | None = None


class Evidence(BaseModel):
    evidence_id: str
    chunk: Chunk
    score: float
    channels: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    rewritten_query: str
    route: str
    answer: str
    mode: str
    evidence: list[Evidence]
    trace: list[dict[str, Any]]
    epoch: int
    cache_hit: bool = False
    insufficient: bool = False
