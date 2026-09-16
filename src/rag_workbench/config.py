from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    data_dir: Path = Path("runtime")
    embedding: str = "tfidf"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    generator: str = "extractive"
    base_url: str = "https://your-provider.example/v1"
    api_key: str = ""
    chat_model: str = "your-chat-model"
    embedding_model: str = "your-embedding-model"
    vlm_model: str = ""
    reranker: str = "lexical"
    local_rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ocr_enabled: bool = True
    chunk_size: int = 700
    chunk_overlap: int = 100
    max_file_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(Path.cwd() / ".env", override=False)
        result = cls()
        for name in result.__dataclass_fields__:
            value = os.getenv(f"RAG_{name.upper()}")
            if value is None:
                continue
            current = getattr(result, name)
            if isinstance(current, bool):
                value = value.lower() in {"true", "1", "yes"}
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, Path):
                value = Path(value)
            setattr(result, name, value)
        if result.chunk_overlap >= result.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return result
