from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

from rag_workbench.config import Settings
from rag_workbench.models import Answer, Chunk, Evidence, ParsedDocument
from rag_workbench.parsing import Parser, split_document
from rag_workbench.providers import Generator, chat
from rag_workbench.retrieval import Snapshot
from rag_workbench.storage import Store


def key_for(namespace: str, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return namespace + ":" + hashlib.sha256(payload.encode()).hexdigest()


def validate_citations(answer: str, evidence: list[Evidence]) -> bool:
    mentioned = set(re.findall(r"\[(E\d+)\]", answer))
    allowed = {item.evidence_id for item in evidence}
    return bool(mentioned) and mentioned.issubset(allowed)


class Workbench:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.settings.data_dir = self.settings.data_dir.resolve()
        self.store = Store(self.settings.data_dir)
        self.parser = Parser(self.settings)
        self.lock = threading.RLock()
        self.snapshots: dict[str, Snapshot] = {}

    @staticmethod
    def validate_kb(kb: str) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", kb):
            raise ValueError("知识库 ID 只能包含字母、数字、下划线和连字符，最多 64 字符")

    def documents(self, kb: str = "default") -> list[dict]:
        self.validate_kb(kb)
        with self.lock:
            return [
                {key: value for key, value in item.items() if key not in {"chunks", "path"}}
                for item in self.store.documents(kb)
            ]

    def _snapshot(self, kb: str) -> Snapshot:
        if kb not in self.snapshots:
            documents = self.store.documents(kb)
            chunks = [Chunk.model_validate(chunk) for doc in documents for chunk in doc["chunks"]]
            stored = self.store.snapshot(kb)
            existing = (
                stored["name"] if stored and stored["fingerprint"] == self._fingerprint() else ""
            )
            snapshot = Snapshot(kb, chunks, self.settings, self.store, existing)
            if not existing:
                try:
                    self.store.activate(
                        kb,
                        documents,
                        self.store.epoch(kb) + (1 if documents else 0),
                        snapshot.collection.name,
                        self._fingerprint(),
                    )
                except Exception:
                    snapshot.close()
                    raise
                if stored:
                    try:
                        snapshot.client.delete_collection(stored["name"])
                    except Exception as exc:
                        self.store.failure(kb, "retired-index-cleanup", type(exc).__name__)
            self.snapshots[kb] = snapshot
        return self.snapshots[kb]

    def _fingerprint(self) -> str:
        return key_for(
            "index",
            [
                "index-v3",
                self.settings.embedding,
                self.settings.embedding_model,
                self.settings.local_embedding_model,
                self.settings.base_url,
                self.settings.reranker,
                self.settings.local_rerank_model,
            ],
        )

    def _activate(self, kb: str, documents: list[dict]) -> None:
        chunks = [Chunk.model_validate(chunk) for doc in documents for chunk in doc["chunks"]]
        snapshot = Snapshot(kb, chunks, self.settings, self.store)
        previous = self.snapshots.get(kb)
        stored_previous = self.store.snapshot(kb)
        try:
            self.store.activate(
                kb,
                documents,
                self.store.epoch(kb) + 1,
                snapshot.collection.name,
                self._fingerprint(),
            )
        except Exception:
            snapshot.close()
            raise
        self.snapshots[kb] = snapshot
        try:
            if previous:
                previous.close()
            elif stored_previous:
                snapshot.client.delete_collection(stored_previous["name"])
        except Exception as exc:
            self.store.failure(kb, "retired-index-cleanup", type(exc).__name__)

    def ingest(
        self, path: Path, kb: str = "default", document_id: str = "", source_name: str = ""
    ) -> dict:
        self.validate_kb(kb)
        path = path.resolve()
        source = source_name or path.name
        if path.stat().st_size > self.settings.max_file_bytes:
            raise ValueError("资料超过大小限制")
        document_id = document_id or hashlib.sha256(source.encode()).hexdigest()[:16]
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        dependencies = []
        if path.suffix.lower() == ".md":
            for reference in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content.decode("utf-8")):
                asset = (path.parent / reference).resolve()
                valid = asset.is_relative_to(path.parent) and asset.is_file()
                dependencies.append(
                    [
                        reference,
                        hashlib.sha256(asset.read_bytes()).hexdigest()
                        if valid
                        else "missing_or_outside",
                    ]
                )
        parser_key = key_for(
            "parse",
            [
                file_hash,
                self.parser.VERSION,
                self.settings.ocr_enabled,
                self.settings.vlm_model,
                self.settings.base_url,
                dependencies,
            ],
        )
        version = key_for(
            "version", [parser_key, self.settings.chunk_size, self.settings.chunk_overlap]
        ).split(":")[1][:24]
        with self.lock:
            documents = self.store.documents(kb)
            previous = next((doc for doc in documents if doc["id"] == document_id), None)
            previous_failed = previous and "failed:" in json.dumps(previous["parse_status"])
            if previous and previous["version"] == version and not previous_failed:
                return {
                    "status": "unchanged",
                    "document": self._public(previous),
                    "epoch": self.store.epoch(kb),
                }
            try:
                # Markdown 的关联图片也可能变化，不能只按正文 Hash 复用解析结果。
                cached = None if path.suffix.lower() == ".md" else self.store.cache_get(parser_key)
                parsed = (
                    ParsedDocument.model_validate(cached) if cached else self.parser.parse(path)
                )
                if not parsed.blocks:
                    raise ValueError(f"未解析出可索引内容，解析状态：{parsed.statuses}")
                if not cached and "failed:" not in json.dumps(parsed.statuses):
                    self.store.cache_put(parser_key, parsed.model_dump())
                chunks = split_document(parsed, document_id, version, source, self.settings)
                if not chunks:
                    raise ValueError("切分后没有有效内容")
                original = self.settings.data_dir / "originals" / (file_hash + path.suffix.lower())
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(content)
                document = {
                    "id": document_id,
                    "source": source,
                    "version": version,
                    "file_hash": file_hash,
                    "path": str(original),
                    "chunks": [chunk.model_dump() for chunk in chunks],
                    "chunk_count": len(chunks),
                    "parse_status": parsed.statuses,
                    "parse_cache_hit": cached is not None,
                }
                updated = [doc for doc in documents if doc["id"] != document_id] + [document]
                self._activate(kb, sorted(updated, key=lambda doc: doc["id"]))
            except Exception as exc:
                self.store.failure(kb, source, str(exc))
                raise
            return {
                "status": "updated" if previous else "created",
                "document": self._public(document),
                "epoch": self.store.epoch(kb),
            }

    @staticmethod
    def _public(document: dict) -> dict:
        return {key: value for key, value in document.items() if key not in {"chunks", "path"}}

    def delete(self, document_id: str, kb: str = "default") -> dict:
        self.validate_kb(kb)
        with self.lock:
            documents = self.store.documents(kb)
            updated = [doc for doc in documents if doc["id"] != document_id]
            if len(updated) == len(documents):
                raise KeyError("文档不存在")
            self._activate(kb, updated)
            return {"status": "deleted", "epoch": self.store.epoch(kb)}

    def search(
        self,
        question: str,
        kb: str = "default",
        mode: str = "hybrid",
        top_k: int = 5,
        rerank: bool = True,
        mmr: bool = False,
        use_cache: bool = True,
    ) -> list[Evidence]:
        self.validate_kb(kb)
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError("检索模式必须是 hybrid、bm25 或 vector")
        with self.lock:
            snapshot = self._snapshot(kb)
            cache_key = key_for(
                "retrieval",
                [
                    kb,
                    self.store.epoch(kb),
                    question,
                    mode,
                    top_k,
                    rerank,
                    mmr,
                    self.settings.embedding,
                    self.settings.reranker,
                    self.settings.local_embedding_model,
                    self.settings.embedding_model,
                    self.settings.local_rerank_model,
                    self.settings.base_url,
                ],
            )
            cached = self.store.cache_get(cache_key) if use_cache else None
            if cached is not None:
                return [Evidence.model_validate(item) for item in cached]
            result = snapshot.search(question, mode, max(1, min(top_k, 20)), rerank, mmr)
            if use_cache:
                self.store.cache_put(cache_key, [item.model_dump() for item in result])
            return result

    def ask(self, question: str, kb: str = "default", session: str = "default") -> Answer:
        self.validate_kb(kb)
        question = question.strip()
        if not question or len(question) > 2000:
            raise ValueError("问题不能为空，且最多 2000 个字符")
        with self.lock:
            self._snapshot(kb)
            epoch = self.store.epoch(kb)
            history = self.store.recent(kb, session)
            query = question
            route = "comparison" if re.search(r"比较|区别|对比", question) else "fact"
            if re.search(r"图中|图片|图表|流程图", question):
                route = "image"
            if re.search(r"^(它|这个|那个|其|这).{0,18}", question):
                if not history:
                    return Answer(
                        question=question,
                        rewritten_query=question,
                        route="clarify",
                        answer="请说明你指的是哪份资料或哪个概念。",
                        mode="clarification",
                        evidence=[],
                        trace=[],
                        epoch=epoch,
                        insufficient=True,
                    )
                query = history[-1]["question"] + " " + question
            cache_key = key_for(
                "answer",
                [
                    kb,
                    epoch,
                    question,
                    query,
                    self.settings.generator,
                    self.settings.chat_model,
                    self.settings.embedding,
                    self.settings.reranker,
                    self.settings.base_url,
                    self.settings.local_embedding_model,
                    self.settings.embedding_model,
                    self.settings.local_rerank_model,
                ],
            )
            cached = self.store.cache_get(cache_key)
            if cached:
                answer = Answer.model_validate(cached)
                answer.cache_hit = True
                self.store.remember(kb, session, question, answer.answer)
                return answer
            started = time.perf_counter()
            evidence = self.search(query, kb)
            trace = [
                {"stage": "route", "route": route, "rewritten_query": query},
                {
                    "stage": "retrieval",
                    "count": len(evidence),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "embedding": self.settings.embedding,
                    "reranker": self.settings.reranker,
                },
            ]
            if route == "comparison":
                pieces = re.split(r"与|和|、| versus | vs ", query)
                subqueries = [
                    re.sub(r"比较|对比|有什么区别|的区别|区别", "", piece).strip(" ？?")
                    for piece in pieces
                ][:3]
                if len(subqueries) > 1:
                    combined = []
                    for subquery in subqueries:
                        if subquery:
                            combined.extend(self.search(subquery, kb, top_k=2))
                    combined.extend(evidence)
                    by_id = {item.chunk.id: item for item in combined}
                    evidence = list(by_id.values())[:5]
                    trace.append({"stage": "comparison_queries", "queries": subqueries})
            if route == "image":
                evidence.sort(key=lambda item: item.chunk.kind in {"ocr", "vision"}, reverse=True)
            if len(evidence) < (2 if route == "comparison" else 1):
                expanded = query.replace("怎么", "如何").replace("好处", "优势")
                if expanded != query:
                    more = self.search(expanded, kb)
                    by_id = {item.chunk.id: item for item in evidence + more}
                    evidence = list(by_id.values())[:5]
                trace.append(
                    {"stage": "supplementary_retrieval", "attempts": 1, "count": len(evidence)}
                )
            for number, item in enumerate(evidence, 1):
                item.evidence_id = f"E{number}"
            insufficient = not evidence
            mode = "extractive"
            if insufficient:
                text = "当前知识库没有足够的相关证据，无法确定。请补充资料或更具体地提问。"
            else:
                text = self._extract(evidence)
                if self.settings.generator == "remote":
                    try:
                        context = "\n\n".join(
                            f"[{item.evidence_id}] {item.chunk.text}" for item in evidence
                        )
                        generated = Generator(self.settings).generate(question, context)
                        structural = validate_citations(generated, evidence)
                        supported = False
                        if structural:
                            review = chat(
                                self.settings,
                                [
                                    {
                                        "role": "user",
                                        "content": "判断回答的全部事实是否被证据支持。"
                                        '仅返回 JSON {"supported":true/false}。'
                                        f"\n回答：{generated}\n证据：{context}",
                                    }
                                ],
                            )
                            review = re.sub(r"^```(?:json)?\s*|\s*```$", "", review.strip())
                            supported = json.loads(review).get("supported") is True
                        trace.append(
                            {
                                "stage": "citation_check",
                                "structural": structural,
                                "model_support_check": supported,
                            }
                        )
                        if structural and supported:
                            text, mode = generated, "generated"
                        else:
                            trace.append({"stage": "fallback", "reason": "citation_not_supported"})
                    except Exception as exc:
                        trace.append({"stage": "fallback", "reason": type(exc).__name__})
                else:
                    trace.append(
                        {
                            "stage": "citation_check",
                            "structural": True,
                            "support": "verbatim_extraction",
                        }
                    )
            trace.append(
                {
                    "stage": "complete",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            answer = Answer(
                question=question,
                rewritten_query=query,
                route=route,
                answer=text,
                mode=mode,
                evidence=evidence,
                trace=trace,
                epoch=epoch,
                insufficient=insufficient,
            )
            self.store.cache_put(cache_key, answer.model_dump())
            self.store.remember(kb, session, question, text)
            reports = self.settings.data_dir / "reports"
            reports.mkdir(exist_ok=True)
            (reports / f"{uuid4().hex}.json").write_text(
                answer.model_dump_json(indent=2), encoding="utf-8"
            )
            return answer

    @staticmethod
    def _extract(evidence: list[Evidence]) -> str:
        return "以下为检索到的原文依据（抽取式模式）：\n\n" + "\n\n".join(
            f"{item.chunk.text} [{item.evidence_id}]" for item in evidence
        )

    def close(self) -> None:
        # 有效 Chroma 集合保留在磁盘，重启时按 SQLite 清单加载。
        self.store.close()
