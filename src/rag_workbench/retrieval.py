from __future__ import annotations

import hashlib
import re
from uuid import uuid4

import chromadb
import jieba
import numpy as np
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from rag_workbench.config import Settings
from rag_workbench.models import Chunk, Evidence
from rag_workbench.providers import NeuralReranker, SemanticEmbedding
from rag_workbench.storage import Store
from rag_workbench.tfidf import Tfidf


def tokenize(text: str) -> list[str]:
    stopwords = {"的", "了", "是", "吗", "呢", "和", "与", "在", "有", "为", "什么", "多少", "多久"}
    return [
        word.lower()
        for word in jieba.lcut(text)
        if word not in stopwords and re.search(r"[a-zA-Z0-9\u4e00-\u9fff]", word)
    ]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return scores


class Snapshot:
    def __init__(
        self,
        kb: str,
        chunks: list[Chunk],
        settings: Settings,
        store: Store,
        existing_name: str = "",
    ):
        self.chunks = chunks
        self.settings = settings
        self.store = store
        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.client = chromadb.PersistentClient(
            path=str(settings.data_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        name = "kb-" + hashlib.sha256(kb.encode()).hexdigest()[:12] + "-" + uuid4().hex[:8]
        self.collection = (
            self.client.get_collection(existing_name)
            if existing_name
            else self.client.create_collection(name, metadata={"hnsw:space": "cosine"})
        )
        self.vectorizer = None
        self.semantic = None
        self.neural = None
        self.bm25 = None
        if not chunks:
            self.vectors = np.zeros((0, 1))
            return
        texts = [chunk.text for chunk in chunks]
        self.bm25 = BM25Okapi([tokenize(text) for text in texts])
        try:
            if settings.embedding == "tfidf":
                self.vectorizer = Tfidf().fit(texts)
                self.vectors = self.vectorizer.transform(texts)
            else:
                self.semantic = SemanticEmbedding(settings)
                self.vectors = self._cached_encode(texts)
            if not existing_name:
                self.collection.add(
                    ids=[chunk.id for chunk in chunks],
                    embeddings=self.vectors.tolist(),
                    documents=texts,
                )
            if settings.reranker == "neural":
                self.neural = NeuralReranker(settings)
        except Exception:
            if not existing_name:
                self.close()
            raise

    def _cached_encode(self, texts: list[str]) -> np.ndarray:
        assert self.semantic is not None
        vectors = []
        for text in texts:
            key = (
                "embedding:"
                + hashlib.sha256((self.semantic.fingerprint + text).encode()).hexdigest()
            )
            cached = self.store.cache_get(key)
            if cached is None:
                cached = self.semantic.encode([text])[0].tolist()
                self.store.cache_put(key, cached)
            vectors.append(cached)
        return np.asarray(vectors, dtype=np.float32)

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        rerank: bool = True,
        mmr: bool = False,
    ) -> list[Evidence]:
        if not self.chunks:
            return []
        tokens = tokenize(query)
        query_terms = set(tokens)
        overlap = [len(query_terms & set(tokenize(chunk.text))) for chunk in self.chunks]
        bm_scores = np.asarray(self.bm25.get_scores(tokens))
        sparse_indices = sorted(
            range(len(self.chunks)), key=lambda i: (bm_scores[i], overlap[i]), reverse=True
        )
        sparse = [self.chunks[i].id for i in sparse_indices if overlap[i] > 0][:20]
        if mode == "vector":
            sparse = []
        query_vector = np.zeros(1)
        if mode != "bm25":
            if self.vectorizer:
                query_vector = self.vectorizer.transform([query])[0]
            else:
                query_vector = self.semantic.encode([query])[0]
        dense = []
        if np.linalg.norm(query_vector) > 0:
            result = self.collection.query(
                query_embeddings=[query_vector.tolist()], n_results=min(20, len(self.chunks))
            )
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True):
                if distance < (0.8 if self.vectorizer else 0.75):
                    dense.append(chunk_id)
        rankings = [sparse] if mode == "bm25" else [dense] if mode == "vector" else [sparse, dense]
        scores = rrf(rankings)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
        if rerank and ordered:
            if self.neural:
                rerank_scores = self.neural.score(query, [self.by_id[key].text for key in ordered])
            else:
                entities = re.findall(
                    r"\b[A-Za-z][A-Za-z0-9_-]*\s+[A-Za-z]*\d+[A-Za-z0-9_-]*\b", query
                )
                rerank_scores = [
                    len(query_terms & set(tokenize(self.by_id[key].text)))
                    / max(len(query_terms), 1)
                    + 0.35
                    * len(query_terms & set(tokenize(self.by_id[key].heading)))
                    / max(len(query_terms), 1)
                    + 0.5
                    * sum(
                        entity.lower() in self.by_id[key].heading.split(" / ")[-1].lower()
                        for entity in entities
                    )
                    + scores[key]
                    for key in ordered
                ]
            ordered = [
                key for _, key in sorted(zip(rerank_scores, ordered, strict=True), reverse=True)
            ]
        if mmr and len(ordered) > 1:
            selected = [ordered.pop(0)]
            indices = {chunk.id: i for i, chunk in enumerate(self.chunks)}
            while ordered and len(selected) < top_k:

                def value(key: str) -> float:
                    vector = self.vectors[indices[key]]
                    redundancy = max(
                        float(np.dot(vector, self.vectors[indices[chosen]]))
                        / max(
                            float(
                                np.linalg.norm(vector)
                                * np.linalg.norm(self.vectors[indices[chosen]])
                            ),
                            1e-12,
                        )
                        for chosen in selected
                    )
                    return 0.7 * scores[key] / max(scores.values()) - 0.3 * redundancy

                best = max(ordered, key=value)
                selected.append(best)
                ordered.remove(best)
            ordered = selected
        return [
            Evidence(
                evidence_id=f"E{i}",
                chunk=self.by_id[key],
                score=scores[key],
                channels=[
                    channel
                    for channel, ranking in (("bm25", sparse), ("vector", dense))
                    if key in ranking
                ],
            )
            for i, key in enumerate(ordered[:top_k], 1)
        ]

    def close(self) -> None:
        self.client.delete_collection(self.collection.name)
