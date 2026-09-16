from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


class Tfidf:
    """小型知识库字符 n-gram TF-IDF，明确定义为词汇统计基线。"""

    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf = np.zeros(0)

    @staticmethod
    def terms(text: str) -> Counter:
        segments = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
        return Counter(
            segment[i : i + n]
            for segment in segments
            for n in (1, 2, 3)
            for i in range(len(segment) - n + 1)
        )

    def fit(self, texts: list[str]) -> Tfidf:
        frequency = Counter()
        for text in texts:
            frequency.update(self.terms(text).keys())
        features = sorted(frequency, key=lambda term: (-frequency[term], term))[: self.max_features]
        self.vocabulary = {term: index for index, term in enumerate(features)}
        self.idf = np.asarray(
            [math.log((len(texts) + 1) / (frequency[term] + 1)) + 1 for term in features],
            dtype=np.float32,
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in self.terms(text).items():
                column = self.vocabulary.get(term)
                if column is not None:
                    vectors[row, column] = (1 + math.log(count)) * self.idf[column]
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
