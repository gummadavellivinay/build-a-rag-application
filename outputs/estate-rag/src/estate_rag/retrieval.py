import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from rank_bm25 import BM25Okapi
from .models import Chunk, Evidence


def tokenize(text: str) -> list[str]:
    # Unicode Latin languages. CJK needs a language-specific tokenizer in Phase 2.
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] += 1 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


class DenseBackend(Protocol):
    def search(self, question: str, tenant: str, k: int) -> list[str]: ...


class Reranker(Protocol):
    def rank(self, question: str, chunks: list[Chunk], k: int) -> list[tuple[Chunk, float]]: ...


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        tenant: str,
        dense: DenseBackend,
        reranker: Reranker,
        candidate_k=20,
        final_k=6,
    ):
        if not chunks or any(c.tenant != tenant for c in chunks):
            raise ValueError("Empty corpus or cross-tenant chunk detected")
        self.chunks, self.tenant = chunks, tenant
        self.by_id = {c.id: c for c in chunks}
        self.dense, self.reranker = dense, reranker
        self.candidate_k, self.final_k = candidate_k, final_k
        self.tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens)

    def lexical(self, question: str):
        terms = tokenize(question)
        scores = self.bm25.get_scores(terms)
        # BM25 can be negative in tiny corpora. Filter by term overlap, not positive score.
        matches = [i for i, words in enumerate(self.tokens) if set(terms).intersection(words)]
        return [self.chunks[i].id for i in sorted(matches, key=lambda i: (-scores[i], i))[: self.candidate_k]]

    def retrieve(self, question: str, mode="hybrid", rerank=True) -> list[Evidence]:
        if mode == "bm25":
            ids = self.lexical(question)
        elif mode == "dense":
            ids = self.dense.search(question, self.tenant, self.candidate_k)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                dense = pool.submit(self.dense.search, question, self.tenant, self.candidate_k)
                sparse = pool.submit(self.lexical, question)
                ids = reciprocal_rank_fusion([dense.result(), sparse.result()])
        candidates = [self.by_id[i] for i in ids if i in self.by_id][: 2 * self.candidate_k]
        ranked = (
            self.reranker.rank(question, candidates, self.final_k)
            if rerank and candidates
            else [(c, None) for c in candidates[: self.final_k]]
        )
        return [
            Evidence(id=c.id, text=c.text, source=c.source, page=c.page, score=score) for c, score in ranked
        ]
