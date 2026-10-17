"""Cloud retrieval ablations. Never use offline test doubles as relevance measurements."""

import argparse
import json
import math
import time
from pathlib import Path
from estate_rag.config import Settings
from estate_rag.runtime import build_runtime


def retrieval_metrics(actual_sources, expected_sources, k=6):
    # Document-level metrics: deduplicate chunks from the same source.
    actual = list(dict.fromkeys(actual_sources))[:k]
    expected = set(expected_sources)
    if not expected:
        return {"recall": None, "mrr": None, "ndcg": None}
    hits = [int(source in expected) for source in actual]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return {
        "recall": len(set(actual) & expected) / len(expected),
        "mrr": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0),
        "ndcg": dcg / ideal,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--answers", action="store_true", help="Also run full graph/hybrid workflows; incurs LLM calls"
    )
    p.add_argument("--out", default="data/eval-results.json")
    args = p.parse_args()
    settings = Settings()
    graph, retriever = build_runtime(settings, "demo")
    vector_graph = build_runtime(settings, "demo", vector_only=True)[0] if args.answers else None
    cases = json.loads((settings.data_dir / "eval.json").read_text())
    rows = []
    for case in cases:
        for mode, rerank in [("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True)]:
            start = time.perf_counter()
            ev = retriever.retrieve(case["question"], mode=mode, rerank=rerank)
            rows.append(
                {
                    "id": case["id"],
                    "variant": f"{mode}-rerank={rerank}",
                    **retrieval_metrics([e.source for e in ev], case["sources"]),
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "sources": [e.source for e in ev],
                }
            )
        if args.answers:
            for mode in ["vector", "graph"] if case["mode"] == "graph" else [case["mode"]]:
                start = time.perf_counter()
                workflow = vector_graph if mode == "vector" else graph
                result = workflow.invoke(
                    {
                        "question": case["question"],
                        "mode": "hybrid" if mode == "vector" else mode,
                        "tenant": "demo",
                    }
                )["result"]
                rows.append(
                    {
                        "id": case["id"],
                        "variant": f"answer-{mode}",
                        "result": result.model_dump(),
                        "expected_abstain": case.get("abstain", False),
                        "abstention_correct": result.abstained == case.get("abstain", False),
                        "latency_ms": (time.perf_counter() - start) * 1000,
                        "human_faithfulness": None,
                    }
                )
    Path(args.out).write_text(json.dumps({"revision": settings.corpus_revision, "rows": rows}, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
