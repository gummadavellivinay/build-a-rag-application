import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from .config import Settings, safe_id


def ingest(settings, tenant, manifest_path, graph_path, strategy):
    from .cloud import embeddings, PineconeDense
    from .ingest import load_corpus
    from .knowledge_graph import KnowledgeGraph

    settings.require_cloud()
    target = settings.snapshot(tenant)
    if target.exists():
        raise ValueError("Immutable snapshot already exists. Set a new CORPUS_REVISION to replace documents.")
    graph_payload = json.loads(graph_path.read_text())
    KnowledgeGraph(graph_payload, tenant)  # Validate before remote writes.
    embedder = embeddings(settings)
    chunks = load_corpus(manifest_path, tenant, settings, strategy, embedder)
    raw = json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2).encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    # One operator/writer per tenant revision; production queue enforces this lock.
    lock = target.with_suffix(".lock")
    with lock.open("x"):
        try:
            PineconeDense(settings, embedder).upsert(chunks, tenant)
            with tempfile.TemporaryDirectory(dir=target.parent) as temp:
                folder = Path(temp) / "snapshot"
                folder.mkdir()
                (folder / "chunks.json").write_bytes(raw)
                (folder / "graph.json").write_text(json.dumps(graph_payload))
                metadata = {
                    "namespace": settings.namespace(tenant),
                    "embedding_model": settings.embedding_model,
                    "dimensions": settings.embedding_dimensions,
                    "strategy": strategy,
                    "chunks_sha256": hashlib.sha256(raw).hexdigest(),
                    "count": len(chunks),
                }
                (folder / "snapshot.json").write_text(json.dumps(metadata, indent=2))
                os.rename(folder, target)
        finally:
            lock.unlink(missing_ok=True)
    print(f"Published {len(chunks)} chunks to {settings.namespace(tenant)}; snapshot: {target}")


def main():
    parser = argparse.ArgumentParser(description="Estate RAG operator CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create-index", help="Creates a billable Pinecone index using .env configuration")
    ing = commands.add_parser("ingest")
    ing.add_argument("--tenant", default="demo")
    ing.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    ing.add_argument("--graph", type=Path, default=Path("data/graph.json"))
    ing.add_argument("--strategy", choices=["fixed", "recursive", "semantic"], default="recursive")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--tenant", default="demo")
    ask.add_argument("--mode", choices=["hybrid", "multimodal", "graph"], default="hybrid")
    args = parser.parse_args()
    settings = Settings()
    if args.command == "create-index":
        from .cloud import ensure_index

        settings.require_cloud()
        print(ensure_index(settings))
    elif args.command == "ingest":
        ingest(settings, safe_id(args.tenant), args.manifest, args.graph, args.strategy)
    else:
        from .runtime import build_runtime
        from .models import Query

        request = Query(question=args.question, mode=args.mode)
        workflow, _ = build_runtime(settings, safe_id(args.tenant))
        result = workflow.invoke({**request.model_dump(), "tenant": args.tenant}, {"recursion_limit": 12})
        print(result["result"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
