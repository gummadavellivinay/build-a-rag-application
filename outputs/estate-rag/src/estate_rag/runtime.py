import hashlib
import json
from .cloud import PineconeDense, PineconeReranker, chat
from .knowledge_graph import KnowledgeGraph
from .models import Chunk
from .retrieval import HybridRetriever
from .tools import Inventory
from .workflow import Reasoner, compile_workflow


def build_runtime(settings, tenant, vector_only=False):
    settings.require_cloud()
    root = settings.snapshot(tenant)
    manifest = json.loads((root / "snapshot.json").read_text())
    if (
        manifest["embedding_model"] != settings.embedding_model
        or manifest["dimensions"] != settings.embedding_dimensions
        or manifest["namespace"] != settings.namespace(tenant)
    ):
        raise ValueError("Snapshot model, dimensions or namespace mismatch; re-ingest a new revision")
    raw = (root / "chunks.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["chunks_sha256"]:
        raise ValueError("Snapshot integrity mismatch")
    chunks = [Chunk.model_validate(c) for c in json.loads(raw)]
    retriever = HybridRetriever(
        chunks,
        tenant,
        PineconeDense(settings),
        PineconeReranker(settings),
        settings.candidate_k,
        settings.final_k,
    )
    graph = KnowledgeGraph(json.loads((root / "graph.json").read_text()), tenant)
    workflow = compile_workflow(
        retriever,
        graph,
        Inventory(settings, tenant).langchain_tool(),
        Reasoner(chat(settings)),
        vector_only=vector_only,
    )
    return workflow, retriever
