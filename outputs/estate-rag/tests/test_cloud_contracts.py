"""Contract tests with mocked network boundaries, not claims of live cloud access."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from estate_rag.config import Settings
from estate_rag.models import Chunk
from estate_rag.cloud import PineconeDense, PineconeReranker, ensure_index


@pytest.fixture
def pc(monkeypatch):
    client = Mock()
    client.has_index.return_value = True
    client.describe_index.return_value = SimpleNamespace(
        dimension=1536, metric="cosine", host="fake", status={"ready": True}
    )
    monkeypatch.setattr("estate_rag.cloud.pinecone", lambda settings: client)
    return client


def test_dense_namespace_filter_and_timeout(pc):
    embedder = Mock()
    embedder.embed_query.return_value = [0.1, 0.2]
    pc.Index.return_value.query.return_value = SimpleNamespace(matches=[SimpleNamespace(id="c1")])
    dense = PineconeDense(Settings(), embedder)
    assert dense.search("garden", "demo", 20) == ["c1"]
    kwargs = pc.Index.return_value.query.call_args.kwargs
    assert kwargs["namespace"] == "4:demo:v1"
    assert kwargs["filter"] == {"tenant": {"$eq": "demo"}}
    assert kwargs["_request_timeout"] == 30


def test_upsert_visibility_and_batch_contract(pc):
    embedder = Mock()
    embedder.embed_documents.return_value = [[0.1, 0.2]]
    pc.Index.return_value.fetch.return_value = SimpleNamespace(vectors={"c1": {}})
    dense = PineconeDense(Settings(), embedder)
    dense.upsert([Chunk(id="c1", text="garden", source="a", tenant="demo")], "demo")
    kwargs = pc.Index.return_value.upsert.call_args.kwargs
    assert kwargs["vectors"][0]["metadata"]["tenant"] == "demo"
    assert kwargs["namespace"] == "4:demo:v1"
    assert pc.Index.return_value.fetch.called


def test_rerank_preserves_original_candidate_mapping(pc):
    pc.inference.rerank.return_value = SimpleNamespace(data=[SimpleNamespace(index=1, score=0.8)])
    chunks = [Chunk(id=str(i), text=str(i), source="a", tenant="demo") for i in range(2)]
    result = PineconeReranker(Settings()).rank("q", chunks, 1)
    assert result[0][0].id == "1" and result[0][1] == 0.8


def test_index_mismatch_rejected(pc):
    pc.describe_index.return_value.dimension = 42
    with pytest.raises(ValueError, match="mismatch"):
        ensure_index(Settings())
    pc.create_index.assert_not_called()


def test_index_creation_has_expected_contract(pc):
    pc.has_index.return_value = False
    assert ensure_index(Settings()) == "fake"
    kwargs = pc.create_index.call_args.kwargs
    assert kwargs["dimension"] == 1536 and kwargs["metric"] == "cosine"
    assert kwargs["deletion_protection"] == "enabled"
