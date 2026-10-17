import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from estate_rag.api import create_app
from estate_rag.config import Settings, safe_id
from estate_rag.ingest import extract, split_documents
from estate_rag.knowledge_graph import KnowledgeGraph
from estate_rag.models import Chunk, Evidence, Plan, Draft, Claim, Verdict, Answer
from estate_rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from estate_rag.tools import Inventory
from estate_rag.workflow import compile_workflow
from langchain_core.documents import Document

DATA = Path(__file__).resolve().parents[1] / "data"


class Dense:
    def search(self, question, tenant, k):
        return ["c2", "c1", "foreign-id"]


class Rerank:
    def rank(self, question, chunks, k):
        return [(c, 0.8) for c in sorted(chunks, key=lambda c: c.id)[:k]]


@pytest.fixture
def retriever():
    return HybridRetriever(
        [
            Chunk(id="c1", text="L-001 solar garden", tenant="demo", source="a.txt"),
            Chunk(id="c2", text="L-002 elevator parking", tenant="demo", source="b.txt"),
            Chunk(id="c3", text="Broker affiliation", tenant="demo", source="c.txt"),
        ],
        "demo",
        Dense(),
        Rerank(),
    )


def test_rrf_deduplicates():
    assert reciprocal_rank_fusion([["a", "a", "b"], ["b", "a"]]) == ["a", "b"]


def test_retrieval_fuses_and_filters_unknown_ids(retriever):
    result = retriever.retrieve("solar")
    assert [e.id for e in result] == ["c1", "c2"]
    assert result[0].source == "a.txt"


def test_no_lexical_match(retriever):
    assert retriever.lexical("unmentioned") == []


def test_cross_tenant_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([Chunk(id="x", text="secret", tenant="other", source="x")], "demo", Dense(), Rerank())


@pytest.mark.parametrize("value", ["../secret", "a/b", "", "a:b", "x" * 65])
def test_safe_ids(value):
    with pytest.raises(ValueError):
        safe_id(value)


def test_namespace_collision_resistance():
    assert Settings(corpus_revision="c").namespace("a-b") != Settings(corpus_revision="b-c").namespace("a")


@pytest.fixture
def kg():
    return KnowledgeGraph(json.loads((DATA / "graph.json").read_text()), "demo")


def test_actual_graph_two_hops(kg):
    assert kg.graph.number_of_nodes() >= 20
    ids, ambiguous = kg.resolve(["L-001"])
    assert ids == ["L-001"] and not ambiguous
    facts = kg.traverse(ids)
    assert any("WORKS_AT" in e.text and "O-01" in e.text for e in facts)
    assert all(e.source for e in facts)


def test_graph_tenant_isolation():
    g = KnowledgeGraph(json.loads((DATA / "graph.json").read_text()), "other")
    assert g.resolve(["L-001"])[0] == []
    assert g.traverse(["L-001"]) == []


def test_ambiguous_entity(kg):
    kg.graph.nodes["L-002"]["aliases"].append("101 Cedar Lane")
    assert kg.resolve(["101 Cedar Lane"]) == ([], ["101 Cedar Lane"])


def test_fixture_tool_label():
    tool = Inventory(Settings(data_dir=DATA), "demo").langchain_tool()
    result = tool.invoke({"listing_id": "L-001"})
    assert result["kind"] == "tool" and "SYNTHETIC" in result["text"]
    with pytest.raises(LookupError):
        Inventory(Settings(data_dir=DATA), "other").lookup("L-001")


class FakeReasoner:
    invalid = False
    unsupported = False
    current = False

    def plan(self, question):
        return Plan(entity_mentions=["L-001"], needs_live_status=self.current)

    def draft(self, question, evidence):
        return Draft(
            claims=[
                Claim(
                    text="Synthetic test claim",
                    evidence_ids=[
                        "missing"
                        if self.invalid
                        else next((e.id for e in evidence if e.kind == "tool"), evidence[0].id)
                    ],
                )
            ]
        )

    def verify(self, question, draft, evidence):
        return Verdict(supported=not self.unsupported, reason="test verdict")


class FixtureTool:
    def invoke(self, args):
        return Evidence(id="tool:L-001", text="Synthetic status", source="fixture", kind="tool").model_dump()


@pytest.mark.parametrize("mode", ["hybrid", "multimodal", "graph"])
def test_compiled_langgraph_modes(retriever, kg, mode):
    app = compile_workflow(retriever, kg, FixtureTool(), FakeReasoner())
    result = app.invoke({"question": "L-001 garden?", "tenant": "demo", "mode": mode})["result"]
    assert not result.abstained and result.evidence
    if mode == "graph":
        assert "graph-2hop" in result.trace


@pytest.mark.parametrize("flag", ["invalid", "unsupported"])
def test_abstains_on_bad_evidence(retriever, kg, flag):
    reasoner = FakeReasoner()
    setattr(reasoner, flag, True)
    app = compile_workflow(retriever, kg, FixtureTool(), reasoner)
    assert app.invoke({"question": "solar", "tenant": "demo", "mode": "hybrid"})["result"].abstained


def test_current_question_calls_tool(retriever, kg):
    reasoner = FakeReasoner()
    reasoner.current = True
    app = compile_workflow(retriever, kg, FixtureTool(), reasoner)
    result = app.invoke({"question": "Is L-001 active?", "tenant": "demo", "mode": "hybrid"})["result"]
    assert "listing-status-tool" in result.trace


def test_tool_outage_abstains(retriever, kg):
    class BrokenTool:
        def invoke(self, args):
            raise TimeoutError()

    reasoner = FakeReasoner()
    reasoner.current = True
    app = compile_workflow(retriever, kg, BrokenTool(), reasoner)
    result = app.invoke({"question": "Is L-001 active?", "tenant": "demo", "mode": "graph"})["result"]
    assert result.abstained and "Inventory unavailable" in result.warnings[0]


def test_pdf_native_spanish():
    docs = extract(DATA / "corpus/spanish.pdf", Settings())
    assert "jardin" in docs[0].page_content and docs[0].metadata["page"] == 1


def test_pdf_image_ocr_routing(monkeypatch):
    monkeypatch.setattr("estate_rag.ingest.ocr_image", lambda image, languages: ("texte francais", 91))
    docs = extract(DATA / "corpus/french-scan.pdf", Settings())
    assert docs[0].metadata["modality"] == "ocr"
    assert docs[0].metadata["ocr_confidence"] == 91


def test_semantic_chunker_uses_vectors():
    class Embeddings:
        def embed_documents(self, texts):
            return [[1, 0] if i == 0 else [0, 1] for i, _ in enumerate(texts)]

    docs = split_documents(
        [Document(page_content="Solar roof. Elevator entrance.")], "semantic", Embeddings()
    )
    assert len(docs) == 2


def test_recursive_chunk_bound():
    import tiktoken

    docs = split_documents([Document(page_content="garden solar. " * 1500)], "recursive")
    assert len(docs) > 1
    assert max(len(tiktoken.get_encoding("cl100k_base").encode(d.page_content)) for d in docs) <= 400


def test_api_auth_and_tenant_binding():
    calls = []

    class Workflow:
        def invoke(self, state, config):
            calls.append(state)
            return {"result": Answer(answer="test", evidence=[], abstained=True)}

    app = create_app(Settings(api_tokens={"secret-test": "demo"}), lambda s, t: (Workflow(), None))
    client = TestClient(app)
    assert client.post("/query", json={"question": "hello"}).status_code == 401
    result = client.post(
        "/query",
        json={"question": "hello", "tenant": "other"},
        headers={"Authorization": "Bearer secret-test"},
    )
    assert result.status_code == 200 and calls[0]["tenant"] == "demo"
    assert (
        client.post(
            "/query", json={"question": "x"}, headers={"Authorization": "Bearer secret-test"}
        ).status_code
        == 422
    )


def test_api_fails_closed_without_tokens():
    assert (
        TestClient(create_app(Settings(api_tokens={}))).post("/query", json={"question": "hello"}).status_code
        == 503
    )


def test_api_limits_and_redacts_errors():
    class Broken:
        def invoke(self, *args):
            raise RuntimeError("secret-value")

    client = TestClient(create_app(Settings(api_tokens={"t": "demo"}), lambda s, t: (Broken(), None)))
    for _ in range(20):
        response = client.post("/query", json={"question": "hello"}, headers={"Authorization": "Bearer t"})
        assert response.status_code == 503 and "secret-value" not in response.text
    assert (
        client.post("/query", json={"question": "hello"}, headers={"Authorization": "Bearer t"}).status_code
        == 429
    )


def test_live_tool_rejects_stale_response(monkeypatch):
    import httpx

    row = dict(
        listing_id="L-001",
        tenant="demo",
        status="active",
        asking_price=1,
        currency="USD",
        updated_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    )

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            return httpx.Response(200, json=row, request=httpx.Request("GET", url))

    monkeypatch.setattr("estate_rag.tools.httpx.Client", Client)
    with pytest.raises(ValueError, match="stale"):
        Inventory(Settings(inventory_base_url="https://inventory.example.test"), "demo").lookup("L-001")


def test_vector_baseline_does_not_use_graph(retriever, kg):
    app = compile_workflow(retriever, kg, FixtureTool(), FakeReasoner(), vector_only=True)
    result = app.invoke({"question": "Which office for L-001?", "tenant": "demo", "mode": "graph"})["result"]
    assert "dense-only" in result.trace and "graph-2hop" not in result.trace


def test_current_answer_must_cite_tool(retriever, kg):
    class UncitedTool(FakeReasoner):
        current = True

        def draft(self, question, evidence):
            return Draft(claims=[Claim(text="Current status", evidence_ids=[evidence[0].id])])

    app = compile_workflow(retriever, kg, FixtureTool(), UncitedTool())
    assert app.invoke({"question": "Is L-001 active?", "tenant": "demo", "mode": "graph"})["result"].abstained


def test_ocr_low_confidence_rejected(monkeypatch):
    from PIL import Image
    from estate_rag.ingest import ocr_image

    monkeypatch.setattr("pytesseract.image_to_data", lambda *a, **k: {"text": ["unclear"], "conf": [10]})
    with pytest.raises(ValueError, match="low confidence"):
        ocr_image(Image.new("RGB", (20, 20)), "eng")


def test_manifest_cannot_escape_corpus(tmp_path):
    from estate_rag.ingest import load_corpus

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"path": "../secret.txt"}]))
    with pytest.raises(ValueError, match="inside"):
        load_corpus(manifest, "demo", Settings())
