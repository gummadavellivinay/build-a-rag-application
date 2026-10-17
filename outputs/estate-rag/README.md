# Estate RAG

A code-heavy real estate assistant using LangChain, LangGraph and Pinecone. It searches listing text and OCR text, combines semantic search with BM25, reranks candidates, and answers relationship questions using a knowledge graph and a controlled inventory tool.

**Delivery status:** implemented application and offline tests, synthetic fixtures, cloud evaluation runner, and Phase 2 deployment baseline. Paid cloud retrieval/generation and real Tesseract OCR still require your credentials/system setup and verification. This is not a claim that production acceptance criteria have passed.

## Start here

1. Read [the project guide](docs/PROJECT_GUIDE.md) for building blocks, design choices and the handout mapping.
2. Follow [Pinecone setup](docs/PINECONE.md) for the index and namespace design.
3. Use [evaluation](docs/EVALUATION.md) for baseline comparisons and [Phase 2](docs/PHASE2.md) for production release work.
4. Use [resources and alternatives](docs/RESOURCES.md) whenever a concept or tool is unfamiliar.

## Install and run offline checks

Use Python 3.12 (supported range: 3.11–3.13). Run from this project directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python scripts/make_demo.py
pytest -q
ruff check .
```

`requirements.lock` captures the tested Python dependency versions; it is not a platform-independent binary/container lock. First use of the token splitter downloads a tiktoken vocabulary. Pre-cache this in a disconnected environment.

No API keys are needed for tests. Test doubles check orchestration; they do **not** demonstrate semantic retrieval quality, actual OCR accuracy or answer faithfulness.

## Enable cloud retrieval and generation

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY, PINECONE_API_KEY and a long random API_TOKENS key.
# Confirm model access, cloud region and billing in your own accounts.
estate-rag create-index
estate-rag ingest --manifest data/manifest-text.json --tenant demo
estate-rag ask 'Which listing mentions solar panels and a private garden?'
estate-rag ask 'Which office employs the broker marketing L-001?' --mode graph
estate-rag ask 'Is L-001 currently active, and what is its asking price?' --mode graph
```

The final query calls the synthetic fixture inventory until `INVENTORY_BASE_URL` is configured. It must not be interpreted as current real market information. Generation and embedding defaults are configurable; changing the embedding model requires a fresh snapshot and a compatible index.

## Enable images and scanned PDFs

Install Tesseract plus the required language packs **outside Python**:

```bash
# macOS, if Homebrew is installed
brew install tesseract tesseract-lang
# Debian/Ubuntu alternative
# sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-fra

tesseract --list-langs
CORPUS_REVISION=v2 estate-rag ingest --manifest data/manifest.json --tenant demo
CORPUS_REVISION=v2 estate-rag ask 'Que equipement est indique dans le document L-003?' --mode multimodal
CORPUS_REVISION=v2 estate-rag ask 'How many parking spaces does the L-002 screenshot show?' --mode multimodal
```

Set `CORPUS_REVISION=v2` in `.env` before serving that revision. This OCR-first implementation retrieves **text inside visual documents**. It does not understand room layouts, photos without text, or diagrams through visual embeddings. See the vision-model alternatives in the resource guide.

## Serve the API

```bash
uvicorn estate_rag.api:app --host 127.0.0.1 --port 8000
# Put the same token configured in API_TOKENS into this local environment variable.
export ESTATE_TOKEN='your-long-random-token'
curl http://127.0.0.1:8000/query \
  -H "Authorization: Bearer $ESTATE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which office employs the broker for L-001?","mode":"graph"}'
```

Explore the API at `http://127.0.0.1:8000/docs`. There is no public upload endpoint. Ingestion is an operator CLI. Authentication determines tenant identity; a caller cannot choose another tenant in the query body.

## Evaluate

```bash
python scripts/evaluate.py --out data/eval-results.json
python scripts/evaluate.py --answers --out data/eval-results-with-answers.json
```

The corpus has 10 synthetic documents, 25 graph nodes and 30 edges. The evaluation set has 20 questions, including 10 graph questions, OCR, Spanish/French, ambiguity and missing facts. Evaluation calls paid services. Full answer evaluation runs several model calls per case; start with a small subset if you want to limit cost.

## Code map

| Building block | File |
|---|---|
| Configuration, tenant namespaces, immutable revisions | `src/estate_rag/config.py` |
| TXT/DOCX/PDF/image extraction, OCR, three chunk strategies | `src/estate_rag/ingest.py` |
| Pinecone creation, ingestion, dense search and reranking | `src/estate_rag/cloud.py` |
| BM25, reciprocal rank fusion, candidate reranking | `src/estate_rag/retrieval.py` |
| Entity resolution and two-hop property graph traversal | `src/estate_rag/knowledge_graph.py` |
| Typed inventory tool and freshness checks | `src/estate_rag/tools.py` |
| LangChain prompt chains and LangGraph orchestration | `src/estate_rag/workflow.py` |
| Snapshot integrity and dependency assembly | `src/estate_rag/runtime.py` |
| Operator commands | `src/estate_rag/cli.py` |
| Authenticated FastAPI deployment baseline | `src/estate_rag/api.py` |
| Synthetic data generation and evaluation | `scripts/` |
| Behavioral tests | `tests/` |

The handout is a requirements reference, not the real estate knowledge corpus. No submission, email, GitHub publication or form interaction has been performed.
