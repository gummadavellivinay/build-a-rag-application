# Resources and design alternatives

Official documentation and original project/research resources are preferred. Core integration pages were checked during implementation; remaining links are further-reading entry points, not a claim that every option has been implemented or benchmarked. Provider availability, prices and APIs can change.

## Implemented building blocks

| Concept or tool | What it contributes | Read more |
|---|---|---|
| RAG | Retrieve external evidence before generating | [Original RAG paper](https://arxiv.org/abs/2005.11401) |
| LangChain | Documents, splitters, model interfaces, prompts and tools | [Overview](https://docs.langchain.com/oss/python/langchain/overview), [tools](https://docs.langchain.com/oss/python/langchain/tools) |
| LangGraph | Explicit typed state and conditional workflow routing | [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| Structured output and Pydantic | Schema-validated plans, claims and tool results | [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output), [Pydantic](https://docs.pydantic.dev/latest/) |
| Recursive splitting | Preserve larger structural boundaries before splitting further | [LangChain recursive splitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) |
| Token splitting and tiktoken | Count model tokens rather than characters | [Token splitters](https://docs.langchain.com/oss/python/integrations/splitters/split_by_token), [tiktoken](https://github.com/openai/tiktoken) |
| Semantic chunking | Group nearby sentences by embedding similarity | [LangChain semantic chunking](https://docs.langchain.com/oss/python/integrations/splitters/semantic-chunker); this project's threshold algorithm is its own small implementation |
| Dense embeddings | Map meaning into vectors | [OpenAI embedding guide](https://developers.openai.com/api/docs/guides/embeddings), [LangChain OpenAI embedding integration](https://docs.langchain.com/oss/python/integrations/text_embedding/openai) |
| ChatOpenAI | Configurable generation client | [Integration](https://docs.langchain.com/oss/python/integrations/chat/openai) |
| Pinecone | Managed semantic vector retrieval | [Create index](https://docs.pinecone.io/guides/index-data/create-an-index), [upsert](https://docs.pinecone.io/guides/index-data/upsert-data), [tenants](https://docs.pinecone.io/guides/index-data/implement-multitenancy) |
| BM25 | Term-frequency/rarity and document-length lexical ranking | [rank_bm25 implementation](https://github.com/dorianbrown/rank_bm25), [BM25 foundations](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) |
| Reciprocal Rank Fusion | Merge ranking lists without mixing score scales | [Original RRF paper](https://dl.acm.org/doi/10.1145/1571941.1572114) |
| Cross-encoder reranking | Jointly score question and candidate passage | [Retrieve and rerank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html), [Pinecone reranking](https://docs.pinecone.io/guides/search/rerank-results) |
| BGE reranker | Multilingual candidate scoring model | [Model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Tesseract and pytesseract | Local OCR with installed language packs | [Command-line and languages](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html), [Python adapter](https://github.com/madmaze/pytesseract) |
| PyMuPDF | PDF text extraction and page rendering | [OCR recipes](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html), [documentation](https://pymupdf.readthedocs.io/) |
| Pillow | Image decoding and orientation | [Handbook](https://pillow.readthedocs.io/en/stable/handbook/index.html) |
| python-docx | Extract Word paragraphs/tables | [Documentation](https://python-docx.readthedocs.io/) |
| NetworkX and breadth-first traversal | Local typed property graph and bounded hops | [Traversal](https://networkx.org/documentation/stable/reference/algorithms/traversal.html), [MultiDiGraph](https://networkx.org/documentation/stable/reference/classes/multidigraph.html) |
| FastAPI and Uvicorn | HTTP API and ASGI server | [Deployment concepts](https://fastapi.tiangolo.com/deployment/concepts/), [Uvicorn settings](https://www.uvicorn.org/settings/) |
| HTTPX | Fixed-endpoint authenticated tool client | [Timeouts](https://www.python-httpx.org/advanced/timeouts/) |
| pytest and Ruff | Behavioral regression tests and static lint | [pytest](https://docs.pytest.org/), [Ruff](https://docs.astral.sh/ruff/) |
| Docker and Compose | Reproducible process packaging | [Docker build](https://docs.docker.com/build/), [Compose](https://docs.docker.com/compose/) |
| GitHub Actions | Automated checks on changes | [Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) |

## Choices and recommendations

| Decision | Options and links | Recommendation for this project |
|---|---|---|
| Orchestration | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), [LlamaIndex workflows](https://developers.llamaindex.ai/python/workflows/), [Haystack](https://docs.haystack.deepset.ai/docs/intro), [n8n](https://docs.n8n.io/advanced-ai/) | LangGraph, as requested; explicit routing makes tool and abstention behavior testable. |
| Chunk strategy | Fixed tokens; recursive; semantic; [parent-document retrieval](https://docs.langchain.com/oss/python/integrations/retrievers/parent_document_retriever); [Docling layout extraction](https://docling-project.github.io/docling/) | Start recursive, compare fixed and semantic using included code; use layout/table parsing when brochures contain structured tables. |
| Embedding provider | [OpenAI](https://developers.openai.com/api/docs/guides/embeddings), [multilingual Sentence Transformers](https://sbert.net/examples/sentence_transformer/training/multilingual/README.html), [BGE-M3](https://huggingface.co/BAAI/bge-m3), [Cohere](https://docs.cohere.com/docs/cohere-embed) | Configurable hosted default for setup simplicity; local multilingual embeddings when privacy or volume justifies infrastructure. Rebuild indexes when changing models. |
| Lexical retrieval | Local rank_bm25; [OpenSearch BM25](https://docs.opensearch.org/latest/search-plugins/keyword-search/); [Pinecone hybrid](https://docs.pinecone.io/guides/search/hybrid-search) | Local for the demo, language-aware distributed lexical search for a larger corpus. |
| Fusion | RRF; weighted normalized scores; single-index dense/sparse alpha | Start RRF; tune weights only on held-out relevance labels. Raw cosine and BM25 scores are not comparable. |
| Reranker | Hosted BGE; [local CrossEncoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html); [Cohere Rerank](https://docs.cohere.com/docs/reranking-with-cohere) | Hosted multilingual default; measure latency/cost before switching to a local model. |
| OCR | Tesseract; [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR); [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview); [Google Document AI](https://cloud.google.com/document-ai/docs/overview) | Tesseract for the teaching pipeline; benchmark layout/table-heavy and low-quality scans before selecting a production service. |
| Visual understanding | OCR text; [OpenAI image input](https://developers.openai.com/api/docs/guides/images-vision); [ColPali](https://github.com/illuin-tech/colpali) | OCR satisfies text-inside-images requirements. Add vision/visual retrieval for diagrams and floor plans only with labelled tests. |
| Multilingual strategy | Original-language embeddings; query translation; dual original/translated chunks | Keep originals for citations; add translation as an evaluated recall expansion, never as authoritative replacement text. |
| Graph storage | NetworkX; [Neo4j Python driver](https://neo4j.com/docs/python-manual/current/); [Amazon Neptune](https://docs.aws.amazon.com/neptune/latest/userguide/intro.html) | NetworkX for the provided graph; durable graph service when concurrent updates/scale require it. |
| Graph retrieval style | Local entity traversal; [Microsoft GraphRAG](https://microsoft.github.io/graphrag/); vector-only retrieval | Local traversal for listing-broker-office paths. Community summaries serve broad corpus-wide themes and are a different design. |
| Entity extraction | Canonical source records; LLM structured extraction; [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) | Prefer canonical listing/broker IDs. Use model extraction to propose records with exact source spans; review merges. |
| Evaluation | Human claim review; [Ragas](https://docs.ragas.io/); [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation) | Deterministic retrieval metrics plus human faithfulness labels first; model judges are secondary signals. |
| Auth | Static service tokens; [OIDC](https://openid.net/developers/how-connect-works/); [FastAPI OAuth2/JWT](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/) | Static tokens only for local/pilot use; tenant membership via OIDC gateway for production. |
| Storage and jobs | Local snapshots; [S3 versioning](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html); [PostgreSQL transactions](https://www.postgresql.org/docs/current/tutorial-transactions.html); [Celery](https://docs.celeryq.dev/) | Immutable objects + transactional revision registry + durable jobs in Phase 2. |
| Limits and observability | [Redis rate limiter](https://redis.io/learn/develop/java/spring/rate-limiting/fixed-window); [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/); [LangSmith observability](https://docs.langchain.com/langsmith/observability) | Distributed gateway limits and redacted stage traces before horizontal scaling. |
| Dependency/release tooling | [uv](https://docs.astral.sh/uv/); [pip requirements](https://pip.pypa.io/en/stable/reference/requirements-file-format/); [Dependabot](https://docs.github.com/en/code-security/dependabot) | Keep tested versions pinned; update in CI with live compatibility and evaluation checks. |

## Suggested reading order

1. RAG paper and LangChain retrieval concepts.
2. Recursive splitting, embeddings, BM25 and RRF.
3. Pinecone index/namespace setup and reranking.
4. OCR language configuration and document extraction.
5. LangGraph state/routing, typed tools and graph traversal.
6. Evaluation, authentication, storage and production operations.

Every option should earn its added latency, cost and maintenance through measured improvement. This repository does not claim one universally best chunk size, graph strategy or model.
