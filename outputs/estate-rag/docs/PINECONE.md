# Pinecone index setup and options

## Recommended implementation

The runnable default uses **one dense Pinecone serverless index plus local BM25**, then RRF and a Pinecone-hosted reranker. It does not store BM25 vectors in Pinecone. This keeps corpus statistics, Unicode tokenization and experiments explicit for a small teaching corpus. Pinecone remains the semantic vector database.

| Setting | Default | Reason |
|---|---|---|
| Name | `estate-dense-v1` | Version when the embedding contract changes. |
| Vector type | Dense | Application supplies embedding vectors. |
| Dimension | 1536 | Explicit output size of configured embedding model. |
| Metric | Cosine | Similarity contract used throughout this project. |
| Deployment | Serverless, AWS, `us-east-1` | Example configuration; choose a region offered by your plan and consistent with data residency. |
| Namespace | `4:demo:v1` | Length-prefixed tenant plus corpus revision. |
| Metadata | Tenant, source, page, listing ID, language, modality | Supports identity and future metadata filters. |
| Raw chunk text | Local immutable snapshot | Avoids putting full source text in vector metadata. Phase 2 moves this to durable storage. |

Create a project and API key in the [Pinecone console](https://app.pinecone.io/). Put the key in `.env`, choose a supported region, and run `estate-rag create-index`. This command can create a billable resource. The setup code validates existing dimension and metric and waits for readiness; it never silently deletes a mismatched index.

Equivalent core SDK code:

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="read-from-your-secret-store")
pc.create_index(
    name="estate-dense-v1",
    vector_type="dense",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    deletion_protection="enabled",
)
```

Use the implementation in `cloud.py` for existence/readiness validation. If your plan requires a different region, edit `.env` before creation. Account availability and billing have not been verified here. [Official index creation guide](https://docs.pinecone.io/guides/index-data/create-an-index).

## Namespace and revision discipline

Ingestion and queries use exactly the same namespace. Authentication supplies the tenant; neither the LLM nor the request body selects it. Local BM25 and graph data are likewise tenant scoped. [Pinecone multitenancy](https://docs.pinecone.io/guides/index-data/implement-multitenancy).

After changed or deleted source files, build a fresh corpus revision. Upsert the complete new revision, wait for record visibility, publish the snapshot, then switch the serving revision. A rollback returns the service to the previous revision. Retire the obsolete namespace and snapshot only after your retention period and verification. The code does not automatically delete old data.

Ingestion is single-writer per tenant/revision, enforced by a local exclusive lock. A killed worker can leave a lock file; confirm no writer is active before removing it. Retrying a partially written revision with changed source content can leave unreachable remote IDs; use a fresh revision after changing failed input. Final retrieval only accepts IDs present in the local published snapshot.

Pinecone upserts and queries are not one transactional commit with the local disk. Visibility checks reduce the publication gap; production needs a revision registry and reconciliation worker. [Upsert documentation](https://docs.pinecone.io/guides/index-data/upsert-data).

## Options when scaling hybrid retrieval

| Option | Advantages | Tradeoff | Recommendation |
|---|---|---|---|
| Dense Pinecone + local BM25 | Simple, exact corpus-level BM25, easy ablations | Every API process loads corpus into RAM; rebuild on updates | Implemented Phase 1 default. |
| Dense Pinecone + OpenSearch BM25 | Distributed text search, analyzers, filters | Two services; synchronize revision and permissions | Good Phase 2 choice for growing multilingual corpora. |
| Separate dense and sparse Pinecone indexes | Independent retrieval, merge IDs with RRF | Extra index and ingestion contract; sparse model may not be BM25 or multilingual | Use when consolidating vector infrastructure matters. |
| Single dense+sparse index | One query with hybrid weights | Index compatibility and score calibration; BM25 statistics lifecycle still matters | Evaluate only with supported metric/API and measured alpha tuning. |
| Pinecone native full-text/BM25 capabilities | Potentially consolidates lexical and semantic data | API/plan/features differ from the pinned classic vector SDK path | Check current docs and benchmark before replacing the working adapter. |

See [Pinecone hybrid search](https://docs.pinecone.io/guides/search/hybrid-search) and [OpenSearch hybrid search](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/).

If using a separate bring-your-own sparse index, the SDK creation pattern is:

```python
pc.create_index(
    name="estate-sparse-v1", vector_type="sparse", metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

This snippet creates only the optional index; it is **not** a second implemented retrieval adapter. Do not create it for the default app. A sparse index has sparse indices/values rather than a fixed dense dimension. Use identical chunk IDs and tenant/revision namespaces in both stores. Fit and persist BM25 corpus statistics if encoding BM25 yourself; a learned sparse encoder is not the same algorithm as BM25. Review language support before choosing a sparse model.

## Reranking

`PineconeReranker` sends candidate text to `pc.inference.rerank`, selecting `bge-reranker-v2-m3` by default. Data therefore leaves the local process for this service as well as for embeddings/generation. Confirm account access, retention requirements and model language coverage. Alternatives include a local cross-encoder or a different hosted multilingual reranker. [Pinecone rerank guide](https://docs.pinecone.io/guides/search/rerank-results).

## Troubleshooting

- Dimension mismatch: confirm model and dimensions; create a fresh index rather than truncating vectors by hand.
- Empty search: confirm credentials, exact namespace/revision, successful snapshot publication and ingestion visibility.
- Poor cross-language recall: evaluate the embedding model on your languages; BM25 is same-language here.
- Model access/rate limits: select an available model and measure throughput; do not replace failed retrieval with invented answers.
- Large local snapshot: move lexical retrieval and text storage to durable services before multiplying workers.
