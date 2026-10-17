import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


def embeddings(settings):
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.openai_api_key.get_secret_value(),
        request_timeout=30,
        max_retries=2,
        check_embedding_ctx_length=True,
    )


def chat(settings):
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=30,
        max_retries=2,
        temperature=0,
    )


def pinecone(settings):
    return Pinecone(api_key=settings.pinecone_api_key.get_secret_value())


def ensure_index(settings):
    pc = pinecone(settings)
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            vector_type="dense",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            deletion_protection="enabled",
        )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        desc = pc.describe_index(settings.pinecone_index)
        if desc.dimension != settings.embedding_dimensions or desc.metric != "cosine":
            raise ValueError("Existing index dimension/metric mismatch; create a new versioned index")
        if desc.status["ready"]:
            return desc.host
        time.sleep(2)
    raise TimeoutError("Pinecone index did not become ready within 180 seconds")


class PineconeDense:
    def __init__(self, settings, embedder=None):
        self.settings = settings
        self.embedder = embedder or embeddings(settings)
        pc = pinecone(settings)
        desc = pc.describe_index(settings.pinecone_index)
        if desc.dimension != settings.embedding_dimensions or desc.metric != "cosine":
            raise ValueError("Embedding/index mismatch")
        self.index = pc.Index(host=desc.host)

    def upsert(self, chunks, tenant):
        namespace = self.settings.namespace(tenant)
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            self.index.upsert(
                vectors=[
                    {
                        "id": c.id,
                        "values": v,
                        "metadata": {
                            "tenant": tenant,
                            "source": c.source,
                            "page": c.page,
                            "listing_id": c.listing_id,
                            "language": c.language,
                            "modality": c.modality,
                        },
                    }
                    for c, v in zip(batch, vectors)
                ],
                namespace=namespace,
                _request_timeout=30,
            )
        # Visibility gate: don't publish the local snapshot before every record is fetchable.
        deadline = time.monotonic() + 120
        ids = [c.id for c in chunks]
        while time.monotonic() < deadline:
            found = set()
            for start in range(0, len(ids), 100):
                response = self.index.fetch(
                    ids=ids[start : start + 100], namespace=namespace, _request_timeout=30
                )
                found.update(response.vectors)
            if set(ids) <= found:
                return
            time.sleep(2)
        raise TimeoutError("Upserts are not yet visible; retry the same ingestion revision")

    def search(self, question, tenant, k):
        result = self.index.query(
            vector=self.embedder.embed_query(question),
            top_k=k,
            namespace=self.settings.namespace(tenant),
            filter={"tenant": {"$eq": tenant}},
            include_metadata=False,
            _request_timeout=30,
        )
        return [match.id for match in result.matches]


class PineconeReranker:
    def __init__(self, settings):
        self.pc, self.model = pinecone(settings), settings.rerank_model

    def rank(self, question, chunks, k):
        if not chunks:
            return []
        result = self.pc.inference.rerank(
            model=self.model,
            query=question,
            documents=[{"id": c.id, "text": c.text} for c in chunks],
            top_n=min(k, len(chunks)),
            return_documents=False,
            parameters={"truncate": "END"},
        )
        return [(chunks[r.index], r.score) for r in result.data]
