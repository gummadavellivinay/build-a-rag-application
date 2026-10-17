"""Single-process deployment baseline; see docs/PHASE2.md before public deployment."""

import hashlib
import logging
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from functools import lru_cache
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .config import Settings, safe_id
from .models import Query, Answer
from .runtime import build_runtime

log = logging.getLogger("estate_rag")


def create_app(settings=None, runtime_factory=build_runtime):
    settings = settings or Settings()
    app = FastAPI(title="Estate RAG", version="0.1.0")
    bearer = HTTPBearer(auto_error=False)
    lock, slots = threading.Lock(), threading.BoundedSemaphore(4)
    requests = defaultdict(deque)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not settings.api_tokens:
            raise HTTPException(503, "API authentication is not configured")
        if credentials is None:
            raise HTTPException(401, "Bearer token required")
        tenant = None
        for token, identity in settings.api_tokens.items():
            if secrets.compare_digest(credentials.credentials, token):
                tenant = safe_id(identity)
        if tenant is None:
            raise HTTPException(401, "Invalid bearer token")
        with lock:
            now = time.monotonic()
            queue = requests[tenant]
            while queue and queue[0] < now - 60:
                queue.popleft()
            if len(queue) >= 20:
                raise HTTPException(429, "Tenant rate limit exceeded", headers={"Retry-After": "60"})
            queue.append(now)
        return tenant

    @lru_cache(maxsize=32)
    def runtime(tenant):
        return runtime_factory(settings, tenant)[0]

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(tenant=Depends(authenticate)):
        try:
            runtime(tenant)
            return {"status": "ready", "revision": settings.corpus_revision}
        except Exception:
            raise HTTPException(503, "Runtime is not ready") from None

    @app.post("/query", response_model=Answer)
    def query(body: Query, tenant=Depends(authenticate)):
        request_id, start = str(uuid.uuid4()), time.monotonic()
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "Capacity exceeded; retry later", headers={"Retry-After": "5"})
        try:
            answer = runtime(tenant).invoke({**body.model_dump(), "tenant": tenant}, {"recursion_limit": 12})[
                "result"
            ]
            # No prompt, source content, API token, or raw tenant identifier is logged.
            log.info(
                "request=%s tenant_hash=%s mode=%s elapsed_ms=%d abstained=%s",
                request_id,
                hashlib.sha256(tenant.encode()).hexdigest()[:12],
                body.mode,
                int((time.monotonic() - start) * 1000),
                answer.abstained,
            )
            return answer
        except Exception:
            log.warning("request=%s dependency_failure=true", request_id)
            raise HTTPException(503, f"Dependency unavailable; reference {request_id}") from None
        finally:
            slots.release()

    return app


app = create_app()
