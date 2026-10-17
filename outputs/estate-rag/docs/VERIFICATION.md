# Verification record

Verified in this task workspace using Python 3.12.15 and the dependency versions in `requirements.lock`.

| Check | Result |
|---|---|
| Editable package installation | Passed |
| Ruff static checks | Passed |
| Python bytecode compilation | Passed |
| CLI command registration | Passed |
| Offline behavioral suite | 38 passed |
| Actual compiled LangGraph routing | Passed with deterministic model/retrieval/tool doubles |
| Graph fixture | 25 nodes, 30 edges; synthetic |
| Native Spanish PDF extraction | Passed |
| Scan-to-OCR routing and low-confidence handling | Passed with mocked OCR output |
| Tenant/auth/rate-limit/error behavior | Passed through FastAPI TestClient |
| Pinecone request contracts | Passed with mocked service boundaries |
| Paid embedding/search/rerank/generation | Not run; credentials and cloud resources not configured |
| Real Tesseract OCR recognition | Not run; system executable not installed |
| Docker image build/runtime | Not run; Docker unavailable on host |
| Production latency/faithfulness targets | Not measured; release gates remain open |

PyMuPDF emitted five upstream SWIG deprecation warnings during tests. They did not fail extraction or the test suite. This verification record is a software correctness record, not an evaluation of live answer quality.
