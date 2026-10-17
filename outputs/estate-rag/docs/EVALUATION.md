# Evaluation and verification

## What was verified locally

The application was installed into an isolated Python 3.12 environment. Behavioral tests exercise the actual compiled LangGraph workflow with deterministic model/retrieval/tool doubles, plus native PDF extraction, chunking, graph traversal and API controls. The test run and static analysis results are recorded in `VERIFICATION.md`.

The model doubles are not semantic models. No live Pinecone index was provisioned; no paid embedding, reranking or generation evaluation was run. Tesseract was not installed on the host, so OCR routing was tested with an injected OCR response; actual OCR recognition remains a setup/acceptance step. The provided scan and screenshot fixtures make that test reproducible once Tesseract is installed.

## Labelled evaluation set

`data/eval.json` contains twenty questions with expected facts, expected sources and abstention labels:

| Cases | Focus | Expected behavior |
|---|---|---|
| Q01–Q06 | Listing → broker → office | Identify the correct two-hop office, with evidence. |
| Q07–Q10 | Reverse edges and relationship combinations | List canonical entities supported by graph paths. |
| Q11–Q13 | Exact ID and listing features | Retrieve the correct listing brochure. |
| Q14 | English question over Spanish PDF | Retrieve and cite original-language document. |
| Q15 | French question over French scan | OCR text supports the answer. |
| Q16 | Screenshot-only parking count | Cite the screenshot evidence. |
| Q17 | Current availability/price intent | Route through the inventory tool; explicitly label synthetic fixture and timestamp. |
| Q18 | Future sale price | Abstain; no supported prediction. |
| Q19 | Missing listing | Abstain or ask for a valid entity. |
| Q20 | Undocumented exact distance | Abstain; “near” does not imply a numeric distance. |

These are synthetic labels, not a representative real-world benchmark. Expand with real approved content, more languages, contradictions, OCR errors, mixed scripts and access-control cases before acceptance.

## Retrieval comparisons

Run `python scripts/evaluate.py`. It evaluates the same questions under:

1. Dense retrieval alone.
2. BM25 alone.
3. Hybrid RRF without reranking.
4. Hybrid RRF with reranking.

Document-level recall@6 measures the fraction of expected source documents found; MRR measures the position of the first expected source; nDCG@6 discounts lower-ranked relevant sources. Chunk results from the same source are deduplicated. With fewer than six unique source documents, the metric uses those retrieved; it does not fill in missing ranks.

Cases without expected documents have `null` retrieval metrics, not a misleading zero. Their quality is assessed through tool/abstention behavior and human review. The runner records per-call latency, but the small sequential benchmark is not a production load test.

To assess chunking, use fixed/recursive/semantic snapshots as described in `PROJECT_GUIDE.md`, and compare exactly the same labels. Report corpus revision, model versions, dimensions, chunk counts and language packs with every result.

## Graph RAG versus vector RAG

`python scripts/evaluate.py --answers` runs both a dense-only answer workflow and the full graph workflow on Q01–Q10. The dense-only baseline disables graph augmentation, reranking and tool execution. The graph workflow uses hybrid retrieval, graph evidence and any allowed tools. This compares end-to-end systems, so differences cannot be attributed solely to graph traversal; for a controlled graph-only ablation, hold retrieval/reranking fixed and disable only graph augmentation in a further experiment.

Read both answers and mark correctness, completeness and citation support. A vector-only system can still answer relationships present explicitly in `relationships.txt`; do not assume graph RAG wins every case. The graph advantage should be tested on missing exact wording and multi-hop composition, while retaining the same source facts for fairness.

## Faithfulness rubric

The response validator checks citation identities; that is not the same as source entailment. Use human review of atomic claims:

```text
faithfulness = supported factual claims / all factual claims
answer completeness = required supported facts supplied / required facts in the label
abstention accuracy = correct abstain-or-answer decisions / all cases
```

Leave faithfulness undefined for an empty answer and score abstention separately. A tool fact must match the tool payload, listing ID and timestamp. A graph fact must match edge direction and provenance. A translated claim must preserve the original meaning.

The JSON output contains `human_faithfulness: null` until you review it. It deliberately does not manufacture an LLM-judge accuracy score or fill in unmeasured results.

## Report template

For each case record question ID, corpus revision, retrieval variant, expected sources, returned sources, retrieval metrics, answer, cited evidence, human correctness/faithfulness, abstention, latency and failure category.

Use failure categories such as OCR omission, bad chunk boundary, lexical mismatch, semantic miss, rerank displacement, unresolved entity, missing relationship, stale tool, unsupported generation or incorrect refusal. Correct the relevant building block, then rerun the unchanged evaluation set and a held-out set.

For the final project recording, show actual results and at least one failure. Do not describe targets from the project one-liner as achieved measurements.
