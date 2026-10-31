# Verification — 31 October 2026

## Executed

- Python 3.12; 23 pytest cases passed, including actual text-PDF extraction and physical page references using pypdf 6.21.4.
- In-process FastAPI upload → SQLite persistence → retrieval → demo excerpt → source resolution → deletion.
- Validation of unsupported, empty, non-UTF-8, null-containing, and oversized uploads; empty and invalid questions.
- Passage overlap and page boundaries, document search scope, empty-corpus abstention, and zero-vector queries.
- Atomic ingestion on provider failure, embedding-configuration mismatch, and rejection of invented citation IDs.
- Mock HTTP checks of OpenAI embedding ordering, Responses API schema request, usage extraction, malformed output, incomplete output, and refusal handling. These are contract tests, not live-provider tests.
- `python -m scripts.demo` completed successfully.
- `python -m scripts.evaluate --demo --output reports/demo-baseline.json` completed all 20 cases and cleaned up its uploaded fixtures.

## Demo baseline

| Measure | Result |
|---|---:|
| Expected evidence retrieved at k=4 | 14 / 14 |
| Unanswerable questions abstained | 3 / 6 |
| Answerable questions receiving an excerpt | 12 / 14 |
| Median request latency on this run | 1.36 ms |

These results describe deterministic word hashing and copied excerpts, not a learned embedding model or LLM. All three handbooks fit within k=4, so retrieval hit rate is an easy plumbing check. Grounding review is pending. Do not present these as RAG model quality or production latency measurements.

## Not executed

- Live OpenAI embeddings or generation: no API credential was supplied; no paid model calls were made.
- PostgreSQL/pgvector or Docker Compose: Docker is not installed in this environment. The SQL backend is included but has not been integration-tested.
- Browser/API-docs interaction, production deployment, authentication, tenant isolation, large-file performance, OCR, or adversarial prompt-injection evaluation.

The Windows sandbox initially blocked the loopback socket used by asyncio/TestClient. Tests were rerun successfully after network permission was granted. Optional PDF dependencies were installed under the task's scratch folder for verification, without changing global Python packages.

## Next verification

Configure a real provider locally, run the same evaluation against an empty database, review every answer against its sources, and save a distinct report. Then repeat against PostgreSQL before describing that backend as tested.

