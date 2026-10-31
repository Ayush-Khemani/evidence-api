# Evidence API — your first AI engineering project

A Python backend that loads technical documents, retrieves passages, and answers questions with evidence. Start by tracing the pipeline, then measure how real embeddings and an LLM change its behavior.

This follows the starting project agreed in your “Upgrading In AI” chat: a small RAG system, direct model calls, visible retrieval, and evaluation before adding agents or a frontend.

## Start here (Windows PowerShell)

Run these commands from this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,pdf]"
.\.venv\Scripts\python.exe -m scripts.demo
.\.venv\Scripts\python.exe -m pytest -q
```

The demo uses an isolated temporary database. No API key, Docker, or model calls are needed. It hashes words into vectors and copies the best evidence passage. **It is a pipeline demonstration, not semantic RAG or a test of an LLM.** Every answer includes `mode: "demo"`.

Run the API:

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn evidence.api:app --host 127.0.0.1 --port 8000
```

Open [API documentation](http://127.0.0.1:8000/docs). In `POST /documents`, try:

```json
{"filename":"handbook.md","text":"Retry failed payment requests at most three times."}
```

Then use `POST /questions`:

```json
{"question":"How many payment retries?","top_k":4}
```

Use `POST /retrieval` with the same body to inspect retrieval without generation. `GET /documents/{id}` shows the stored passages. Delete your upload when finished.

To upload a file (raw bytes, not multipart):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload?filename=payments.md" -H "Content-Type: application/octet-stream" --data-binary "@sample_docs/payments.md"
```

Supported: UTF-8 Markdown/text, text PDFs with the PDF extra installed. Physical PDF page numbers are preserved; text and Markdown have `page: null`. Scans need OCR and are outside this milestone. Uploads are synchronous: a successful response means all passages are indexed. Failed indexing commits no document. Limits: 2 MB, 100 PDF pages, 250 passages.

## Turn on real RAG

Edit your local `.env`:

```dotenv
RAG_MODE=openai
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
SQLITE_PATH=openai.sqlite3
```

Restart the API and re-upload your documents. Keep the key in `.env`, never in source control or chat. This mode makes paid model API requests. Model availability depends on your API account; the model name is configurable. The example is a configurable baseline, not a claim about the newest or best model.

The real path calls the embeddings endpoint, retrieves by cosine similarity, and calls the Responses API with a strict answer schema. SQLite supports real embeddings too, using an exact vector scan in Python. Start there if Docker is unavailable. Database metadata prevents mixing demo vectors and real embeddings, or embeddings from different models/dimensions. Changing the embedding configuration requires a new database and re-indexing.

The server resolves filenames, page numbers, and evidence quotes from its own storage. The model supplies only citation IDs. Unknown IDs and answers without citations are rejected. **Valid citations do not prove that the answer is supported.** Review answer grounding separately.

## PostgreSQL and pgvector

When Docker Desktop is available:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[postgres]"
docker compose up -d --wait db
```

Add to `.env`:

```dotenv
DATABASE_URL=postgresql://rag:rag-local-only@localhost:5432/rag
```

Restart the API. A configured `DATABASE_URL` selects Postgres; otherwise SQLite is used. This local database has separate data from SQLite, so re-upload documents. Use one embedding configuration per database. Local credentials in Compose are for local development.

The store uses pgvector's cosine operator and exact search. There is deliberately no approximate vector index for this small corpus. Later, compare exact search against HNSW on a larger dataset and measure recall before choosing an index. Connections are opened per operation for clarity; connection pooling and migrations are later work.

## Evaluate before improving

Run a free pipeline baseline:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --demo --output reports/demo-baseline.json
```

To evaluate a running real API, start it against an **empty** database and run:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --output reports/real-baseline.json
```

The evaluator uploads three fictional handbooks, asks 20 questions (14 answerable, 6 unanswerable), and removes only the documents it uploaded. It refuses a nonempty corpus to avoid contaminating the result. Real mode makes paid embedding and answer calls. Reports contain document text and answers; inspect before sharing.

Recorded metrics: expected-evidence retrieval hit rate at k, abstention on unanswerable questions, response rate on answerable questions, and median request latency. Expected evidence uses filename plus a gold text snippet so it survives new passage IDs. Grounding review remains explicitly pending. `generation_usage` excludes embedding tokens; the report does not claim a total dollar cost. Use provider usage and current prices for that.

The supplied corpus is tiny and often puts an entire handbook in one passage. A high retrieval hit rate at k=4 is weak evidence because all three handbooks fit. Try `--top-k 1`, add distractor documents, and write new questions before making portfolio claims. The score threshold is a heuristic, not a probability or a reliable proof of missing evidence. Calibrate it on a development set, then evaluate on held-out questions.

## Read the code in this order

1. `evidence/documents.py`: extraction, page boundaries, overlapping word windows.
2. `evidence/providers.py`: what the demo does and how real embedding/generation requests differ.
3. `evidence/store.py`: persistence, cosine similarity, and pgvector queries.
4. `evidence/service.py`: upload → embed → retrieve → generate → resolve sources.
5. `evidence/api.py`: requests, validation, limits, lifecycle, and errors.
6. `scripts/evaluate.py`: metrics and the boundary between automated checks and human judgment.

See [LEARNING_PLAN.md](LEARNING_PLAN.md) for your first sessions and experiments. See [VERIFICATION.md](VERIFICATION.md) for what was actually run.

## Current scope

This is a local, single-user learning backend. It has no authentication, tenant isolation, OCR, background workers, deployment configuration, or production monitoring. Document-scoped retrieval is a search filter, not authorization. Bind it to localhost while learning. Uploaded text is treated as evidence in the model instructions, but prompt injection resistance is not guaranteed. Word-based chunking can still exceed provider token limits for pathological inputs; failures are returned as provider errors without partial indexing.

Next extensions after the baseline: measured chunking improvements, token accounting, PDF extraction evaluation, then a narrow read-only tool call. Add write actions with an explicit review step when you reach that stage.

## Official references checked

- [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings): embedding requests and dimensions.
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs): strict Responses API output schema.
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/): startup and cleanup.
- [pgvector](https://github.com/pgvector/pgvector): cosine distance and exact vector search.
- [Psycopg usage](https://www.psycopg.org/psycopg3/docs/basic/usage.html): connection contexts and transactions.

