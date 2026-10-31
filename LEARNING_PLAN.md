# Build understanding alongside the code

Your first target is to explain and diagnose one working RAG pipeline. The starter gives you something concrete to inspect; your experiments and measured findings make it your project.

## Session 1 — trace the evidence (45–60 minutes)

Run `python -m scripts.demo`. Find the payment question's top passage, score, and source quote. Follow that passage through `ingest`, `retrieve`, and `answer` in `evidence/service.py`.

Answer in your own words:

- What is stored during ingestion? What runs only when a question arrives?
- Why is an embedding a vector? Why does the demo's word hashing fail on synonyms?
- What information is lost when we normalize whitespace?
- Why is `page` null for Markdown?
- Why must the server resolve source metadata instead of trusting the model?

Small change to make yourself: add a second sample document with a different retry policy. Ask the question again. Explain why retrieval order and document scope matter.

## Session 2 — real embeddings and generation (60 minutes)

Configure OpenAI mode with a separate database, restart, and upload the same sample documents. Inspect `/retrieval` before `/questions`. Compare direct wording with a paraphrase such as “How often should I replace my credentials?”

Explain the difference between semantic retrieval and text generation. Identify exactly what is sent to the provider. Explain why this is not model training. Inspect generation usage and check provider usage for embedding requests.

## Session 3 — find failures (60 minutes)

Run the 20-question evaluation against an empty real-mode database. Read every incorrect answer and abstention. For each failure, assign a cause: extraction, chunking, retrieval, threshold, generation, or ambiguous gold label. Review cited text against every factual claim.

Do not treat a correct source ID as proof of grounding. Record at least one case where the relevant document exists but the question cannot be answered.

## Sessions 4–5 — one controlled experiment

Choose a variable: passage size (60 versus 120 words), top k (1 versus 4), or overlap (0 versus 20 words). Keep the other settings fixed. For chunking changes, re-index into a new empty database. Repeat the same questions and save separate reports. Use a separate held-out set after tuning.

Write a short experiment note:

| Change | Expected effect | Actual result | Example failure | Decision |
|---|---|---|---|---|
| Fill this with your experiment | | | | |

Add at least five distractor documents and five questions you wrote yourself. A three-document fixture is useful for plumbing, but too easy to establish retrieval quality.

## Sessions 6–7 — make it discussable

Try PostgreSQL/pgvector, run the same evaluation, and compare behavior with exact SQLite search. Add an integration test against a disposable Postgres database. Explain why an approximate index is unnecessary for a tiny corpus.

Record a two-minute demonstration: a supported answer, an unanswerable question, retrieved evidence, and a failure you measured and improved. Write the README findings using actual numbers and limitations.

Before adding agents, explain the entire document → passage → vector → nearest neighbors → prompt → answer → citation path without reading the code. Then add one read-only tool, such as looking up a payment status in a fictional local dataset.

## Keep the rest of your preparation moving

The earlier chat proposed roughly one hour building, forty minutes DSA/SQL, and twenty minutes explaining a concept. Use that as a flexible starting rhythm. The next milestone is a measured real-model baseline, not more technologies.

