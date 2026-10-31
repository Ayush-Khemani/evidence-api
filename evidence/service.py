from time import perf_counter
from uuid import uuid4

from .documents import chunk_pages, extract, safe_name
from .providers import ProviderError


class RAGService:
    def __init__(self, settings, store, provider):
        self.settings, self.store, self.provider = settings, store, provider

    def ingest(self, filename, content):
        if len(content) > self.settings.max_bytes:
            raise ValueError("File exceeds 2 MB limit")
        filename = safe_name(filename)
        chunks = chunk_pages(extract(filename, content), self.settings.chunk_words, self.settings.overlap)
        if len(chunks) > self.settings.max_chunks:
            raise ValueError("Too many passages; use a smaller document (max 250 chunks)")
        for chunk in chunks:
            chunk["id"] = str(uuid4())
        vectors = self.provider.embed([chunk["text"] for chunk in chunks])
        doc = {"id": str(uuid4()), "filename": filename, "status": "ready", "chunk_count": len(chunks)}
        # Commit only after all embeddings succeed: no partly indexed documents.
        self.store.save(doc, chunks, vectors)
        return doc

    def retrieve(self, question, top_k=4, document_id=None):
        if document_id and not self.store.document(document_id):
            raise KeyError("Document not found")
        vector = self.provider.embed([question])[0]
        return self.store.search(vector, top_k, document_id)

    def answer(self, question, top_k=4, document_id=None):
        start = perf_counter()
        passages = self.retrieve(question, top_k, document_id)
        eligible = [p for p in passages if p["score"] >= self.settings.min_score]
        sources, usage = [], {}
        answer, abstained = "The documents do not contain enough evidence to answer this question.", True
        if eligible:
            draft, usage = self.provider.generate(question, eligible)
            allowed = {p["id"]: p for p in eligible}
            if any(cid not in allowed for cid in draft.citation_ids):
                raise ProviderError("Model cited a passage that was not retrieved")
            if not draft.abstained and not draft.citation_ids:
                raise ProviderError("Model answered without evidence citations")
            answer, abstained = draft.answer, draft.abstained
            if not abstained:
                for cid in dict.fromkeys(draft.citation_ids):
                    p = allowed[cid]
                    sources.append({"passage_id": cid, "document_id": p["document_id"],
                                    "filename": p["filename"], "page": p["page"], "quote": p["text"]})
        return {"answer": answer, "abstained": abstained, "sources": sources,
                "retrieved_passages": passages, "mode": self.settings.mode,
                "latency_ms": round((perf_counter() - start) * 1000, 2), "generation_usage": usage}

