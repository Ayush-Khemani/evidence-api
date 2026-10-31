from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .providers import DemoProvider, OpenAIProvider, ProviderError
from .service import RAGService
from .store import Store


class TextDocument(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    document_id: str | None = Field(default=None, max_length=100)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


def create_app(settings=None, provider=None):
    @asynccontextmanager
    async def lifespan(app):
        config = settings or Settings.from_env()
        store = Store(config)
        store.initialize()
        model = provider or (DemoProvider() if config.mode == "demo" else OpenAIProvider(config))
        app.state.service = RAGService(config, store, model)
        try:
            yield
        finally:
            model.close()

    app = FastAPI(title="Evidence API", version="0.1.0", lifespan=lifespan,
                  description="RAG learning backend. Demo mode uses word hashing and copied excerpts; openai mode uses real models.")

    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def input_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": "Document not found"})

    @app.get("/health")
    def health(request: Request):
        service = request.app.state.service
        return {"status": "ok", "mode": service.settings.mode,
                "storage": "postgres-pgvector" if service.store.pg else "sqlite-exact-search",
                "configuration": {"embedding": service.settings.fingerprint,
                    "generation_model": service.settings.model if service.settings.mode == "openai" else None,
                    "chunk_words": service.settings.chunk_words, "overlap": service.settings.overlap,
                    "min_score": service.settings.min_score}}

    @app.post("/documents", status_code=201)
    def add_document(body: TextDocument, request: Request):
        if body.filename.lower().endswith(".pdf"):
            raise ValueError("Use /documents/upload for PDF bytes")
        return request.app.state.service.ingest(body.filename, body.text.encode("utf-8"))

    @app.post("/documents/upload", status_code=201)
    async def upload(request: Request, filename: str = Query(min_length=1, max_length=200)):
        """Send raw file bytes with Content-Type: application/octet-stream."""
        service = request.app.state.service
        data = bytearray()
        async for block in request.stream():
            data.extend(block)
            if len(data) > service.settings.max_bytes:
                raise HTTPException(413, "File exceeds 2 MB limit")
        return await run_in_threadpool(service.ingest, filename, bytes(data))

    @app.get("/documents")
    def documents(request: Request):
        return request.app.state.service.store.documents()

    @app.get("/documents/{document_id}")
    def document(document_id: str, request: Request):
        doc = request.app.state.service.store.document(document_id)
        if doc is None:
            raise HTTPException(404, "Document not found")
        return doc

    @app.delete("/documents/{document_id}", status_code=204)
    def delete(document_id: str, request: Request):
        if not request.app.state.service.store.delete(document_id):
            raise HTTPException(404, "Document not found")

    @app.post("/retrieval")
    def retrieval(body: Question, request: Request):
        return {"passages": request.app.state.service.retrieve(**body.model_dump())}

    @app.post("/questions")
    def questions(body: Question, request: Request):
        return request.app.state.service.answer(**body.model_dump())

    return app


app = create_app()

