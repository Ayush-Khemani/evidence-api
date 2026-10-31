import json

import httpx
import pytest
from fastapi.testclient import TestClient

from evidence.api import create_app
from evidence.config import Settings
from evidence.documents import chunk_pages
from evidence.providers import DemoProvider, Draft, OpenAIProvider, ProviderError
from evidence.store import Store


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(sqlite_path=str(tmp_path / "test.sqlite3")))) as session:
        yield session


def upload(client, text="Payment retries: retry failed payments at most three times."):
    response = client.post("/documents", json={"filename": "payments.md", "text": text})
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_retrieve_answer_and_delete(client):
    doc = upload(client)
    result = client.post("/questions", json={"question": "Payment retries?", "document_id": doc["id"]}).json()
    assert result["mode"] == "demo"
    assert not result["abstained"]
    assert "three times" in result["answer"]
    assert result["sources"][0]["quote"] == result["retrieved_passages"][0]["text"]
    assert result["sources"][0]["page"] is None
    assert client.delete(f'/documents/{doc["id"]}').status_code == 204
    assert client.get(f'/documents/{doc["id"]}').status_code == 404
    assert client.post("/retrieval", json={"question": "payments"}).json()["passages"] == []


def test_empty_corpus_abstains_without_generation(client):
    result = client.post("/questions", json={"question": "Where are my documents?"}).json()
    assert result["abstained"] and result["sources"] == []


def test_scope_excludes_other_document(client):
    first = upload(client, "Payment retries: three attempts.")
    second = upload(client, "Authentication uses bearer tokens.")
    result = client.post("/retrieval", json={"question": "payment", "document_id": second["id"]}).json()
    assert all(p["document_id"] == second["id"] for p in result["passages"])
    assert all(p["document_id"] != first["id"] for p in result["passages"])


def test_chunk_overlap_and_page_boundaries():
    chunks = chunk_pages([(1, "one two three four five six"), (2, "seven eight")], 4, 1)
    assert chunks == [{"page": 1, "text": "one two three four"},
                      {"page": 1, "text": "four five six"}, {"page": 2, "text": "seven eight"}]


@pytest.mark.parametrize("body", [
    {"question": "   "}, {"question": "x", "top_k": 0}, {"question": "x", "top_k": 11},
])
def test_bad_questions_rejected(client, body):
    assert client.post("/questions", json=body).status_code == 422


@pytest.mark.parametrize("filename,content", [
    ("test.exe", b"not text"), ("empty.md", b"  "), ("bad.txt", b"\xff"), ("null.txt", b"a\x00b"),
])
def test_bad_uploads_leave_no_documents(client, filename, content):
    assert client.post("/documents/upload", params={"filename": filename}, content=content).status_code == 422
    assert client.get("/documents").json() == []


def test_oversized_stream_rejected(client):
    assert client.post("/documents/upload", params={"filename": "large.md"}, content=b"x" * 2_000_001).status_code == 413


def test_paths_normalized_and_content_persists(client):
    doc = client.post("/documents", json={"filename": "../example.md", "text": "Useful content"}).json()
    assert doc["filename"] == "example.md"
    assert client.get(f'/documents/{doc["id"]}').json()["chunks"][0]["text"] == "Useful content"


class BadCitation(DemoProvider):
    def generate(self, question, passages):
        return Draft(answer="A fabricated answer", abstained=False, citation_ids=["invented-id"]), {}


def test_unknown_citations_fail_closed(tmp_path):
    with TestClient(create_app(Settings(sqlite_path=str(tmp_path / "db")), BadCitation())) as client:
        upload(client)
        assert client.post("/questions", json={"question": "Payment retries?"}).status_code == 502


class FailedEmbedding(DemoProvider):
    def embed(self, texts):
        raise ProviderError("Provider unavailable")


def test_provider_failure_is_atomic(tmp_path):
    with TestClient(create_app(Settings(sqlite_path=str(tmp_path / "db")), FailedEmbedding())) as client:
        assert client.post("/documents", json={"filename": "a.md", "text": "payment retries"}).status_code == 502
        assert client.get("/documents").json() == []


def test_embedding_fingerprint_mismatch(tmp_path):
    path = str(tmp_path / "db")
    Store(Settings(sqlite_path=path)).initialize()
    with pytest.raises(ValueError, match="configuration differs"):
        Store(Settings(mode="openai", api_key="test-only", sqlite_path=path)).initialize()


def test_openai_contract_with_mock_http():
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0.0, 1.0]},
                                                       {"index": 0, "embedding": [1.0, 0.0]}]})
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"answer": "Three retries.", "abstained": False,
                                                         "citation_ids": ["passage-1"]})}]}],
            "usage": {"input_tokens": 12, "output_tokens": 8}})
    provider = OpenAIProvider(Settings(mode="openai", api_key="test-only", dimensions=2),
        httpx.Client(base_url="https://example.invalid/v1/", transport=httpx.MockTransport(handler)))
    assert provider.embed(["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
    draft, usage = provider.generate("Retries?", [{"id": "passage-1", "text": "Three retries."}])
    assert draft.citation_ids == ["passage-1"] and usage["input_tokens"] == 12
    assert calls[1]["text"]["format"]["strict"] is True
    assert calls[1]["store"] is False
    provider.close()


@pytest.mark.parametrize("response", [
    {"status": "incomplete", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "not json"}]}]},
])
def test_refusal_incomplete_and_malformed_output(response):
    provider = OpenAIProvider(Settings(mode="openai", api_key="test-only"),
        httpx.Client(base_url="https://example.invalid/", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response))))
    with pytest.raises(ProviderError):
        provider.generate("question", [{"id": "a", "text": "data"}])
    provider.close()


def test_stopwords_only_query_has_no_candidates(client):
    upload(client)
    result = client.post("/questions", json={"question": "the and of"}).json()
    assert result["abstained"] and result["retrieved_passages"] == []


def test_pdf_text_and_page_number(client):
    from io import BytesIO
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 30 700 Td (Payment retries: three times.) Tj ET")
    page[NameObject("/Contents")] = stream
    buffer = BytesIO()
    writer.write(buffer)
    response = client.post("/documents/upload", params={"filename": "handbook.pdf"}, content=buffer.getvalue())
    assert response.status_code == 201, response.text
    chunks = client.get(f'/documents/{response.json()["id"]}').json()["chunks"]
    assert chunks[0]["page"] == 1 and "three times" in chunks[0]["text"]


def test_blank_pdf_is_rejected(client):
    from io import BytesIO
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=600, height=800)
    buffer = BytesIO()
    writer.write(buffer)
    response = client.post("/documents/upload", params={"filename": "scan.pdf"}, content=buffer.getvalue())
    assert response.status_code == 422 and "No extractable text" in response.text

