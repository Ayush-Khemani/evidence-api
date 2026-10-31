from io import BytesIO
from pathlib import PurePosixPath


def safe_name(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or len(name) > 200 or any(ord(c) < 32 for c in name):
        raise ValueError("Provide a filename of at most 200 characters")
    if PurePosixPath(name).suffix.lower() not in {".md", ".txt", ".pdf"}:
        raise ValueError("Supported files: UTF-8 .md, .txt, and text-based .pdf")
    return name


def extract(filename: str, content: bytes) -> list[tuple[int | None, str]]:
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError('PDF support requires: pip install ".[pdf]"') from exc
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted or len(reader.pages) > 100:
                raise ValueError("Use an unencrypted PDF with at most 100 pages")
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception as exc:
            raise ValueError("Cannot read PDF; use an unencrypted text PDF (max 100 pages)") from exc
    else:
        try:
            pages = [(None, content.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must be UTF-8 encoded") from exc
    if not any(text.strip() for _, text in pages):
        raise ValueError("No extractable text; scanned PDFs need OCR, which is not included")
    if any("\x00" in text for _, text in pages):
        raise ValueError("Document contains null characters")
    return pages


def chunk_pages(pages, size: int, overlap: int):
    """Word windows stay within physical PDF pages; Markdown has no page number."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    chunks = []
    for page, text in pages:
        words = text.split()
        for start in range(0, len(words), size - overlap):
            chunks.append({"page": page, "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
    return chunks

