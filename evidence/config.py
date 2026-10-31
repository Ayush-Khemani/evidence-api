import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    sqlite_path: str = "demo.sqlite3"
    database_url: str = ""
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    dimensions: int = 1536
    chunk_words: int = 120
    overlap: int = 20
    min_score: float = 0.25
    max_bytes: int = 2_000_000
    max_chunks: int = 250

    def __post_init__(self):
        if self.mode not in {"demo", "openai"}:
            raise ValueError("RAG_MODE must be demo or openai")
        if not 0 <= self.overlap < self.chunk_words <= 500:
            raise ValueError("Require 0 <= overlap < chunk_words <= 500")
        if not 1 <= self.dimensions <= 3072 or not -1 <= self.min_score <= 1:
            raise ValueError("Invalid embedding dimensions or score threshold")
        if self.mode == "openai" and not self.api_key:
            raise ValueError("Set OPENAI_API_KEY locally in .env for openai mode")

    @property
    def fingerprint(self):
        # Different embedding spaces must never be compared, even at equal size.
        return "demo-hash-v1:256" if self.mode == "demo" else f"openai:{self.embedding_model}:{self.dimensions}"

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            mode=os.getenv("RAG_MODE", "demo"),
            sqlite_path=os.getenv("SQLITE_PATH", "demo.sqlite3"),
            database_url=os.getenv("DATABASE_URL", ""),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            chunk_words=int(os.getenv("CHUNK_WORDS", "120")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "20")),
            min_score=float(os.getenv("MIN_SCORE", "0.25")),
        )

