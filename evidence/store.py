import json
import math
import sqlite3
from contextlib import contextmanager


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ; re-index documents")
    denom = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(a*b for a, b in zip(left, right)) / denom if denom else 0.0


class Store:
    """SQLite exact search for an immediate start; PostgreSQL uses pgvector SQL."""
    def __init__(self, settings):
        self.settings = settings
        self.pg = bool(settings.database_url)

    @contextmanager
    def connection(self):
        if self.pg:
            import psycopg
            from psycopg.rows import dict_row
            with psycopg.connect(self.settings.database_url, row_factory=dict_row, connect_timeout=5) as conn:
                yield conn
        else:
            conn = sqlite3.connect(self.settings.sqlite_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def execute(self, conn, sql, params=()):
        return conn.execute(sql.replace("?", "%s") if self.pg else sql, params)

    def initialize(self):
        with self.connection() as conn:
            if self.pg:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.execute(conn, "INSERT INTO settings VALUES ('embedding', ?) ON CONFLICT(name) DO NOTHING",
                         (self.settings.fingerprint,))
            row = conn.execute("SELECT value FROM settings WHERE name='embedding'").fetchone()
            if row["value"] != self.settings.fingerprint:
                raise ValueError("Database embedding configuration differs. Use a separate database and re-upload documents.")
            conn.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL)")
            vector_type = "vector" if self.pg else "TEXT"
            conn.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page INTEGER, ordinal INTEGER NOT NULL, text TEXT NOT NULL, embedding {vector_type} NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id)")

    def save(self, doc, chunks, vectors):
        with self.connection() as conn:
            self.execute(conn, "INSERT INTO documents VALUES (?, ?, 'ready')", (doc["id"], doc["filename"]))
            for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                cast = "?::vector" if self.pg else "?"
                self.execute(conn, f"INSERT INTO chunks VALUES (?, ?, ?, ?, ?, {cast})",
                             (chunk["id"], doc["id"], chunk["page"], i, chunk["text"], json.dumps(vector)))

    def documents(self):
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY id")]

    def document(self, doc_id):
        with self.connection() as conn:
            row = self.execute(conn, "SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                return None
            doc = dict(row)
            doc["chunks"] = [dict(r) for r in self.execute(conn,
                "SELECT id, page, ordinal, text FROM chunks WHERE document_id=? ORDER BY ordinal", (doc_id,))]
            return doc

    def delete(self, doc_id):
        with self.connection() as conn:
            return self.execute(conn, "DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0

    def search(self, vector, top_k, doc_id=None):
        if not any(vector):
            return []
        where = " WHERE c.document_id=?" if doc_id else ""
        params = (doc_id,) if doc_id else ()
        columns = "c.id, c.document_id, d.filename, c.page, c.text"
        with self.connection() as conn:
            if self.pg:
                rows = self.execute(conn,
                    f"SELECT {columns}, 1 - (c.embedding <=> ?::vector) AS score "
                    f"FROM chunks c JOIN documents d ON d.id=c.document_id{where} "
                    "ORDER BY c.embedding <=> ?::vector, c.id LIMIT ?",
                    (json.dumps(vector), *params, json.dumps(vector), top_k))
                return [dict(r) for r in rows if math.isfinite(r["score"])]
            rows = self.execute(conn, f"SELECT {columns}, c.embedding FROM chunks c "
                                f"JOIN documents d ON d.id=c.document_id{where}", params)
            results = []
            for row in rows:
                item = dict(row)
                item["score"] = cosine(vector, json.loads(item.pop("embedding")))
                results.append(item)
            return sorted(results, key=lambda r: (-r["score"], r["id"]))[:top_k]

