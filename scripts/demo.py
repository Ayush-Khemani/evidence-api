"""No server, credentials, or paid requests required."""
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from evidence.api import create_app
from evidence.config import Settings


def main():
    with tempfile.TemporaryDirectory() as directory:
        settings = Settings(sqlite_path=str(Path(directory) / "demo.sqlite3"))
        with TestClient(create_app(settings)) as client:
            for path in sorted(Path("sample_docs").glob("*.md")):
                response = client.post("/documents", json={"filename": path.name, "text": path.read_text()})
                response.raise_for_status()
            result = client.post("/questions", json={"question": "How many payment retries?"})
            result.raise_for_status()
            print(json.dumps(result.json(), indent=2))
            print("\nDEMO ONLY: copied evidence, no semantic embedding model or LLM was used.")


if __name__ == "__main__":
    main()

