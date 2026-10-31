"""Evaluate a running local API. Uploads sample fixtures; deletes only its own uploads."""
import argparse
import json
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=4)
    parser.add_argument("--output", default="reports/evaluation.json")
    parser.add_argument("--demo", action="store_true", help="Use an isolated in-process demo; no server or model calls")
    args = parser.parse_args()
    if args.demo:
        from fastapi.testclient import TestClient
        from evidence.api import create_app
        from evidence.config import Settings
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Settings(sqlite_path=str(Path(directory) / "eval.sqlite3")))) as client:
                evaluate(client, args)
    else:
        with httpx.Client(base_url=args.url, timeout=120) as client:
            evaluate(client, args)


def evaluate(client, args):
    rows, uploaded = [], []
    health = client.get("/health")
    health.raise_for_status()
    existing = client.get("/documents")
    existing.raise_for_status()
    if existing.json():
        raise SystemExit("Use an empty database so other documents cannot contaminate the evaluation.")
    try:
        for path in sorted(Path("sample_docs").glob("*.md")):
            response = client.post("/documents", json={"filename": path.name, "text": path.read_text()})
            response.raise_for_status()
            uploaded.append(response.json()["id"])
        for case in json.loads(Path("eval/questions.json").read_text()):
            response = client.post("/questions", json={"question": case["question"], "top_k": args.top_k})
            response.raise_for_status()
            result = response.json()
            hit = None if case["unanswerable"] else any(
                p["filename"] == case["filename"] and case["evidence"].lower() in p["text"].lower()
                for p in result["retrieved_passages"])
            rows.append({"case": case, "retrieval_hit": hit,
                         "abstention_correct": result["abstained"] == case["unanswerable"],
                         "human_grounding_review": "pending", "result": result})
    finally:
        for doc_id in uploaded:
            client.delete(f"/documents/{doc_id}").raise_for_status()
    positives = [r for r in rows if not r["case"]["unanswerable"]]
    negatives = [r for r in rows if r["case"]["unanswerable"]]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "environment": health.json(),
              "top_k": args.top_k, "cases": len(rows),
              "retrieval_hit_rate_at_k": sum(r["retrieval_hit"] for r in positives) / len(positives),
              "unanswerable_abstention_rate": sum(r["result"]["abstained"] for r in negatives) / len(negatives),
              "answerable_response_rate": sum(not r["result"]["abstained"] for r in positives) / len(positives),
              "median_latency_ms": statistics.median(r["result"]["latency_ms"] for r in rows),
              "grounding": "Not automatically scored. Review answers against evidence manually.",
              "cost": "Not calculated. generation_usage excludes embedding tokens; consult provider usage.",
              "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

