import argparse
import json
import sys
from pathlib import Path
from unittest import result

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextor.config import Config
from contextor.pipeline import ContextorPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    pipeline = ContextorPipeline(config)

    with open(args.qa) as f:
        qa_pairs = json.load(f)

    retrieval_hits = 0
    grounded_count = 0
    total_flagged_claims = 0

    print(f"Running {len(qa_pairs)} queries against corpus '{config.collection_name}'...\n")

    for item in qa_pairs:
        result = pipeline.query(item["question"])
        retrieved_sources = {c["source"] for c in result["retrieved_chunks"]}

        retrieved_ok = item["expected_source"] in retrieved_sources
        if retrieved_ok:
            retrieval_hits += 1

        if result["grounded"]:
            grounded_count += 1
        total_flagged_claims += len(result["flagged_claims"])

        status = "✓" if retrieved_ok else "✗"
        grounded_status = "grounded" if result["grounded"] else "FLAGGED"
        print(f"[{status}] [{grounded_status}] {item['question']}")

    n = len(qa_pairs)
    print(f"\n--- Results for {config.collection_name} ---")
    print(f"Retrieval precision@k: {retrieval_hits}/{n} ({retrieval_hits / n:.1%})")
    print(f"Grounding rate: {grounded_count}/{n} ({grounded_count / n:.1%})")
    print(f"Total flagged claims across all answers: {total_flagged_claims}")


if __name__ == "__main__":
    main()