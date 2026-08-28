import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextor.config import Config
from contextor.pipeline import ContextorPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    pipeline = ContextorPipeline(config)

    result = pipeline.query(args.question)

    print("\n--- Answer ---")
    print(result["answer"])

    print("\n--- Retrieved chunks ---")
    for hit in result["retrieved_chunks"]:
        print(f"[{hit['source']}] {hit['text'][:150]} ...")

    if result["flagged_claims"]:
        print("\n--- Flagged (unsupported) Claims ---")
        for claim in result["flagged_claims"]:
            print(f"- {claim}")

if __name__ == "__main__":
    main()