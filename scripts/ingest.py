import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextor.config import Config
from contextor.pipeline import ContextorPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    pipeline = ContextorPipeline(config)

    print(f"Ingesting corpus from {config.corpus_path} ...")
    n_chunks = pipeline.ingest()
    print(f"Indexed {n_chunks} chunks into collection '{config.collection_name}'.")

if __name__ == "__main__":
    main()