from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import yaml

@dataclass
class Config:
    raw: dict

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        load_dotenv()
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw)

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def corpus_path(self) -> Path:
        return Path(self.raw["corpus"]["path"])

    @property
    def collection_name(self) -> str:
        return self.raw["corpus"]["name"]

    @property
    def persist_dir(self) -> Path:
        return Path(self.raw["vectorstore"]["persist_dir"])