import os
from abc import ABC, abstractmethod

SYSTEM_PROMPT = """You answer questions using ONLY the provided context chunks.
Every claim must be traceable to a chunk. Cite the source file for each claim 
in brackets, e.g. [source: rules.md]. If the context does not contain the 
answer, say so explicitly instead of guessing."""

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: list[dict]) -> str:
        ...

class AntrhopicGenerator(BaseGenerator):
    def __init__(self, model: str, max_tokens: int, temperature: float):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        import anthropic

        context_block = "\n\n".join(
            f"[source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        client = anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]}
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OllamaGenerator(BaseGenerator):
    def __init__(self, model: str, base_url: str, temperature: float):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        import requests

        context_block = "\n\n".join(
            f"[source: {c['source']}]\n{c['text']}" for c in context_chunks
        )

        full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_block}\n\nQuestion: {query}"

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
        )

        response.raise_for_status()
        return response.json()["response"]

def create_generator(config) -> BaseGenerator:
    provider = config["generation"]["provider"]

    if provider == "anthropic":
        settings = config["generation"]["anthropic"]
        return AntrhopicGenerator(
            model=settings["model"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )

    elif provider == "ollama":
        settings = config["generation"]["ollama"]
        return OllamaGenerator(
            model=settings["model"],
            base_url=settings["base_url"],
            temperature=settings["temperature"],
        )

    else:
        raise ValueError(f"Unknown generation provider: {provider}")