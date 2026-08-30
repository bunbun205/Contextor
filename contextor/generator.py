import os
from abc import ABC, abstractmethod

KNOWN_MODELS = {
    "anthropic": [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-haiku-4-5-20251001",
    ],
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "o4-mini",
        "o3",
    ],
    "gemini": [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
    ],
    "ollama": [
        "llama3.1:8b",
        "llama3.1:70b",
        "mistral:7b",
        "phi3:14b",
        "gemma2:9b",
        "qwen2.5:14b",
    ],
}

SYSTEM_PROMPT = """You answer questions using ONLY the provided context chunks.
Every claim must be traceable to a chunk. Cite the source file for each claim 
in brackets, e.g. [source: rules.md]. If the context does not contain the 
answer, say so explicitly instead of guessing."""

def _warn_if_unknown_model(provider: str, model: str) -> None:
    known = KNOWN_MODELS.get(provider, [])
    if known and model not in known:
        print(
            f"Note: '{model}' isn't in the known {provider} models list "
            f"({', '.join(known)}). Continuing anyway — this may be a "
            f"valid newer model, or worth double-checking for a typo."
        )

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: list[dict]) -> str:
        ...

class AnthropicGenerator(BaseGenerator):
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

class OPenAIGenerator(BaseGenerator):
    def __init__(self, model: str, max_tokens: str, temperature: float):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        from openai import OpenAI

        context_block = "\n\n".join(
            f"[source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "sustem", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content


class GeminiGenerator(BaseGenerator):
    def __init__(self, model: str, max_tokens: int, temperature: float):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        import os
        from google import genai
        from google.genai import types

        context_block = "\n\n".join(
            f"[source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=int(str(self.max_tokens).strip(",")),
            ),
        )
        return response.text


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
    settings = config["generation"].get(provider, {})
    model = settings.get("model")

    if model:
        _warn_if_unknown_model(provider, model)

    if provider == "anthropic":
        return AnthropicGenerator(
            model=settings["model"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )

    elif provider == "openai":
        return OPenAIGenerator(
            model=settings["model"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )

    elif provider == "gemini":
        return GeminiGenerator(
            model=settings["model"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )

    elif provider == "ollama":
        return OllamaGenerator(
            model=settings["model"],
            base_url=settings["base_url"],
            temperature=settings["temperature"],
        )

    else:
        raise ValueError(f"Unknown generation provider: {provider!r}")