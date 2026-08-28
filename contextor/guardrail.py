class Guardrail:
    def __init__(self, enabled: bool, strict_mode: bool):
        self.enabled = enabled
        self.strict_mode = strict_mode

    def check(self, answer: str, context_chunks: list[dict]) -> dict:
        if not self.enabled:
            return {"answer": answer, "flagged_claims": [], "grounded": True}

        return {"answer": answer, "flagged_claims": [], "grounded": True}