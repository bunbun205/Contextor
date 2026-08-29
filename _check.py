from contextor.generator import _warn_if_unknown_model

_warn_if_unknown_model("anthropic", "claude-sonnet-5")   # should print nothing
_warn_if_unknown_model("openai", "gpt-5.4-mini")           # should print nothing
_warn_if_unknown_model("ollama", "mistrl:7b")              # typo — should warn