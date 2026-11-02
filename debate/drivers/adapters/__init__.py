from typing import Dict, Any
from debate.drivers.adapters.claude import ClaudeAdapter
from debate.drivers.adapters.chatgpt import ChatGPTAdapter
from debate.drivers.adapters.gemini import GeminiAdapter
from debate.drivers.adapters.deepseek import DeepSeekAdapter

ADAPTERS: Dict[str, Any] = {
    "claude": ClaudeAdapter(),
    "chatgpt": ChatGPTAdapter(),
    "gemini": GeminiAdapter(),
    "deepseek": DeepSeekAdapter(),
}

def get_adapter(name: str):
    name_clean = name.lower().strip()
    if name_clean not in ADAPTERS:
        raise ValueError(f"Unknown model adapter: '{name}'. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name_clean]

__all__ = ["get_adapter", "ClaudeAdapter", "ChatGPTAdapter", "GeminiAdapter", "DeepSeekAdapter"]
