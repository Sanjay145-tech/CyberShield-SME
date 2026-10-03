"""Thin wrapper around a local Ollama model ."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import config


class OllamaLLM:
    def __init__(self, model=config.LLM_MODEL, temperature=config.TEMPERATURE):
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature, "seed": 42},
        }
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(config.OLLAMA_URL, json=body, timeout=300)
            r.raise_for_status()
        except requests.ConnectionError:
            sys.exit("Can't reach Ollama. Start it with `ollama serve` "
                     f"and pull the model with `ollama pull {self.model}`.")
        return r.json()["message"]["content"].strip()
