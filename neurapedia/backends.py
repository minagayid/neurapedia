"""Optional local text backends for summarizing structured measurements."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen


SYSTEM_PROMPT = (
    "You summarize structured neuroimaging measurements for research review. "
    "Do not diagnose, estimate disease probability, recommend treatment, or invent evidence. "
    "Always state that the result is not a diagnosis and requires qualified human review."
)


class DeterministicReportBackend:
    def summarize(self, payload: dict[str, Any]) -> str:
        image = payload.get("image", {})
        anomalies = payload.get("anomalies", [])
        return (
            "Research summary\n"
            f"Patient label: {image.get('patient_id', 'unknown')}\n"
            f"Observations: {len(anomalies)}\n"
            "Interpretation: requires human review; this is not a diagnosis."
        )


class OpenAICompatibleBackend:
    """Call an optional local OpenAI-compatible server such as Ollama or vLLM."""

    def __init__(self, endpoint: str, model: str = "gpt-oss-20b", api_key: str | None = None, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def build_request(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }

    def summarize(self, prompt: str) -> str:
        url = self.endpoint if self.endpoint.endswith("/chat/completions") else self.endpoint + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=json.dumps(self.build_request(prompt)).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("local text backend returned an unexpected response") from error

