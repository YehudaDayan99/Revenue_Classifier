from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class LLMError(RuntimeError):
    pass


def _coerce_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception as e:
        raise LLMError(f"Failed to parse model JSON output: {e}\nRaw:\n{text[:2000]}") from e


@dataclass(frozen=True)
class OpenAIChatClient:
    """Small OpenAI client using plain HTTP (no extra dependency).

    Requires: OPENAI_API_KEY in environment (or passed in).
    """

    model: str = "gpt-4.1-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 120
    max_retries: int = 3

    def _key(self) -> str:
        k = self.api_key or os.getenv("OPENAI_API_KEY")
        if not k:
            raise LLMError("OPENAI_API_KEY is not set")
        return k

    def _session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update(
            {
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            }
        )
        return s

    def json_call(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Call Chat Completions and require strict JSON output."""
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        url = f"{self.base_url}/chat/completions"
        s = self._session()

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = s.post(url, data=json.dumps(payload), timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise LLMError(f"OpenAI HTTP {r.status_code}: {r.text[:2000]}")
                data = r.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content or not isinstance(content, str):
                    raise LLMError(f"OpenAI returned empty content: {data}")
                return _coerce_json(content.strip())
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.8 * (2 ** (attempt - 1)))
                    continue
                raise

        raise LLMError(f"OpenAI call failed: {last_err}")

