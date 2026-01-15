from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from revseg.secrets import get_openai_api_key


class LLMError(RuntimeError):
    pass


def _coerce_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception as e:
        raise LLMError(f"Failed to parse model JSON output: {e}\nRaw:\n{text[:2000]}") from e


_RETRY_AFTER_HINT_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


def _parse_retry_after_seconds(msg: str) -> Optional[float]:
    if not msg:
        return None
    m = _RETRY_AFTER_HINT_RE.search(msg)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


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
    # Throttle to avoid RPM limits (defaults match low-tier limits like 3 RPM).
    rate_limit_rpm: Optional[float] = 3.0
    # Mutable single-element list used for tracking last call time even in frozen dataclass.
    _last_call_ts: list[float] = field(default_factory=list, repr=False)

    def _key(self) -> str:
        k = self.api_key or get_openai_api_key()
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
        # Best-effort client-side RPM throttle
        if self.rate_limit_rpm and self.rate_limit_rpm > 0:
            min_interval = 60.0 / float(self.rate_limit_rpm)
            now = time.time()
            if self._last_call_ts and (now - self._last_call_ts[0]) < min_interval:
                time.sleep(min_interval - (now - self._last_call_ts[0]) + 0.05)
            if self._last_call_ts:
                self._last_call_ts[0] = time.time()
            else:
                self._last_call_ts.append(time.time())

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
                if r.status_code == 429:
                    # Honor server hint if present, otherwise exponential backoff.
                    hint = _parse_retry_after_seconds(r.text)
                    sleep_s = (hint + 0.5) if hint is not None else (1.5 * (2 ** (attempt - 1)))
                    if attempt < self.max_retries:
                        time.sleep(sleep_s)
                        continue
                    raise LLMError(f"OpenAI HTTP 429: {r.text[:2000]}")
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

