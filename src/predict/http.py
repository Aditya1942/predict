"""HTTP client with SQLite caching and per-host rate limiting."""

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from predict.errors import SourceError
from predict.store import Store

_SECRET_PARAMS = frozenset({"apiKey"})


class CachedHttp:
    def __init__(
        self,
        store: Store,
        client: httpx.Client,
        *,
        now: Callable[[], datetime],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._store = store
        self._client = client
        self._now = now
        self._sleep = sleep
        self._last_request: dict[str, datetime] = {}

    def get_text(
        self,
        url: str,
        *,
        ttl: timedelta,
        params: dict | None = None,
        headers: dict | None = None,
        min_interval: float = 0.0,
    ) -> str:
        key = _cache_key(url, params)
        cached = self._store.cache_get(key, ttl, self._now())
        if cached is not None:
            return cached
        host = httpx.URL(url).host
        self._throttle(host, min_interval)
        try:
            response = self._client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise SourceError(host, str(exc) or type(exc).__name__) from exc
        if not response.is_success:
            raise SourceError(host, f"HTTP {response.status_code}")
        self._store.cache_put(key, response.text, self._now())
        return response.text

    def get_json(
        self,
        url: str,
        *,
        ttl: timedelta,
        params: dict | None = None,
        headers: dict | None = None,
        min_interval: float = 0.0,
    ) -> Any:
        return json.loads(self.get_text(url, ttl=ttl, params=params, headers=headers, min_interval=min_interval))

    def _throttle(self, host: str, min_interval: float) -> None:
        last = self._last_request.get(host)
        if last is not None and min_interval > 0:
            wait = min_interval - (self._now() - last).total_seconds()
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._now()


def _cache_key(url: str, params: dict | None) -> str:
    if not params:
        return url
    kept = sorted((k, v) for k, v in params.items() if k not in _SECRET_PARAMS)
    if not kept:
        return url
    return f"{url}?{urlencode(kept, doseq=True)}"
