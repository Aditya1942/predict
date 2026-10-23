"""SQLite store: HTTP cache now, more tables from later tasks."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    fetched_at TEXT NOT NULL
)
"""


class Store:
    def __init__(self, path: Path | str) -> None:
        location = str(path)
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(location)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def cache_get(self, key: str, max_age: timedelta, now: datetime) -> str | None:
        """The cached body if it is strictly younger than `max_age`, else None."""
        hit = self.cache_latest(key)
        if hit is None:
            return None
        body, fetched_at = hit
        return body if now - fetched_at < max_age else None

    def cache_put(self, key: str, body: str, now: datetime) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO http_cache (key, body, fetched_at) VALUES (?, ?, ?)",
                (key, body, now.astimezone(UTC).isoformat()),
            )

    def cache_latest(self, key: str) -> tuple[str, datetime] | None:
        """The most recent cached body and its fetch time, regardless of age."""
        row = self._conn.execute("SELECT body, fetched_at FROM http_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return row[0], datetime.fromisoformat(row[1])
