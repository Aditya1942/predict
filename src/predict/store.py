"""SQLite store: HTTP cache and saved Predictions, more tables from later tasks."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from predict.domain import Prediction

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS http_cache (
        key TEXT PRIMARY KEY,
        body TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    )
    """,
    # `body` is the whole Prediction as JSON; the other columns are copies for querying.
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER NOT NULL,
        kickoff TEXT NOT NULL,
        created_at TEXT NOT NULL,
        shared_at TEXT,
        body TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS predictions_match ON predictions (match_id)",
)


def _utc(dt: datetime | None) -> str | None:
    return None if dt is None else dt.astimezone(UTC).isoformat()


class Store:
    def __init__(self, path: Path | str) -> None:
        location = str(path)
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(location)
        with self._conn:
            for statement in _SCHEMA:
                self._conn.execute(statement)

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
                (key, body, _utc(now)),
            )

    def cache_latest(self, key: str) -> tuple[str, datetime] | None:
        """The most recent cached body and its fetch time, regardless of age."""
        row = self._conn.execute("SELECT body, fetched_at FROM http_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return row[0], datetime.fromisoformat(row[1])

    def save_prediction(self, p: Prediction) -> int:
        """Insert a new row and return its id; any id already on `p` is ignored."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO predictions (match_id, kickoff, created_at, shared_at, body) VALUES (?, ?, ?, ?, ?)",
                _columns(p),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def update_prediction(self, p: Prediction) -> None:
        """Rewrite the saved row with this Prediction's current values."""
        if p.id is None:
            raise ValueError("update_prediction needs a saved Prediction (no id)")
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE predictions SET match_id = ?, kickoff = ?, created_at = ?, shared_at = ?, body = ?"
                " WHERE id = ?",
                (*_columns(p), p.id),
            )
        if cursor.rowcount == 0:
            raise ValueError(f"no saved Prediction with id {p.id}")

    def predictions_for_match(self, match_id: int) -> list[Prediction]:
        """Every Prediction saved for the Match, oldest first."""
        rows = self._conn.execute("SELECT id, body FROM predictions WHERE match_id = ? ORDER BY id", (match_id,))
        return [_prediction(*row) for row in rows.fetchall()]

    def all_predictions(self) -> list[Prediction]:
        rows = self._conn.execute("SELECT id, body FROM predictions ORDER BY id")
        return [_prediction(*row) for row in rows.fetchall()]


def _columns(p: Prediction) -> tuple[int, str | None, str | None, str | None, str]:
    """match_id, kickoff, created_at, shared_at and the JSON body, as stored. The id lives in its own column."""
    return (
        p.match.id,
        _utc(p.match.kickoff),
        _utc(p.created_at),
        _utc(p.shared_at),
        p.model_dump_json(exclude={"id"}),
    )


def _prediction(row_id: int, body: str) -> Prediction:
    return Prediction.model_validate_json(body).model_copy(update={"id": row_id})
