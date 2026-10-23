"""SQLite store: HTTP cache and saved Predictions, more tables from later tasks."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from predict.domain import MatchDossier, Prediction, Result

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
    # `body` is the whole MatchDossier as JSON.
    """
    CREATE TABLE IF NOT EXISTS dossiers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        claude_model TEXT NOT NULL,
        body TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS dossiers_match ON dossiers (match_id)",
    # A Match is settled once: `finished` with its 90-minute score, or `void` with none.
    """
    CREATE TABLE IF NOT EXISTS results (
        match_id INTEGER PRIMARY KEY,
        status TEXT NOT NULL,
        home_goals INTEGER,
        away_goals INTEGER,
        settled_at TEXT NOT NULL
    )
    """,
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

    def save_dossier(self, match_id: int, dossier: MatchDossier) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO dossiers (match_id, created_at, claude_model, body) VALUES (?, ?, ?, ?)",
                (match_id, _utc(dossier.created_at), dossier.claude_model, dossier.model_dump_json()),
            )

    def latest_dossier(self, match_id: int) -> MatchDossier | None:
        """The most recently saved News Dossier for the Match."""
        row = self._conn.execute(
            "SELECT body FROM dossiers WHERE match_id = ? ORDER BY id DESC LIMIT 1", (match_id,)
        ).fetchone()
        return None if row is None else MatchDossier.model_validate_json(row[0])

    def save_result(
        self, match_id: int, status: Literal["finished", "void"], result: Result | None, at: datetime
    ) -> None:
        """Settle the Match: a finished one needs its Result and a void one has none. Replaces an earlier settlement."""
        if (status == "finished") != (result is not None):
            raise ValueError(f"a finished Match has a Result and a void one has none (got {status!r}, {result})")
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO results (match_id, status, home_goals, away_goals, settled_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    match_id,
                    status,
                    None if result is None else result.home_goals,
                    None if result is None else result.away_goals,
                    _utc(at),
                ),
            )

    def results(self) -> dict[int, tuple[str, Result | None]]:
        """Every settled Match by id: its status and, when finished, its Result."""
        rows = self._conn.execute("SELECT match_id, status, home_goals, away_goals FROM results").fetchall()
        return {
            match_id: (status, None if home is None else Result(home_goals=home, away_goals=away))
            for match_id, status, home, away in rows
        }

    def unsettled_match_ids(self, kicked_off_before: datetime) -> list[int]:
        """Ids of predicted Matches with no result whose latest kickoff is before the given time, in id order."""
        rows = self._conn.execute(
            "SELECT match_id FROM predictions WHERE match_id NOT IN (SELECT match_id FROM results)"
            " GROUP BY match_id HAVING MAX(kickoff) < ? ORDER BY match_id",
            (_utc(kicked_off_before),),
        )
        return [row[0] for row in rows.fetchall()]


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
