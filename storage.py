"""SQLite persistence for objects, visual observations, and audio notes.

Paths inside the database are stored relative to ``data_dir`` whenever possible,
so the data directory can be copied or moved without breaking records.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    created_at: str
    name: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name or f"object-{self.object_id[:8]}"


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    object_id: str
    created_at: str
    image_path: str
    features_path: str
    embedding: tuple[float, ...]
    embedding_model: str | None
    image_width: int
    image_height: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoteRecord:
    note_id: str
    object_id: str
    created_at: str
    audio_path: str
    duration: float
    transcript: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    object_id       TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    name            TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id  TEXT PRIMARY KEY,
    object_id       TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    image_path      TEXT NOT NULL,
    features_path   TEXT NOT NULL,
    embedding_json  TEXT NOT NULL,
    embedding_model TEXT,
    image_width     INTEGER NOT NULL DEFAULT 0,
    image_height    INTEGER NOT NULL DEFAULT 0,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS notes (
    note_id         TEXT PRIMARY KEY,
    object_id       TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    audio_path      TEXT NOT NULL,
    duration        REAL NOT NULL,
    transcript      TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_observations_object
    ON observations(object_id);
CREATE INDEX IF NOT EXISTS idx_notes_object
    ON notes(object_id);
CREATE INDEX IF NOT EXISTS idx_notes_created
    ON notes(created_at);
"""


class Storage:
    """Thread-safe-enough SQLite store for a single-process desktop app."""

    def __init__(self, data_dir: Path | str, db_path: Path | str | None = None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.data_dir / "objectecho.sqlite3"
        if not self.db_path.is_absolute():
            self.db_path = (self.data_dir / self.db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=20.0, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ paths
    def to_stored_path(self, path: Path | str) -> str:
        """Prefer a portable path relative to data_dir; fall back to absolute."""
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = (self.data_dir / resolved).resolve()
        else:
            resolved = resolved.resolve()
        try:
            return resolved.relative_to(self.data_dir).as_posix()
        except ValueError:
            return str(resolved)

    def resolve_path(self, stored_path: str) -> Path:
        path = Path(stored_path)
        return path if path.is_absolute() else (self.data_dir / path)

    # ---------------------------------------------------------------- objects
    def create_object(
        self,
        object_id: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObjectRecord:
        oid = object_id or str(uuid.uuid4())
        created = utc_now()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    "INSERT INTO objects(object_id, created_at, name, metadata_json) "
                    "VALUES (?, ?, ?, ?)",
                    (oid, created, name, meta_json),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Object already exists: {oid}") from exc
        return ObjectRecord(oid, created, name, metadata or {})

    def get_object(self, object_id: str) -> ObjectRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM objects WHERE object_id = ?", (object_id,)
            ).fetchone()
        return self._object_from_row(row) if row else None

    def list_objects(self) -> list[ObjectRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM objects ORDER BY created_at, rowid"
            ).fetchall()
        return [self._object_from_row(row) for row in rows]

    def delete_object(self, object_id: str) -> None:
        """Delete a DB object and its records (files are left untouched)."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM objects WHERE object_id = ?", (object_id,)
            )
            if cur.rowcount == 0:
                raise ValueError(f"Unknown object: {object_id}")

    # ----------------------------------------------------------- observations
    def add_observation(
        self,
        object_id: str,
        image_path: Path | str,
        features_path: Path | str,
        embedding: Iterable[float] | np.ndarray,
        embedding_model: str | None = None,
        image_width: int = 0,
        image_height: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ObservationRecord:
        if self.get_object(object_id) is None:
            raise ValueError(f"Cannot add observation: unknown object {object_id}")
        arr = np.asarray(list(embedding) if not isinstance(embedding, np.ndarray) else embedding,
                          dtype=np.float32).reshape(-1)
        if arr.size and not np.all(np.isfinite(arr)):
            raise ValueError("Embedding contains non-finite values")
        embedding_json = json.dumps([round(float(x), 7) for x in arr.tolist()])
        obs_id = str(uuid.uuid4())
        created = utc_now()
        image_stored = self.to_stored_path(image_path)
        features_stored = self.to_stored_path(features_path)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO observations("
                "observation_id, object_id, created_at, image_path, features_path, "
                "embedding_json, embedding_model, image_width, image_height, metadata_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    obs_id,
                    object_id,
                    created,
                    image_stored,
                    features_stored,
                    embedding_json,
                    embedding_model,
                    int(image_width),
                    int(image_height),
                    meta_json,
                ),
            )
        return ObservationRecord(
            obs_id,
            object_id,
            created,
            image_stored,
            features_stored,
            tuple(float(x) for x in arr),
            embedding_model,
            int(image_width),
            int(image_height),
            metadata or {},
        )

    def list_observations(self, object_id: str | None = None) -> list[ObservationRecord]:
        with self._lock:
            if object_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM observations ORDER BY object_id, created_at, rowid"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM observations WHERE object_id = ? "
                    "ORDER BY created_at, rowid",
                    (object_id,),
                ).fetchall()
        return [self._observation_from_row(row) for row in rows]

    def count_observations(self, object_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM observations WHERE object_id = ?",
                (object_id,),
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------ notes
    def add_note(
        self,
        object_id: str,
        audio_path: Path | str,
        duration: float,
        transcript: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NoteRecord:
        if self.get_object(object_id) is None:
            raise ValueError(f"Cannot add note: unknown object {object_id}")
        if duration < 0 or not np.isfinite(duration):
            raise ValueError("Note duration must be a finite non-negative number")
        note_id = str(uuid.uuid4())
        created = utc_now()
        audio_stored = self.to_stored_path(audio_path)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO notes(note_id, object_id, created_at, audio_path, "
                "duration, transcript, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (note_id, object_id, created, audio_stored, float(duration),
                 transcript, meta_json),
            )
        return NoteRecord(
            note_id, object_id, created, audio_stored, float(duration),
            transcript, metadata or {}
        )

    def list_notes(self, object_id: str | None = None) -> list[NoteRecord]:
        with self._lock:
            if object_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM notes ORDER BY created_at, rowid"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM notes WHERE object_id = ? ORDER BY created_at, rowid",
                    (object_id,),
                ).fetchall()
        return [self._note_from_row(row) for row in rows]

    def count_notes(self, object_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM notes WHERE object_id = ?", (object_id,)
            ).fetchone()
        return int(row["n"])

    def notes_by_object(self) -> dict[str, list[NoteRecord]]:
        result: dict[str, list[NoteRecord]] = {}
        for note in self.list_notes():
            result.setdefault(note.object_id, []).append(note)
        return result

    # -------------------------------------------------------------- conversions
    @staticmethod
    def _object_from_row(row: sqlite3.Row) -> ObjectRecord:
        return ObjectRecord(
            row["object_id"], row["created_at"], row["name"],
            _json_dict(row["metadata_json"]),
        )

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> ObservationRecord:
        try:
            embedding = tuple(float(x) for x in json.loads(row["embedding_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            embedding = ()
        return ObservationRecord(
            row["observation_id"],
            row["object_id"],
            row["created_at"],
            row["image_path"],
            row["features_path"],
            embedding,
            row["embedding_model"],
            int(row["image_width"]),
            int(row["image_height"]),
            _json_dict(row["metadata_json"]),
        )

    @staticmethod
    def _note_from_row(row: sqlite3.Row) -> NoteRecord:
        return NoteRecord(
            row["note_id"],
            row["object_id"],
            row["created_at"],
            row["audio_path"],
            float(row["duration"]),
            row["transcript"],
            _json_dict(row["metadata_json"]),
        )
