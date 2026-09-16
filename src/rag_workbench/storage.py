from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "metadata.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS documents(
                kb TEXT, id TEXT, source TEXT, version TEXT, payload TEXT,
                PRIMARY KEY(kb,id));
            CREATE TABLE IF NOT EXISTS epochs(kb TEXT PRIMARY KEY, epoch INTEGER);
            CREATE TABLE IF NOT EXISTS snapshots(kb TEXT PRIMARY KEY, name TEXT, fingerprint TEXT);
            CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE IF NOT EXISTS history(
                id INTEGER PRIMARY KEY, kb TEXT, session TEXT, question TEXT, answer TEXT);
            CREATE TABLE IF NOT EXISTS failures(
                id INTEGER PRIMARY KEY, kb TEXT, source TEXT, error TEXT);
        """)
        self.db.commit()

    def documents(self, kb: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT payload FROM documents WHERE kb=? ORDER BY id", (kb,))
        return [json.loads(row[0]) for row in rows]

    def epoch(self, kb: str) -> int:
        row = self.db.execute("SELECT epoch FROM epochs WHERE kb=?", (kb,)).fetchone()
        return int(row[0]) if row else 0

    def activate(
        self, kb: str, documents: list[dict[str, Any]], epoch: int, name: str, fingerprint: str
    ) -> None:
        try:
            self.db.execute("BEGIN")
            self.db.execute("DELETE FROM documents WHERE kb=?", (kb,))
            self.db.executemany(
                "INSERT INTO documents VALUES(?,?,?,?,?)",
                [
                    (
                        kb,
                        item["id"],
                        item["source"],
                        item["version"],
                        json.dumps(item, ensure_ascii=False),
                    )
                    for item in documents
                ],
            )
            self.db.execute("INSERT OR REPLACE INTO epochs VALUES(?,?)", (kb, epoch))
            self.db.execute(
                "INSERT OR REPLACE INTO snapshots VALUES(?,?,?)", (kb, name, fingerprint)
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def snapshot(self, kb: str) -> dict | None:
        row = self.db.execute(
            "SELECT name, fingerprint FROM snapshots WHERE kb=?", (kb,)
        ).fetchone()
        return dict(row) if row else None

    def cache_get(self, key: str) -> Any:
        row = self.db.execute("SELECT payload FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO cache VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False))
        )
        self.db.commit()

    def recent(self, kb: str, session: str) -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT question, answer FROM history WHERE kb=? AND session=? "
            "ORDER BY id DESC LIMIT 3",
            (kb, session),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def remember(self, kb: str, session: str, question: str, answer: str) -> None:
        self.db.execute(
            "INSERT INTO history(kb,session,question,answer) VALUES(?,?,?,?)",
            (kb, session, question, answer),
        )
        self.db.commit()

    def failure(self, kb: str, source: str, error: str) -> None:
        self.db.execute("INSERT INTO failures(kb,source,error) VALUES(?,?,?)", (kb, source, error))
        self.db.commit()

    def close(self) -> None:
        self.db.close()
