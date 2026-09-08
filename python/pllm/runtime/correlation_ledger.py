from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


class CorrelationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CorrelationReservation:
    correlation_id: int
    model_id: str
    stage_id: str
    key_id: str
    reserved_at: float


class DurableCorrelationLedger:
    """Crash safe single use state for prepared correlation material.

    A reservation is deliberately destructive. Once an entry leaves AVAILABLE,
    it can never become available again. A timeout, cancellation, process crash,
    or ambiguous network result burns the entry rather than risking reuse.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma journal_mode=WAL")
        self.db.execute("pragma synchronous=FULL")
        self.db.execute("pragma busy_timeout=30000")
        self.db.executescript(
            """
            create table if not exists correlations(
              id integer primary key autoincrement,
              model_id text not null,
              stage_id text not null,
              key_id text not null,
              payload_path text,
              state text not null check(state in ('available','reserved','consumed','burned')),
              created_at real not null,
              reserved_at real,
              consumed_at real
            );
            create index if not exists correlations_lookup
              on correlations(model_id,stage_id,key_id,state,id);
            create table if not exists events(
              sequence integer primary key autoincrement,
              correlation_id integer not null,
              event text not null,
              created_at real not null
            );
            """
        )

    def close(self) -> None:
        self.db.close()

    def add(
        self,
        model_id: str,
        stage_id: str,
        key_id: str,
        *,
        count: int,
        payload_prefix: str = "",
    ) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        now = time.time()
        self.db.execute("begin immediate")
        try:
            for index in range(count):
                cursor = self.db.execute(
                    "insert into correlations(model_id,stage_id,key_id,payload_path,state,created_at) values(?,?,?,?,?,?)",
                    (
                        model_id,
                        stage_id,
                        key_id,
                        f"{payload_prefix}{index}" if payload_prefix else None,
                        "available",
                        now,
                    ),
                )
                cid = int(cursor.lastrowid)
                self.db.execute(
                    "insert into events(correlation_id,event,created_at) values(?,?,?)",
                    (cid, "created", now),
                )
            self.db.execute("commit")
        except Exception:
            self.db.execute("rollback")
            raise

    def reserve(self, model_id: str, stage_id: str, key_id: str) -> CorrelationReservation:
        now = time.time()
        self.db.execute("begin immediate")
        try:
            row = self.db.execute(
                "select id from correlations where model_id=? and stage_id=? and key_id=? and state='available' order by id limit 1",
                (model_id, stage_id, key_id),
            ).fetchone()
            if row is None:
                raise CorrelationUnavailable(f"no available correlation for {model_id}/{stage_id}/{key_id}")
            cid = int(row["id"])
            updated = self.db.execute(
                "update correlations set state='reserved',reserved_at=? where id=? and state='available'",
                (now, cid),
            ).rowcount
            if updated != 1:
                raise CorrelationUnavailable("correlation was reserved concurrently")
            self.db.execute(
                "insert into events(correlation_id,event,created_at) values(?,?,?)",
                (cid, "reserved", now),
            )
            self.db.execute("commit")
        except Exception:
            self.db.execute("rollback")
            raise
        return CorrelationReservation(cid, model_id, stage_id, key_id, now)

    def consume(self, correlation_id: int) -> None:
        now = time.time()
        self.db.execute("begin immediate")
        try:
            updated = self.db.execute(
                "update correlations set state='consumed',consumed_at=? where id=? and state='reserved'",
                (now, int(correlation_id)),
            ).rowcount
            if updated != 1:
                raise CorrelationUnavailable("correlation is not reserved")
            self.db.execute(
                "insert into events(correlation_id,event,created_at) values(?,?,?)",
                (int(correlation_id), "consumed", now),
            )
            self.db.execute("commit")
        except Exception:
            self.db.execute("rollback")
            raise

    def burn_stale_reservations(self, *, older_than_seconds: float) -> int:
        cutoff = time.time() - float(older_than_seconds)
        now = time.time()
        self.db.execute("begin immediate")
        try:
            rows = self.db.execute(
                "select id from correlations where state='reserved' and reserved_at<?",
                (cutoff,),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            for cid in ids:
                self.db.execute(
                    "update correlations set state='burned',consumed_at=? where id=? and state='reserved'",
                    (now, cid),
                )
                self.db.execute(
                    "insert into events(correlation_id,event,created_at) values(?,?,?)",
                    (cid, "burned", now),
                )
            self.db.execute("commit")
            return len(ids)
        except Exception:
            self.db.execute("rollback")
            raise

    def counts(self, model_id: str, stage_id: str, key_id: str) -> dict[str, int]:
        rows = self.db.execute(
            "select state,count(*) as count from correlations where model_id=? and stage_id=? and key_id=? group by state",
            (model_id, stage_id, key_id),
        ).fetchall()
        result = {state: 0 for state in ("available", "reserved", "consumed", "burned")}
        for row in rows:
            result[str(row["state"])] = int(row["count"])
        return result

    def event_chain(self) -> list[tuple[int, int, str]]:
        return [
            (int(row["sequence"]), int(row["correlation_id"]), str(row["event"]))
            for row in self.db.execute(
                "select sequence,correlation_id,event from events order by sequence"
            )
        ]
