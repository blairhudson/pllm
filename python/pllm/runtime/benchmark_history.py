from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 3
DEFAULT_LIMIT = 200
MAX_LIMIT = 2000

_PRIVACY_FIELDS = frozenset(
    {
        "plaintext_prompt_bytes_sent",
        "plaintext_token_ids_sent",
        "public_context_bytes",
        "encrypted_correlation_upload_bytes",
        "encrypted_correlation_download_bytes",
        "masked_online_upload_bytes",
        "masked_online_download_bytes",
        "correlation_count",
        "online_steps",
        "inference_stage_calls",
        "token_lookup_cache_hits",
        "token_lookup_cache_misses",
        "kv_continuation_hits",
        "kv_continuation_misses",
        "direct_fhe_upload_bytes",
        "direct_fhe_download_bytes",
        "direct_fhe_steps",
        "preparation_upload_bytes",
        "preparation_download_bytes",
        "inference_upload_bytes",
        "inference_download_bytes",
        "preparation_server_ns",
        "inference_server_ns",
        "correction_push_bytes",
        "correction_push_ns",
        "session_authorization_upload_bytes",
        "session_authorization_download_bytes",
        "preparation_attempts",
        "preparation_rows",
        "preparation_requests_during_online",
        "preparation_failures",
        "bundle_network_bytes",
        "bundle_cache_hits",
        "bundle_cache_misses",
        "bundle_cache_corruptions",
    }
)
_PROCESS_ROLES = frozenset({"client", "preparation", "inference"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SAFE_FAILURE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")


def default_history_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    home = Path(env.get("HOME") or Path.home()).expanduser()
    configured = env.get("XDG_STATE_HOME")
    root = Path(configured).expanduser() if configured else home / ".local" / "state"
    if not root.is_absolute():
        root = home / ".local" / "state"
    return root / "pllm" / "benchmark-history.sqlite3"


def _seconds(start: int | None, end: int | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0, end - start) / 1_000_000_000


def _duration(explicit: float | None, start: int | None, end: int | None) -> float | None:
    return explicit if explicit is not None else _seconds(start, end)


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    run_id: str
    status: str
    model_id: str
    max_output_tokens: int
    cold: bool
    warm: bool
    started_at_ns: int
    finished_at_ns: int
    model_fingerprint: str | None = None
    full_seconds: float | None = None
    preparation_seconds: float | None = None
    transition_seconds: float | None = None
    online_seconds: float | None = None
    ttft_seconds: float | None = None
    generation_seconds: float | None = None
    preparation_started_at_ns: int | None = None
    preparation_finished_at_ns: int | None = None
    online_started_at_ns: int | None = None
    first_token_at_ns: int | None = None
    last_token_at_ns: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_usage_authoritative: bool = False
    inventory_required: int = 0
    inventory_generated: int = 0
    inventory_reused: int = 0
    inventory_consumed: int = 0
    inventory_burned: int = 0
    privacy_delta: Mapping[str, int] = field(default_factory=dict)
    process_metrics: Mapping[str, Mapping[str, float | int | None]] = field(default_factory=dict)
    failure_type: str | None = None

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.run_id):
            raise ValueError("run_id contains unsupported characters")
        if self.status not in {"completed", "failed"}:
            raise ValueError("run status must be completed or failed")
        if (
            not self.model_id
            or len(self.model_id) > 1024
            or any(ord(char) < 32 for char in self.model_id)
        ):
            raise ValueError("model_id is invalid")
        if self.model_fingerprint is not None and (
            not self.model_fingerprint
            or len(self.model_fingerprint) > 256
            or any(ord(char) < 32 for char in self.model_fingerprint)
        ):
            raise ValueError("model_fingerprint is invalid")
        if not 1 <= self.max_output_tokens <= 512:
            raise ValueError("max_output_tokens is outside dashboard bounds")
        if self.started_at_ns <= 0 or self.finished_at_ns < self.started_at_ns:
            raise ValueError("run timestamps are invalid")
        if self.cold == self.warm:
            raise ValueError("exactly one of cold and warm must be true")
        for value in (
            self.full_seconds,
            self.preparation_seconds,
            self.transition_seconds,
            self.online_seconds,
            self.ttft_seconds,
            self.generation_seconds,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("run durations must be finite and non-negative")
        for value in (
            self.input_tokens,
            self.output_tokens,
            self.inventory_required,
            self.inventory_generated,
            self.inventory_reused,
            self.inventory_consumed,
            self.inventory_burned,
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError("run counts must be non-negative integers")
        unknown_privacy = set(self.privacy_delta).difference(_PRIVACY_FIELDS)
        if unknown_privacy:
            raise ValueError(f"unsupported privacy metrics: {sorted(unknown_privacy)}")
        if any(not isinstance(value, int) or value < 0 for value in self.privacy_delta.values()):
            raise ValueError("privacy deltas must be non-negative integers")
        if set(self.process_metrics).difference(_PROCESS_ROLES):
            raise ValueError("unsupported process role")
        for metrics in self.process_metrics.values():
            if set(metrics).difference({"cpu_seconds", "rss_peak_bytes"}):
                raise ValueError("unsupported process metric")
            cpu_value = metrics.get("cpu_seconds")
            rss_value = metrics.get("rss_peak_bytes")
            if cpu_value is not None and (
                not math.isfinite(float(cpu_value)) or float(cpu_value) < 0
            ):
                raise ValueError("process metrics must be non-negative")
            if rss_value is not None and int(rss_value) < 0:
                raise ValueError("process metrics must be non-negative")
        if self.failure_type is not None and not _SAFE_FAILURE.fullmatch(self.failure_type):
            raise ValueError("failure_type is invalid")

    def to_dict(self) -> dict[str, Any]:
        generation_seconds = _duration(
            self.generation_seconds, self.first_token_at_ns, self.last_token_at_ns
        )
        generated_after_first = max(0, (self.output_tokens or 0) - 1)
        tokens_per_second = (
            generated_after_first / generation_seconds
            if generated_after_first > 0
            and generation_seconds is not None
            and generation_seconds > 0
            else None
        )
        total_tokens = (
            self.input_tokens + self.output_tokens
            if self.input_tokens is not None and self.output_tokens is not None
            else None
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "max_output_tokens": self.max_output_tokens,
            "context_tokens": self.input_tokens,
            "cold": self.cold,
            "warm": self.warm,
            "timestamps": {
                "started_at_ns": self.started_at_ns,
                "preparation_started_at_ns": self.preparation_started_at_ns,
                "preparation_finished_at_ns": self.preparation_finished_at_ns,
                "online_started_at_ns": self.online_started_at_ns,
                "first_token_at_ns": self.first_token_at_ns,
                "last_token_at_ns": self.last_token_at_ns,
                "finished_at_ns": self.finished_at_ns,
            },
            "durations": {
                "full_seconds": _duration(
                    self.full_seconds, self.started_at_ns, self.finished_at_ns
                ),
                "preparation_seconds": _duration(
                    self.preparation_seconds,
                    self.preparation_started_at_ns, self.preparation_finished_at_ns
                ),
                "transition_seconds": self.transition_seconds,
                "online_seconds": _duration(
                    self.online_seconds, self.online_started_at_ns, self.finished_at_ns
                ),
                "ttft_seconds": _duration(
                    self.ttft_seconds, self.online_started_at_ns, self.first_token_at_ns
                ),
                "generation_seconds": generation_seconds,
                "tokens_per_second": tokens_per_second,
            },
            "tokens": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": total_tokens,
                "authoritative": self.token_usage_authoritative,
            },
            "inventory": {
                "required": self.inventory_required,
                "generated": self.inventory_generated,
                "reused": self.inventory_reused,
                "consumed": self.inventory_consumed,
                "burned": self.inventory_burned,
            },
            "privacy": {key: int(value) for key, value in sorted(self.privacy_delta.items())},
            "processes": {
                role: {
                    "cpu_seconds": float(cpu) if (cpu := metrics.get("cpu_seconds")) is not None else None,
                    "rss_peak_bytes": int(rss) if (rss := metrics.get("rss_peak_bytes")) is not None else None,
                }
                for role, metrics in sorted(self.process_metrics.items())
            },
            "failure": {"type": self.failure_type} if self.failure_type else None,
        }

    def evidence_report(
        self,
        *,
        privacy_cohort: str,
        numeric_cohort: str,
        environment: Mapping[str, Any],
        plan_lock_digest: str | None = None,
        origin: str = "imported_archive",
        evidence_paths: tuple[str, ...] = (),
    ):
        """Convert a completed real-role run into Rust-validated evidence."""
        if self.status != "completed":
            raise ValueError("failed runs cannot become benchmark evidence")
        observations: list[dict[str, Any]] = []

        def observe(
            role: str,
            phase: str,
            metric: str,
            value: float | int | None,
            unit: str,
        ) -> None:
            if value is not None:
                observations.append(
                    {
                        "role": role,
                        "phase": phase,
                        "origin": origin,
                        "metric": metric,
                        "unit": unit,
                        "value": value,
                        "evidence_paths": list(evidence_paths),
                    }
                )

        durations = self.to_dict()["durations"]
        observe("client", "online", "latency", durations["full_seconds"], "seconds")
        observe("client", "online", "time_to_first_token", durations["ttft_seconds"], "seconds")
        observe(
            "client",
            "online",
            "generation_latency",
            durations["generation_seconds"],
            "seconds",
        )
        observe(
            "client",
            "online",
            "throughput",
            durations["tokens_per_second"],
            "tokens_per_second",
        )
        observe(
            "preparation",
            "offline",
            "latency",
            durations["preparation_seconds"],
            "seconds",
        )
        observe("inference", "online", "latency", durations["online_seconds"], "seconds")
        if not observations:
            raise ValueError("completed run has no measured durations")

        from pllm.evidence import deployment_benchmark

        return deployment_benchmark(
            {
                "options": {
                    "id": self.run_id,
                    "plan_lock_digest": plan_lock_digest,
                    "privacy_cohort": privacy_cohort,
                    "numeric_cohort": numeric_cohort,
                    "environment": dict(environment),
                },
                "observations": observations,
            }
        )


class BenchmarkHistory:
    def __init__(self, path: str | Path | None = None) -> None:
        resolved: str | Path = default_history_path() if path is None else path
        self.path = ":memory:" if str(resolved) == ":memory:" else Path(resolved).expanduser()
        if self.path != ":memory:":
            database_path = Path(self.path)
            database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_CREAT | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(database_path, flags, 0o600)
            os.close(descriptor)
            database_path.chmod(0o600)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=5,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self.path == ":memory:":
            self._connection.execute("PRAGMA journal_mode = MEMORY")
        else:
            self._connection.execute("PRAGMA journal_mode = DELETE")
        self._connection.execute("PRAGMA synchronous = FULL")

    def _migrate(self) -> None:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
                if version > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"benchmark history schema {version} is newer than supported "
                        f"{SCHEMA_VERSION}"
                    )
                if version == 0:
                    self._connection.execute(
                        """
                        CREATE TABLE benchmark_runs (
                            run_id TEXT PRIMARY KEY NOT NULL,
                            schema_version INTEGER NOT NULL,
                            status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
                            model_id TEXT NOT NULL,
                            context_tokens INTEGER,
                            cold INTEGER NOT NULL CHECK (cold IN (0, 1)),
                            started_at_ns INTEGER NOT NULL,
                            record_json TEXT NOT NULL
                        )
                        """
                    )
                    self._connection.execute(
                        """
                        CREATE INDEX benchmark_runs_comparison
                        ON benchmark_runs(model_id, context_tokens, cold, started_at_ns DESC)
                        """
                    )
                    self._connection.execute(
                        """
                        CREATE TRIGGER benchmark_runs_no_update
                        BEFORE UPDATE ON benchmark_runs
                        BEGIN
                            SELECT RAISE(ABORT, 'benchmark records are immutable');
                        END
                        """
                    )
                    self._connection.execute(
                        """
                        CREATE TRIGGER benchmark_runs_no_delete
                        BEFORE DELETE ON benchmark_runs
                        BEGIN
                            SELECT RAISE(ABORT, 'benchmark records are immutable');
                        END
                        """
                    )
                if version < 2:
                    self._connection.execute(
                        """
                        CREATE TRIGGER IF NOT EXISTS benchmark_runs_no_replace
                        BEFORE INSERT ON benchmark_runs
                        WHEN EXISTS (
                            SELECT 1 FROM benchmark_runs WHERE run_id = NEW.run_id
                        )
                        BEGIN
                            SELECT RAISE(ABORT, 'benchmark records are immutable');
                        END
                        """
                    )
                if version < 3:
                    self._connection.execute("DROP INDEX IF EXISTS benchmark_runs_comparison")
                    self._connection.execute(
                        "CREATE INDEX benchmark_runs_comparison "
                        "ON benchmark_runs(status, model_id, cold, context_tokens, "
                        "started_at_ns DESC, run_id DESC)"
                    )
                    self._connection.execute(
                        "CREATE INDEX IF NOT EXISTS benchmark_runs_history "
                        "ON benchmark_runs(started_at_ns DESC, run_id DESC)"
                    )
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                columns = {
                    str(row[1])
                    for row in self._connection.execute("PRAGMA table_info(benchmark_runs)")
                }
                expected = {
                    "run_id",
                    "schema_version",
                    "status",
                    "model_id",
                    "context_tokens",
                    "cold",
                    "started_at_ns",
                    "record_json",
                }
                if columns != expected:
                    raise RuntimeError("benchmark history schema does not match expected columns")
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def add(self, run: BenchmarkRun) -> dict[str, Any]:
        record = run.to_dict()
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO benchmark_runs(
                    run_id, schema_version, status, model_id, context_tokens,
                    cold, started_at_ns, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    SCHEMA_VERSION,
                    run.status,
                    run.model_id,
                    run.input_tokens,
                    int(run.cold),
                    run.started_at_ns,
                    payload,
                ),
            )
        return record

    def contains(self, run_id: str) -> bool:
        if not _SAFE_ID.fullmatch(run_id):
            return False
        with self._lock:
            return (
                self._connection.execute(
                    "SELECT 1 FROM benchmark_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                is not None
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        if not _SAFE_ID.fullmatch(run_id):
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT record_json FROM benchmark_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["record_json"]) if row is not None else None

    def list(
        self,
        *,
        limit: int = DEFAULT_LIMIT,
        model_id: str | None = None,
        status: str | None = None,
        context_tokens: int | None = None,
        min_context_tokens: int | None = None,
        max_context_tokens: int | None = None,
        cold: bool | None = None,
        before_started_at_ns: int | None = None,
        before_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if status is not None and status not in {"completed", "failed"}:
            raise ValueError("status must be completed or failed")
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value, operator in (
            ("model_id", model_id, "="),
            ("status", status, "="),
            ("context_tokens", context_tokens, "="),
            ("context_tokens", min_context_tokens, ">="),
            ("context_tokens", max_context_tokens, "<="),
            ("cold", int(cold) if cold is not None else None, "="),
        ):
            if value is not None:
                clauses.append(f"{column} {operator} ?")
                parameters.append(value)
        if (before_started_at_ns is None) != (before_run_id is None):
            raise ValueError("both pagination cursor fields are required")
        if before_started_at_ns is not None and before_run_id is not None:
            if before_started_at_ns <= 0 or not _SAFE_ID.fullmatch(before_run_id):
                raise ValueError("pagination cursor is invalid")
            clauses.append("(started_at_ns < ? OR (started_at_ns = ? AND run_id < ?))")
            parameters.extend((before_started_at_ns, before_started_at_ns, before_run_id))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                "SELECT record_json FROM benchmark_runs"
                + where
                + " ORDER BY started_at_ns DESC, run_id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
