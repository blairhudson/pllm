import sqlite3
import stat
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)

from pllm._cli.app import build_parser
from pllm.runtime import dashboard as dashboard_module
from pllm.runtime.benchmark_history import (
    SCHEMA_VERSION,
    BenchmarkHistory,
    BenchmarkRun,
    default_history_path,
)
from pllm.runtime.client import PrivacyAudit
from pllm.runtime.dashboard import (
    DashboardConfig,
    DashboardRuntime,
    OTelStore,
    create_dashboard_app,
)


def _record(
    run_id: str = "run_test",
    *,
    model_id: str = "model-a",
    input_tokens: int | None = 12,
    cold: bool = True,
    started_at_ns: int = 1_000_000_000,
) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=run_id,
        status="completed",
        model_id=model_id,
        model_fingerprint="f" * 64,
        max_output_tokens=8,
        cold=cold,
        warm=not cold,
        started_at_ns=started_at_ns,
        preparation_started_at_ns=started_at_ns + 10,
        preparation_finished_at_ns=started_at_ns + 20,
        online_started_at_ns=started_at_ns + 30,
        first_token_at_ns=started_at_ns + 40,
        last_token_at_ns=started_at_ns + 50,
        finished_at_ns=started_at_ns + 60,
        input_tokens=input_tokens,
        output_tokens=2,
        token_usage_authoritative=True,
        inventory_required=4,
        inventory_generated=0,
        inventory_reused=4,
        inventory_consumed=3,
        inventory_burned=1,
        privacy_delta={"online_steps": 2},
        process_metrics={
            "client": {"cpu_seconds": 0.25, "rss_peak_bytes": 1024},
        },
    )


def test_completed_run_exports_rust_validated_role_evidence() -> None:
    report = _record().evidence_report(
        privacy_cohort="seeded-preparation",
        numeric_cohort="mixed-exact-ring",
        environment={"host": "loopback-fixture"},
        plan_lock_digest="a" * 64,
    )
    document = report.to_dict()

    assert report.schema_version == "pllm.deployment_benchmark_report.v1"
    assert document["plan_lock_digest"] == "a" * 64
    measurements = document["measurements"]
    assert {item["role"] for item in measurements} == {
        "client",
        "preparation",
        "inference",
    }
    assert next(item for item in measurements if item["role"] == "preparation")[
        "phase"
    ] == "offline"


def test_failed_run_cannot_be_benchmark_evidence() -> None:
    record = replace(_record(), status="failed", failure_type="RuntimeError")
    with pytest.raises(ValueError, match="failed runs"):
        record.evidence_report(
            privacy_cohort="seeded-preparation",
            numeric_cohort="mixed-exact-ring",
            environment={"host": "loopback-fixture"},
        )


def test_history_uses_xdg_versioned_immutable_sqlite_and_filters(tmp_path: Path) -> None:
    expected = tmp_path / "state" / "pllm" / "benchmark-history.sqlite3"
    assert default_history_path({"XDG_STATE_HOME": str(tmp_path / "state")}) == expected

    database = tmp_path / "runs.sqlite3"
    history = BenchmarkHistory(database)
    history.add(_record())
    assert history.contains("run_test")
    assert not history.contains("missing")
    history.add(
        _record(
            "run_second",
            model_id="model-b",
            input_tokens=24,
            cold=False,
            started_at_ns=2_000_000_000,
        )
    )
    first_page = history.list(limit=1)
    assert [row["run_id"] for row in first_page] == ["run_second"]
    second_page = history.list(
        limit=1,
        before_started_at_ns=first_page[-1]["timestamps"]["started_at_ns"],
        before_run_id=first_page[-1]["run_id"],
    )
    assert [row["run_id"] for row in second_page] == ["run_test"]
    assert [row["run_id"] for row in history.list(model_id="model-a")] == ["run_test"]
    assert [row["run_id"] for row in history.list(context_tokens=24, cold=False)] == ["run_second"]
    history.close()

    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        connection.execute("UPDATE benchmark_runs SET status = 'failed'")
    connection.rollback()
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        connection.execute("DELETE FROM benchmark_runs")
    connection.rollback()
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        connection.execute(
            "INSERT OR REPLACE INTO benchmark_runs "
            "SELECT * FROM benchmark_runs LIMIT 1"
        )
    connection.close()
    assert stat.S_IMODE(database.stat().st_mode) == 0o600
    assert not database.with_name(f"{database.name}-wal").exists()
    assert not database.with_name(f"{database.name}-shm").exists()

    reopened = BenchmarkHistory(database)
    assert reopened.get("run_test") == _record().to_dict()
    reopened.close()

    memory = BenchmarkHistory(":memory:")
    memory.add(_record("run_memory"))
    assert memory.get("run_memory")["status"] == "completed"
    memory.close()


def test_history_schema_initialization_is_atomic_across_connections(tmp_path: Path) -> None:
    database = tmp_path / "concurrent.sqlite3"
    barrier = threading.Barrier(4)
    failures: list[BaseException] = []

    def initialize() -> None:
        try:
            barrier.wait()
            history = BenchmarkHistory(database)
            history.close()
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=initialize) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name = 'benchmark_runs'"
    ).fetchone()[0] == 1
    connection.close()


def _process_export(cpu_seconds: float, rss_bytes: int) -> bytes:
    request = ExportMetricsServiceRequest()
    resource = request.resource_metrics.add()
    service = resource.resource.attributes.add()
    service.key = "service.name"
    service.value.string_value = "pllm-client"
    scope = resource.scope_metrics.add()
    cpu = scope.metrics.add()
    cpu.name = "process.cpu.time"
    point = cpu.sum.data_points.add()
    point.as_double = cpu_seconds
    kind = point.attributes.add()
    kind.key = "type"
    kind.value.string_value = "user"
    memory = scope.metrics.add()
    memory.name = "process.memory.usage"
    memory.gauge.data_points.add().as_int = rss_bytes
    return request.SerializeToString()


def test_otel_run_window_captures_cpu_delta_and_rss_peak() -> None:
    store = OTelStore()
    store.ingest_metrics(_process_export(5.0, 100))
    store.begin_run_window("run_window")
    store.ingest_metrics(_process_export(7.5, 300))
    store.ingest_metrics(_process_export(8.0, 200))

    metrics = store.finish_run_window("run_window")

    assert metrics["client"] == {"cpu_seconds": 3.0, "rss_peak_bytes": 300}
    assert metrics["preparation"] == {"cpu_seconds": None, "rss_peak_bytes": None}
    assert store.finish_run_window("run_window") == {}


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000_000_000
        self.lock = threading.Lock()

    def time_ns(self) -> int:
        with self.lock:
            value = self.value
            self.value += 100_000_000
            return value

    def sleep(self, _seconds: float) -> None:
        with self.lock:
            self.value += 30_000_000_000

    def monotonic_ns(self) -> int:
        return self.time_ns()


class _BenchmarkClient:
    def __init__(self) -> None:
        self.privacy_audit = PrivacyAudit()
        self.consumed = 0
        self.burned = 0
        self.fail = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.responses = SimpleNamespace(create=self._create)

    def prepared_inventory_status(self, _model: str) -> dict[str, object]:
        return {
            "id": "inventory_secret_901",
            "status": "ready",
            "capacity": 64,
            "available": 64 - self.consumed - self.burned,
            "reserved": 0,
            "burned": self.burned,
            "consumed": self.consumed,
        }

    @staticmethod
    def prepared_rows_for_response(
        _prompt: str,
        _max_output_tokens: int,
        *,
        model: str,
    ) -> int:
        assert model == "model-secure"
        return 5

    def preprocess(self, _model: str, *, count: int) -> dict[str, object]:
        self.privacy_audit.preparation_attempts += 1
        self.privacy_audit.preparation_rows += count
        return {"generated": 0}

    def _create(self, **_kwargs: object) -> list[SimpleNamespace]:
        self.entered.set()
        assert self.release.wait(2)
        if self.fail:
            raise RuntimeError(
                "private prompt alpha at https://secret.invalid/session/session_secret_777"
            )
        self.privacy_audit.online_steps += 3
        self.consumed += 4
        self.burned += 1
        return [
            SimpleNamespace(type="response.output_text.delta", delta="private output omega"),
            SimpleNamespace(type="response.output_text.delta", delta="more private output"),
            SimpleNamespace(
                type="response.completed",
                response={
                    "id": "response_secret_456",
                    "session_id": "session_secret_777",
                    "endpoint": "https://secret.invalid",
                    "pid": 424242,
                    "output": [{"text": "private output omega"}],
                    "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                },
            ),
        ]


def test_runtime_persists_exact_sanitized_runs_and_recovers_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(dashboard_module.time, "time_ns", clock.time_ns)
    monkeypatch.setattr(dashboard_module.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(dashboard_module.time, "sleep", clock.sleep)
    database = tmp_path / "runtime.sqlite3"
    history = BenchmarkHistory(database)
    runtime = DashboardRuntime(
        DashboardConfig("127.0.0.1", 8791, None, "model-secure", 8, database),
        OTelStore(),
        history,
    )
    client = _BenchmarkClient()
    runtime._client = client
    runtime._topology = SimpleNamespace(
        started=True,
        closed=False,
        requires_preparation=True,
        statuses=(SimpleNamespace(running=True),),
    )
    runtime._state["phase"] = "ready"
    runtime._model_fingerprint = "a" * 64

    completed_id = runtime.begin("private prompt alpha", 8)
    assert client.entered.wait(2)
    worker = runtime._worker
    assert worker is not None
    client.release.set()
    worker.join(2)
    assert not worker.is_alive()

    completed = history.get(completed_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["tokens"] == {
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
        "authoritative": True,
    }
    assert completed["inventory"] == {
        "required": 5,
        "generated": 0,
        "reused": 5,
        "consumed": 4,
        "burned": 1,
    }
    assert completed["privacy"]["preparation_rows"] == 5
    assert completed["privacy"]["online_steps"] == 3
    assert (
        completed["timestamps"]["finished_at_ns"] - completed["timestamps"]["last_token_at_ns"]
        < 1_000_000_000
    )
    generation = completed["durations"]["generation_seconds"]
    assert completed["durations"]["tokens_per_second"] == pytest.approx(2 / generation)
    assert runtime._state["phase"] == "ready"
    assert runtime._state["run_id"] == completed_id

    client.fail = True
    client.entered.clear()
    client.release.clear()
    failed_id = runtime.begin("private prompt alpha", 8)
    assert client.entered.wait(2)
    worker = runtime._worker
    assert worker is not None
    client.release.set()
    worker.join(2)
    assert not worker.is_alive()

    failed = history.get(failed_id)
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["failure"] == {"type": "RuntimeError"}
    assert failed["cold"] is False
    assert failed["warm"] is True
    assert runtime._state["phase"] == "ready"

    connection = sqlite3.connect(database)
    stored = "\n".join(
        row[0] for row in connection.execute("SELECT record_json FROM benchmark_runs")
    )
    connection.close()
    history.close()

    async def no_start(_runtime: DashboardRuntime) -> None:
        return None

    async def no_stop(_runtime: DashboardRuntime) -> None:
        return None

    monkeypatch.setattr(DashboardRuntime, "start", no_start)
    monkeypatch.setattr(DashboardRuntime, "stop", no_stop)
    app = create_dashboard_app(
        DashboardConfig("127.0.0.1", 8791, None, "model-secure", 8, database)
    )
    with TestClient(app, base_url="http://127.0.0.1:8791") as api_client:
        response = api_client.get("/api/runs", params={"model_id": "model-secure"})
        assert response.status_code == 200
        api_records = response.text

    for secret in (
        "private prompt alpha",
        "private output omega",
        "inventory_secret_901",
        "response_secret_456",
        "session_secret_777",
        "https://secret.invalid",
        "424242",
    ):
        assert secret not in stored
        assert secret not in api_records


def test_history_api_is_bounded_filterable_and_run_post_returns_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_start(_runtime: DashboardRuntime) -> None:
        return None

    async def no_stop(_runtime: DashboardRuntime) -> None:
        return None

    monkeypatch.setattr(DashboardRuntime, "start", no_start)
    monkeypatch.setattr(DashboardRuntime, "stop", no_stop)
    config = DashboardConfig("127.0.0.1", 8791, None, "model-a", 8, ":memory:")
    app = create_dashboard_app(config)
    app.state.benchmark_history.add(_record())
    monkeypatch.setattr(
        app.state.dashboard_runtime,
        "begin",
        lambda _prompt, _maximum, request_id=None: request_id or "run_api_safe",
    )

    with TestClient(app, base_url="http://127.0.0.1:8791") as client:
        response = client.get("/api/runs", params={"model_id": "model-a", "cold": "true"})
        assert response.status_code == 200
        assert [row["run_id"] for row in response.json()["runs"]] == ["run_test"]
        assert client.get("/api/runs/run_test").json()["context_tokens"] == 12
        assert client.get("/api/runs/missing").status_code == 404
        assert client.get("/api/runs", params={"limit": 2001}).status_code == 422
        started = client.post(
            "/api/run",
            json={
                "prompt": "must not echo this",
                "max_output_tokens": 4,
                "request_id": "run_api_safe",
            },
        )
        assert started.status_code == 202
        assert started.json()["status"] == "started"
        assert started.json()["run_id"] == "run_api_safe"
        assert isinstance(started.json()["protocol_cursor"], int)
        assert "must not echo this" not in started.text
        assert client.post(
            "/api/run",
            json={"prompt": "p", "max_output_tokens": True},
        ).status_code == 400
        assert client.post(
            "/api/run",
            json={"prompt": "p", "max_output_tokens": 1, "unexpected": "value"},
        ).status_code == 400

        assert client.post(
            "/api/run",
            content="{}",
            headers={"content-type": "text/plain"},
        ).status_code == 415
        assert client.post(
            "/api/run",
            content=b"{" + (b"x" * 21_000),
            headers={"content-type": "application/json"},
        ).status_code == 413
        assert client.get(
            "/api/runs",
            headers={"origin": "https://attacker.invalid"},
        ).status_code == 403
        assert client.get(
            "/api/runs",
            headers={"host": "attacker.invalid"},
        ).status_code == 403
        assert client.post(
            "/v1/metrics",
            content=ExportMetricsServiceRequest().SerializeToString(),
            headers={"content-type": "application/x-protobuf"},
        ).status_code == 401
        assert client.post(
            "/v1/metrics",
            content=ExportMetricsServiceRequest().SerializeToString(),
            headers={
                "content-type": "application/x-protobuf",
                "x-pllm-otel-token": config.otel_token,
            },
        ).status_code == 200

    args = build_parser().parse_args(["dev", "dashboard", "--history-db", ":memory:"])
    assert args.history_db == ":memory:"
