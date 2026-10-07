"""Atomic, text-free checkpoints for one benchmark invocation.

An incomplete document preserves individual reports but is never a comparison.
Only the ordinary comparison builder can replace it with a final report.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .errors import LocalIOError


def write_atomic_json(path: Path, document: dict, *, replace: bool) -> None:
    payload = json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as destination:
            temporary = Path(destination.name)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Publish the first document without racing another writer.
            os.link(temporary, path)
    except FileExistsError as exc:
        raise LocalIOError("OUTPUT_EXISTS",
            f"output already exists; use --force to replace it: {path}") from exc
    except OSError as exc:
        raise LocalIOError("OUTPUT_WRITE", f"cannot write output: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class BenchmarkOutput:
    def __init__(self, path: Path, *, force: bool, configuration: dict, experiments: list) -> None:
        self.path = path
        self.replace = force
        self.created = False
        self.active: int | None = None
        self.document = {
            "schema_version": "pllm.benchmark_progress.v1",
            "complete": False,
            "status": "running",
            "configuration": configuration,
            "candidates": [
                {"index": index, "experiment": None if experiment is None else {
                    "name": experiment.name,
                    "configuration_digest": experiment.configuration_digest(),
                    "pipeline_digest": experiment.pipeline.digest(),
                }, "status": "pending", "report": None}
                for index, experiment in enumerate(experiments, start=1)
            ],
            "checks": {"complete": False, "passed": False},
        }

    def _write(self, document: dict) -> None:
        write_atomic_json(self.path, document, replace=self.replace)
        self.replace = self.created = True

    def _checkpoint(self) -> None:
        self.document["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(self.document)

    def start(self, index: int) -> None:
        self.active = index - 1
        self.document["candidates"][self.active]["status"] = "running"
        self._checkpoint()

    def completed(self, report: dict) -> None:
        # Reject non-JSON results before changing the last durable checkpoint.
        json.dumps(report, allow_nan=False)
        candidate = self.document["candidates"][self.active]
        candidate.update(status="completed", report=report)
        self.active = None
        self._checkpoint()

    def failed(self, error: BaseException) -> None:
        if not self.created:
            return
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        self.document["status"] = status
        # Exception messages can contain private request text or credentials.
        self.document["failure"] = {"type": type(error).__name__}
        if self.active is not None:
            self.document["candidates"][self.active]["status"] = status
        self._checkpoint()

    def finish(self, report: dict) -> None:
        self._write(report)
