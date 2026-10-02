"""Docker execution backend for the existing local role supervisor.

Only public CPU roles run here. Checkpoints are read-only, credentials travel in
environment variables, and each container owns a separate network namespace.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from urllib.parse import urlsplit

from .servers import LocalTopology, TopologyError


def _docker(arguments, *, environment=None, timeout=30):
    executable = shutil.which("docker")
    if executable is None:
        raise TopologyError("--docker requires the Docker CLI and a running Linux engine")
    result = subprocess.run([executable, *arguments], env=environment, text=True,
                            capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        # Error output can contain daemon environment/configuration. Do not archive it.
        raise TopologyError(f"Docker {arguments[0]} failed ({result.returncode})")
    return (result.stdout + (result.stderr if arguments[0] == "logs" else "")).strip()


def ensure_image(image=None):
    if _docker(["info", "--format", "{{.OSType}}"], timeout=20) != "linux":
        raise TopologyError("--docker requires a Linux Docker engine")
    if image is not None:
        if type(image) is not str or not image or image.startswith("-"):
            raise ValueError("docker_image must be a nonempty image reference")
        return image, _docker(["image", "inspect", image, "--format", "{{.Id}}"])
    root = next((path for path in (Path.cwd(), *Path.cwd().parents)
                 if (path / "deploy/Dockerfile.runtime").is_file() and
                    (path / "Cargo.lock").is_file()), None)
    if root is None:
        raise TopologyError("Docker image build requires a source checkout; supply docker_image for an installed wheel")
    digest = hashlib.sha256()
    files = [root / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
             "pyproject.toml", "uv.lock", "deploy/Dockerfile.runtime", "deploy/entrypoint.sh", ".dockerignore")]
    for directory in ("python/pllm", "crates", "schemas"):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file()
                     and path.suffix in {".py", ".pyi", ".rs", ".toml", ".json", ".h"})
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    tag = f"pllm-benchmark:{digest.hexdigest()[:20]}"
    try:
        identifier = _docker(["image", "inspect", tag, "--format", "{{.Id}}"])
    except TopologyError:
        _docker(["build", "--build-arg", "PLLM_EXTRAS=", "-f", str(root / "deploy/Dockerfile.runtime"),
                 "-t", tag, str(root)], timeout=600)
        identifier = _docker(["image", "inspect", tag, "--format", "{{.Id}}"])
    return tag, identifier


class _Container:
    pid = None  # Container PIDs belong to the VM, never the client's PID namespace.

    def __init__(self, name, *, created=True):
        self.name = name
        self.created = created
        self.closed = False

    def poll(self):
        if self.closed:
            return 0
        state = json.loads(_docker(["inspect", self.name, "--format", "{{json .State}}"]))
        return None if state["Running"] else int(state["ExitCode"])

    def close(self):
        if self.closed:
            return
        if not self.created and self.name not in _docker([
            "ps", "--all", "--filter", f"name=^{self.name}$", "--format", "{{.Names}}"
        ]).splitlines():
            self.closed = True
            return
        try:
            _docker(["stop", "--time", "5", self.name], timeout=15)
        finally:
            _docker(["rm", "--force", self.name], timeout=30)
            self.closed = True


class DockerTopology(LocalTopology):
    __slots__ = ("_docker_image", "_docker_image_id", "_docker_network", "_docker_names", "_docker_mount",
                 "_docker_hub", "_docker_blobs")

    def __init__(self, *args, docker_image=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self._privacy_mode not in {"public", "offset_public", "client_only"} or (
            self._correlation_mode != "bfv" or self._tenseal_path is not None
        ):
            raise TopologyError("Docker benchmark backend supports public compiled CPU roles")
        if self._experiment is not None:
            kernel = self._experiment.pipeline.components.get("kernels")
            if kernel is not None and kernel.component != "pllm/cpu":
                raise TopologyError("Docker benchmark backend requires an explicit CPU-compatible kernel")
        self._docker_image = docker_image
        self._docker_image_id = None
        self._docker_network = None
        self._docker_names = {}
        self._docker_mount = None
        self._docker_hub = None
        self._docker_blobs = ()

    def start(self):
        if self._started and not self._closed:
            return self
        if self._closed:
            raise TopologyError("Docker topology is closed")
        if not self._role_ids:
            return super().start()
        from pllm.model_loader import resolve_model
        from pllm.configuration import Model

        try:
            self._docker_image, self._docker_image_id = ensure_image(self._docker_image)
            source = resolve_model(self._model, cache_dir=self._hf_cache_dir)
            if source.path is None:
                raise TopologyError("Docker roles require a resolved checkpoint")
            path = Path(source.path).resolve()
            # Hub snapshots refer to sibling blobs. Mount this model repository,
            # not the complete Hub cache or the user's credential directory.
            self._docker_mount = path.parent.parent if path.parent.name == "snapshots" else path
            self._docker_hub = path.parent.parent.parent if path.parent.name == "snapshots" else None
            # New Hub layouts can keep this snapshot's blobs outside its model
            # repository. Bind only locked checkpoint files, never that shared store.
            from pllm.model_loader import _source_files
            self._docker_blobs = tuple(sorted({file.resolve() for file in _source_files(path)
                                              if not file.resolve().is_relative_to(self._docker_mount)}))
            if "," in str(self._docker_mount):
                raise TopologyError("Docker checkpoint mount paths cannot contain commas")
            if self._model.kind != "huggingface" or self._docker_hub is None:
                self._model = Model.path(str(path), model_id=self._model_id)
            token = secrets.token_hex(12)
            self._docker_network = "pllm-benchmark-" + token
            _docker(["network", "create", "--label", "pllm.benchmark=true", self._docker_network])
            self._docker_names = {role: f"{self._docker_network}-{role}" for role in self._role_ids}
            return super().start()
        except BaseException:
            self.close()
            raise

    def _spawn(self, role, command):
        if self._stopping.is_set():
            raise TopologyError("Docker topology stopped during startup")
        environment = self._environment(role)
        forwarded = []
        if role == "preparation":
            port = urlsplit(self._inference_url).port
            forwarded.append({"port": port, "host": self._docker_names["inference"], "target_port": port})
        if self._telemetry_endpoint is not None:
            port = urlsplit(self._telemetry_endpoint).port
            forwarded.append({"port": port, "host": "host.docker.internal", "target_port": port})
        # The fixed loopback proxy preserves existing origin and push-URL checks.
        # It forwards opaque TCP only inside this explicitly co-located backend.
        arguments = command[1:]
        if "--host" in arguments:
            arguments[arguments.index("--host") + 1] = "0.0.0.0"
        else:
            arguments += ["--host", "0.0.0.0"]
        # Forward owned role settings, never arbitrary host credentials or payload variables.
        keys = {"PLLM_ROLE_EXPERIMENT_JSON", "PLLM_API_KEY", "PLLM_PREPARED_INVENTORY_ROWS"}
        if role == "inference":
            keys.add("PLLM_PROVIDER_PUSH_API_KEY")
        elif role == "preparation":
            keys.update({"PLLM_INFERENCE_URL", "PLLM_PUSH_API_KEY"})
        else:
            keys.update({"PLLM_OFFSET_WORKER_API_KEY", "PLLM_OFFSET_EXPERIMENT_JSON"})
        if self._telemetry_endpoint is not None:
            keys.update({"OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_METRIC_EXPORT_INTERVAL",
                         "OTEL_EXPORTER_OTLP_HEADERS", "OTEL_SERVICE_NAME"})
        allowed = {name: environment[name] for name in keys if name in environment}
        allowed.update({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "PLLM_DOCKER_COMMAND": json.dumps(arguments),
                        "PLLM_DOCKER_FORWARD": json.dumps(forwarded),
                        "OMP_NUM_THREADS": str(self._engine_threads or 1),
                         "OPENBLAS_NUM_THREADS": str(self._engine_threads or 1)})
        if self._docker_hub is not None:
            allowed["HF_HUB_CACHE"] = str(self._docker_hub)
        name = self._docker_names[role]
        port = urlsplit(self._role_urls[role]).port
        options = ["create", "--name", name, "--label", "pllm.benchmark=true",
                   "--network", self._docker_network, "--add-host", "host.docker.internal:host-gateway",
                   "--publish", f"127.0.0.1:{port}:{port}", "--cap-drop=ALL",
                   "--security-opt=no-new-privileges", "--pids-limit=256",
                   "--cpus", str(self._engine_threads or 1),
                   "--mount", f"type=bind,src={self._docker_mount},dst={self._docker_mount},readonly",
                    "--entrypoint", "/opt/pllm/.venv/bin/python"]
        for blob in self._docker_blobs:
            if "," in str(blob):
                raise TopologyError("Docker checkpoint blob paths cannot contain commas")
            options += ["--mount", f"type=bind,src={blob},dst={blob},readonly"]
        for key in sorted(allowed):
            options += ["--env", key]
        with self._process_lock:
            if self._stopping.is_set():
                raise TopologyError("Docker topology stopped during startup")
            # A timed-out CLI may have created its container. Retain that owned
            # name before creation so close can discover and remove it.
            process = _Container(name, created=False)
            self._processes[role] = process
            _docker([*options, self._docker_image, "-m", "pllm.runtime.docker_role"],
                    environment={**os.environ, **allowed})
            process.created = True
            _docker(["start", name])
        return process

    @staticmethod
    def _stop(process):
        process.close()

    def _redacted_tail(self, role):
        try:
            text = _docker(["logs", "--tail", "30", self._docker_names[role]])
        except TopologyError:
            return ""
        for secret in (self._inference_key, self._preparation_key, self._push_key, *self._role_credentials.values()):
            if secret:
                text = text.replace(secret, "<redacted>")
        return text[-4000:]

    def resource_samples(self):
        samples = {}
        for role, process in self._processes.items():
            value = json.loads(_docker(["exec", process.name, "/opt/pllm/.venv/bin/python",
                                      "-m", "pllm.runtime.docker_role", "--sample"]))
            samples[role] = value
        return {"schema": "pllm.docker_resources.v1", "image_id": self._docker_image_id,
                "scope": "co-located-provider-containers", "roles": samples,
                "full_wire_bytes": None}

    def close(self):
        try:
            super().close()
        finally:
            if self._docker_network is not None:
                if self._docker_network in _docker([
                    "network", "ls", "--filter", f"name=^{self._docker_network}$", "--format", "{{.Name}}"
                ]).splitlines():
                    _docker(["network", "rm", self._docker_network])
                self._docker_network = None
