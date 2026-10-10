"""Inert, model-neutral declarations shared by benchmark controllers and workers.

Keep this module independent of NumPy, model loaders and serving imports. A worker
must be able to measure initialization without first importing the entire SDK.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import sys
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bounded_int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def identity(value, name="identity"):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,191}", value) is None:
        raise ValueError(f"{name} must be a bounded public identity")
    return value


def choice(value, choices, name):
    if type(value) is not str or value not in choices:
        raise ValueError(f"unsupported {name}")
    return value


class TargetUnavailable(Exception):
    """Public reason code; never carry exception strings or private inputs."""

    def __init__(self, code, *, status="unsupported", details=None):
        self.code, self.status, self.details = code, status, details or {}


def _model(experiment, config):
    from pllm.configuration import Experiment

    if not isinstance(experiment, Experiment):
        raise TypeError("model targets require an Experiment")
    if experiment.budget is None:
        raise ValueError("model targets require an execution budget")
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a public model configuration mapping")
    value = {"experiment": experiment.to_spec(), "profile": experiment.pipeline.profile,
             "config": None if config is None else json.loads(canonical(dict(config)))}
    if len(canonical(value)) > 2 << 20:
        raise ValueError("model target metadata exceeds 2 MiB")
    return value


@dataclass(frozen=True, slots=True)
class Sampling:
    """Independent processes, repetitions within each process, and hard limits."""

    processes: int = 3
    repetitions: int = 1
    warmups: int = 0
    timeout_seconds: float = 300
    order: str = "balanced"
    threads: int = 1
    memory_bytes: int = 512 << 20

    def __post_init__(self):
        for name, low, high in (("processes", 1, 32), ("repetitions", 1, 64),
                                ("warmups", 0, 16), ("threads", 1, 64),
                                ("memory_bytes", 16 << 20, 16 << 30)):
            bounded_int(getattr(self, name), low, high, name)
        if (type(self.timeout_seconds) not in {int, float}
                or not math.isfinite(self.timeout_seconds)
                or not 0 < self.timeout_seconds <= 86400):
            raise ValueError("timeout_seconds must be finite and in (0, 86400]")
        choice(self.order, {"balanced", "ordered"}, "sample order")

    def to_spec(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BuildRef:
    """An installed interpreter and optional observed implementation lock.

    Construction does not execute or inspect it. ``capture`` explicitly obtains
    a lock in an independent metadata process, outside measured target workers.
    """

    python: str = sys.executable
    expected_digest: str | None = None

    def __post_init__(self):
        if type(self.python) is not str or not self.python or "\0" in self.python:
            raise ValueError("python must name an installed interpreter")
        if self.expected_digest is not None and (
            type(self.expected_digest) is not str
            or re.fullmatch("[0-9a-f]{64}", self.expected_digest) is None
        ):
            raise ValueError("expected_digest must be a SHA-256 digest")

    @classmethod
    def capture(cls, python: str = sys.executable):
        from pllm.metrics.benchmark_runner import _interpreter, inspect_build
        declared = cls(python)
        path = _interpreter(declared.python)
        observed = inspect_build(cls(path))
        return cls(path, observed["digest"])

    def to_spec(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Scenario:
    """Public request lifecycle; prompt values are supplied only at execution."""

    name: str
    kind: str = "cold"
    count: int = 1
    max_output_tokens: int = 8

    def __post_init__(self):
        identity(self.name, "scenario name")
        choice(self.kind, {"cold", "repeat", "sequence"}, "scenario kind")
        bounded_int(self.count, 1, 64, "request count")
        bounded_int(self.max_output_tokens, 1, 1024, "output tokens")
        if self.kind == "cold" and self.count != 1:
            raise ValueError("cold scenarios have exactly one request")

    @classmethod
    def cold(cls, name="request", *, max_output_tokens=8):
        return cls(name, "cold", 1, max_output_tokens)

    @classmethod
    def repeat(cls, name="request", *, count=2, max_output_tokens=8):
        return cls(name, "repeat", count, max_output_tokens)

    @classmethod
    def sequence(cls, name="requests", *, count, max_output_tokens=8):
        return cls(name, "sequence", count, max_output_tokens)

    def to_spec(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TargetBudget:
    """A predeclared metric threshold, separate from protocol admission."""

    metric: str
    maximum: float | None = None
    minimum: float | None = None

    def __post_init__(self):
        identity(self.metric, "budget metric")
        if (self.maximum is None) == (self.minimum is None):
            raise ValueError("supply exactly one maximum or minimum")
        value = self.maximum if self.maximum is not None else self.minimum
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError("budget threshold must be finite and nonnegative")

    def to_spec(self):
        return asdict(self)


@dataclass(frozen=True, slots=True, init=False)
class BenchmarkTarget:
    """A bounded installed executor, not an arbitrary timing callback."""

    _bytes: bytes

    def __init__(self, kind: str, parameters: Mapping[str, Any]):
        choice(kind, {"runtime", "geometry", "artifacts", "stage", "preparation",
                      "operator", "state", "response", "projection"}, "benchmark target")
        if not isinstance(parameters, Mapping):
            raise TypeError("target parameters must be a mapping")
        document = {"schema": "pllm.benchmark_target.v1", "kind": kind,
                    "parameters": dict(parameters)}
        payload = canonical(document)
        if len(payload) > 4 << 20:
            raise ValueError("benchmark target metadata exceeds 4 MiB")
        _validate_target(document)
        object.__setattr__(self, "_bytes", payload)

    @property
    def kind(self):
        return self.to_spec()["kind"]

    @property
    def digest(self):
        return hashlib.sha256(b"pllm.benchmark_target.v1\0" + self._bytes).hexdigest()

    def to_spec(self):
        return json.loads(self._bytes)

    @classmethod
    def from_spec(cls, spec):
        if (not isinstance(spec, Mapping) or set(spec) != {"schema", "kind", "parameters"}
                or spec["schema"] != "pllm.benchmark_target.v1"):
            raise ValueError("invalid benchmark target record")
        return cls(spec["kind"], spec["parameters"])

    def with_params(self, **changes):
        document = self.to_spec()
        params = document["parameters"]
        model_changes = {key: value for key, value in changes.items()
                         if key.startswith(("pipeline__", "budget__", "deployment__"))}
        if model_changes:
            if "model" not in params:
                raise ValueError("this target has no Experiment")
            from pllm.configuration import Experiment
            model = params["model"]
            experiment = Experiment.from_spec(model["experiment"]).with_params(
                pipeline__profile=model["profile"])
            # Serialized pipelines retain their component contract, not the
            # original Python subclass. Preserve public slot aliases on replay.
            normalized = {}
            for key, value in model_changes.items():
                path = key.split("__")
                if path[0] == "pipeline" and path[1] in experiment.pipeline.components:
                    path.insert(1, "components")
                name = "__".join(path)
                if name in normalized:
                    raise ValueError("duplicate pipeline parameter aliases")
                normalized[name] = value
            paths = [key.split("__") for key in normalized]
            for index, path in enumerate(paths):
                for other in paths[index + 1:]:
                    length = min(len(path), len(other))
                    if path[:length] == other[:length]:
                        raise ValueError("overlapping parameter paths are not allowed")
            removed = [path[2] for path in paths if len(path) == 3
                       and path[:2] == ["pipeline", "components"]
                       and normalized["__".join(path)] is None]
            for slot in removed:
                if slot not in experiment.pipeline.components:
                    raise ValueError("cannot remove an unknown pipeline component")
                del normalized[f"pipeline__components__{slot}"]
            experiment = experiment.with_params(**normalized)
            if removed:
                experiment = experiment.with_params(pipeline__components={
                    key: value for key, value in experiment.pipeline.components.items() if key not in removed})
            params["model"] = _model(experiment, model["config"])
        for key, value in changes.items():
            if key in model_changes:
                continue
            if key not in params or key == "model":
                raise ValueError(f"unknown target parameter {key!r}")
            params[key] = value
        return type(self)(document["kind"], params)

    @classmethod
    def runtime(cls, role="client"):
        return cls("runtime", {"role": role})

    @classmethod
    def geometry(cls, experiment, *, config=None):
        return cls("geometry", {"model": _model(experiment, config)})

    @classmethod
    def stage(cls, experiment, *, role="mlp_gate_up", layer=0, phase="prefill",
              rows=16, weights="synthetic", seed=7, execution="protocol_local",
              kernel="native", storage="resident", config=None):
        return cls("stage", dict(model=_model(experiment, config), role=role, layer=layer,
            phase=phase, rows=rows, weights=weights, seed=seed, execution=execution,
            kernel=kernel, storage=storage))

    @classmethod
    def preparation(cls, experiment, *, role="mlp_gate_up", layer=0, rows=16,
                    weights="synthetic", seed=7, kernel="native", storage="resident",
                    chunk_rows=0, output_columns=None, credit_window=None, config=None):
        parameters = dict(model=_model(experiment, config), role=role,
            layer=layer, rows=rows, weights=weights, seed=seed, kernel=kernel,
            storage=storage, chunk_rows=chunk_rows)
        if output_columns is not None:
            parameters["output_columns"] = output_columns
        if credit_window is not None:
            parameters["credit_window"] = credit_window
        return cls("preparation", parameters)

    @classmethod
    def artifacts(cls, experiment, *, role="lm_head", layer=None, weights="synthetic",
                  seed=7, storage="shared", operation="import", config=None):
        return cls("artifacts", dict(model=_model(experiment, config), role=role, layer=layer,
            weights=weights, seed=seed, storage=storage, operation=operation))

    @classmethod
    def operator(cls, component, *, lanes=64, geometry=None):
        from pllm.configuration import ComponentRef
        if not isinstance(component, ComponentRef):
            raise TypeError("operator targets require a component")
        return cls("operator", {"component": component.to_spec(), "lanes": lanes,
                                "geometry": None if geometry is None else dict(geometry)})

    @classmethod
    def state(cls, experiment, *, operation="roundtrip", storage="memory",
              tokens=16, seed=7, config=None):
        return cls("state", dict(model=_model(experiment, config), operation=operation,
                                 storage=storage, tokens=tokens, seed=seed))

    @classmethod
    def response(cls, experiment, *, scenario=None, backend="native", wan=None, client_stage=None,
                 client_mlp=None, client_metadata=False):
        if scenario is None:
            scenario = Scenario.cold()
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        parameters = {"model": _model(experiment, None),
            "scenario": scenario.to_spec(), "backend": backend,
            "wan": None if wan is None else wan.to_spec()}
        if client_stage is not None:
            parameters["client_stage"] = dict(client_stage)
        if client_mlp is not None:
            parameters["client_mlp"] = dict(client_mlp)
        if type(client_metadata) is not bool:
            raise ValueError("client metadata attribution must be a boolean")
        if client_metadata:
            parameters["client_metadata"] = True
        return cls("response", parameters)

    @classmethod
    def projection(cls, experiment, *, construction, parameters, config=None):
        return cls("projection", {"model": _model(experiment, config),
            "construction": construction, "assumptions": dict(parameters)})


def _validate_target(document):
    kind, p = document["kind"], document["parameters"]
    stage = {"model", "role", "layer", "rows", "weights", "seed", "kernel", "storage"}
    fields = {
        "runtime": {"role"}, "geometry": {"model"},
        "stage": stage | {"phase", "execution"},
        "preparation": stage | {"chunk_rows"},
        "artifacts": {"model", "role", "layer", "weights", "seed", "storage", "operation"},
        "operator": {"component", "lanes", "geometry"},
        "state": {"model", "operation", "storage", "tokens", "seed"},
        "response": {"model", "scenario", "backend", "wan"},
        "projection": {"model", "construction", "assumptions"},
    }
    if kind == "response" and "client_stage" in p:
        fields[kind] = fields[kind] | {"client_stage"}
    if kind == "response" and "client_mlp" in p:
        fields[kind] = fields[kind] | {"client_mlp"}
    if kind == "response" and "client_metadata" in p:
        fields[kind] = fields[kind] | {"client_metadata"}
    if kind == "preparation" and "output_columns" in p:
        fields[kind] = fields[kind] | {"output_columns"}
        if "credit_window" in p:
            fields[kind] = fields[kind] | {"credit_window"}
    if set(p) != fields[kind]:
        raise ValueError("benchmark target fields differ from its contract")
    if "model" in p:
        m = p["model"]
        if not isinstance(m, dict) or set(m) != {"experiment", "profile", "config"}:
            raise ValueError("invalid target model declaration")
        from pllm.configuration import Experiment
        _model(Experiment.from_spec(m["experiment"]).with_params(pipeline__profile=m["profile"]), m["config"])
    if kind == "runtime":
        choice(p["role"], {"python", "sdk", "numpy", "transport", "native", "client", "prepared_client", "preparation"}, "runtime role")
    if kind in {"stage", "preparation", "artifacts"}:
        identity(p["role"], "semantic role")
        if p["layer"] is not None:
            bounded_int(p["layer"], 0, 1023, "layer")
        choice(p["weights"], {"synthetic", "checkpoint"}, "weight provenance")
    if "seed" in p:
        bounded_int(p["seed"], 0, (1 << 32) - 1, "public fixture seed")
    if kind in {"stage", "preparation"}:
        bounded_int(p["rows"], 1, 4096, "rows")
        choice(p["kernel"], {"native", "digit4", "digit8", "digit8_native"}, "integer kernel")
        choice(p["storage"], {"resident", "paged"}, "weight storage")
    if kind == "stage":
        choice(p["phase"], {"prefill", "decode"}, "decoder phase")
        choice(p["execution"], {"clear", "protocol_local", "transport"}, "execution scope")
        if p["phase"] == "decode" and p["rows"] != 1:
            raise ValueError("decode stage targets require one row")
    elif kind == "preparation":
        bounded_int(p["chunk_rows"], 0, 4096, "mask row chunk")
        if "output_columns" in p:
            bounded_int(p["output_columns"], 0, 65536, "output stripe columns")
            bounded_int(p.get("credit_window", 1), 1, 8, "stripe credit window")
            if p["chunk_rows"] or p["storage"] != "resident" or p["kernel"] not in {"native", "digit8_native"}:
                raise ValueError("output stripes require resident native kernels and full input batches")
    elif kind == "artifacts":
        choice(p["storage"], {"shared", "copied"}, "snapshot ownership")
        choice(p["operation"], {"import", "bind"}, "artifact operation")
    elif kind == "operator":
        from pllm.components import create_component
        c = p["component"]
        if not isinstance(c, dict) or set(c) != {"component", "params"}:
            raise ValueError("invalid operator component")
        create_component(c["component"], c["params"])
        bounded_int(p["lanes"], 1, 256, "lanes")
        if p["geometry"] is not None and not isinstance(p["geometry"], dict):
            raise ValueError("operator geometry must be a mapping")
    elif kind == "state":
        choice(p["operation"], {"roundtrip", "attention", "restart", "decoder_restart"}, "state operation")
        choice(p["storage"], {"memory", "sealed_local", "sealed_stream"}, "state storage")
        bounded_int(p["tokens"], 1, 1 << 20, "state tokens")
    elif kind == "response":
        Scenario(**p["scenario"])
        if "client_metadata" in p:
            if (p["client_metadata"] is not True or p["backend"] != "native"
                    or "client_stage" in p or "client_mlp" in p):
                raise ValueError("metadata attribution requires a separate native response case")
        if "client_mlp" in p:
            selected = p["client_mlp"]
            if not isinstance(selected, dict) or set(selected) != {"tile_rows"}:
                raise ValueError("invalid research MLP tiling selection")
            bounded_int(selected["tile_rows"], 0, 256, "MLP tile rows")
            if p["backend"] != "native" or "client_stage" in p:
                raise ValueError("research MLP tiling requires separate native response cases")
        choice(p["backend"], {"native", "docker", "auto"}, "serving backend")
        if p["wan"] is not None:
            from pllm.deployment import WanConditions
            WanConditions.from_spec(p["wan"])
        if "client_stage" in p:
            selected = p["client_stage"]
            if not isinstance(selected, dict) or set(selected) != {"owner", "role", "layer"}:
                raise ValueError("invalid research client-stage selection")
            choice(selected["owner"], {"python", "native"}, "client-stage owner")
            identity(selected["role"], "semantic role")
            bounded_int(selected["layer"], 0, 1023, "layer")
            if p["backend"] != "native":
                raise ValueError("research client-stage responses require native roles")
    elif kind == "projection":
        choice(p["construction"], {"layer_kv", "restart"}, "projection construction")
        if not isinstance(p["assumptions"], dict):
            raise ValueError("projection assumptions must be a mapping")
        fields = ({"fixed_bytes", "live_layers"} if p["construction"] == "layer_kv"
                  else {"bootstrap_bytes", "publication_bytes", "control_bytes",
                        "resumed_bytes", "resumptions"})
        if set(p["assumptions"]) != fields:
            raise ValueError("projection assumptions differ from construction")
        for key, value in p["assumptions"].items():
            bounded_int(value, 1 if key in {"live_layers", "resumptions"} else 0, 1 << 50, key)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    target: BenchmarkTarget
    build: BuildRef = BuildRef()
    budgets: tuple[TargetBudget, ...] = ()

    def __post_init__(self):
        identity(self.name, "case name")
        if not isinstance(self.target, BenchmarkTarget) or not isinstance(self.build, BuildRef):
            raise TypeError("case requires a BenchmarkTarget and BuildRef")
        if not isinstance(self.budgets, (tuple, list)) or any(not isinstance(b, TargetBudget) for b in self.budgets):
            raise TypeError("budgets must contain TargetBudget records")
        if len({b.metric for b in self.budgets}) != len(self.budgets):
            raise ValueError("case budget metrics must be unique")
        object.__setattr__(self, "budgets", tuple(self.budgets))

    def to_spec(self):
        return {"name": self.name, "target": self.target.to_spec(), "build": self.build.to_spec(),
                "budgets": [budget.to_spec() for budget in self.budgets]}

    @classmethod
    def from_spec(cls, spec):
        if not isinstance(spec, Mapping) or set(spec) != {"name", "target", "build", "budgets"}:
            raise ValueError("invalid benchmark case")
        return cls(spec["name"], BenchmarkTarget.from_spec(spec["target"]),
                   BuildRef(**spec["build"]), tuple(TargetBudget(**b) for b in spec["budgets"]))


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    cases: tuple[BenchmarkCase, ...]
    sampling: Sampling = Sampling()
    name: str = "benchmark"

    def __post_init__(self):
        identity(self.name, "suite name")
        if not isinstance(self.cases, (tuple, list)) or not 1 <= len(self.cases) <= 256:
            raise ValueError("suite requires 1..256 cases")
        if any(not isinstance(case, BenchmarkCase) for case in self.cases):
            raise TypeError("suite cases must be BenchmarkCase records")
        if len({case.name for case in self.cases}) != len(self.cases):
            raise ValueError("case names must be unique")
        if not isinstance(self.sampling, Sampling):
            raise TypeError("sampling must be Sampling")
        if len(self.cases) * self.sampling.processes > 1024:
            raise ValueError("suite exceeds 1024 isolated trials")
        object.__setattr__(self, "cases", tuple(self.cases))

    def to_spec(self):
        return {"schema": "pllm.benchmark_suite.v1", "name": self.name,
                "cases": [case.to_spec() for case in self.cases],
                "sampling": self.sampling.to_spec()}

    @classmethod
    def from_spec(cls, spec):
        if (not isinstance(spec, Mapping) or set(spec) != {"schema", "name", "cases", "sampling"}
                or spec["schema"] != "pllm.benchmark_suite.v1"):
            raise ValueError("invalid benchmark suite")
        return cls(tuple(BenchmarkCase.from_spec(c) for c in spec["cases"]),
                   Sampling(**spec["sampling"]), spec["name"])

    @property
    def digest(self):
        return digest(self.to_spec())
