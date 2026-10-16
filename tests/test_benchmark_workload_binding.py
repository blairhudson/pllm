"""Worker identity semantics only: no model loading, providers or benchmark run.

Source overlay tests uninstalled changes without builds/installing a wheel.
"""
from copy import deepcopy
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_source_worker(monkeypatch):
    import pllm
    for name in ("_benchmark_workload", "_benchmark_worker"):
        qualified = "pllm." + name
        spec = importlib.util.spec_from_file_location(qualified, ROOT / "python" / "pllm" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        monkeypatch.setattr(pllm, name, module, raising=False)
        spec.loader.exec_module(module)
    return sys.modules["pllm._benchmark_worker"]


@pytest.fixture
def worker(monkeypatch):
    return load_source_worker(monkeypatch)


def target(audit=True):
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.metrics import BenchmarkTarget
    from pllm.profiles import ClientOnlyCpu
    exp = Experiment("binding-only", ClientOnlyCpu(Model.tiny()), Deployment.local(root="local://binding"), ExecutionBudget(1, 128, 2))
    return BenchmarkTarget.response(exp, state_audit=audit)


def canonical(value):
    # Independent of production canonicalization/builders.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def measurement(worker, tmp_path, *, prompt="private sentinel Ω not for publication", salt=b"x" * 32):
    from pllm.metrics import Sampling
    m = worker.Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": salt.hex(),
                            "inputs": {"prompt": prompt}, "directory": str(tmp_path)})
    context = prompt.encode() if isinstance(prompt, str) else canonical(prompt)
    prompt_hash = hashlib.sha256(b"pllm.benchmark.prompt.v1\0" + salt + context).hexdigest()
    m.fixture = {"kind": "checkpoint", "digest": "1" * 64, "metadata": {"cached_checkpoint": True}}
    m.details = {"report": {"configuration": {"prompt_digest": prompt_hash, "source_lock_digest": "1" * 64}}}
    m.comparison.update(workload_digest="a" * 64, measurement_digest="b" * 64)
    return m


@pytest.mark.parametrize("prompt", ["private sentinel Ω not for publication", [{"role": "user", "content": "private Ω message"}]])
def test_actual_measurement_domain_separated_private_commit_and_replay(worker, tmp_path, prompt):
    m = measurement(worker, tmp_path, prompt=prompt)
    m.finalize_workload(target())
    binding = m.details["workload_binding"]
    independent = hmac.new(b"x" * 32, canonical({"schema": "pllm.benchmark.private_inputs.v1", "inputs": {"prompt": prompt}}), hashlib.sha256).hexdigest()
    assert binding["private_inputs_commitment"] == independent
    assert binding["public_workload_digest"] == "a" * 64
    assert m.comparison["workload_digest"] == hashlib.sha256(canonical(binding)).hexdigest()
    assert m.comparison["workload_digest"] != "a" * 64  # Regression: old test faked this equality.
    witness = m.details["workload_measurement"]
    assert witness == {"schema": "pllm.response_state_audit.workload_measurement.v1",
        "workload_binding_schema": "pllm.response_state_audit.workload_binding.v1", "target_measurement_digest": "b" * 64}
    assert m.comparison["measurement_digest"] == hashlib.sha256(canonical(witness)).hexdigest()
    assert binding["fixture_digest"] == m.fixture["digest"] == "1" * 64
    assert binding["prompt_commitment"] == m.details["report"]["configuration"]["prompt_digest"]
    published = json.dumps({"details": m.details, "comparison": m.comparison, "fixture": m.fixture})
    assert "private sentinel" not in published and "private Ω message" not in published
    assert (b"x" * 32).hex() not in published


def test_salt_or_private_inputs_change_identity_without_public_token_changes(worker, tmp_path):
    values = []
    for prompt, salt in (("private A", b"x"*32), ("private A", b"x"*32), ("private B", b"x"*32), ("private A", b"y"*32)):
        m = measurement(worker, tmp_path, prompt=prompt, salt=salt)
        m.finalize_workload(target())
        values.append((m.details["workload_binding"], m.comparison["workload_digest"]))
    assert values[0] == values[1]
    assert values[0] != values[2] and values[0] != values[3]
    assert all(binding["public_workload_digest"] == "a"*64 for binding, _ in values)


@pytest.mark.parametrize("completed", [False, True])
def test_ordinary_targets_keep_exact_historical_hmac(worker, tmp_path, completed):
    m = measurement(worker, tmp_path)
    expected = hmac.new(b"x"*32, canonical({"target_workload": "a"*64, "inputs": m.inputs}), hashlib.sha256).hexdigest()
    m.finalize_workload(target(False), completed=completed)
    assert m.comparison["workload_digest"] == expected
    assert m.comparison["measurement_digest"] == "b"*64
    assert "workload_binding" not in m.details


def test_failed_state_audit_does_not_invent_witness(worker, tmp_path):
    m = measurement(worker, tmp_path)
    m.details = {}
    m.finalize_workload(target(), completed=False)
    assert "workload_binding" not in m.details


@pytest.mark.parametrize("field", ["prompt_digest", "source_lock_digest"])
def test_actual_measurement_rejects_prompt_or_fixture_mismatch_before_publication(worker, tmp_path, field):
    m = measurement(worker, tmp_path)
    m.details["report"]["configuration"][field] = "f" * 64
    before = deepcopy((m.details, m.comparison))
    with pytest.raises(ValueError, match="fixture/private prompt"):
        m.finalize_workload(target())
    assert (m.details, m.comparison) == before
