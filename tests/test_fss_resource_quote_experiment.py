"""Mock-only quote checks plus inert recipe declarations; no checkpoint/description run."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest


@pytest.fixture
def study(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/papers/fusefss_resource_quote.py"
    spec = importlib.util.spec_from_file_location("fss_resource_quote_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Exercise the experiment's adapter closures, not admission or native work.
    topology = SimpleNamespace(_stopping=SimpleNamespace(is_set=lambda: False),
                               _observe_failure=Mock())
    frozen = SimpleNamespace(snapshot_pins={"source": "pin"})
    custody = object()
    module._test_topology, module._test_source, module._test_custody = topology, frozen, custody
    monkeypatch.setattr(module, "build_roles", Mock(return_value=topology))
    monkeypatch.setattr(module, "resolve_client_source", Mock(return_value=frozen))
    monkeypatch.setattr(module, "_fss_qualification_price", Mock(return_value=object()))

    def funded(owner, *, resolve, price, produce, consume, **policy):
        assert owner is topology
        lifecycle = lambda phase: None
        source = resolve(lifecycle)
        assert source is frozen
        pins = source.snapshot_pins
        price(source, pins, 0)
        price(source, pins, 70001)
        result = produce(source, pins, custody, lifecycle)
        return consume(result, source, pins), source, pins

    monkeypatch.setattr(module, "_funded_source_qualification", Mock(side_effect=funded))
    return module


@pytest.mark.parametrize("extent,refused", [
    ("complete", False), ("complete", True), ("partial", True),
])
@pytest.mark.parametrize("write_output", [False, True])
@pytest.mark.parametrize("work_policy", [None, "mixed-work-40"])
@pytest.mark.parametrize("observe_peer_frames", [False, True])
def test_report_preserves_raw_description_and_forces_no_admission(
    study, monkeypatch, capsys, tmp_path, extent, refused, write_output, work_policy, observe_peer_frames,
):
    # Synthetic public DTOs only, including full-width u128 values. In particular,
    # partial kernel/work counters are deliberately not a coherent prefix.
    large = (1 << 127) + 123
    first = {
        "stage": "ResourceQuote", "gate": "MaterialBytes", "node": 7,
        "kernel": 3, "required": large, "used": 19, "limit": 23,
    } if refused else None
    stop = {
        "stage": "ResourceQuote", "gate": "PublicCopyBytes" if observe_peer_frames else "FrameBytes", "node": 8,
        "kernel": 4, "required": large + 1, "used": None, "limit": None,
    } if extent == "partial" else None
    description = {
        "schema": "pllm.fss.source_description.v1",
        "extent": extent,
        "resources": {
            "kernels": 3, "work": large + 2,
            "private_material_cumulative_bytes": large + 3,
            "primitive_requests_per_party": 17,
        },
        "first_material_refusal": first,
        "first_material_selection": 0 if refused else None,
        "selection": 0, "node": 8,
        "stop_error": "Resource" if extent == "partial" else None,
        "stop_context": stop,
        "material_wire_bytes": None,
        # Even a supplied admission flag cannot override the experiment's refusal
        # to make any execution-admission claim.
        "execution_admitted": True,
    }
    if work_policy is not None:
        description.update(work_policy=work_policy, work_ceiling=1 << 40)
    if observe_peer_frames:
        description.update(observe_peer_frames=True, first_frame_refusal={
            "stage": "ResourceQuote", "gate": "FrameBytes", "node": 7,
            "kernel": 3, "required": large + 7, "used": None, "limit": 8 << 30,
        } if refused else None, first_frame_selection=0 if refused else None)
    before = deepcopy(description)
    configuration = object()
    resolve = Mock(return_value=SimpleNamespace(configuration=configuration))
    describe = Mock(return_value=description)
    monkeypatch.setattr(study, "resolve_target", resolve)
    monkeypatch.setattr(study, "describe_source", describe)
    output = tmp_path / "quote.json"

    args = ["--output", str(output)] if write_output else []
    if work_policy is not None:
        args += ["--work-policy", work_policy]
    if observe_peer_frames:
        args += ["--observe-peer-frames"]
    study.main(args)

    resolve.assert_called_once_with(
        "examples/papers/fusefss.py:qwen_empty_system",
        factory=True, no_input=True, trust_python=True, output_format="json",
    )
    options = {}
    if work_policy is not None:
        options["work_policy"] = work_policy
    if observe_peer_frames:
        options["observe_peer_frames"] = True
    describe.assert_called_once_with(configuration, _source=study._test_source,
        _expected_pins=study._test_source.snapshot_pins, _custody=study._test_custody,
        input_rows=14, outputs=1, cancelled=ANY, lifecycle=ANY,
        on_failure=study._test_topology._observe_failure, **options)
    study.build_roles.assert_called_once_with(configuration, memory_budget_bytes=None,
                                              memory_safety_margin_percent=25)
    assert describe.call_args.args[0] is configuration
    assert description == before
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    actual = json.loads(captured.out)
    assert actual == {**before, "execution_admitted": False}
    assert actual["execution_admitted"] is False
    assert actual["material_wire_bytes"] is None
    if not observe_peer_frames:
        assert not {"observe_peer_frames", "first_frame_refusal", "first_frame_selection"} & actual.keys()
    assert type(actual["resources"]["work"]) is int
    assert actual["resources"]["work"] == large + 2
    assert str(large + 2) in captured.out
    if write_output:
        assert output.read_text(encoding="utf-8") == captured.out
    else:
        assert not output.exists()


def test_unknown_work_policy_rejected_before_factory(study, monkeypatch):
    resolve = Mock()
    describe = Mock()
    monkeypatch.setattr(study, "resolve_target", resolve)
    monkeypatch.setattr(study, "describe_source", describe)
    with pytest.raises(SystemExit) as failure:
        study.main(["--work-policy", "unbounded"])
    assert failure.value.code == 2
    resolve.assert_not_called()
    describe.assert_not_called()


def test_quote_factory_declaration_keeps_saved_funding_envelope(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.papers.fusefss import qwen_empty_system
    from pllm.configuration import Deployment, ExecutionBudget, Experiment, Pipeline
    from pllm.profiles import FuseFssSourceCpu

    actual = qwen_empty_system()
    legacy = Experiment("fusefss-qwen-empty-system",
        FuseFssSourceCpu(actual.pipeline.model,
            numeric_profile="fusefss.dense-rope-checked-ranges-ring50-q12.v1",
            capacity="decoder", material_delivery="batch-v1", online_schedule="tensor-v1"),
        Deployment.local(root=".pllm/papers/fusefss-checked-ranges"),
        ExecutionBudget(requests=1, max_input_tokens=15, max_new_tokens=1))
    assert type(actual.pipeline) is Pipeline
    assert actual.canonical_bytes() == legacy.canonical_bytes()
    assert actual.configuration_digest() == legacy.configuration_digest()
    # Saved funding envelope is 15/1; quote's actual 14/1 cut is not run here.
    assert actual.budget == legacy.budget


class TruthyOption:
    def __init__(self):
        self.bool_calls = 0

    def __bool__(self):
        self.bool_calls += 1
        return True


class TruthinessTrap(TruthyOption):
    def __bool__(self):
        self.bool_calls += 1
        raise AssertionError("option truthiness must not be evaluated")


@pytest.mark.parametrize("observe_peer_frames", [
    None, 0, 1, "true", [], TruthyOption(), TruthinessTrap(),
], ids=["none", "zero", "one", "string", "container", "truthy", "trap"])
def test_helper_rejects_nonbool_frame_option_before_factory(study, monkeypatch, observe_peer_frames):
    resolve = Mock()
    describe = Mock()
    monkeypatch.setattr(study, "resolve_target", resolve)
    monkeypatch.setattr(study, "describe_source", describe)
    with pytest.raises(study.FssSourceError, match="^FuseFSS observe_peer_frames requires bool$"):
        study.resource_quote(observe_peer_frames=observe_peer_frames)
    resolve.assert_not_called()
    describe.assert_not_called()
    if isinstance(observe_peer_frames, TruthyOption):
        assert observe_peer_frames.bool_calls == 0
