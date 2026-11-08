"""Genuine native byte getters through production bounded parsing/admission.

Run after rebuilding the native wheel. No synthetic contract or executable owner.
"""
import sys

import pytest

from pllm import _native
from pllm.configuration import Deployment, Experiment, ForwardBudget, Model
from pllm.numeric import SourceFloat32
from pllm.profiles import ClientOnlyCpu
from pllm.runtime import forward_binding as binding, source_io
from pllm.runtime.bundle_storage import CHUNK_BYTES, FilePayload
from pllm.runtime.forward_inputs import ForwardInputs
from pllm.runtime.source_io import SourceReadPolicy
from pllm.sources.generated_encoder import write_checkpoint


# Positive functional setup only: reuse the observed 128-MiB headroom of the
# existing large-held-metadata/cap-independence test. Not a production allowance
# or promised frontend 64-MiB cap; direct native setup and boundary caps stay put.
FUNCTIONAL_SETUP_BYTES = 128 << 20


@pytest.fixture
def genuine(tmp_path):
    root = write_checkpoint(tmp_path / "source")
    spec = Experiment("json-admission", ClientOnlyCpu(Model(str(root)), numeric=SourceFloat32()),
                      Deployment.local(root=str(root / "deployment")), ForwardBudget(2, 8))
    contract = _native.ForwardContract.compile(
        (root / "config.json").read_bytes(), spec.pipeline.canonical_bytes(),
        max_sequence=8, memory_bytes=64 << 20)
    assert type(contract) is _native.ForwardContract
    return root, spec, contract


def test_genuine_documents_prior_owners_exact_and_short_caps(genuine, monkeypatch):
    root, spec, contract = genuine
    quote = binding._quote(contract, 8, admit=lambda size: None)
    held = quote["metadata_bytes"]
    held += SourceReadPolicy(bounded_json=True)._json_census((root / "config.json").read_bytes()).retained_bytes
    loads, hook = source_io.json.loads, source_io._object
    for getter in (contract.plan_bytes, contract.schedule_bytes, contract.requirements_bytes):
        raw = getter()
        assert type(raw) is bytes
        census = SourceReadPolicy(bounded_json=True)._json_census(raw)
        cap = held + census.parse_peak_bytes
        calls = []
        def admit(size):
            calls.append(("admit", held + size))
            if held + size > cap - 1:
                raise binding.ForwardBindingError("controlled tight cap")
        def forbidden(*args, **kwargs):
            pytest.fail("short parse admission reached decoder/hook/freeze")
        with monkeypatch.context() as patch:
            patch.setattr(source_io.json, "loads", forbidden)
            patch.setattr(source_io, "_object", forbidden)
            patch.setattr(source_io, "_bounded_loads", forbidden)
            patch.setattr(source_io, "_BOUNDED_OBJECT", forbidden)
            patch.setattr(binding, "_freeze", forbidden)
            with pytest.raises(binding.ForwardBindingError, match="tight cap"):
                SourceReadPolicy(bounded_json=True, metadata_admit=admit)._parse_json(raw)
        assert calls == [("admit", cap)]
        def exact(size):
            assert held + size <= cap
        transferred = []
        value, actual = SourceReadPolicy(bounded_json=True, metadata_admit=exact)._parse_json(raw, retain=transferred.append)
        assert actual == census and transferred == [census.retained_bytes]
        assert value == loads(raw, object_pairs_hook=hook)
        held += census.retained_bytes
        # Same destinations production holds before deriving/finally freezing.
        held += census.freeze_bytes


def test_quote_uses_small_production_envelope_before_decode(genuine, monkeypatch):
    _, _, contract = genuine
    calls = []
    def reject(size):
        calls.append(size)
        raise binding.ForwardBindingError("quote cap")
    def forbidden(*args, **kwargs):
        pytest.fail("quote refusal reached decoder or pair hook")
    with monkeypatch.context() as patch:
        patch.setattr(source_io.json, "loads", forbidden)
        patch.setattr(source_io, "_object", forbidden)
        patch.setattr(source_io, "_bounded_loads", forbidden)
        patch.setattr(source_io, "_BOUNDED_OBJECT", forbidden)
        with pytest.raises(binding.ForwardBindingError, match="quote cap"):
            binding._quote(contract, 8, admit=reject)
    assert calls == [binding._QUOTE_BOOTSTRAP_BYTES]
    actual = []
    quote = binding._quote(contract, 8, admit=actual.append)
    assert quote["sequence_length"] == 8
    assert max(actual) == binding._QUOTE_BOOTSTRAP_BYTES
    census = SourceReadPolicy(bounded_json=True)._json_census(contract.quote(8))
    assert census.parse_peak_bytes <= binding._QUOTE_PARSE_BYTES
    assert (census.parse_peak_bytes + census.retained_bytes
            <= binding._QUOTE_BOOTSTRAP_BYTES <= binding._NATIVE_NUMERICAL_FLOOR_BYTES)


def test_preload_holds_documents_and_freeze_before_compile_or_description(genuine, monkeypatch):
    _, spec, contract = genuine
    raws = [contract.plan_bytes(), contract.schedule_bytes(), contract.requirements_bytes()]
    observed = []
    original = SourceReadPolicy._parse_json
    def parse(policy, raw, **kwargs):
        assert policy.bounded_json is True
        if raw in raws:
            admission = policy.metadata_admit.__self__
            before = admission.held_bytes
            value, census = original(policy, raw, **kwargs)
            assert admission.held_bytes == before + census.retained_bytes
            assert admission.peak_bytes >= before + census.parse_peak_bytes + 8 * CHUNK_BYTES
            observed.append((raws.index(raw), before, census))
            return value, census
        return original(policy, raw, **kwargs)
    monkeypatch.setattr(SourceReadPolicy, "_parse_json", parse)
    description = binding.describe_forward_source(spec, memory_bytes=FUNCTIONAL_SETUP_BYTES)
    assert [index for index, _, _ in observed] == [0, 1, 2]
    for previous, current in zip(observed, observed[1:]):
        assert current[1] >= previous[1] + previous[2].retained_bytes + previous[2].freeze_bytes
    assert description["metadata_bytes"] >= sum(c.retained_bytes + c.freeze_bytes for _, _, c in observed)
    reservation = description["lifetime_reservation"]
    cap = reservation["admission_bytes"]
    original_read = FilePayload.read_at
    def metadata_only(snapshot, offset, size):
        assert offset in (0, 8), "short complete cap reached coefficient RAM"
        return original_read(snapshot, offset, size)
    with monkeypatch.context() as patch:
        patch.setattr(FilePayload, "read_at", metadata_only)
        patch.setattr(binding, "_freeze", lambda *_: pytest.fail("short complete cap reached freeze"))
        with pytest.raises(binding.ForwardBindingError):
            spec.compile_forward(memory_bytes=cap - 1)
    original_preload, original_freeze = binding._preload, binding._freeze
    loaded_records, frozen_records = [], []
    frozen_bytes = sum(SourceReadPolicy(bounded_json=True)._json_census(raw).freeze_bytes for raw in raws)
    def preload(*args, **kwargs):
        loaded = original_preload(*args, **kwargs)
        loaded_records.append(loaded)
        return loaded
    def freeze(value):
        loaded = loaded_records[-1]
        assert loaded.reservation["required_bytes"] >= loaded.reservation["metadata_phase_peak_bytes"]
        assert loaded.metadata_bytes >= frozen_bytes
        frozen_records.append(loaded)
        return original_freeze(value)
    with monkeypatch.context() as patch:
        patch.setattr(binding, "_preload", preload)
        patch.setattr(binding, "_freeze", freeze)
        with spec.compile_forward(memory_bytes=cap) as model:
            assert model.lifetime_reservation == reservation
            assert binding.describe_forward_source(spec, memory_bytes=cap)["lifetime_reservation"] == reservation
            assert "pooler_output" in model.forward(ForwardInputs([1, 2])).outputs
    assert len(loaded_records) == 2
    assert all(any(frozen is loaded for frozen in frozen_records) for loaded in loaded_records)


def test_execution_quote_admitted_before_parser_and_attempt_unspent(genuine, monkeypatch):
    _, spec, _ = genuine
    with spec.compile_forward(memory_bytes=FUNCTIONAL_SETUP_BYTES) as model:
        requests = model._requests
        def reject(*args, **kwargs):
            raise binding.ForwardBindingError("prequote capacity")
        def forbidden(*args, **kwargs):
            pytest.fail("execution quote decoded before admission")
        with monkeypatch.context() as patch:
            patch.setattr(binding, "_admit", reject)
            patch.setattr(source_io.json, "loads", forbidden)
            patch.setattr(source_io, "_object", forbidden)
            patch.setattr(source_io, "_bounded_loads", forbidden)
            patch.setattr(source_io, "_BOUNDED_OBJECT", forbidden)
            with pytest.raises(binding.ForwardBindingError, match="prequote capacity"):
                model.forward(ForwardInputs([1]))
        assert model._requests == requests and not model._active
        assert "pooler_output" in model.forward(ForwardInputs([1])).outputs


@pytest.mark.parametrize("caller,held", [
    (64 << 20, 0), (64 << 20, 4 << 20), (sys.maxsize, sys.maxsize // 3),
])
def test_bootstrap_partition_formula_only(caller, held):
    budget = binding._native_bootstrap_budget(caller, held)
    assert budget == (4 * caller) // 5 - held - binding._METADATA_PHASE_BYTES
    assert (held + binding._METADATA_PHASE_BYTES + budget) * 5 <= caller * 4


@pytest.mark.parametrize("caller,held", [
    (0, 0), (True, 0), (sys.maxsize + 1, 0), (64 << 20, -1),
    (64 << 20, sys.maxsize + 1), (8 << 20, 0), (64 << 20, 64 << 20),
])
def test_bootstrap_partition_refuses_nonpositive_or_unaddressable(caller, held):
    with pytest.raises(binding.ForwardBindingError, match="bootstrap"):
        binding._native_bootstrap_budget(caller, held)


def test_bootstrap_fails_closed_if_production_quote_outgrows_native_floor(monkeypatch):
    # Perturb a capacity constant, never native ownership/authority.
    monkeypatch.setattr(binding, "_QUOTE_BOOTSTRAP_BYTES", binding._NATIVE_NUMERICAL_FLOOR_BYTES + 1)
    with pytest.raises(binding.ForwardBindingError, match="bootstrap"):
        binding._native_bootstrap_budget(64 << 20, 0)


@pytest.mark.parametrize("damage", ["missing", "zero", "boolean", "over_budget", "no_floor"])
def test_native_metadata_and_budget_mandatory_inspection_validation_only(genuine, damage):
    _, _, contract = genuine
    quote = dict(binding._quote(contract, 8, admit=lambda size: None))
    budget = contract.memory_bytes()
    if damage == "missing":
        del quote["metadata_bytes"]
    elif damage == "zero":
        quote["metadata_bytes"] = 0
    elif damage == "boolean":
        quote["metadata_bytes"] = True
    elif damage == "over_budget":
        budget = quote["required_bytes"] - 1
    else:
        quote["metadata_bytes"] = quote["required_bytes"] - binding._NATIVE_NUMERICAL_FLOOR_BYTES + 1
    # A damaged descriptive dict is never passed to any executable factory.
    with pytest.raises(binding.ForwardBindingError, match="quote is invalid"):
        binding._validate_native_quote(quote, 8, budget)


def test_first_quote_partition_proof_and_cap_independent_complete_recompile(genuine, monkeypatch):
    from test_forward_runtime_ownership import _write_resolver_metadata
    root, spec, _ = genuine
    _write_resolver_metadata(root, repo_id="r" * (1 << 20))
    partition, quotes, decodes = [], [], []
    original_budget, original_quote = binding._native_bootstrap_budget, binding._quote
    original_loads = source_io._bounded_loads
    original_parse = SourceReadPolicy._parse_json
    def budget(caller, held):
        result = original_budget(caller, held)
        partition.append((caller, held, result))
        return result
    def quote(contract, sequence, **kwargs):
        caller, held, native_budget = partition[-1]
        assert type(contract) is _native.ForwardContract
        assert contract.memory_bytes() == native_budget
        assert native_budget == (4 * caller) // 5 - held - binding._METADATA_PHASE_BYTES
        assert held >= 4 << 20  # Actual retained resolver identity, not nominal cap.
        assert binding._QUOTE_BOOTSTRAP_BYTES <= binding._NATIVE_NUMERICAL_FLOOR_BYTES
        def parse(policy, raw, **parse_kwargs):
            # Before raw.decode: native admission already authorizes M+W <=
            # subbudget; Q<=W absorbs unknown M without inspecting opaque owner.
            assert policy.bounded_json is True
            assert held + binding._METADATA_PHASE_BYTES + native_budget <= (4 * caller) // 5
            return original_parse(policy, raw, **parse_kwargs)
        def loads(decoded):
            decodes.append(native_budget)
            return original_loads(decoded)  # Genuine captured C scanner.
        with monkeypatch.context() as patch:
            patch.setattr(SourceReadPolicy, "_parse_json", parse)
            patch.setattr(source_io, "_bounded_loads", loads)
            result = original_quote(contract, sequence, **kwargs)
        assert result["required_bytes"] <= native_budget
        assert (held + binding._METADATA_PHASE_BYTES + result["metadata_bytes"]
                + binding._QUOTE_BOOTSTRAP_BYTES <= (4 * caller) // 5)
        quotes.append((caller, held, result))
        return result
    monkeypatch.setattr(binding, "_native_bootstrap_budget", budget)
    monkeypatch.setattr(binding, "_quote", quote)
    first = binding.describe_forward_source(spec, memory_bytes=128 << 20)
    second = binding.describe_forward_source(spec, memory_bytes=192 << 20)
    reservation = first["lifetime_reservation"]
    assert second["lifetime_reservation"] == reservation
    caller, held, resources = quotes[0]
    prerequisite = held + binding._METADATA_PHASE_BYTES + resources["required_bytes"]
    assert reservation["native_compile_prerequisite_bytes"] == prerequisite
    assert reservation["metadata_phase_peak_bytes"] >= prerequisite
    assert reservation["required_bytes"] == max(
        reservation["metadata_phase_peak_bytes"], prerequisite, reservation["import_peak_bytes"],
        reservation["max_call_bytes"] + spec.budget.requests * (
            reservation["output_bytes_per_result"] + reservation["result_metadata_bytes_per_result"]))
    cap = reservation["admission_bytes"]
    original_read = FilePayload.read_at
    def metadata_only(snapshot, offset, size):
        assert offset in (0, 8), "one-byte-short bootstrap/complete cap reached coefficients"
        return original_read(snapshot, offset, size)
    with monkeypatch.context() as patch:
        patch.setattr(FilePayload, "read_at", metadata_only)
        with pytest.raises((binding.ForwardBindingError, ValueError)):
            spec.compile_forward(memory_bytes=cap - 1)
    with spec.compile_forward(memory_bytes=cap) as model:
        assert model.contract.memory_bytes() == partition[-1][2] < cap
        assert model._memory_bytes == cap
        assert model.lifetime_reservation == reservation
    assert decodes and len(quotes) >= 3


def test_quote_exact_census_and_actual_forward_callback_freeze_code_slots():
    from pllm.runtime import forward_execution
    assert binding._QUOTE_POLICY.bounded_json is True
    assert binding._QUOTE_CENSUS.longest_token == 1024
    assert binding._QUOTE_CENSUS.retained_bytes == 30753
    assert binding._QUOTE_PARSE_BYTES == 622585
    assert binding._QUOTE_BOOTSTRAP_BYTES == 653338 < 8 << 20
    for functions in (binding._FORWARD_JSON_FRAME_FUNCTIONS,
                      forward_execution._EXECUTION_JSON_FRAME_FUNCTIONS):
        inventory = source_io._code_slot_inventory(functions)
        assert inventory and all(slots <= 137 for _, slots in inventory), inventory
    assert binding._FORWARD_JSON_FRAMES_SUPPORTED
    assert forward_execution._EXECUTION_JSON_FRAMES_SUPPORTED


def test_first_metadata_and_every_source_lock_snapshot_header_hash_route_bounded(genuine, monkeypatch):
    from test_forward_runtime_ownership import _write_resolver_metadata
    root, spec, _ = genuine
    _write_resolver_metadata(root, repo_id="bounded-source-routes")
    seen = set()
    read, parse = SourceReadPolicy.read_bytes, SourceReadPolicy._parse_json
    chunks, directory = SourceReadPolicy.chunks, SourceReadPolicy.directory_files
    def reading(policy, path, **kwargs):
        assert policy.bounded_json is True
        seen.add("read")
        return read(policy, path, **kwargs)
    def parsing(policy, raw, **kwargs):
        assert policy.bounded_json is True
        seen.add("parse")
        return parse(policy, raw, **kwargs)
    def streaming(policy, path):
        assert policy.bounded_json is True
        seen.add("chunks")
        yield from chunks(policy, path)
    def inventory(policy, path, **kwargs):
        assert policy.bounded_json is True
        seen.add("inventory")
        return directory(policy, path, **kwargs)
    monkeypatch.setattr(SourceReadPolicy, "read_bytes", reading)
    monkeypatch.setattr(SourceReadPolicy, "_parse_json", parsing)
    monkeypatch.setattr(SourceReadPolicy, "chunks", streaming)
    monkeypatch.setattr(SourceReadPolicy, "directory_files", inventory)
    description = binding.describe_forward_source(spec, memory_bytes=FUNCTIONAL_SETUP_BYTES)
    assert description["execution_admitted"] is False
    assert seen == {"read", "parse", "chunks", "inventory"}
