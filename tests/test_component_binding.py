"""Explicit declarations, native code correspondence and client bridge pricing.

These normal regression definitions require a freshly built extension. Merely
authoring them is not execution, qualification, or independent review evidence.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline, _native
from pllm.components import create_component, get
from pllm.configuration import ComponentRef, ConfigurationError, ForwardBudget
from pllm.numeric import Q11I16Ring64Boundary, SourceFloat32
from pllm.profiles import (OpenWeightVerifiableCpu, resolve_runtime_composition,
                           resolve_decoder_runtime_composition)
from pllm.protocols import TwoShareInput, VerifiedPublicF32Lookup, ClientF32DecoderRegions
from pllm.runtime import benchmark_memory as memory, delegated_decoder as decoder
from pllm.runtime import delegated_transport as transport
from pllm.runtime import _component_binding as binding
from test_delegated_decoder import config, fixture
from test_delegated_transport_binding import admitted


def bindings():
    return {"input": TwoShareInput(), "lookup": VerifiedPublicF32Lookup(),
            "regions": ClientF32DecoderRegions(), "numeric": Q11I16Ring64Boundary()}


def pipeline(**params):
    return OpenWeightVerifiableCpu(Model.path("unused"), **params)


def explicit(base):
    return Pipeline(base.model, components={**base.components, **bindings()})


@pytest.mark.parametrize("component", [TwoShareInput(), VerifiedPublicF32Lookup(),
                                       ClientF32DecoderRegions(), Q11I16Ring64Boundary()])
def test_zero_parameter_public_binding(component):
    assert component.get_params() == {}
    assert get(component.component) is type(component)
    assert create_component(component.component, {}) == component
    assert component.describe().parameter_schema["additionalProperties"] is False
    with pytest.raises(ConfigurationError, match="unknown fields"):
        create_component(component.component, {"sharing": "seeded"})
    with pytest.raises(TypeError):
        type(component)(sharing="seeded")


@pytest.mark.parametrize("sharing", ["full", "seeded"])
def test_legacy_bytes_unchanged_explicit_nine_equivalent_and_distinct(sharing):
    from pllm.protocols import OpenWeightVerifiableOutsourcing

    base = pipeline(linear=OpenWeightVerifiableOutsourcing(sharing=sharing))
    before, spec, pin = base.canonical_bytes(), deepcopy(base.to_spec()), base.digest()
    composed = explicit(base)
    assert len(base.components) == 5 and len(composed.components) == 9
    assert resolve_runtime_composition(base) == resolve_runtime_composition(composed)
    assert resolve_decoder_runtime_composition(composed) == resolve_runtime_composition(base)
    assert base.canonical_bytes() == before and base.to_spec() == spec and base.digest() == pin
    assert composed.digest() != pin
    assert Pipeline.from_spec(composed.to_spec()).canonical_bytes() == composed.canonical_bytes()
    Experiment("binding", composed, Deployment.local(root="."), ExecutionBudget(1, 2, 2))
    with pytest.raises(ConfigurationError, match="forward"):
        Experiment("binding", composed, Deployment.local(root="."), ForwardBudget(1, 2))


@pytest.mark.parametrize("missing", ["input", "lookup", "regions", "numeric",
                                     "state", "topology", "verification", "kernels", "linear"])
def test_missing_binding_rejected_during_configuration(missing):
    base = pipeline()
    components = {**base.components, **bindings()}
    del components[missing]
    with pytest.raises(ConfigurationError):
        Pipeline(base.model, components=components)


@pytest.mark.parametrize("change", ["unknown", "mixed", "extra", "roles", "sharing"])
def test_unknown_mixed_and_incompatible_declarations_reject(change):
    base = pipeline()
    components = {**base.components, **bindings()}
    if change == "unknown":
        components["lookup"] = ComponentRef("external/unknown/v1", {})
    elif change == "mixed":
        components["numeric"] = SourceFloat32()
    elif change == "extra":
        components["extra"] = TwoShareInput()
    elif change == "roles":
        components["topology"] = ComponentRef("external/roles/v1", {})
    else:
        components["linear"] = ComponentRef("pllm/open-weight-verifiable-outsourcing/v1",
                                             {"sharing": "unknown"})
    with pytest.raises(ConfigurationError):
        Pipeline(base.model, components=components)


def test_unrelated_numeric_and_source_float32_still_forward_only():
    from pllm.kernels import Cpu
    from pllm.protocols import CleartextLinear
    from pllm.roles import ClientOnlyRoles

    forward = Pipeline(Model.path("unused"), components={
        "numeric": SourceFloat32(), "linear": CleartextLinear(),
        "kernels": Cpu(threads=1), "topology": ClientOnlyRoles()})
    Experiment("forward", forward, Deployment.local(root="."), ForwardBudget(1, 2))
    with pytest.raises(ConfigurationError, match="ForwardBudget"):
        Experiment("forward", forward, Deployment.local(root="."), ExecutionBudget(1, 2, 2))
    with pytest.raises(ConfigurationError, match="forward-only"):
        resolve_decoder_runtime_composition(forward)
    with pytest.raises(ConfigurationError):
        Pipeline(forward.model, components={**forward.components,
                 "numeric": ComponentRef("external/numeric/v1", {})})


# Deliberately independent expected table: catches Rust/PyO3/Python-code drift,
# not two fakes agreeing. Uses generated tiny native session and transport owner.
EXPECTED = {"token_lookup": 1, "reshape": 2, "rms_norm": 3, "linear": 4,
            "rotary_embedding": 5, "kv_cache_append": 6, "attention_scores": 7,
            "attention_scale": 8, "causal_mask": 9, "softmax": 10,
            "attention_values": 11, "residual_add": 12, "silu": 13,
            "multiply": 14, "output_head": 15, "last_token": 16,
            "greedy_token_selection": 17, "token_feedback": 18, "cache_suffix": 19}


def assert_native_codes(owner, contract):
    for phase in ("prefill", "decode"):
        graph = contract["plan"][phase]
        expected_ids = {op["id"] for op in graph["operations"]}
        scheduled = [key for step in contract[phase]["steps"] for key in step["operation_ids"]]
        assert set(scheduled) == expected_ids and len(scheduled) == len(expected_ids)
        for op in graph["operations"]:
            expected = EXPECTED[op["operator"]]
            for _ in range(3):
                code = owner.bound_step(phase, op["id"])
                assert type(code) is int and code == expected
                assert binding.checked_operator(code, op) == expected
    for phase, key in (("unknown", "token_lookup"), ("prefill", "unknown")):
        with pytest.raises(ValueError, match="^invalid bound decoder step$"):
            owner.bound_step(phase, key)


def test_real_pyo3_session_codes_reads_grant_no_completion_and_abort_retires():
    cfg = config()
    contract, _, tensors = fixture(cfg, rows=1, decode=0)
    encoded = decoder._encode_config(cfg)
    owner = _native._GkmDecoderSession(encoded, 1, 0, tensors)
    assert_native_codes(owner, contract)
    with pytest.raises(ValueError):
        owner.finish()  # No delegated work performed despite repeated reads.
    owner = _native._GkmDecoderSession(encoded, 1, 0, tensors)
    owner.abort()
    with pytest.raises(ValueError, match="^invalid bound decoder step$"):
        owner.bound_step("prefill", "token_lookup")


def test_real_pyo3_transport_codes_reads_grant_no_completion_and_abort_retires(admitted):
    args, contract, tensors = admitted
    owner = _native._GkmTransportClient(*args, tensors)
    assert_native_codes(owner, contract)
    with pytest.raises(ValueError):
        owner.finish()
    owner = _native._GkmTransportClient(*args, tensors)
    owner.abort()
    with pytest.raises(ValueError, match="^invalid bound decoder step$"):
        owner.bound_step("prefill", "token_lookup")


@pytest.mark.parametrize("code", [0, 20, 255, True, None, "4"])
def test_unknown_code_never_becomes_linear(code):
    with pytest.raises(ValueError, match="invalid bound decoder step"):
        binding.checked_operator(code, {"operator": "linear"})


class _ObservedSession:
    def __init__(self, owner, *, bad_code=None):
        self.owner, self.bad_code = owner, bad_code
        self.codes, self.linears, self.aborts = [], [], 0

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def bound_step(self, phase, operation_id):
        self.codes.append((phase, operation_id))
        return (self.owner.bound_step(phase, operation_id)
                if self.bad_code is None else self.bad_code)

    def linear(self, stage, values):
        self.linears.append(stage)
        return self.owner.linear(stage, values)

    def abort(self):
        self.aborts += 1
        self.owner.abort()


def test_dispatch_consumes_native_codes_not_schedule_metadata_fused_once_bias_kv_feedback():
    from test_delegated_decoder import clear_oracle

    cfg = config()
    contract, arrays, tensors = fixture(cfg, rows=2, decode=1)
    metadata = deepcopy(contract)
    for phase in ("prefill", "decode"):
        for step in metadata[phase]["steps"]:
            step["executor"] = "not_dispatch_authority"
            step["operators"] = ["not_dispatch_authority"] * len(step["operation_ids"])
    assert any(len(step["operation_ids"]) > 1 for step in contract["prefill"]["steps"])
    assert any(member["bias"] is not None for matrix in contract["matrices"] for member in matrix["members"])
    owner = _ObservedSession(_native._GkmDecoderSession(decoder._encode_config(cfg), 2, 1, tensors))
    request = decoder._Request(owner, metadata)
    try:
        prefill, token = request.phase((0, 4))
        decode, next_token = request.phase((token,))
        expected = clear_oracle(cfg, arrays, (0, 4), 1)
        assert (prefill, decode) == expected[0]
        assert (token, next_token) == expected[2]
        assert request.position == expected[4] == 3
        for cache, (shape, key, value) in zip(request.caches, expected[1], strict=True):
            assert tuple(cache.key[:cache.length].shape) == shape
            assert cache.key[:cache.length].astype("<f4", copy=False).tobytes() == key
            assert cache.value[:cache.length].astype("<f4", copy=False).tobytes() == value
        # Prefill B rows per body stage, one row at head; decode one per stage.
        assert len(owner.linears) == 3 * len(contract["matrices"]) - 1
        for phase in ("prefill", "decode"):
            assert {key for seen_phase, key in owner.codes if seen_phase == phase} == {
                op["id"] for op in contract["plan"][phase]["operations"]}
        owner.finish()
    finally:
        request.retire()


@pytest.mark.parametrize("bad_code", [255, 4])
def test_unknown_or_operator_mismatched_native_code_aborts_window(bad_code):
    cfg = config()
    contract, _, tensors = fixture(cfg, rows=1, decode=0)
    owner = _ObservedSession(_native._GkmDecoderSession(decoder._encode_config(cfg), 1, 0, tensors),
                             bad_code=bad_code)
    with pytest.raises(ValueError, match="invalid bound decoder step"):
        decoder._execute_verified_window(owner, contract, (0,))
    assert owner.aborts and not owner.linears
    with pytest.raises(ValueError, match="^invalid bound decoder step$"):
        owner.owner.bound_step("prefill", "token_lookup")


@pytest.mark.parametrize("operator,code", EXPECTED.items())
def test_checked_adapters_exact_operator(operator, code):
    assert binding.checked_operator(code, {"operator": operator}) == code
    with pytest.raises(ValueError, match="invalid bound decoder step"):
        binding.checked_operator(code, {"operator": "not_" + operator})


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("margin", [0, 25])
def test_bridge_quote_once_client_only_after_payload_replacement(capacity, batch, margin, monkeypatch):
    from pllm.protocols import OpenWeightVerifiableOutsourcing

    cfg = config()
    base = pipeline(linear=OpenWeightVerifiableOutsourcing(
        **({"capacity": "decoder"} if capacity == "decoder" else {}),
        execution="source_stage_batch_v1" if batch else None))
    experiment = Experiment("memory-binding", base, Deployment.local(root="."),
                            ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=2))
    raw = _native.gkm_runtime_contract(experiment.canonical_bytes(), decoder._encode_config(cfg),
                                       2, 2, "1" * 64, "2" * 64)
    contract = json.loads(raw)["decoder"]
    assert binding.has_decoder_bridge(contract)
    source = SimpleNamespace(capacity=capacity, contract=contract, runtime_contract_bytes=raw,
                             execution_payload_estimate_bytes=contract["execution_payload_estimate_bytes"])
    args = cfg, base, 2, 2, 0, None, 0, margin
    quote = memory._delegated_memory(*args, runtime_contract=raw)
    client = transport.request_memory_bytes(source)
    workers = {role: transport.request_memory_bytes(source, worker=True, role=role)
               for role in ("worker_a", "worker_b")}
    with monkeypatch.context() as control:
        control.setattr(binding, "CLIENT_BRIDGE_ESTIMATE_BYTES", 0)
        control.setattr(transport, "CLIENT_BRIDGE_ESTIMATE_BYTES", 0)
        baseline = memory._delegated_memory(*args, runtime_contract=raw)
        assert client - transport.request_memory_bytes(source) == 65_536
        assert workers == {role: transport.request_memory_bytes(source, worker=True, role=role)
                           for role in workers}
    delta = 65_536 if margin == 0 else 81_920
    assert quote["client_peak_bytes"] - baseline["client_peak_bytes"] == delta
    assert quote["native_total_peak_bytes"] - baseline["native_total_peak_bytes"] == delta
    assert quote["provider_peak_bytes"] == baseline["provider_peak_bytes"]
    assert quote["verification_allocation"] == baseline["verification_allocation"]
    assert quote["components"]["component_binding_client_bridge_estimate_bytes"] == 65_536
    assert baseline["components"]["component_binding_client_bridge_estimate_bytes"] == 0
    assert quote["components"]["owned_provider_retirement_raw_peak_bytes"] == 1 << 20


def test_nondecoder_label_not_bridge_authority():
    for contract in (None, {}, {"capacity": "decoder"},
                     {"schema_version": "pllm.gkm.decoder-reference.v1"}):
        assert not binding.has_decoder_bridge(contract)
        source = SimpleNamespace(capacity="capsule", contract=contract,
                                 execution_payload_estimate_bytes=1024)
        assert transport.request_memory_bytes(source) == 1024 + (64 << 20)


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("role", [None, "client", "worker_a", "worker_b"])
def test_property_source_quote_decodes_once_and_releases_graph_before_batch(capacity, batch, role, monkeypatch):
    from pllm.protocols import OpenWeightVerifiableOutsourcing
    from pllm.runtime.delegated_source import AdmittedSource

    experiment = Experiment("property-quote", pipeline(linear=OpenWeightVerifiableOutsourcing(
        capacity=capacity, execution="source_stage_batch_v1" if batch else None)),
        Deployment.local(root="."), ExecutionBudget(1, 2, 2))
    raw = _native.gkm_runtime_contract(experiment.canonical_bytes(), decoder._encode_config(config()),
                                       2, 2, "1" * 64, "2" * 64)
    runtime = json.loads(raw)
    contract = runtime["decoder"]
    # Actual immutable property owner, not a stored SimpleNamespace dictionary.
    # Other source fields are irrelevant to this metadata-only quote.
    source = object.__new__(AdmittedSource)
    object.__setattr__(source, "contract_bytes", json.dumps(contract, sort_keys=True,
                                                          separators=(",", ":")).encode())
    object.__setattr__(source, "runtime_contract_bytes", raw)
    assert source.contract == source.contract and source.contract is not source.contract
    worker = role != "client"
    payload = contract["execution_payload_estimate_bytes"]
    if capacity == "decoder":
        payload = contract["decoder_resources"]["worker_payload_bytes" if worker else "client_payload_bytes"]
    prices = transport.batch_memory_bytes(runtime, capacity)
    if prices:
        payload = (max(prices["worker_payload_bytes"].values()) if role is None else
                   prices["worker_payload_bytes"][role] if worker else prices["client_payload_bytes"])
    loads, live = {"contract": 0, "runtime": 0}, [0]
    real_loads, real_batch = json.loads, transport.batch_memory_bytes

    class Lifetime:
        def __init__(self):
            live[0] += 1

        def __del__(self):
            live[0] -= 1

    def counted_loads(encoded, *args, **kwargs):
        if encoded == raw:
            assert live[0] == 0, "full source document/resource subtree overlaps runtime decode"
            loads["runtime"] += 1
        document = real_loads(encoded, *args, **kwargs)
        if encoded == source.contract_bytes:
            loads["contract"] += 1
            document["_lifetime_probe"] = Lifetime()
            if "decoder_resources" in document:
                document["decoder_resources"]["_lifetime_probe"] = Lifetime()
        return document

    def checked_batch(document, selected_capacity):
        assert live[0] == 0, "source graph or resource subtree retained through batch prices"
        return real_batch(document, selected_capacity)

    monkeypatch.setattr(json, "loads", counted_loads)
    monkeypatch.setattr(transport, "batch_memory_bytes", checked_batch)
    assert transport.request_memory_bytes(source, worker=worker,
        role=None if role in (None, "client") else role) == payload + (64 << 20) + (0 if worker else 65_536)
    assert loads == {"contract": 1, "runtime": 1}
    assert live[0] == 0


@pytest.mark.parametrize("failure", ["payload", "client_resources", "capacity_profile", "runtime_batch"])
def test_property_source_quote_errors_release_metadata_while_exception_alive(failure, monkeypatch):
    import weakref
    from pllm.protocols import OpenWeightVerifiableOutsourcing
    from pllm.runtime.delegated_source import AdmittedSource, DelegatedSourceError

    experiment = Experiment("property-quote-errors", pipeline(linear=OpenWeightVerifiableOutsourcing(
        capacity="decoder", execution="source_stage_batch_v1")),
        Deployment.local(root="."), ExecutionBudget(1, 2, 2))
    raw = _native.gkm_runtime_contract(experiment.canonical_bytes(), decoder._encode_config(config()),
                                       2, 2, "1" * 64, "2" * 64)
    document = json.loads(raw)["decoder"]
    if failure == "payload":
        document["execution_payload_estimate_bytes"] = 0
    elif failure == "client_resources":
        document["decoder_resources"]["client_payload_bytes"] += 1
    elif failure == "capacity_profile":
        document["decoder_resources"]["profile"] = "invalid"
        # Competing payload error must NOT outrank capacity-profile rejection.
        document["execution_payload_estimate_bytes"] = 0
    encoded_contract = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    del document  # Mutations/expected data never retain the later decoded source.
    events, references = [], {}
    injected = RuntimeError("fixed runtime/batch quote failure")
    real_loads = json.loads

    class Lifetime:
        pass

    def assert_source_released():
        assert set(references) == {"graph", "resources"}
        assert references["graph"]() is None
        assert references["resources"]() is None

    class PropertySource(AdmittedSource):
        __slots__ = ()

        @property
        def runtime_contract_bytes(self):
            events.append("runtime_access")
            assert_source_released()
            return raw

    source = object.__new__(PropertySource)
    object.__setattr__(source, "contract_bytes", encoded_contract)
    # Inherited actual property returns independently decoded, noncached dicts.
    assert source.contract is not source.contract

    def counted_loads(encoded, *values, **kwargs):
        if encoded == encoded_contract:
            events.append("source_decode")
            decoded = real_loads(encoded, *values, **kwargs)
            decoded["plan"]["prefill"]["_lifetime_probe"] = Lifetime()
            decoded["decoder_resources"]["_lifetime_probe"] = Lifetime()
            references["graph"] = weakref.ref(decoded["plan"]["prefill"]["_lifetime_probe"])
            references["resources"] = weakref.ref(decoded["decoder_resources"]["_lifetime_probe"])
            return decoded
        assert encoded == raw
        events.append("runtime_decode")
        assert_source_released()
        return real_loads(encoded, *values, **kwargs)

    def failed_batch(runtime_document, selected_capacity):
        events.append("batch_pricing")
        assert selected_capacity == "decoder"
        assert_source_released()
        # This exception may retain its PREEXISTING runtime-pricing arguments;
        # only temporary source metadata is required to have been released.
        raise injected

    expected_type = DelegatedSourceError if failure == "capacity_profile" else (
        RuntimeError if failure == "runtime_batch" else ValueError)
    expected_text = ("GKM execution payload has no admitted bound" if failure == "payload" else
                     "GKM native resource contract differs" if failure == "client_resources" else
                     "GKM invalid native capacity contract" if failure == "capacity_profile" else
                     "fixed runtime/batch quote failure")
    monkeypatch.setattr(json, "loads", counted_loads)
    monkeypatch.setattr(transport, "batch_memory_bytes", failed_batch)
    with pytest.raises(expected_type, match="^" + expected_text + "$") as caught:
        transport.request_memory_bytes(source)
    # ExceptionInfo keeps the actual exception AND unmodified traceback alive.
    # No GC, frame clearing, weak proxy dictionaries, or document-bearing records.
    assert type(caught.value) is expected_type
    assert str(caught.value) == expected_text
    assert_source_released()
    if failure == "runtime_batch":
        assert caught.value is injected
        assert events == ["source_decode", "runtime_access", "runtime_decode", "batch_pricing"]
    else:
        assert events == ["source_decode"]  # Neither runtime access nor runtime decoding.


def test_binding_error_failed_real_transport_retirement_keeps_original_custody(admitted, monkeypatch):
    """Real tiny native client, peers/I/O, ledger, reservation and cleanup slot.

    Definition only until separately authorized. Provider admission is bypassed;
    no provider request/network/trained model executes. The close wrapper injects
    failure BEFORE real retirement; its later ACK means local closure, not a
    remote-destruction acknowledgement or native recovery proof.
    """
    import threading
    from pllm.deployment import resources

    args, contract, tensors = admitted
    runtime = _native.gkm_runtime_contract(*args)
    capacity = resources._ClientCapacity()
    state, primary, provider_calls = {}, [], []
    allow_close, closure_ack = threading.Event(), threading.Event()
    real_reserve = resources.reserve_incremental_memory
    real_start, real_close = transport._Peers.start, transport._Peers.close
    real_execute = transport._execute_verified_window

    class WrongBinding:
        def __init__(self):
            self.native = _native._GkmTransportClient(*args, tensors)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def bound_step(self, phase, operation_id):
            assert self.native.bound_step(phase, operation_id) == 1
            return 255  # The actual checked adapter must select the binding error.

    class Source:
        capacity = "capsule"
        execution_payload_estimate_bytes = contract["execution_payload_estimate_bytes"]
        runtime_contract_bytes = runtime

        def runtime_contract(self, *unused):
            return runtime

        def load_transport(self, *unused, **kwargs):
            return WrongBinding()

    source = Source()
    source.contract = contract
    amount = transport.request_memory_bytes(source)

    def observe():
        return resources.ClientResources(observed_at_ms=0, available_memory_bytes=2 * amount,
            process_memory_bytes=1, reserve_bytes=0, reserved_memory_bytes=sum(capacity._owners.values()))

    def reserve(size):
        reservation = real_reserve(size)
        state.update(reservation=reservation, token=reservation._token, manager=reservation._manager)
        return reservation

    def start(owner, *values, **kwargs):
        real_start(owner, *values, **kwargs)
        state["owner"] = owner
        state["io"] = owner.io
        assert type(owner) is transport._Peers and type(owner.io) is transport._IOOwner

    def admit(owner, bound, publication, pin, starts, experiment, source, request_id, rows, outputs):
        # Native publication exists, but never send it to real/mock providers.
        owner.request_id = request_id.hex()
        assert not any(owner.live)

    async def no_provider(*values, **kwargs):
        provider_calls.append(True)
        raise AssertionError("binding failure must precede every provider request")

    def close(owner):
        assert owner is state["owner"] and owner.io is state["io"]
        assert capacity._owners[state["token"]] == amount
        if not allow_close.is_set():
            raise RuntimeError("transport retirement unacknowledged")
        real_close(owner)
        assert owner._close_succeeded and owner.io._close_succeeded
        assert owner.io.loop.is_closed() and not owner.io.thread.is_alive()
        assert capacity._owners[state["token"]] == amount  # No premature refund.
        closure_ack.set()

    def execute(*values, **kwargs):
        try:
            return real_execute(*values, **kwargs)
        except BaseException as error:
            primary.append(error)
            raise

    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", observe)
    monkeypatch.setattr(transport, "reserve_incremental_memory", reserve)
    monkeypatch.setattr(transport._Peers, "start", start)
    monkeypatch.setattr(transport._Peers, "admit", admit)
    monkeypatch.setattr(transport._Peers, "_post", no_provider)
    monkeypatch.setattr(transport._Peers, "close", close)
    monkeypatch.setattr(transport, "_execute_verified_window", execute)
    slot = resources.reserve_cleanup_slot()
    control_token, control_amount = slot._token, slot.control_bytes
    binding_tuple = None
    try:
        with pytest.raises(ValueError) as caught:
            transport.execute_request(None, source, (0,), 1,
                {"worker_a": ("http://127.0.0.1:1", "a" * 32),
                 "worker_b": ("http://127.0.0.1:2", "b" * 32)}, 1, lambda: False)
        assert type(caught.value) is ValueError
        assert str(caught.value) == "invalid bound decoder step"
        assert primary == [caught.value]
        assert caught.value.__cause__ is caught.value.__context__ is None
        assert caught.value.__notes__ == ["GKM cleanup failed: RuntimeError"]
        reservation, token, owner = state["reservation"], state["token"], state["owner"]
        retire = reservation.drain_failed
        slot.retain(reservation, retire)
        binding_tuple = slot._binding

        def charged():
            assert state["reservation"] is reservation and reservation._token is token
            assert reservation._manager is state["manager"]
            assert token.retained is reservation and reservation._failed == (owner,)
            assert reservation.pending_retirement and not reservation._exited
            assert slot._binding is binding_tuple and slot.owns(reservation, retire=retire)
            assert capacity._owners == {control_token: control_amount, token: amount}
            assert not closure_ack.is_set()

        charged()
        with pytest.raises(RuntimeError, match="^transport retirement unacknowledged$"):
            slot.drain()
        charged()
        assert caught.value is primary[0]  # Failed cleanup never replaced binding primary.
        allow_close.set()
        slot.drain()
        assert closure_ack.is_set() and slot.closed and slot._binding is None
        assert not reservation.pending_retirement and reservation._exited
        assert token.retained is None and not capacity._owners and not capacity._cleanup
        assert not provider_calls
    finally:
        allow_close.set()
        reservation = state.get("reservation")
        if reservation is not None and reservation.pending_retirement:
            if slot._binding is None:
                slot.retain(reservation, reservation.drain_failed)
            slot.drain()
        elif not slot.closed:
            if "owner" in state and not state["owner"]._close_succeeded:
                real_close(state["owner"])
            slot.drain()


def test_declaration_and_config_rejection_precede_host_and_tensor_values(monkeypatch):
    def forbidden_host():
        raise AssertionError("host read before declaration admission")

    monkeypatch.setattr(memory, "host_memory", forbidden_host)
    base = pipeline()
    incomplete = {**base.components, **bindings()}
    del incomplete["lookup"]
    with pytest.raises(ConfigurationError):
        Pipeline(base.model, components=incomplete)
    composed = Experiment("early-binding", explicit(base), Deployment.local(root="."),
                          ExecutionBudget(1, 1, 1))
    # Invalid config is rejected before native source shape/value reads. This is
    # NOT an unsupported-operator coverage test; Rust contract mutations cover that.
    # PyDict is accepted by the bridge, but its object value cannot be tensor bytes.
    args = composed.canonical_bytes(), b'{}', 1, 1, "1" * 64, "2" * 64
    with pytest.raises(ValueError) as empty:
        _native._GkmTransportClient(*args, {})
    with pytest.raises(ValueError) as poison:
        _native._GkmTransportClient(*args, {"unread-tensor": object()})
    assert str(empty.value) == str(poison.value)


def test_composed_factory_preserves_all_existing_declarations(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.papers import gkm

    factories = (gkm.experiment, gkm.seeded, gkm.smoke, gkm.seeded_smoke, gkm.bounded_response)
    before = tuple(factory().canonical_bytes() for factory in factories)
    composed, legacy = gkm.composed_bounded_response(), gkm.bounded_response()
    assert composed.name == "gkm-composed-bounded-response"
    assert composed.deployment == legacy.deployment and composed.budget == legacy.budget
    assert composed.pipeline.model == legacy.pipeline.model
    assert {key: value for key, value in composed.pipeline.components.items() if key not in bindings()} == legacy.pipeline.components
    assert composed.configuration_digest() != legacy.configuration_digest()
    assert before == tuple(factory().canonical_bytes() for factory in factories)


def test_legacy_and_selected_factories_complete_real_tiny_transport_windows(tmp_path, monkeypatch):
    """Generated tensors only: no checkpoint/source qualification or physical claim.

    Callback echoes plus the clear oracle test ordinary greedy semantics, not an
    adversarial non-argmax canary. Both windows retain native resource guards.
    """
    import importlib.util
    from pathlib import Path
    import sys
    from types import MappingProxyType
    from test_delegated_decoder import clear_oracle

    # Direct-file declarations use installed SDK imports, never a checkout path
    # or examples.* cache entry. Even a preexisting fake common cannot leak in.
    papers = Path(__file__).resolve().parents[1] / "examples" / "papers"
    missing = object()
    previous = sys.modules.get("common", missing)
    try:
        spec = importlib.util.spec_from_file_location("common", papers / "common.py")
        common = importlib.util.module_from_spec(spec)
        sys.modules["common"] = common
        spec.loader.exec_module(common)
        spec = importlib.util.spec_from_file_location("_binding_gkm", papers / "gkm.py")
        gkm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gkm)
    finally:
        if previous is missing:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous

    cfg = deepcopy(config())
    cfg["max_position_embeddings"] = 64  # Declared envelope 32 + 1 fits.
    _, arrays, tensors = fixture(cfg, rows=2, decode=1, capacity="decoder")
    expected = clear_oracle(cfg, arrays, (0, 4), 1)
    model, budget = Model.path(str(tmp_path)), ExecutionBudget(1, 32, 2)
    experiments = (gkm.experiment(model=model, budget=budget),
                   gkm.benchmark_experiment(model=model, budget=budget))
    assert tuple(len(item.pipeline.components) for item in experiments) == (5, 17)
    results, runtimes, acceptances = [], [], []
    records = decoder.semantic_executor._LOCAL_IMPLEMENTATIONS
    local_operators = set(EXPECTED) - {"token_lookup", "linear", "output_head"}
    assert set(records) == local_operators and len(local_operators) == 16

    class SynchronousPeers:
        batch = None

        def __init__(self, workers):
            self.workers, self.phase = workers, "prefill"
            self.busy = self.finished = False
            self.exchanges = 0

        def exchange(self, frames):
            assert not self.busy and not self.finished
            self.busy = True
            try:
                replies = tuple(worker.execute(frame)
                                for worker, frame in zip(self.workers, frames, strict=True))
                assert len(replies) == 2 and all(type(reply) is bytes for reply in replies)
                self.exchanges += 1
                return replies
            finally:
                self.busy = False

        def finish(self):
            assert not self.busy and not self.finished
            completions = tuple(json.loads(worker.finish()) for worker in self.workers)
            self.finished = True
            return completions

        def check(self):
            assert self.finished and not self.busy and self.phase == "decode"
            assert self.exchanges == contract["window_steps"]

    class Observed(_ObservedSession):
        def bound_step(self, phase, operation_id):
            code = self.owner.bound_step(phase, operation_id)
            operation = operations[phase][operation_id]
            assert type(code) is int and code == EXPECTED[operation["operator"]]
            self.codes.append((phase, operation_id))
            events.append(("bound", phase, operation_id, code))
            return code

        def lookup(self, token):
            lookups.append(token)
            return self.owner.lookup(token)

    real_retire = decoder._Request.retire
    for index, experiment in enumerate(experiments):
        # Nonzero digest strings label this generated fixture only. No source
        # files were hashed, admitted, or verified by these declarations.
        args = (experiment.canonical_bytes(), decoder._encode_config(cfg),
                2, 2, "01" * 32, "02" * 32)
        runtime = json.loads(_native.gkm_runtime_contract(*args))
        contract = runtime["decoder"]
        runtimes.append(runtime)
        assert contract["input_rows"] == 2 and contract["decode_steps"] == 1
        assert any(member["bias"] is not None for matrix in contract["matrices"]
                   for member in matrix["members"])
        assert contract["table"] == "model.embed_tokens.weight"
        operations = {phase: {op["id"]: op for op in contract["plan"][phase]["operations"]}
                      for phase in ("prefill", "decode")}
        events, lookups, callbacks, selections, feedback, retired = [], [], [], [], [], []
        client, workers = None, []

        def observe_record(operator, selector, original):
            def callback(request, operation, values, *args, **kwargs):
                phase = "prefill" if request.position == 0 else "decode"
                assert operation["operator"] == operator == selector
                assert events[-1] == ("bound", phase, operation["id"], EXPECTED[operator])
                result = original(request, operation, values, *args, **kwargs)
                callbacks.append((phase, operation["id"], operator))
                events.append(("callback", phase, operation["id"], operator))
                if operator == "greedy_token_selection":
                    assert result.shape == (1,)
                    selections.append(result.item())
                elif operator == "token_feedback":
                    # IR shape [1, 1] does not require changing actual array [1].
                    assert result is values[operation["inputs"][0]]
                    feedback.append(result.item())
                return result  # Exact original callback result, never a canary.
            return selector, callback

        def retire(request):
            try:
                return real_retire(request)
            finally:
                assert all(cache.key is None and cache.value is None and cache.length == 0
                           for cache in request.caches)
                assert not request._coefficients
                retired.append(True)

        try:
            client = _native._GkmTransportClient(*args, tensors)
            publication, pin = client.publication(), client.publication_digest()
            starts = client.starts()
            for role in (0, 1):
                workers.append(_native._GkmTransportWorker(
                    *args, tensors, publication, pin, role, starts[role]))
            peers = SynchronousPeers(workers)
            owner = Observed(transport._NativeExchange(client, peers, len(contract["matrices"]) - 1))
            with monkeypatch.context() as control:
                control.setattr(decoder._Request, "retire", retire)
                if index == 1:
                    control.setattr(decoder.semantic_executor, "_LOCAL_IMPLEMENTATIONS", MappingProxyType({
                        operator: observe_record(operator, selector, original)
                        for operator, (selector, original) in records.items()
                    }))
                result = decoder._execute_verified_window(owner, contract, (0, 4))
            assert decoder.semantic_executor._LOCAL_IMPLEMENTATIONS is records
            assert retired == [True] and owner.aborts >= 1
            assert (result.logits, result.kv, result.selected, result.executed, result.position) == expected
            assert len(result.selected) == 2 and result.position == 3
            assert result.executed == (0, 4, result.selected[0])
            assert tuple(lookups) == result.executed  # Final selection remains pending.
            for phase in ("prefill", "decode"):
                assert {key for seen, key in owner.codes if seen == phase} == set(operations[phase])
            if index == 1:
                assert {operator for _, _, operator in callbacks} == local_operators
                for phase in ("prefill", "decode"):
                    assert {key for seen, key, _ in callbacks if seen == phase} == {
                        key for key, op in operations[phase].items() if op["operator"] in local_operators}
                assert tuple(selections) == tuple(feedback) == result.selected
            else:
                assert not callbacks
            acceptance = json.loads(result.acceptance)
            assert acceptance["steps"] == contract["window_steps"]
            acceptances.append(acceptance)
            results.append(result)
        finally:
            if client is not None:
                client.abort()
            for worker in workers:
                worker.abort()
        for native in (client, *workers):
            with pytest.raises(ValueError, match=r"\AGKM transport owner is retired\Z"):
                native.request_id()
        with pytest.raises(ValueError, match=r"\Ainvalid bound decoder step\Z"):
            client.bound_step("prefill", "token_lookup")

    assert runtimes[0]["composition"] != runtimes[1]["composition"]
    assert runtimes[0]["configuration"] != runtimes[1]["configuration"]
    for key in ("decoder", "source_lock", "tokenizer"):
        assert runtimes[0][key] == runtimes[1][key]
    for key in ("decoder_contract", "source"):
        assert acceptances[0][key] == acceptances[1][key]
    assert (results[0].logits, results[0].kv, results[0].selected,
            results[0].executed, results[0].position) == (
        results[1].logits, results[1].kv, results[1].selected,
        results[1].executed, results[1].position)


@pytest.mark.parametrize("mutation,error", [
    ("selection", r"\AGKM decoder reference: incompatible SDK component binding\Z"),
    ("activation", r"\Aunsupported model: hidden_act relu is not silu\Z"),
])
def test_selected_transport_rejects_contract_before_empty_tensors(tmp_path, mutation, error):
    import importlib.util
    from pathlib import Path
    import sys

    papers = Path(__file__).resolve().parents[1] / "examples" / "papers"
    missing = object()
    previous = sys.modules.get("common", missing)
    try:
        spec = importlib.util.spec_from_file_location("common", papers / "common.py")
        common = importlib.util.module_from_spec(spec)
        sys.modules["common"] = common
        spec.loader.exec_module(common)
        spec = importlib.util.spec_from_file_location("_binding_gkm_rejection", papers / "gkm.py")
        gkm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gkm)
    finally:
        if previous is missing:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous

    experiment = gkm.benchmark_experiment(model=Model.path(str(tmp_path)),
                                           budget=ExecutionBudget(1, 32, 2))
    document = json.loads(experiment.canonical_bytes())
    cfg = deepcopy(config())
    cfg["max_position_embeddings"] = 64
    assert len(document["pipeline"]["components"]) == 17
    if mutation == "selection":
        document["pipeline"]["components"]["selection"]["component"] = "external/selection/v1"
    else:
        cfg["hidden_act"] = "relu"
    assert len(document["pipeline"]["components"]) == 17
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    # Fixture identity strings only, not source-content verification. An
    # accidentally admitted contract reaches tensor-set mismatch and MUST fail
    # these anchored error matches instead of masquerading as early rejection.
    args = encoded, decoder._encode_config(cfg), 2, 2, "01" * 32, "02" * 32
    with pytest.raises(ValueError, match=error):
        _native._GkmTransportClient(*args, {})


# Configuration-only definitions. Installed SDK constructors and real admission
# helpers remain intact; sentinels prohibit source I/O and native resolution.
@pytest.fixture
def gkm_configuration_only(monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from pllm import model_loader
    from pllm.runtime import delegated_source

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration admission crossed source/native boundary")

    monkeypatch.setattr(_native, "resolve_experiment", forbidden)
    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(model_loader, "load_model", forbidden)
    monkeypatch.setattr(delegated_source, "admit_source", forbidden)
    papers = Path(__file__).resolve().parents[1] / "examples" / "papers"
    missing = object()
    previous = sys.modules.get("common", missing)
    try:
        spec = importlib.util.spec_from_file_location("common", papers / "common.py")
        common = importlib.util.module_from_spec(spec)
        sys.modules["common"] = common
        spec.loader.exec_module(common)
        spec = importlib.util.spec_from_file_location("_binding_gkm_configuration", papers / "gkm.py")
        gkm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gkm)
    finally:
        if previous is missing:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    return gkm


# Independent slot inventory, not imported from the production matcher's table.
_GKM_TARGET_SLOTS = (
    "linear", "input", "lookup", "numeric", "rmsnorm", "rotary", "attention",
    "activation", "multiply", "bias", "add", "head", "selection",
    "verification", "topology", "state", "kernels",
)


def _gkm_target(gkm):
    return gkm.benchmark_experiment(model=Model.hf("unused/configuration-only"),
                                    budget=ExecutionBudget(1, 32, 2))


def _decoder_record_locations(pipeline_record):
    # Only envelopes walked by require_decoder_source_record, not arbitrary
    # nested metadata. The schema envelope need not resolve an ExperimentProfile.
    return (pipeline_record, {"pipeline": pipeline_record},
            {"experiment": {"pipeline": pipeline_record}},
            {"schema": "pllm.source.v1", "experiment": {"pipeline": pipeline_record}})


def _reject_gkm_records(record):
    from pllm._decoder_admission import require_decoder_source_record

    for envelope in _decoder_record_locations(record):
        with pytest.raises(ConfigurationError, match="forward-only"):
            require_decoder_source_record(envelope)


@pytest.mark.parametrize("kind", [None, "huggingface", "safetensors"])
def test_gkm_configuration_real_target_classification_and_guards(gkm_configuration_only, kind):
    from pllm._decoder_admission import require_decoder_pipeline, require_decoder_source_record

    experiment = _gkm_target(gkm_configuration_only)
    target = experiment.pipeline
    assert type(target) is Pipeline and type(experiment) is Experiment
    assert set(target.components) == set(_GKM_TARGET_SLOTS)
    record = target.to_spec()
    assert "kind" not in record["model"]  # Valid Model.to_spec default omission.
    if kind is not None:
        record["model"]["kind"] = kind
    rebuilt = Pipeline.from_spec(record)
    classification = resolve_runtime_composition(rebuilt)
    assert classification is not None
    assert classification.runtime_kind == "gkm"
    assert classification.client_runtime == "compiled_gkm_v1"
    assert classification.privacy_mode == "gkm_public"
    assert classification.privacy_protocol == "gkm_i16_q11_ring64"
    assert classification.numeric_profile == (
        "gkm.nearest-even.2048.i16.ring64.signed.f32-before-div22."
        "le-finite-f32.sha256.secp256k1.v1")
    assert classification.local_numeric_profile == "semantic.float32.dense-full-kv.client-bias-after-div22.v1"
    assert classification.sharing == "full"
    assert classification.delegation_capacity == "decoder"
    assert classification.generation_policy == "fixed_length_greedy"
    assert classification.verification_window == "request_end"
    assert classification.setup_owner == "client"
    assert not classification.requires_preparation
    assert (classification.weight_bits, classification.activation_bits) == (16, 64)
    assert resolve_decoder_runtime_composition(rebuilt) == classification
    require_decoder_pipeline(rebuilt)
    rebuilt_experiment = Experiment(experiment.name, rebuilt, experiment.deployment, experiment.budget)
    for value in (rebuilt, rebuilt_experiment, rebuilt_experiment.to_spec(),
                  {"pipeline": rebuilt}, {"experiment": rebuilt_experiment},
                  *_decoder_record_locations(record)):
        require_decoder_source_record(value)
    roundtrip = Experiment.from_json(rebuilt_experiment.to_json())
    assert roundtrip.to_json() == rebuilt_experiment.to_json()
    assert roundtrip.canonical_bytes() == rebuilt_experiment.canonical_bytes()
    assert roundtrip.configuration_digest() == rebuilt_experiment.configuration_digest()
    assert roundtrip.pipeline.digest() == rebuilt.digest()
    if kind != "safetensors":
        assert rebuilt.canonical_bytes() == target.canonical_bytes()
        assert rebuilt.digest() == target.digest()
        assert rebuilt_experiment.to_json() == experiment.to_json()


@pytest.mark.parametrize("slot", _GKM_TARGET_SLOTS)
@pytest.mark.parametrize("mutation", ["missing", "wrong-id", "unexpected-param"])
def test_gkm_configuration_closed_target_members(gkm_configuration_only, slot, mutation):
    experiment = _gkm_target(gkm_configuration_only)
    record = experiment.pipeline.to_spec()
    if mutation == "missing":
        del record["components"][slot]
    elif mutation == "wrong-id":
        record["components"][slot]["component"] = "external/wrong-member/v1"
    else:
        record["components"][slot]["params"]["unexpected"] = "not-admitted"
    # Malformed raw input reaches the guard directly, never a sanitizing ctor.
    if mutation == "missing" and slot == "numeric":
        from pllm._decoder_admission import require_decoder_source_record

        # Numeric-intent guard is not a whole-composition validator.
        for envelope in _decoder_record_locations(record):
            assert require_decoder_source_record(envelope) is None
    else:
        _reject_gkm_records(record)
    components = {name: ComponentRef(value["component"], value["params"])
                  for name, value in record["components"].items()}
    with pytest.raises(ConfigurationError):
        Pipeline(experiment.pipeline.model, components=components)


@pytest.mark.parametrize("slot,changes", [
    ("linear", {"sharing": "seeded"}),
    ("linear", {"capacity": "capsule"}),
    ("linear", {"execution": "source_stage_batch_v1"}),
    ("linear", {"generation": "sample"}),
    ("rmsnorm", {"epsilon": "fixed"}),
    ("rotary", {"configuration": "fixed"}),
    ("attention", {"heads": "fixed"}),
    ("attention", {"kv": "attention"}),
    ("activation", {"function": "relu"}),
    ("bias", {"placement": "before-reconstruction"}),
    ("head", {"linear": "head"}),
    ("head", {"linear": "state"}),
    ("head", {"rows": "all"}),
    ("selection", {"eos": "stop"}),
    ("selection", {"logits": "top-k"}),
    ("verification", {"acceptance": "stage"}),
    ("state", {"retention": "window"}),
    ("kernels", {"threads": True}),
    ("kernels", {"threads": 1.0}),
    ("kernels", {"threads": 2}),
])
def test_gkm_configuration_target_params_and_links_closed(gkm_configuration_only, slot, changes):
    experiment = _gkm_target(gkm_configuration_only)
    record = experiment.pipeline.to_spec()
    record["components"][slot]["params"].update(changes)
    _reject_gkm_records(record)
    components = {name: ComponentRef(value["component"], value["params"])
                  for name, value in record["components"].items()}
    with pytest.raises(ConfigurationError):
        Pipeline(experiment.pipeline.model, components=components)


@pytest.mark.parametrize("slot,param", [
    ("linear", "sharing"), ("linear", "generation"), ("linear", "capacity"),
    ("rmsnorm", "epsilon"), ("rotary", "configuration"),
    ("attention", "heads"), ("attention", "kv"), ("activation", "function"),
    ("bias", "placement"), ("head", "linear"), ("head", "rows"),
    ("selection", "eos"), ("selection", "logits"),
    ("verification", "acceptance"), ("kernels", "threads"),
])
def test_gkm_configuration_raw_required_params_not_defaulted(gkm_configuration_only, slot, param):
    record = _gkm_target(gkm_configuration_only).pipeline.to_spec()
    del record["components"][slot]["params"][param]
    # A decoder guard must not repair this record via component defaults.
    _reject_gkm_records(record)


@pytest.mark.parametrize("extra", ["extra", "regions"])
def test_gkm_configuration_target_extra_slot_rejected(gkm_configuration_only, extra):
    experiment = _gkm_target(gkm_configuration_only)
    record = experiment.pipeline.to_spec()
    record["components"][extra] = ClientF32DecoderRegions().to_spec()
    _reject_gkm_records(record)
    with pytest.raises(ConfigurationError):
        Pipeline(experiment.pipeline.model,
                 components={**experiment.pipeline.components, extra: ClientF32DecoderRegions()})


@pytest.mark.parametrize("mutation", [
    "missing-model", "empty-model", "nonmapping-model", "missing-source",
    "bad-source", "extra-model-field", "bad-local-only", "bad-kind",
    "missing-component-id", "missing-params", "extra-member-field",
    "nonmapping-member", "nonmapping-params", "null-params",
])
def test_gkm_configuration_raw_malformed_records_not_discarded(gkm_configuration_only, mutation):
    record = _gkm_target(gkm_configuration_only).pipeline.to_spec()
    if mutation == "missing-model":
        del record["model"]
    elif mutation == "empty-model":
        record["model"] = {}
    elif mutation == "nonmapping-model":
        record["model"] = "unused"
    elif mutation == "missing-source":
        record["model"] = {"kind": "huggingface"}
    elif mutation == "bad-source":
        record["model"]["source"] = 1
    elif mutation == "extra-model-field":
        record["model"]["unknown"] = True
    elif mutation == "bad-local-only":
        record["model"]["local_files_only"] = 1
    elif mutation == "bad-kind":
        record["model"]["kind"] = None
    elif mutation == "missing-component-id":
        del record["components"]["multiply"]["component"]
    elif mutation == "missing-params":
        del record["components"]["multiply"]["params"]
    elif mutation == "extra-member-field":
        record["components"]["multiply"]["unknown"] = True
    elif mutation == "nonmapping-member":
        record["components"]["multiply"] = "pllm/f32-multiply/v1"
    else:
        record["components"]["multiply"]["params"] = [] if mutation == "nonmapping-params" else None
    _reject_gkm_records(record)


@pytest.mark.parametrize("kind", ["tiny", "gguf", "vllm", "ollama"])
def test_gkm_configuration_target_source_kind_closed(gkm_configuration_only, kind):
    experiment = _gkm_target(gkm_configuration_only)
    model = Model.tiny() if kind == "tiny" else Model("unused", kind=kind)
    record = experiment.pipeline.to_spec()
    record["model"] = model.to_spec()
    _reject_gkm_records(record)
    with pytest.raises(ConfigurationError):
        Pipeline(model, components=experiment.pipeline.components)


@pytest.mark.parametrize("slot", _GKM_TARGET_SLOTS)
def test_gkm_configuration_source_float32_any_member_rejected_first(gkm_configuration_only, slot):
    from pllm._decoder_admission import require_decoder_source_record

    record = _gkm_target(gkm_configuration_only).pipeline.to_spec()
    record["components"][slot] = SourceFloat32().to_spec()
    for envelope in _decoder_record_locations(record):
        with pytest.raises(ConfigurationError, match=r"\ASourceFloat32 is forward-only, not a decoder numeric mode\Z"):
            require_decoder_source_record(envelope)


def test_gkm_configuration_unrelated_numeric_and_known_walk_only(gkm_configuration_only):
    from pllm._decoder_admission import require_decoder_source_record

    target = _gkm_target(gkm_configuration_only).pipeline.to_spec()
    for components in ({"numeric": Q11I16Ring64Boundary().to_spec()},
                       {"numeric": ComponentRef("external/numeric/v1", {}).to_spec()}):
        _reject_gkm_records({"model": target["model"], "components": components})
    # A checkpoint's arbitrary metadata is not an execution-intent envelope.
    require_decoder_source_record({"metadata": {"pipeline": {
        "components": {"numeric": SourceFloat32().to_spec()}}}})


def test_gkm_configuration_forward_budget_and_schema_still_reject(gkm_configuration_only):
    from pllm._decoder_admission import require_decoder_pipeline, require_decoder_source_record
    from pllm.kernels import Cpu
    from pllm.protocols import CleartextLinear
    from pllm.roles import ClientOnlyRoles

    experiment = _gkm_target(gkm_configuration_only)
    with pytest.raises(ConfigurationError, match="forward"):
        Experiment("wrong-budget", experiment.pipeline, experiment.deployment, ForwardBudget(1, 32))
    forward = Pipeline(experiment.pipeline.model, components={
        "numeric": SourceFloat32(), "linear": CleartextLinear(),
        "kernels": Cpu(threads=1), "topology": ClientOnlyRoles()})
    forward_experiment = Experiment("forward", forward, experiment.deployment, ForwardBudget(1, 32))
    with pytest.raises(ConfigurationError, match="forward-only"):
        require_decoder_pipeline(forward)
    for value in (forward_experiment, forward_experiment.to_spec(),
                  {"experiment": forward_experiment},
                  {"schema": "pllm.experiment.forward.v1", "pipeline": experiment.pipeline.to_spec()},
                  {"experiment": {"schema": "pllm.experiment.forward.v1",
                                  "pipeline": experiment.pipeline.to_spec()}}):
        with pytest.raises(ConfigurationError, match="forward-only"):
            require_decoder_source_record(value)


@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
def test_gkm_configuration_legacy_choices_keep_identity_and_guards(
        gkm_configuration_only, sharing, capacity, execution):
    from pllm._decoder_admission import require_decoder_pipeline, require_decoder_source_record
    from pllm.protocols import OpenWeightVerifiableOutsourcing

    base = pipeline(linear=OpenWeightVerifiableOutsourcing(
        sharing=sharing, capacity=capacity, execution=execution))
    before = base.to_spec(), base.canonical_bytes(), base.digest()
    composed = explicit(base)
    for candidate in (base, composed):
        classification = resolve_runtime_composition(candidate)
        assert classification is not None and classification.runtime_kind == "gkm"
        assert classification.sharing == sharing and classification.delegation_capacity == capacity
        assert resolve_decoder_runtime_composition(candidate) == classification
        require_decoder_pipeline(candidate)
        for record in _decoder_record_locations(candidate.to_spec()):
            require_decoder_source_record(record)
        experiment = Experiment("legacy-pure", candidate, Deployment.local(root="."), ExecutionBudget(1, 32, 2))
        require_decoder_source_record(experiment)
        restored = Experiment.from_json(experiment.to_json())
        assert restored.canonical_bytes() == experiment.canonical_bytes()
        assert restored.configuration_digest() == experiment.configuration_digest()
    assert resolve_runtime_composition(base) == resolve_runtime_composition(composed)
    assert before == (base.to_spec(), base.canonical_bytes(), base.digest())
    assert composed.digest() != base.digest()
