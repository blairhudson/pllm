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
