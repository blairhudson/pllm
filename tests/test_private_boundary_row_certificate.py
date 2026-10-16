"""Tiny source certificates, native heterogeneous codec and lifecycle guards."""
import hashlib
import json
import struct
import zlib
from dataclasses import replace

import numpy as np
import pytest

from pllm.configuration import ConfigurationError
from pllm.roles import PrivateLookupAndFullHead
from pllm.runtime import private_boundary_certificate as cert
from pllm.runtime.benchmark_memory import private_boundary_memory
from pllm.runtime.private_boundary_head_codec import HeadOutputCodec, HEADER
from pllm.runtime.private_boundary_protocol import BoundaryError, ROLES, canonical

from test_private_boundary_head_codec import args, descriptor, encode_pair, shares


def fixture_certificate(path, d, weights):
    """Explicit test-only trusted publisher, not a production source importer."""
    widths, bounds = cert.matrix_layout(weights, d)
    metadata = {"schema": cert.SCHEMA, "source": cert.source_identity(d),
        "numeric": cert.NUMERIC, "layout": "zlib-u8-minimal-signed-widths-v1",
        "widths_digest": hashlib.sha256(widths).hexdigest(), "bounds_digest": bounds,
        "implementation": cert.publisher_implementation()}
    return write_certificate(path, metadata, zlib.compress(widths, 9))


def write_certificate(path, metadata, layout):
    raw = canonical(metadata)
    raw = cert.HEADER.pack(cert.MAGIC, len(raw)) + raw + layout
    path.write_bytes(raw)
    return {"path": str(path), "digest": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


@pytest.fixture
def certified(tmp_path):
    weights = np.array([[0, 0, 0], [1, 0, 0], [-128, 0, 0], [127, -127, 3], [0, -2, 0]], np.int8)
    d = descriptor(head_output_encoding="u32", head_weight_digest=hashlib.sha256(weights.tobytes()).hexdigest())
    pin = fixture_certificate(tmp_path / "head.cert", d, weights)
    d = replace(d, head_output_encoding="row_residues", head_certificate_digest=pin["digest"])
    return d, pin, weights


def test_exact_every_row_endpoints_random_and_float32(certified):
    d, pin, weights = certified
    codec = HeadOutputCodec(d, certificate=pin, weights=weights)
    bounds = [127 * sum(abs(int(x)) for x in row) for row in weights]
    assert list(codec.widths) == [max(1, b.bit_length() + 1) for b in bounds]
    assert codec.widths[0] == 1 and codec.width == 0
    assert codec.payload_bytes == (sum(codec.widths) + 7) // 8
    rng = np.random.default_rng(91)
    for values in ([0] * 5, bounds, [-b for b in bounds],
                   *[[int(rng.integers(-b, b + 1)) for b in bounds] for _ in range(8)]):
        frames = encode_pair(codec, shares(values), args(d))
        actual = codec.decode(*frames, **args(d))
        assert actual.tolist() == [values]
        assert actual.dtype == np.int64
        scales = np.array([1, .1, .001, .5, .25], np.float32)
        assert ((actual.astype(np.float32) * np.float32(.125)) * scales).tobytes() == (
            (np.array([values], np.int64).astype(np.float32) * np.float32(.125)) * scales).tobytes()
    # One fewer bit fails at the tight positive endpoint (except zero row).
    for bound, width in zip(bounds[1:], codec.widths[1:]):
        assert bound >= 1 << (width - 2)


def test_pins_required_and_defaults_preserved(certified):
    d, pin, _ = certified
    assert PrivateLookupAndFullHead().params == {}
    for bad in (None, {}, {**pin, "path": "relative"}, {**pin, "size_bytes": True}, {**pin, "digest": "f"}):
        with pytest.raises((BoundaryError, ConfigurationError)):
            PrivateLookupAndFullHead(head_output_encoding="row_residues", head_certificate=bad)
    with pytest.raises(ConfigurationError):
        PrivateLookupAndFullHead(head_certificate=pin)
    with pytest.raises(BoundaryError):
        HeadOutputCodec(d)
    for dispatch in ("serial", "parallel"):
        for kernel in ("existing", "byte_dot"):
            selection = PrivateLookupAndFullHead(dispatch, "row_residues", head_kernel=kernel, head_certificate=pin)
            assert PrivateLookupAndFullHead.from_params(selection.params) == selection
    assert type(d).from_dict(d.to_dict()) == d


def test_artifact_hash_checked_before_parsing_or_expansion(certified, monkeypatch):
    d, pin, _ = certified
    monkeypatch.setattr(cert.json, "loads", lambda *a, **k: pytest.fail("parsed unauthenticated artifact"))
    path = __import__("pathlib").Path(pin["path"])
    path.write_bytes(b"x" * pin["size_bytes"])
    with pytest.raises(BoundaryError, match="authentication"):
        cert.import_certificate(pin, d)
    with pytest.raises(BoundaryError, match="pin differs"):
        cert.import_certificate({**pin, "digest": "f" * 64}, d)


@pytest.mark.parametrize("field", ["source_digest", "checkpoint_digest", "model_config_digest",
    "head_weight_digest", "head_scales_digest", "lookup_digest", "hidden", "vocabulary", "signed_output_bound"])
def test_stale_or_substituted_source_rejected(certified, field):
    d, pin, _ = certified
    value = getattr(d, field)
    changed = replace(d, **{field: "f" * 64 if isinstance(value, str) else value + (1 if field == "hidden" else -1)})
    with pytest.raises(BoundaryError, match="source or numeric"):
        cert.import_certificate(pin, changed)


@pytest.mark.parametrize("malform", ["bomb", "trailing", "padding", "zero", "wide", "bounds", "numeric", "source-bool", "duplicate", "implementation", "nested", "array"])
def test_authenticated_but_malformed_publisher_artifact_rejected(certified, tmp_path, malform):
    d, pin, weights = certified
    raw = __import__("pathlib").Path(pin["path"]).read_bytes()
    _, size = cert.HEADER.unpack_from(raw)
    metadata = json.loads(raw[cert.HEADER.size:cert.HEADER.size + size])
    layout = raw[cert.HEADER.size + size:]
    if malform == "bomb":
        layout = zlib.compress(b"\x01" * 100000)
    elif malform == "trailing":
        layout += zlib.compress(b"garbage")
    elif malform == "padding":
        layout = layout[:-1]
    elif malform in ("zero", "wide"):
        widths = bytes([0 if malform == "zero" else 33]) * d.vocabulary
        metadata["widths_digest"] = hashlib.sha256(widths).hexdigest()
        layout = zlib.compress(widths)
    elif malform == "bounds":
        metadata["bounds_digest"] = "0" * 64
    elif malform == "numeric":
        metadata["numeric"] = "observed-activation-range"
    elif malform == "source-bool":
        metadata["source"]["weight_bits"] = True
    elif malform == "implementation":
        metadata["implementation"]["pllm._native"] = "untrusted"
    elif malform == "nested":
        metadata["source"]["hidden"] = {"nested": {"bad": 1}}
    elif malform == "array":
        metadata["source"]["hidden"] = [1] * 100
    else:
        raw_metadata = canonical(metadata)
        raw_metadata = b'{"schema":"forged",' + raw_metadata[1:]
        payload = cert.HEADER.pack(cert.MAGIC, len(raw_metadata)) + raw_metadata + layout
        path = tmp_path / "duplicate.cert"
        path.write_bytes(payload)
        pin = {"path": str(path), "digest": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    if malform != "duplicate":
        pin = write_certificate(tmp_path / "malformed.cert", metadata, layout)
    d = replace(d, head_certificate_digest=pin["digest"])
    with pytest.raises(BoundaryError):
        cert.import_certificate(pin, d, weights=weights)


def test_workers_recompute_each_row_not_just_maximum(certified, tmp_path):
    d, pin, weights = certified
    changed = weights.copy()
    changed[1, 1] = 127
    # Even caller-approved certificate cannot override actual native row bounds.
    with pytest.raises(BoundaryError, match="actual immutable"):
        cert.import_certificate(pin, d, weights=changed)


def test_layout_source_session_downgrade_replay_padding_and_checksum(certified):
    d, pin, _ = certified
    codec = HeadOutputCodec(d, certificate=pin)
    kw = args(d)
    frames = encode_pair(codec, shares([0, 127, -16256, 32639, -127]), kw)
    for change in ({"epoch": 3}, {"sequence": 2}, {"paired_session": "d" * 32}):
        with pytest.raises(BoundaryError):
            codec.decode(*frames, **{**kw, **change})
    for bad in (frames[0][:-1], frames[0] + b"x", frames[0][::-1], shares([0] * 5)[0]):
        with pytest.raises(BoundaryError):
            codec.decode(bad, frames[1], **kw)
    with pytest.raises(BoundaryError):
        codec.decode(*reversed(frames), **kw)
    uniform = HeadOutputCodec(replace(d, head_output_encoding="uniform_residues", head_certificate_digest=""))
    with pytest.raises(BoundaryError):
        uniform.decode(*frames, **args(uniform.descriptor))
    assert frames[0][:8] == b"PLLMBH02"
    used = sum(codec.widths) % 8
    assert used
    magic, party, width, binding, _ = HEADER.unpack_from(frames[0])
    payload = bytearray(frames[0][HEADER.size:])
    payload[-1] |= 0x80
    bad = HEADER.pack(magic, party, width, binding, hashlib.sha256(codec.domain + binding + payload).digest()) + payload
    with pytest.raises(BoundaryError, match="padding"):
        codec.decode(bad, frames[1], **kw)


def test_admission_prices_live_layout_native_decoders_publication_and_delivery(certified):
    d, pin, _ = certified
    for dispatch in ("serial", "parallel"):
        plain = private_boundary_memory(5, 3, lookup_rows=8, head_calls=4, head_dispatch=dispatch)
        row = private_boundary_memory(5, 3, lookup_rows=8, head_calls=4, head_dispatch=dispatch,
            head_output_encoding="row_residues", head_certificate=pin)
        assert row["worker_peak_bytes"] > plain["worker_peak_bytes"]
        assert row["client_work_bytes"] > plain["client_work_bytes"]
        assert row["client_head_codec_work_bytes"] >= 17 * 5 + 4 * row["head_payload_bytes_per_worker"]
        assert row["certificate_three_recipient_delivery_bytes"] == 3 * pin["size_bytes"]
        assert row["head_layout_bytes_per_owner"] == 5


def test_production_publisher_bridge_and_worker_native_comparison(experiment, imported_boundary, tmp_path):
    from pllm.runtime.private_boundary_loader import build_private_boundary_contract
    d, _ = build_private_boundary_contract(imported_boundary, experiment)
    pin = cert.publish(imported_boundary, experiment, tmp_path / "published.cert")
    selected = replace(d, head_output_encoding="row_residues", head_certificate_digest=pin["digest"])
    assert cert.import_certificate(pin, selected, weights=imported_boundary.head.compiled_weight.weight_view())
    with pytest.raises(FileExistsError):
        cert.publish(imported_boundary, experiment, tmp_path / "published.cert")


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_generated_source_publisher_matches_independent_thin_bundle(tmp_path, family):
    import msgpack
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.roles import PreparedBoundaryRoles
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.transformer_client import ClientBundle
    from test_private_boundary_bundle import generated_boundary
    engine, _, document, plan, composition = generated_boundary(tmp_path, family)
    compiled = compile_runtime_model(plan, ClientBundle.unpack(msgpack.packb(document, use_bin_type=True)), composition=composition)
    recipe = Experiment("publisher", MaskedLinearCpu(Model.path(str(tmp_path / "model"), model_id="private-boundary"),
        boundary=PrivateLookupAndFullHead(), topology=PreparedBoundaryRoles()),
        Deployment.local(root=str(tmp_path / "publisher")),
        ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=4))
    pin = PrivateLookupAndFullHead.publish_head_certificate(recipe, tmp_path / "source.cert")
    descriptor = replace(compiled.private_boundary_descriptor(), head_output_encoding="row_residues",
                         head_certificate_digest=pin["digest"])
    widths = cert.import_certificate(pin, descriptor)
    head = engine.models["private-boundary"].stages["lm_head"]
    assert widths == cert.matrix_layout(head.compiled_weight.weight_view(), descriptor)[0]
    assert all(s.client_weight is None for s in compiled._bundle.stages.values())


# These fixtures intentionally model the loader bridge without claiming a new
# native lowering build. Generated decoder/SDK tests exercise that separate gate.
from test_private_boundary_topology import experiment, imported_boundary


@pytest.mark.parametrize("dispatch", ["serial", "parallel"])
@pytest.mark.parametrize("kernel", ["existing", "byte_dot"])
def test_actual_native_workers_exact_full_logits_and_retirement(tmp_path, dispatch, kernel):
    import asyncio
    import httpx
    from pllm import _native
    from pllm.runtime.private_boundary_protocol import BoundaryAdmission
    from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from test_private_boundary_worker import KEYS, app_for, fixture
    d, head, scales, roots = fixture()
    if not hasattr(_native.PrivatePageServer, "bind_context"):
        pytest.skip("installed native context API absent; no fallback")
    if kernel == "byte_dot" and not hasattr(head.owner._executor, "wrap32_byte_dot"):
        pytest.skip("installed native byte-dot API absent; no fallback")
    pin = fixture_certificate(tmp_path / "worker.cert", d, head.weight_view())
    d = replace(d, head_dispatch=dispatch, head_kernel=kernel,
                head_output_encoding="row_residues", head_certificate_digest=pin["digest"])
    admission = BoundaryAdmission(d.digest, "c" * 64, 4, 2)
    def admit(actual, requested):
        if actual != d or requested != admission:
            raise BoundaryError("independent fixture admission differs")
        return admission
    with pytest.raises(BoundaryError):
        app_for(d, head, scales, roots[0], admit)
    clients = []
    for p, role in enumerate(ROLES):
        app = app_for(d, head, scales, roots[p], admit, role, head_certificate=pin)
        def exchange(request, app=app):
            async def call():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=str(request.url)) as client:
                    response = await client.request(request.method, str(request.url), content=request.read(), headers=request.headers)
                    return httpx.Response(response.status_code, content=response.content)
            return asyncio.run(call())
        clients.append(httpx.Client(base_url=f"http://localhost:{8100 + p}", transport=httpx.MockTransport(exchange)))
    t = PrivateBoundaryTransport(worker_a=clients[0], worker_b=clients[1], api_key_a=KEYS[ROLES[0]],
        api_key_b=KEYS[ROLES[1]], descriptor=d, head_scales=scales, head_certificate=pin)
    try:
        t.open(admission)
        x = np.array([[1.25, -.5, 3]], np.float32)
        q = quantize_activation_per_row(x, bits=8)
        expected = dequantize_matmul(head.clear(q.values), q.scales, scales)
        for _ in range(2):
            assert t.linear("lm_head", x).tobytes() == expected.tobytes()
        t.complete()
        with pytest.raises(BoundaryError):
            t.open(admission)
        with pytest.raises(BoundaryError):
            t.linear("lm_head", x)
    finally:
        t.close()
        for client in clients:
            client.close()


@pytest.fixture
def row_harnesses(tmp_path):
    """Mock HTTP lifecycle faults around real native products and row codec."""
    import httpx
    from pllm.runtime.private_boundary_protocol import BoundaryAdmission, pack_frame, unpack_frame
    from test_private_boundary_parallel import ParallelHarness
    from test_private_boundary_transport import Harness
    import threading
    class RowHarness(ParallelHarness):
        def handle(self, request):
            if not request.url.path.endswith("/execute"):
                return super().handle(request)
            party = int(request.url.port) - 8001
            row = unpack_frame(request.content)
            with self.guard:
                self.rows.append((party, row, threading.current_thread()))
            self.before(request, row, party)
            response = self.encode_response(Harness.handle(self, request), row, party)
            return self.after(response, row, party)
    owners = []
    def make():
        h = RowHarness()
        owners.append(h)
        pin = fixture_certificate(tmp_path / f"layout-{len(owners)}.cert", h.d, h.weights)
        h.d = replace(h.d, head_output_encoding="row_residues", head_certificate_digest=pin["digest"])
        h.admission = replace(h.admission, descriptor_digest=h.d.digest)
        codec = HeadOutputCodec(h.d, certificate=pin, weights=h.weights)
        def after(response, row, party):
            reply = unpack_frame(response.content, response=True)
            frame = codec.encode(reply["frames"][0], admission=h.admission, role=ROLES[party],
                paired_session=row["session"], epoch=row["epoch"], sequence=row["sequence"])
            return httpx.Response(200, content=pack_frame(role=ROLES[party], paired_session=row["session"],
                epoch=row["epoch"], sequence=row["sequence"], operation="head", frames=[frame], response=True))
        h.encode_response = after
        transport = h.transport
        h.transport = lambda **overrides: transport(head_certificate=pin, **overrides)
        return h
    yield make
    for h in owners:
        h.close()


@pytest.mark.parametrize("party", [0, 1])
def test_row_replay_burns_pair(row_harnesses, party):
    import test_private_boundary_parallel as lifecycle
    lifecycle.test_previous_step_reply_is_rejected_as_replay(row_harnesses, party)


@pytest.mark.parametrize("party", [0, 1])
def test_row_partial_failure_drains_pair(row_harnesses, party):
    import httpx
    import test_private_boundary_parallel as lifecycle
    lifecycle.test_partial_stream_failure_drains_other_party_and_charges_bytes(row_harnesses, party, httpx.ReadError)


@pytest.mark.parametrize("action", ["abort", "close"])
@pytest.mark.parametrize("owned", [False, True])
def test_row_cancellation_drains_pair(row_harnesses, monkeypatch, action, owned):
    import test_private_boundary_parallel as lifecycle
    lifecycle.test_external_cancellation_drains_both_without_output(row_harnesses, monkeypatch, action, owned)


def test_metadata_return_uses_same_authenticated_parse(certified, monkeypatch):
    d, pin, weights = certified
    parsed = []
    original = cert._metadata
    def observe(raw):
        parsed.append(raw)
        return original(raw)
    monkeypatch.setattr(cert, "_metadata", observe)
    widths, metadata = cert.import_certificate(pin, d, weights=weights, with_metadata=True)
    assert len(parsed) == 1
    assert metadata["source"] == cert.source_identity(d)
    assert metadata["numeric"] == cert.NUMERIC
    assert metadata["widths_digest"] == hashlib.sha256(widths).hexdigest()
    assert metadata["bounds_digest"] == cert.matrix_layout(weights, d)[1]
    assert metadata["implementation"] == cert.publisher_implementation()
    assert metadata["artifact_digest"] == pin["digest"]
    assert metadata["artifact_size_bytes"] == pin["size_bytes"]
    assert cert.import_certificate(pin, d) == widths
    changed = weights.copy()
    changed[1, 1] = 127
    with pytest.raises(BoundaryError, match="actual immutable"):
        cert.import_certificate(pin, d, weights=changed, with_metadata=True)
    with pytest.raises(BoundaryError, match="metadata selection"):
        cert.import_certificate(pin, d, with_metadata=1)


@pytest.fixture
def publisher_recipe(tmp_path):
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.roles import PreparedBoundaryRoles
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    root = create_tiny_llama_checkpoint(tmp_path / "source", model_type="qwen2",
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=258,
        with_qkv_bias=True)
    return Experiment("publisher-reservation", MaskedLinearCpu(Model.path(str(root), model_id="publisher"),
        boundary=PrivateLookupAndFullHead(), topology=PreparedBoundaryRoles()),
        Deployment.local(root=str(tmp_path / "publisher")),
        ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=4))


@pytest.fixture
def publisher_capacity(monkeypatch):
    from pllm.deployment import resources
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: resources.ClientResources(
        100, 64 << 30, 128 << 20, 1 << 30, sum(capacity._owners.values()), 64 << 30))
    return capacity


@pytest.mark.parametrize("constraint", ["host", "cgroup"])
def test_publisher_denies_before_stage_or_any_tensor_values(
        tmp_path, publisher_recipe, publisher_capacity, monkeypatch, constraint):
    from pllm.deployment.resources import ClientResources
    from pllm.deployment.network import NetworkError
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    def forbidden(*args, **kwargs):
        pytest.fail("tensor values/importer ran before publisher capacity admission")
    monkeypatch.setattr(MaskedTransformerEngine, "_load_stage", forbidden)
    for method in ("get", "get_slice", "iter_slices"):
        monkeypatch.setattr(SafeTensorStore, method, forbidden)
    monkeypatch.setattr(publisher_capacity, "observe", lambda: ClientResources(
        100, (64 << 30) if constraint == "cgroup" else (1 << 30) + 1,
        128 << 20, 1 << 30, sum(publisher_capacity._owners.values()),
        1 if constraint == "cgroup" else None))
    path = tmp_path / "denied.cert"
    with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
        PrivateLookupAndFullHead.publish_head_certificate(publisher_recipe, path)
    assert not path.exists() and not publisher_capacity._owners


@pytest.mark.parametrize("failure", [None, "import", "tokenizer", "scan", "fsync", "link"])
def test_publisher_holds_owner_through_publication_and_cleans_every_failure(
        tmp_path, publisher_recipe, publisher_capacity, monkeypatch, failure):
    from types import SimpleNamespace
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    events = []
    original_stage = MaskedTransformerEngine._load_stage
    original_tokenizer = MaskedTransformerEngine._load_tokenizer_descriptor
    original_scan, original_fsync, original_link = cert.matrix_layout, cert.os.fsync, cert.os.link
    original_clear = SafeTensorStore.clear_cache
    def held(name):
        assert publisher_capacity._owners, f"reservation absent during {name}"
        events.append(name)
        if failure == name:
            raise OSError("injected publication failure")
    def stage(*args, **kwargs):
        held("import")
        return original_stage(*args, **kwargs)
    def tokenizer(self, *args, **kwargs):
        held("tokenizer")
        return original_tokenizer(*args, **kwargs)
    def scan(*args, **kwargs):
        held("scan")
        return original_scan(*args, **kwargs)
    def fsync(*args):
        held("fsync")
        return original_fsync(*args)
    def link(*args):
        held("link")
        return original_link(*args)
    def clear(store):
        held("clear")
        original_clear(store)
        assert not store._cache
    monkeypatch.setattr(MaskedTransformerEngine, "_load_stage", stage)
    monkeypatch.setattr(MaskedTransformerEngine, "_load_tokenizer_descriptor", tokenizer)
    monkeypatch.setattr(SafeTensorStore, "clear_cache", clear)
    monkeypatch.setattr(cert, "matrix_layout", scan)
    monkeypatch.setattr(cert, "os", SimpleNamespace(fsync=fsync, link=link,
        fdopen=cert.os.fdopen, unlink=cert.os.unlink))
    path = tmp_path / "reserved.cert"
    if failure is None:
        pin = PrivateLookupAndFullHead.publish_head_certificate(publisher_recipe, path)
        assert pin["digest"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert {"import", "tokenizer", "clear", "scan", "fsync", "link"} <= set(events)
    else:
        with pytest.raises(OSError, match="injected publication failure"):
            PrivateLookupAndFullHead.publish_head_certificate(publisher_recipe, path)
        assert not path.exists()
        assert "clear" in events
    assert not publisher_capacity._owners
    assert not list(tmp_path.glob(".pllm-head-certificate-*"))


def test_publisher_concurrent_calls_cannot_double_spend_headroom(
        tmp_path, publisher_recipe, publisher_capacity, monkeypatch):
    import threading
    from pllm.deployment.network import NetworkError
    from pllm.deployment.resources import ClientResources
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    entered, release = threading.Event(), threading.Event()
    prices, results, errors = [], [], []
    original_price, original_stage = cert.publication_memory, MaskedTransformerEngine._load_stage
    def price(*args):
        result = original_price(*args)
        prices.append(result["incremental_memory_bytes"])
        return result
    monkeypatch.setattr(cert, "publication_memory", price)
    monkeypatch.setattr(publisher_capacity, "observe", lambda: ClientResources(
        100, prices[0] + (1 << 30), 128 << 20, 1 << 30,
        sum(publisher_capacity._owners.values()), prices[0]))
    def stage(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_stage(*args, **kwargs)
    monkeypatch.setattr(MaskedTransformerEngine, "_load_stage", stage)
    def first():
        try:
            results.append(PrivateLookupAndFullHead.publish_head_certificate(publisher_recipe, tmp_path / "first.cert"))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            PrivateLookupAndFullHead.publish_head_certificate(publisher_recipe, tmp_path / "second.cert")
        assert len(prices) == 2 and prices[0] == prices[1]
        assert sum(publisher_capacity._owners.values()) == prices[0]
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors
    assert len(results) == 1 and not publisher_capacity._owners
    assert not (tmp_path / "second.cert").exists()


def test_publisher_prices_only_validated_head_metadata(publisher_recipe, monkeypatch):
    from pllm.model_loader import resolve_model
    from pllm.runtime.models import StageSpec
    from pllm.runtime.safetensors_store import SafeTensorStore
    source = resolve_model(publisher_recipe.pipeline.model)
    spec = StageSpec("lm_head", "lm_head", 8, 258, weight_keys=("model.embed_tokens.weight",))
    store = SafeTensorStore(source.path)
    for method in ("get", "get_slice", "iter_slices"):
        monkeypatch.setattr(store, method, lambda *a, **k: pytest.fail("read tensor values while pricing"))
    price = cert.publication_memory(source, spec, store)
    assert price["source_f32_head_allowance_bytes"] == 4 * 258 * 8
    assert price["quantized_and_native_copy_bytes"] == 4 * 258 * 8
    assert price["scales_and_row_metadata_bytes"] == 32 * 258
    assert price["tokenizer_work_bytes"] > 0
    assert price["incremental_memory_bytes"] == sum(v for k, v in price.items() if k != "incremental_memory_bytes")
    with pytest.raises(BoundaryError, match="shape differs"):
        cert.publication_memory(source, replace(spec, out_features=257), store)


@pytest.mark.parametrize("kind", ["tiny", "online"])
def test_publisher_rejects_source_materialization_before_reservation(publisher_recipe, monkeypatch, tmp_path, kind):
    from pllm import Model
    from pllm.runtime import private_boundary_loader as loader
    spec = publisher_recipe.to_spec()
    spec["pipeline"]["model"] = (Model.tiny() if kind == "tiny" else Model("untrusted-publication/not-local")).to_spec()
    selected = type(publisher_recipe).from_spec(spec)
    monkeypatch.setattr(loader, "resolve_model", lambda *a, **k: pytest.fail("resolved a materializing source"))
    with pytest.raises(BoundaryError, match="existing local or explicitly offline"):
        PrivateLookupAndFullHead.publish_head_certificate(selected, tmp_path / "not-created.cert")
    assert not (tmp_path / "not-created.cert").exists()


def test_publisher_rejects_undeclared_bias_using_metadata_only(publisher_recipe, publisher_capacity, monkeypatch, tmp_path):
    from pathlib import Path
    from safetensors.numpy import save_file
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    save_file({"model.embed_tokens.bias": np.zeros(258, np.float32)},
              str(Path(publisher_recipe.pipeline.model.source) / "extra.safetensors"))
    def forbidden(*args, **kwargs):
        pytest.fail("loaded tensor values before rejecting undeclared bias")
    monkeypatch.setattr(MaskedTransformerEngine, "_load_stage", forbidden)
    for name in ("get", "get_slice", "iter_slices"):
        monkeypatch.setattr(SafeTensorStore, name, forbidden)
    with pytest.raises(BoundaryError, match="undeclared bias"):
        PrivateLookupAndFullHead.publish_head_certificate(publisher_recipe, tmp_path / "bias.cert")
    assert not publisher_capacity._owners and not (tmp_path / "bias.cert").exists()
