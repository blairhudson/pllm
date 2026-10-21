"""Real native batch bindings: complete windows and consuming argument parsing."""

import itertools
import json
import struct

import pytest

from pllm import _native

from test_delegated_transport_binding import _canonical, admitted  # noqa: F401


@pytest.fixture(params=tuple(itertools.product(("full", "seeded"),
                                               ("capsule", "decoder"), (1, 2))))
def batch_source(request, admitted):
    sharing, capacity, rows = request.param
    scalar_args, _, tensors = admitted
    experiment = json.loads(scalar_args[0])
    parameters = experiment["pipeline"]["components"]["linear"]["params"]
    parameters.update(sharing=sharing, execution="source_stage_batch_v1")
    if capacity == "decoder":
        parameters["capacity"] = capacity
    experiment["budget"].update(max_input_tokens=rows, max_new_tokens=2)
    args = (_canonical(experiment), scalar_args[1], rows, 2, *scalar_args[4:])
    contract = json.loads(_native.gkm_runtime_contract(*args))
    return args, contract, tensors


class BatchWindow:
    def __init__(self, source):
        self.args, self.contract, tensors = source
        def import_source():
            importer = _native._GkmTransportImport(*self.args)
            try:
                for name in sorted(self.contract["decoder"]["source_shapes"]):
                    data = tensors[name]
                    for offset in range(0, len(data), 64 << 10):
                        importer.push(name, offset, data[offset:offset + (64 << 10)])
                return importer
            except BaseException:
                importer.abort()
                raise

        importer = import_source()
        try:
            self.client = importer.client()
        finally:
            importer.abort()
        self.workers = []
        try:
            publication = self.client.publication()
            pin = self.client.publication_digest()
            starts = self.client.starts()
            for role in range(2):
                importer = import_source()
                try:
                    self.workers.append(importer.worker(publication, pin, role, starts[role]))
                finally:
                    importer.abort()
        except BaseException:
            self.close()
            raise

    def replies(self, frames, descriptor):
        assert tuple(map(len, frames)) == tuple(descriptor["request_bytes"])
        result = tuple(worker.execute(frame)
                       for worker, frame in zip(self.workers, frames, strict=True))
        assert tuple(map(len, result)) == tuple(descriptor["reply_bytes"])
        return result

    def lookup(self):
        descriptor = self.contract["batch_execution"]["descriptors"][0]
        frames = self.client.lookup_batch((0,) * descriptor["rows"])
        return self.client.accept_batch(*self.replies(frames, descriptor))

    def close(self):
        self.client.abort()
        for worker in self.workers:
            worker.abort()


@pytest.fixture
def batch_window(batch_source):
    window = BatchWindow(batch_source)
    try:
        yield window
    finally:
        window.close()


def test_complete_batch_bindings_match_local_arithmetic(batch_window, batch_source):
    args, contract, tensors = batch_source
    reference = _native._GkmDecoderSession(
        args[1], args[2], args[3] - 1, tensors,
        seeded=json.loads(args[0])["pipeline"]["components"]["linear"]["params"]["sharing"]
        == "seeded")
    try:
        assert batch_window.client.execution() == "source_stage_batch_v1"
        for descriptor in contract["batch_execution"]["descriptors"]:
            rows = descriptor["rows"]
            if descriptor["kind"] == "lookup":
                tokens = (0,) * rows
                expected = reference.lookup_batch(tokens=tokens)
                frames = batch_window.client.lookup_batch(tokens=tokens)
            else:
                stage = descriptor["stage"]
                values = struct.pack("<f", 0.25) * (rows * descriptor["columns"])
                expected = reference.linear_batch(stage=stage, values=values)
                frames = batch_window.client.linear_batch(stage=stage, values=values)
            assert all(frame.startswith(b"PLGKMB01") for frame in frames)
            replies = batch_window.replies(frames, descriptor)
            actual = batch_window.client.accept_batch(zero=replies[0], one=replies[1])
            assert actual == expected
            assert len(actual) == 4 * rows * descriptor["output_width"]
        for worker in batch_window.workers:
            worker.finish()
        accepted = json.loads(batch_window.client.finish())
        reference.finish()
        assert accepted["steps"] == contract["decoder"]["window_steps"]
    finally:
        reference.abort()


@pytest.mark.parametrize("method", ["lookup_batch", "linear_batch", "accept_batch"])
@pytest.mark.parametrize("style", ["extra", "missing", "unknown"])
def test_batch_signature_errors_consume_client(batch_window, method, style):
    invoke = getattr(batch_window.client, method)
    with pytest.raises((TypeError, ValueError)):
        if style == "extra":
            invoke(None, None, None)
        elif style == "missing":
            invoke()
        else:
            invoke(unexpected=None)
    with pytest.raises(ValueError, match="retired"):
        batch_window.client.request_id()


class IndexCallback:
    def __index__(self):
        raise AssertionError("Batch token parsing must not call user index code")


@pytest.mark.parametrize("kind", ["list", "short", "long", "bool", "negative", "index"])
def test_batch_token_preflight_consumes_client(batch_window, kind):
    rows = batch_window.args[2]
    tokens = (0,) * rows
    if kind == "list":
        tokens = list(tokens)
    elif kind == "short":
        tokens = tokens[:-1]
    elif kind == "long":
        tokens += (0,)
    else:
        invalid = {"bool": True, "negative": -1, "index": IndexCallback()}[kind]
        tokens = (invalid,) + tokens[1:]
    with pytest.raises((TypeError, ValueError, OverflowError)):
        batch_window.client.lookup_batch(tokens)
    with pytest.raises(ValueError, match="retired"):
        batch_window.client.execution()


@pytest.mark.parametrize("kind", ["short", "long", "mutable", "nan", "infinite",
                                  "bound", "boolean-stage"])
def test_batch_linear_preflight_consumes_client(batch_window, kind):
    batch_window.lookup()
    descriptor = batch_window.contract["batch_execution"]["descriptors"][1]
    stage = descriptor["stage"]
    count = descriptor["rows"] * descriptor["columns"]
    values = bytes(4 * count)
    if kind == "short":
        values = values[:-4]
    elif kind == "long":
        values += bytes(4)
    elif kind == "mutable":
        values = bytearray(values)
    elif kind == "boolean-stage":
        stage = False
    else:
        value = {"nan": float("nan"), "infinite": float("inf"), "bound": 2**21}[kind]
        values = struct.pack("<f", value) + values[4:]
    with pytest.raises((TypeError, ValueError, OverflowError)):
        batch_window.client.linear_batch(stage, values)
    with pytest.raises(ValueError, match="retired"):
        batch_window.client.execution()


@pytest.mark.parametrize("method", ["lookup_batch", "linear_batch"])
def test_reference_batch_signature_errors_consume_owner(batch_source, method):
    args, _, tensors = batch_source
    reference = _native._GkmDecoderSession(args[1], args[2], 1, tensors)
    try:
        with pytest.raises((TypeError, ValueError)):
            getattr(reference, method)(unexpected=None)
        with pytest.raises(ValueError, match="retired"):
            reference.contract()
    finally:
        reference.abort()
