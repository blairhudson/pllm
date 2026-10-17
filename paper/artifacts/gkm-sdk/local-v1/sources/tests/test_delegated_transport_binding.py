"""Argument errors must consume genuine one-use native transport owners."""

import json
import math
import struct

import pytest

from pllm import _native


_PARAMETERS = {
    "publication": (), "starts": (), "lookup": ("token",),
    "linear": ("stage", "input"), "accept": ("zero", "one"),
    "execute": ("request",), "finish": (), "abort": (),
}
_METHODS = [("client", method) for method in (
    "publication", "starts", "lookup", "linear", "accept", "finish", "abort",
)] + [("worker", method) for method in ("execute", "finish", "abort")]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture(scope="module")
def admitted():
    # One row, one output, one tiny layer; no source files, servers or model load.
    experiment = _canonical({
        "schema": "pllm.experiment.v2", "name": "transport-binding",
        "pipeline": {"model": {"source": "fixture", "kind": "safetensors"}, "components": {
            "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
            "linear": {"component": "pllm/open-weight-verifiable-outsourcing/v1",
                       "params": {"sharing": "full", "generation": "fixed_length_greedy"}},
            "topology": {"component": "pllm/two-online-delegation-workers/v1", "params": {}},
            "state": {"component": "pllm/client-local-kv", "params": {}},
            "verification": {"component": "pllm/published-linear-verify/v1",
                             "params": {"acceptance": "request"}},
        }},
        "deployment": {"kind": "local", "root": "./transport-binding-fixture"},
        "budget": {"requests": 1, "max_input_tokens": 1, "max_new_tokens": 1},
    })
    config = _canonical({
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 16, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
    })
    args = experiment, config, 1, 1, "01" * 32, "02" * 32
    decoder = json.loads(_native.gkm_runtime_contract(*args))["decoder"]
    tensors = {name: struct.pack("<f", 0.125) * math.prod(shape)
               for name, shape in decoder["source_shapes"].items()}
    table = decoder["table"]
    tensors[table] = struct.pack("<f", -0.0) + tensors[table][4:]
    return args, decoder, tensors


class _Window:
    def __init__(self, admitted):
        self.args, self.decoder, self.tensors = admitted
        self.client = _native._GkmTransportClient(*self.args, self.tensors)
        self.workers = []
        self.style = "positional"

    def call(self, owner, method, *values):
        invoke = getattr(owner, method)
        names = _PARAMETERS[method]
        if self.style == "named":
            # Reverse insertion order so parsing cannot depend on keyword order.
            return invoke(**dict(zip(reversed(names), reversed(values), strict=True)))
        if self.style == "mixed" and values:
            return invoke(values[0], **dict(zip(names[1:], values[1:], strict=True)))
        return invoke(*values)

    def begin(self):
        publication = self.call(self.client, "publication")
        pin = self.client.publication_digest()
        starts = self.call(self.client, "starts")
        for role in range(2):
            self.workers.append(_native._GkmTransportWorker(
                *self.args, self.tensors, publication, pin, role, starts[role],
            ))

    def replies(self, frames):
        return tuple(self.call(worker, "execute", frame)
                     for worker, frame in zip(self.workers, frames, strict=True))

    def lookup(self):
        frames = self.call(self.client, "lookup", 0)
        row = self.call(self.client, "accept", *self.replies(frames))
        assert row == self.tensors[self.decoder["table"]][:self.decoder["width"] * 4]

    def complete(self):
        self.begin()
        self.lookup()
        assert self.decoder["window_steps"] == len(self.decoder["matrices"]) + 1
        for stage, matrix in enumerate(self.decoder["matrices"]):
            frames = self.call(self.client, "linear", stage, bytes(matrix["columns"] * 4))
            result = self.call(self.client, "accept", *self.replies(frames))
            assert result == bytes(matrix["rows"] * 4)

    def ready(self, side, method):
        values = ()
        if method == "finish":
            self.complete()
        elif side == "worker" or method in {"lookup", "linear", "accept", "abort"}:
            self.begin()
            if method == "lookup":
                values = (0,)
            elif method == "linear":
                self.lookup()
                values = (0, bytes(self.decoder["matrices"][0]["columns"] * 4))
            elif method in {"accept", "execute", "abort"}:
                frames = self.client.lookup(0)
                if method == "accept":
                    values = self.replies(frames)
                elif method == "execute":
                    values = (frames[0],)
        return (self.client if side == "client" else self.workers[0]), values

    def close(self):
        self.client.abort()
        for worker in self.workers:
            worker.abort()


@pytest.fixture
def window(admitted):
    owner = _Window(admitted)
    try:
        yield owner
    finally:
        owner.close()


def _signature_cases():
    for side, method in _METHODS:
        names = _PARAMETERS[method]
        cases = ["extra_positional", "unexpected_keyword"]
        if names:
            cases += ["missing", "unknown_only"]
            cases += [f"duplicate_{index}" for index in range(len(names))]
        if len(names) == 2:
            cases.append("missing_all")
        for case in cases:
            yield pytest.param(side, method, case, id=f"{side}-{method}-{case}")


def _assert_retired(owner):
    with pytest.raises(ValueError, match="retired"):
        owner.request_id()
    with pytest.raises(ValueError, match="retired"):
        owner.finish()
    owner.abort()
    owner.abort()  # Valid abort stays idempotent after every failure.


@pytest.mark.parametrize("side,method,case", tuple(_signature_cases()))
def test_signature_errors_retire_owner(window, side, method, case):
    owner, values = window.ready(side, method)
    names = _PARAMETERS[method]
    args, kwargs = values, {}
    if case == "extra_positional":
        args += (None,)
    elif case == "unexpected_keyword":
        kwargs = {"unexpected": True}
    elif case == "missing":
        args = args[:-1]
    elif case == "missing_all":
        args = ()
    elif case == "unknown_only":
        args, kwargs = (), {"unexpected": True}
    else:
        index = int(case.removeprefix("duplicate_"))
        kwargs = {names[index]: values[index]}
    with pytest.raises(TypeError):
        getattr(owner, method)(*args, **kwargs)
    _assert_retired(owner)
    if method != "abort":
        with pytest.raises(ValueError, match="retired"):
            getattr(owner, method)(*values)


def test_keyword_subclass_cannot_alias_a_parameter(window):
    class Keyword(str):
        def __hash__(self):
            return hash("token")

        def __eq__(self, other):
            return other == "token"

    owner, _ = window.ready("client", "lookup")
    with pytest.raises(TypeError):
        owner.lookup(**{Keyword("unexpected"): 0})
    _assert_retired(owner)


@pytest.mark.parametrize("side,method,index,bad", [
    ("client", "lookup", 0, True),
    ("client", "lookup", 0, "0"),
    ("client", "lookup", 0, -1),
    ("client", "linear", 0, False),
    ("client", "linear", 1, bytearray()),
    ("client", "accept", 0, bytearray()),
    ("client", "accept", 1, bytearray()),
    ("worker", "execute", 0, bytearray()),
])
def test_value_conversion_errors_still_retire_owner(window, side, method, index, bad):
    owner, values = window.ready(side, method)
    args = list(values)
    args[index] = bad
    with pytest.raises((TypeError, ValueError, OverflowError)):
        getattr(owner, method)(*args)
    _assert_retired(owner)


@pytest.mark.parametrize("style", ["positional", "named", "mixed"])
def test_valid_call_forms_complete_native_acceptance(window, style):
    window.style = style
    window.complete()
    request = window.client.request_id().hex()
    completions = [json.loads(window.call(worker, "finish")) for worker in window.workers]
    acceptance = json.loads(window.call(window.client, "finish"))
    assert acceptance["schema_version"] == "pllm.gkm.sdk-acceptance.v1"
    assert acceptance["request"] == request
    assert acceptance["steps"] == window.decoder["window_steps"]
    for role, completion in enumerate(completions):
        assert completion["schema_version"] == "pllm.gkm.sdk-worker-completion.v1"
        assert completion["worker"] == role
        for key in ("contract", "request", "publication", "steps"):
            assert completion[key] == acceptance[key]
    for owner in [window.client, *window.workers]:
        _assert_retired(owner)
