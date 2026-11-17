"""Inert local prompt admission. Custody is not source or budget authority."""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace


_input_binding = ContextVar("pllm_local_input_binding", default=None)


@dataclass(frozen=True)
class _InputBinding:
    original: object
    selected: object
    original_canonical: bytes
    selected_canonical: bytes
    ids: tuple
    original_label: str | None
    selected_label: str | None
    source_preview: object = None


def retain_source_preview(binding, preview):
    if binding.source_preview is not None:
        if preview is not binding.source_preview:
            raise ValueError("token input source preview custody changed")
        return binding
    return replace(binding, source_preview=preview)


def bind_input(original, selected, admitted):
    """Custody/comparison only; both Experiments come from existing owners."""
    expected = original.with_params(deployment__root=selected.deployment.root)
    if (expected.canonical_bytes() != selected.canonical_bytes()
            or original.pipeline.profile != selected.pipeline.profile):
        raise ValueError("selected Experiment is not the exact owned root transformation")
    captured = admit_input(admitted, original.budget)
    if captured["ids"] is not admitted["ids"]:
        raise ValueError("input binding requires already owned immutable IDs")
    return _InputBinding(original, selected, original.canonical_bytes(),
                         selected.canonical_bytes(), admitted["ids"],
                         original.pipeline.profile, selected.pipeline.profile)


def check_input_binding(admitted, experiment, binding=None):
    binding = _input_binding.get() if binding is None else binding
    if binding is None:
        return
    if (experiment is None or admitted["ids"] is not binding.ids
            or experiment.canonical_bytes() != binding.selected_canonical
            or binding.original.canonical_bytes() != binding.original_canonical
            or experiment.pipeline.profile != binding.selected_label
            or binding.original.pipeline.profile != binding.original_label
            or binding.selected.pipeline.profile != binding.selected_label
            or binding.original.with_params(deployment__root=binding.selected.deployment.root
                ).canonical_bytes() != binding.selected_canonical):
        raise ValueError("token input complete Experiment/owned IDs association changed")
    admit_input(admitted, binding.original.budget)
    if binding.source_preview is not None:
        binding.source_preview[2]()


@contextmanager
def input_binding_scope(binding):
    token = _input_binding.set(binding)
    try:
        yield
    finally:
        _input_binding.reset(token)


class _ResponseInputs(dict):
    """Retain normalized inline identity beside IDs; target carries it over IPC."""
    def __init__(self, values, experiment):
        super().__init__(values)
        self.experiment = experiment
        self.canonical = experiment.canonical_bytes()
        self.label = experiment.pipeline.profile


class _PreparedInput:
    """Rejection marker only. Never grants budget, source, or input authority."""


def reject_prepared_input(value):
    if isinstance(value, _PreparedInput):
        raise ValueError("prepared prompt custody is not a transport input")


class _OwnedIds(tuple):
    def __deepcopy__(self, memo):
        return self


def is_token_input(value):
    # Any object input is reserved locally; malformed tags must not fall back to text.
    return isinstance(value, dict)


def _limit(budget, name):
    value = budget.get(name) if isinstance(budget, Mapping) else getattr(budget, name, None)
    if type(value) is not int or value < 1:
        raise ValueError("token IDs require an already admitted execution budget")
    return value


def _token_shape(value):
    if type(value) is not dict or len(value) != 2:
        raise ValueError("internal input requires exactly kind=token_ids and ids")
    # Bounded item capture never hashes/compares a caller key, even during mutation.
    items, fields, kind, ids = iter(value.items()), 0, None, None
    try:
        for index in range(3):
            pair = next(items, None)
            if pair is None:
                break
            key, item = pair
            if index == 2 or type(key) is not str:
                raise ValueError("internal input requires exactly kind=token_ids and ids")
            if key == "kind":
                fields |= 1
                kind = item
            elif key == "ids":
                fields |= 2
                ids = item
            else:
                raise ValueError("internal input requires exactly kind=token_ids and ids")
    except RuntimeError as error:
        raise ValueError("token input changed during admission") from error
    if fields != 3 or type(kind) is not str or kind != "token_ids":
        raise ValueError("internal input requires exactly kind=token_ids and ids")
    return ids


def admit_input(value, budget, *, output_tokens=None, requests=1, experiment=None):
    """Bounded capture; immutable owner supplies authority, never the request."""
    reject_prepared_input(value)
    if not is_token_input(value):
        return None
    ids = _token_shape(value)
    maximum = _limit(budget, "max_input_tokens")
    outputs = _limit(budget, "max_new_tokens")
    capacity = _limit(budget, "requests")
    if (type(requests) is not int or not 1 <= requests <= capacity
            or output_tokens is not None and (
                type(output_tokens) is not int or not 1 <= output_tokens <= outputs)):
        raise ValueError("token ID request exceeds execution budget")
    if type(ids) not in (list, _OwnedIds):
        raise ValueError("token IDs must be a nonempty bounded list")
    count = len(ids)
    # Existing native prompt seal remains 128 bytes / 32 u32s. No widening.
    if not 1 <= count <= min(maximum, 32):
        raise ValueError("token IDs exceed input budget or native prompt seal")
    captured = []
    try:
        for index in range(count):
            item = ids[index]
            if type(item) is not int or not 0 <= item < 2**32:
                raise ValueError("token IDs must be strict unsigned 32-bit integers")
            captured.append(item)
        if len(ids) != count or _token_shape(value) is not ids:
            raise ValueError("token input changed during admission")
        for index, item in enumerate(captured):
            current = ids[index]
            if type(current) is not int or current != item:
                raise ValueError("token input changed during admission")
    except (IndexError, RuntimeError) as error:
        raise ValueError("token input changed during admission") from error
    owned = ids if type(ids) is _OwnedIds else _OwnedIds(captured)
    admitted = {"kind": "token_ids", "ids": owned}
    if experiment is not None:
        check_input_binding(admitted, experiment)
    return admitted


def admit_body(body, budget, *, requests=1, experiment=None):
    value = body.get("input")
    output_tokens = body.get("max_output_tokens")
    if is_token_input(value) and output_tokens is None:
        output_tokens = _limit(budget, "max_new_tokens")
    admitted = admit_input(value, budget, output_tokens=output_tokens, requests=requests,
                           experiment=experiment)
    if admitted is None:
        return body, None
    if any(body.get(key) for key in ("instructions", "tools", "previous_response_id")):
        raise ValueError("token IDs require a fresh prompt without text instructions or tools")
    return {**body, "input": admitted, "max_output_tokens": output_tokens}, admitted


def check_vocabulary(admitted, vocabulary):
    if (type(vocabulary) is not int or vocabulary < 1
            or any(item >= vocabulary for item in admitted["ids"])):
        raise ValueError("token IDs exceed approved source vocabulary")


def _preflight_text(value):
    """Bounded legacy text shape, before any recursive JSON/copy operation."""
    if isinstance(value, str):
        if not 1 <= len(value) <= 16384 or not value.strip():
            raise ValueError("prompt must contain 1 to 16384 bytes")
        return value
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("prompt must be text or bounded text messages")
    size = len(value)
    captured = []
    total = 0
    for index in range(size):
        row = value[index]
        if not isinstance(row, dict) or len(row) != 2 or set(row) != {"role", "content"}:
            raise ValueError("prompt messages require only a text role and content")
        role, content = row["role"], row["content"]
        if (not isinstance(role, str) or role not in {"system", "developer", "user", "assistant"}
                or not isinstance(content, str)):
            raise ValueError("prompt messages require only a text role and content")
        total += len(content)
        if total > 16384:
            raise ValueError("prompt messages exceed their text bound")
        captured.append({"role": role, "content": content})
    if len(value) != size or not any(row["content"].strip() for row in captured):
        raise ValueError("prompt messages changed or contain no text")
    return captured


def preflight_response(parameters, inputs, *, experiment_loader=None):
    """Controller/consumer preflight, before input copies, JSON or cold imports."""
    scenario = parameters["scenario"]
    field = "prompts" if scenario["kind"] == "sequence" else "prompt"
    if (not isinstance(inputs, Mapping) or len(inputs) != 1
            or any(type(key) is not str for key in inputs) or field not in inputs):
        raise ValueError("response inputs require exactly the scenario prompt field")
    value = inputs.get(field)
    reject_prepared_input(value)
    if field == "prompts" and not isinstance(value, (list, tuple)):
        raise ValueError("response sequence must be a list or tuple")
    values = value if field == "prompts" else (value,)
    count = len(values)
    if field == "prompts" and count != scenario["count"]:
        raise ValueError("sequence length differs from scenario")
    # Bound sequence iteration before looking at any prompt.
    if count > 32:
        raise ValueError("response sequence exceeds its bound")
    captured = []
    experiment = None
    for index in range(count):
        context = values[index]
        reject_prepared_input(context)
        if is_token_input(context):
            _token_shape(context)
            if parameters.get("concurrent_requests", 1) != 1:
                raise ValueError("token IDs do not support concurrent response groups")
            if parameters.get("local_dispatch") is not True:
                raise ValueError("token IDs are local-only")
            declaration = parameters.get("model", {}).get("experiment")
            if type(declaration) is not dict or type(declaration.get("pipeline")) is not dict:
                raise ValueError("token IDs require an inline Experiment, not a factory")
            if experiment_loader is None:
                raise ValueError("token IDs require normalized inline Experiment authority")
            if experiment is None:
                experiment = experiment_loader(parameters["model"])
                if type(inputs) is _ResponseInputs and (
                        inputs.canonical != experiment.canonical_bytes()
                        or inputs.label != experiment.pipeline.profile):
                    raise ValueError("inline Experiment changed after input capture")
            context = admit_input(context, experiment.budget,
                output_tokens=scenario["max_output_tokens"], requests=scenario["count"])
        captured.append(context)
    if len(values) != count:
        raise ValueError("response sequence changed during admission")
    # Replacing the prompt retains owned IDs through every later copy/serialization.
    captured = [context if is_token_input(context) else _preflight_text(context) for context in captured]
    result = {field: captured if field == "prompts" else captured[0]}
    return result if experiment is None else _ResponseInputs(result, experiment)


def admit_benchmark_prompts(prompt, sequence, warmup, *, experiment, local_dispatch,
                            outputs, warmups, repetitions):
    if type(local_dispatch) is not bool:
        raise ValueError("local dispatch must be a boolean")
    size = None if sequence is None else len(sequence) if isinstance(sequence, (list, tuple)) else 0
    if sequence is not None and not 1 <= size <= 32:
        raise ValueError("prompt sequence requires 1-32 contexts")
    count = warmups + repetitions * (size if sequence is not None else 1)
    budget = None if experiment is None else experiment.budget
    def capture(value):
        reject_prepared_input(value)
        if not is_token_input(value):
            return value
        if not local_dispatch:
            raise ValueError("token IDs cannot use HTTP dispatch")
        return admit_input(value, budget, output_tokens=outputs, requests=count)
    prompt = capture(prompt)
    if sequence is not None:
        captured = tuple(capture(sequence[index]) for index in range(size))
        if len(sequence) != size:
            raise ValueError("prompt sequence changed during admission")
        sequence = captured
    warmup = None if warmup is None else capture(warmup)
    return prompt, sequence, warmup
