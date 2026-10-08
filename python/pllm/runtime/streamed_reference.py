"""Bounded, module-streamed FP32 weights for the independent Torch reference.

The upstream model still executes every operator. Only parameter ownership changes:
checkpoint rows are copied into one active module, then retired after its forward.
This is a local quality diagnostic, not a serving or encrypted execution backend.
"""
from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from pllm.runtime.safetensors_store import SafeTensorStore

CHUNK_BYTES = 8 << 20
MAX_WEIGHT_BYTES = 2 << 30
MAX_BUFFER_BYTES = 16 << 20
MAX_OUTPUT_BYTES = 512 << 20


class StreamedReferenceError(ValueError):
    pass


class StreamedFloat32Reference:
    """Context-owned eager reference; all checkpoint parameters rest on meta."""

    def __init__(self, path: str | Path, *, max_input_tokens: int = 128,
                 max_decode_steps: int = 0,
                 max_weight_bytes: int = MAX_WEIGHT_BYTES):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        if (type(max_input_tokens) is not int or not 1 <= max_input_tokens <= 128
                or type(max_decode_steps) is not int or not 0 <= max_decode_steps <= 8
                or type(max_weight_bytes) is not int
                or not 0 < max_weight_bytes <= MAX_WEIGHT_BYTES):
            raise StreamedReferenceError("invalid streamed reference bounds")
        self._torch = torch
        self._store = SafeTensorStore(path)
        self._max_input_tokens = max_input_tokens
        self._max_decode_steps = max_decode_steps
        self._hooks = []
        self._owners = []
        self._closed = False
        self._guard = None
        self._active_bytes = 0
        self.stats = {"parameter_loads": 0, "loaded_float32_bytes": 0,
                      "load_seconds": 0., "peak_active_weight_bytes": 0}
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if max_input_tokens * config.vocab_size * 4 > MAX_OUTPUT_BYTES:
            raise StreamedReferenceError("streamed reference logits exceed 512 MiB")
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=False, dtype=torch.float32,
                attn_implementation="eager",
            ).eval()
        try:
            # Rebuild only parameter-free, config-initialized buffer leaves. This
            # also replaces upstream aliases of nonpersistent rotary buffers.
            buffer_bytes = 0
            for name, module in list(self.model.named_modules()):
                buffers = dict(module.named_buffers(recurse=False))
                if not buffers:
                    continue
                sizes = {key: (tuple(value.shape), value.dtype) for key, value in buffers.items()}
                buffer_bytes += sum(value.numel() * value.element_size() for value in buffers.values())
                if (buffer_bytes > MAX_BUFFER_BYTES or list(module.parameters())
                        or list(module.children()) or not name or not hasattr(module, "config")
                        or set(inspect.signature(type(module)).parameters) - {"config", "device"}):
                    raise StreamedReferenceError("unsupported reference buffer owner")
                with torch.device("cpu"):
                    replacement = type(module)(module.config)
                actual = dict(replacement.named_buffers(recurse=False))
                if (list(replacement.parameters()) or set(actual) != set(sizes)
                        or any(value.device.type != "cpu"
                               or (tuple(value.shape), value.dtype) != sizes[key]
                               for key, value in actual.items())):
                    raise StreamedReferenceError("reference buffer reconstruction changed its contract")
                self.model.set_submodule(name, replacement)

            aliases = {}
            for name, parameter in self.model.named_parameters(remove_duplicate=False):
                aliases.setdefault(id(parameter), []).append(name)
            source_keys = set(self._store.keys)
            sizes = {}
            for name, module in self.model.named_modules():
                entries = []
                for local, parameter in module.named_parameters(recurse=False):
                    full = f"{name}.{local}" if name else local
                    candidates = [key for key in aliases[id(parameter)] if key in source_keys]
                    key = full if full in source_keys else next(iter(candidates), None)
                    if (key is None or tuple(parameter.shape) != self._store.tensor_shape(key)
                            or self._store.tensor_dtype(key) not in {"F32", "F16", "BF16"}
                            or not parameter.is_meta or parameter.ndim not in {1, 2}):
                        raise StreamedReferenceError("reference parameter is missing or incompatible")
                    entries.append((local, key, parameter))
                if entries:
                    size = sum(parameter.numel() * 4 for _, _, parameter in entries)
                    sizes[name] = size
                    self._owners.append((module, entries, size))
            # A parent with direct parameters can remain live during child calls.
            self.weight_bound = max((sum(size for parent, size in sizes.items()
                                    if parent == name or not parent or name.startswith(parent + "."))
                                     for name in sizes), default=0)
            if not 0 < self.weight_bound <= max_weight_bytes:
                raise StreamedReferenceError("active reference weights exceed the admitted bound")
            self.buffer_bytes = buffer_bytes
            self._files = {location.path: self._signature(location.path)
                           for location in self._store._locations.values()}
            # Additional owners after model construction: weights, overlapping
            # conversion chunks, logits, eager attention/activation workspaces,
            # and bounded library scratch. The host's own reserve stays separate.
            work = (max_input_tokens * config.vocab_size * 4
                     + 16 * max_input_tokens * max(config.hidden_size, config.intermediate_size) * 4
                     + 4 * config.num_attention_heads * max_input_tokens ** 2 * 4)
            # Dynamic eager KV plus old/new concatenation and transient owners.
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            self.kv_bound_bytes = (2 * config.num_hidden_layers
                                   * getattr(config, "num_key_value_heads", config.num_attention_heads)
                                   * head_dim * (max_input_tokens + max_decode_steps) * 4
                                   if max_decode_steps else 0)
            work += 3 * self.kv_bound_bytes
            # Safetensors maps source rows during conversion. Conservatively
            # price another full FP32 active-weight owner for touched source pages.
            self.allocation_estimate_bytes = (
                (2 * self.weight_bound + 3 * CHUNK_BYTES + work + (256 << 20)) * 5 + 3) // 4
            from pllm.runtime.benchmark_memory import MemoryWatchdog, host_memory

            host = host_memory()
            if self.allocation_estimate_bytes > host.available - host.reserve:
                raise StreamedReferenceError("streamed reference exceeds post-reserve host headroom")
            self._guard = MemoryWatchdog(
                {"host": asdict(host) | {"reserve_bytes": host.reserve}}, lambda _: None)
            self._guard.start()
            for module, entries, size in self._owners:
                self._hooks.append(module.register_forward_pre_hook(self._loader(entries, size)))
                self._hooks.append(module.register_forward_hook(
                    self._retirer(entries, size), always_call=True))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _load(self, key, shape):
        columns = int(np.prod(shape[1:], dtype=np.int64))
        if columns * 4 > CHUNK_BYTES:
            raise StreamedReferenceError("reference row exceeds the loading buffer bound")
        result = np.empty(shape, dtype=np.float32)
        rows = max(1, CHUNK_BYTES // (4 * columns))
        selections = [slice(start, min(start + rows, shape[0]))
                      for start in range(0, shape[0], rows)]
        for selection, value in zip(selections, self._store.iter_slices(key, selections), strict=True):
            if self._guard.error:
                raise StreamedReferenceError(self._guard.error)
            result[selection] = value
        return self._torch.nn.Parameter(self._torch.from_numpy(result), requires_grad=False)

    def _loader(self, entries, size):
        def load(module, _inputs):
            if self._active_bytes + size > self.weight_bound:
                raise StreamedReferenceError("reference parameter lifetime exceeded its bound")
            started = time.perf_counter()
            for local, key, empty in entries:
                if not module._parameters[local].is_meta:
                    raise StreamedReferenceError("reference parameter was not retired")
                module._parameters[local] = self._load(key, tuple(empty.shape))
                self._active_bytes += empty.numel() * 4
                self.stats["parameter_loads"] += 1
                self.stats["loaded_float32_bytes"] += empty.numel() * 4
                self.stats["peak_active_weight_bytes"] = max(
                    self.stats["peak_active_weight_bytes"], self._active_bytes)
            self.stats["load_seconds"] += time.perf_counter() - started
        return load

    def _retirer(self, entries, _size):
        def retire(module, _inputs, _output):
            for local, _key, empty in entries:
                if not module._parameters[local].is_meta:
                    self._active_bytes -= empty.numel() * 4
                    module._parameters[local] = empty
        return retire

    def logits(self, token_ids):
        return self.trajectory(token_ids, ())[0]

    def trajectory(self, token_ids, decode_ids):
        """Execute prefill and bounded same-token cached decode; retire each owner."""
        if self._closed:
            raise StreamedReferenceError("streamed reference is closed")
        ids = tuple(token_ids)
        continuation = tuple(decode_ids)
        if (not 1 <= len(ids) <= self._max_input_tokens
                or len(continuation) > self._max_decode_steps
                or any(type(value) is not int or not 0 <= value < self.model.config.vocab_size
                        for value in ids + continuation)):
            raise StreamedReferenceError("reference token input exceeds its bounds")
        output = cache = None
        try:
            self._guard.check()
            if any(self._signature(path) != signature for path, signature in self._files.items()):
                raise StreamedReferenceError("reference checkpoint changed after admission")
            kwargs = {"input_ids": self._torch.tensor([ids]), "use_cache": bool(continuation)}
            with self._torch.inference_mode():
                output = self.model(**kwargs)
                result = [output.logits[0, -1].float().numpy().copy()]
                cache = output.past_key_values if continuation else None
                del output
                for token in continuation:
                    self._guard.check()
                    output = self.model(input_ids=self._torch.tensor([[token]]),
                                        past_key_values=cache, use_cache=True)
                    result.append(output.logits[0, -1].float().numpy().copy())
                    cache = output.past_key_values
                    del output
                cache = None
            self._guard.check()
            if self._guard.error:
                raise StreamedReferenceError(self._guard.error)
            if any(self._signature(path) != signature for path, signature in self._files.items()):
                raise StreamedReferenceError("reference checkpoint changed during execution")
            return result
        finally:
            output = cache = None
            self._retire_all()

    def _retire_all(self):
        for module, entries, _size in self._owners:
            for local, _key, empty in entries:
                module._parameters[local] = empty
        self._active_bytes = 0
        self._store.clear_cache()

    def close(self):
        if self._guard is not None:
            self._guard.close()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._retire_all()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
