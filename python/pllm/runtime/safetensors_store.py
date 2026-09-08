from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import ml_dtypes  # noqa: F401 - registers NumPy bfloat16 for Safetensors


class TensorStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TensorLocation:
    key: str
    path: Path


class SafeTensorStore:
    """Lazy reader for single-file and sharded Hugging Face/MLX checkpoints.

    Tensor names are resolved by exact key first and then by unique suffix. The
    suffix mode is intentional: model wrappers commonly add prefixes such as
    ``model.language_model`` without changing the architecture-level names.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise TensorStoreError(f"checkpoint path does not exist: {self.root}")
        self._locations = self._discover()
        self._cache: dict[str, np.ndarray] = {}
        self._metadata: dict[str, tuple[tuple[int, ...], str]] = {}
        self._file_signatures: dict[Path, tuple[int, int]] = {}
        self.quantization = self._load_quantization_config()


    def _load_quantization_config(self) -> dict[str, object]:
        config_path = self.root / "config.json"
        if not config_path.exists():
            return {}
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        for key in ("quantization", "quantization_config"):
            value = config.get(key)
            if isinstance(value, dict):
                return dict(value)
        text = config.get("text_config")
        if isinstance(text, dict):
            for key in ("quantization", "quantization_config"):
                value = text.get(key)
                if isinstance(value, dict):
                    return dict(value)
        return {}

    def drop(self, *keys: str) -> None:
        """Release materialised source tensors after stage compilation."""
        for item in keys:
            try:
                key = self.resolve(item)
            except TensorStoreError:
                continue
            self._cache.pop(key, None)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _discover(self) -> dict[str, TensorLocation]:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - dependency declaration covers normal install
            raise TensorStoreError("safetensors is required to load model weights") from exc

        index_candidates = sorted(self.root.glob("*.safetensors.index.json"))
        locations: dict[str, TensorLocation] = {}
        if index_candidates:
            index = json.loads(index_candidates[0].read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise TensorStoreError("invalid safetensors index: missing weight_map")
            paths = {str(file_name): self.root / str(file_name) for file_name in weight_map.values()}
            for path in paths.values():
                if not path.exists():
                    raise TensorStoreError(f"missing shard {path}")
            for key, file_name in weight_map.items():
                path = paths[str(file_name)]
                locations[str(key)] = TensorLocation(str(key), path)
            return locations

        files = sorted(self.root.glob("*.safetensors"))
        if not files:
            raise TensorStoreError(f"no .safetensors weights found in {self.root}")
        for path in files:
            with safe_open(path, framework="np") as handle:
                for key in handle.keys():
                    if key in locations:
                        raise TensorStoreError(f"duplicate tensor key {key!r}")
                    locations[key] = TensorLocation(key, path)
        return locations

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._locations)

    def contains(self, key_or_suffix: str) -> bool:
        try:
            self.resolve(key_or_suffix)
            return True
        except TensorStoreError:
            return False

    def resolve(self, key_or_suffix: str) -> str:
        if key_or_suffix in self._locations:
            return key_or_suffix
        suffix = key_or_suffix if key_or_suffix.startswith(".") else "." + key_or_suffix
        matches = [key for key in self._locations if key.endswith(suffix)]
        if not matches:
            # A few checkpoints omit the separator before top-level names.
            matches = [key for key in self._locations if key.endswith(key_or_suffix)]
        if len(matches) != 1:
            detail = "not found" if not matches else f"ambiguous: {matches[:8]}"
            raise TensorStoreError(f"tensor {key_or_suffix!r} {detail}")
        return matches[0]

    def resolve_first(self, candidates: Iterable[str]) -> str:
        errors: list[str] = []
        for candidate in candidates:
            try:
                return self.resolve(candidate)
            except TensorStoreError as exc:
                errors.append(str(exc))
        raise TensorStoreError("; ".join(errors) or "no tensor candidates supplied")


    def _tensor_metadata(self, key_or_suffix: str) -> tuple[tuple[int, ...], str]:
        key = self.resolve(key_or_suffix)
        if key not in self._metadata:
            from safetensors import safe_open

            location = self._locations[key]
            with safe_open(location.path, framework="np") as handle:
                for shard_key in handle.keys():
                    tensor = handle.get_slice(shard_key)
                    self._metadata[shard_key] = (
                        tuple(int(value) for value in tensor.get_shape()),
                        str(tensor.get_dtype()),
                    )
        return self._metadata[key]

    def tensor_shape(self, key_or_suffix: str) -> tuple[int, ...]:
        return self._tensor_metadata(key_or_suffix)[0]

    def tensor_dtype(self, key_or_suffix: str) -> str:
        return self._tensor_metadata(key_or_suffix)[1]

    @staticmethod
    def _convert(value: object, dtype: np.dtype | type | None) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype.name == "bfloat16":
            array = array.astype(np.float32)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return np.ascontiguousarray(array)

    def get_slice(
        self,
        key_or_suffix: str,
        selection: object,
        *,
        dtype: np.dtype | type | None = np.float32,
    ) -> np.ndarray:
        """Read only the requested Safetensors region without materialising the tensor."""
        key = self.resolve(key_or_suffix)
        from safetensors import safe_open

        location = self._locations[key]
        with safe_open(location.path, framework="np") as handle:
            value = handle.get_slice(key)[selection]
        return self._convert(value, dtype)

    def iter_slices(
        self,
        key_or_suffix: str,
        selections: Iterable[object],
        *,
        dtype: np.dtype | type | None = np.float32,
    ) -> Iterator[np.ndarray]:
        """Stream regions while keeping the source shard open once."""
        key = self.resolve(key_or_suffix)
        from safetensors import safe_open

        location = self._locations[key]
        with safe_open(location.path, framework="np") as handle:
            tensor = handle.get_slice(key)
            for selection in selections:
                yield self._convert(tensor[selection], dtype)

    def source_signature(self, keys: Iterable[str]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in keys:
            key = self.resolve(item)
            location = self._locations[key]
            signature = self._file_signatures.get(location.path)
            if signature is None:
                stat = location.path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                self._file_signatures[location.path] = signature
            result.append({
                "key": key,
                "file": location.path.name,
                "size": signature[0],
                "mtime_ns": signature[1],
                "shape": list(self.tensor_shape(key)),
                "dtype": self.tensor_dtype(key),
            })
        return result

    def get(self, key_or_suffix: str, *, dtype: np.dtype | type | None = np.float32) -> np.ndarray:
        key = self.resolve(key_or_suffix)
        if key not in self._cache:
            from safetensors import safe_open

            location = self._locations[key]
            with safe_open(location.path, framework="np") as handle:
                value = np.asarray(handle.get_tensor(key))
            # NumPy does not consistently expose bfloat16 across supported versions.
            if value.dtype.name == "bfloat16":
                value = value.astype(np.float32)
            self._cache[key] = np.ascontiguousarray(value)
        value = self._cache[key]
        return np.ascontiguousarray(value.astype(dtype, copy=False)) if dtype is not None else value

    def get_optional(self, candidates: Iterable[str], *, dtype: np.dtype | type = np.float32) -> np.ndarray | None:
        try:
            key = self.resolve_first(candidates)
        except TensorStoreError:
            return None
        return self.get(key, dtype=dtype)

    def get_linear(self, candidates: Iterable[str]) -> np.ndarray:
        """Load a dense matrix or dequantize common packed checkpoint layouts.

        Supported packed forms:

        * this project's ``weight_packed`` signed-nibble QAT representation;
        * native MLX affine quantization, where ``weight`` is uint32-packed and
          sibling ``scales``/``biases`` tensors hold per-group dequantization
          parameters (``q * scale + bias``).
        """

        candidates = tuple(candidates)
        dense_error: TensorStoreError | None = None
        try:
            key = self.resolve_first(candidates)
            raw = self.get(key, dtype=None)
            if np.issubdtype(raw.dtype, np.floating):
                return np.ascontiguousarray(raw.astype(np.float32, copy=False))
            if raw.ndim == 2 and np.issubdtype(raw.dtype, np.integer):
                stem = key[:-7] if key.endswith(".weight") else key
                scales = self.get_optional((stem + ".scales",), dtype=np.float32)
                biases = self.get_optional((stem + ".biases",), dtype=np.float32)
                if scales is not None:
                    return self._dequantize_mlx_affine(raw, scales, biases, stem=stem)
                dense_error = TensorStoreError(
                    f"integer tensor {key!r} has no supported quantization metadata"
                )
            else:
                dense_error = TensorStoreError(
                    f"unsupported tensor dtype/shape for {key!r}: {raw.dtype} {raw.shape}"
                )
        except TensorStoreError as exc:
            dense_error = exc

        for dense_key in candidates:
            stem = dense_key[:-7] if dense_key.endswith(".weight") else dense_key
            packed_candidates = (stem + ".weight_packed", dense_key + "_packed")
            try:
                packed_key = self.resolve_first(packed_candidates)
            except TensorStoreError:
                continue
            packed = self.get(packed_key, dtype=None)
            shape_value = self.get_optional(
                (stem + ".weight_shape", stem + ".shape"), dtype=np.int64
            )
            if shape_value is None or shape_value.size != 2:
                raise TensorStoreError(f"packed tensor {packed_key!r} has no two-element weight_shape")
            out_features, in_features = (int(v) for v in shape_value.reshape(-1))
            raw = np.ascontiguousarray(packed).view(np.uint8).reshape(-1)
            nibbles = np.empty(raw.size * 2, dtype=np.int8)
            nibbles[0::2] = (raw & 0x0F).astype(np.int8)
            nibbles[1::2] = (raw >> 4).astype(np.int8)
            nibbles[nibbles >= 8] -= 16
            quantized = nibbles[: out_features * in_features].reshape(out_features, in_features)
            scale = self.get_optional(
                (stem + ".weight_scale", stem + ".scale"), dtype=np.float32
            )
            if scale is None:
                raise TensorStoreError(f"packed tensor {packed_key!r} has no weight scale")
            if scale.size == 1:
                return quantized.astype(np.float32) * float(scale.reshape(-1)[0])
            if scale.shape == (out_features,) or scale.shape == (out_features, 1):
                return quantized.astype(np.float32) * scale.reshape(out_features, 1)
            raise TensorStoreError(
                f"unsupported groupwise scale shape {scale.shape} for {packed_key!r}"
            )
        assert dense_error is not None
        raise dense_error

    def _dequantize_mlx_affine(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        biases: np.ndarray | None,
        *,
        stem: str,
    ) -> np.ndarray:
        bits = int(self.quantization.get("bits", 4))
        group_size = int(self.quantization.get("group_size", 64))
        if bits <= 0 or bits > 8 or 32 % bits:
            raise TensorStoreError(f"unsupported MLX quantization bit width {bits} for {stem!r}")
        if group_size <= 0:
            raise TensorStoreError(f"invalid MLX group size {group_size} for {stem!r}")

        words = np.ascontiguousarray(packed).astype(np.uint32, copy=False)
        out_features, packed_in = words.shape
        pack_factor = 32 // bits
        mask = np.uint32((1 << bits) - 1)
        shifts = (np.arange(pack_factor, dtype=np.uint32) * np.uint32(bits)).reshape(1, 1, -1)
        qvalues = ((words[..., None] >> shifts) & mask).reshape(out_features, packed_in * pack_factor)

        scales = np.asarray(scales, dtype=np.float32)
        if scales.ndim == 1:
            scales = scales.reshape(out_features, -1)
        elif scales.shape[0] != out_features and scales.ndim == 2 and scales.shape[1] == out_features:
            scales = scales.T
        if scales.ndim != 2 or scales.shape[0] != out_features:
            raise TensorStoreError(f"invalid MLX scale shape {scales.shape} for {stem!r}")
        n_groups = scales.shape[1]
        expected_groups = (qvalues.shape[1] + group_size - 1) // group_size
        if n_groups != expected_groups:
            raise TensorStoreError(
                f"MLX scale groups {n_groups} do not cover {qvalues.shape[1]} inputs "
                f"at group_size={group_size} for {stem!r}"
            )

        if biases is None:
            # MLX symmetric quantization uses the midpoint as the implicit zero.
            biases = -float(1 << (bits - 1)) * scales
        else:
            biases = np.asarray(biases, dtype=np.float32)
            if biases.ndim == 1:
                biases = biases.reshape(out_features, -1)
            elif biases.shape[0] != out_features and biases.ndim == 2 and biases.shape[1] == out_features:
                biases = biases.T
            if biases.shape != scales.shape:
                raise TensorStoreError(
                    f"MLX bias shape {biases.shape} does not match scales {scales.shape} for {stem!r}"
                )

        expanded_scales = np.repeat(scales, group_size, axis=1)[:, : qvalues.shape[1]]
        expanded_biases = np.repeat(biases, group_size, axis=1)[:, : qvalues.shape[1]]
        return np.ascontiguousarray(
            qvalues.astype(np.float32) * expanded_scales + expanded_biases
        )
