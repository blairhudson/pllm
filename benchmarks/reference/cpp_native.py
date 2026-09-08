from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np


class NativeKernelError(RuntimeError):
    pass


class MaskedGEMM:
    """Exact same-weight integer GEMM with optional AVX2/OpenMP kernels."""

    _build_lock = threading.Lock()

    def __init__(self, library: str | Path | None = None, *, threads: int = 0, build: bool = True) -> None:
        self.threads = int(threads or max(1, min(os.cpu_count() or 1, 8)))
        self.library_path: Path | None = None
        self.lib: ctypes.CDLL | None = None
        self._modular_function = None
        self._modular_limb12_function = None
        self._wrap32_function = None
        self._clear_function = None
        if library is not None:
            self._load(Path(library))
            return

        package_dir = Path(__file__).resolve().parent
        root = package_dir.parent
        environment_library = os.environ.get("PLLM_NATIVE_LIBRARY")
        candidates = tuple(filter(None, (
            Path(environment_library) if environment_library else None,
            package_dir / _library_name(),
            package_dir / "libhe_openai_masked_gemm.so",
            root / "native" / _library_name(),
            root / "native" / "libhe_openai_masked_gemm.so",
        )))
        for candidate in candidates:
            if candidate.exists():
                self._load(candidate)
                return
        if build:
            try:
                self._load(build_native_library())
            except Exception:
                # NumPy remains a correctness fallback on unsupported systems.
                self.lib = None

    @property
    def native(self) -> bool:
        return self._modular_function is not None

    @property
    def available(self) -> bool:
        return self.native

    def __call__(self, weights: np.ndarray, inputs: np.ndarray, modulus: int) -> np.ndarray:
        return self.modular(weights, inputs, modulus)

    def modular(self, weights: np.ndarray, inputs: np.ndarray, modulus: int) -> np.ndarray:
        w = np.ascontiguousarray(weights, dtype=np.int8)
        x = np.ascontiguousarray(inputs, dtype=np.uint32)
        _validate_shapes(w, x)
        if modulus <= 1 or modulus >= 2**31:
            raise NativeKernelError("prime-ring modulus must satisfy 1 < modulus < 2^31")
        if self._modular_function is None:
            raw = x.astype(np.int64) @ w.astype(np.int64).T
            return (raw % int(modulus)).astype(np.uint32)
        output = np.empty((x.shape[0], w.shape[0]), dtype=np.uint32)
        function = self._select_modular_kernel(w, x, modulus)
        function(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            x.shape[0],
            w.shape[0],
            w.shape[1],
            int(modulus),
            self.threads,
        )
        return output


    def _select_modular_kernel(self, weights: np.ndarray, inputs: np.ndarray, modulus: int):
        """Select the exact AVX2 kernel for the current stage shape.

        The 12-bit limb path doubles SIMD density for wide residues. It wins on
        very wide inputs and large output banks, while the shared-weight batch
        kernel remains better for common medium Gemma projections.
        """
        limb = self._modular_limb12_function
        if limb is None or modulus > 0xFFFFFF:
            return self._modular_function
        batch = int(inputs.shape[0])
        out_features, in_features = map(int, weights.shape)
        if batch <= 2 and in_features >= 1024:
            return limb
        if batch >= 8 and (in_features >= 8192 or out_features >= 16384):
            return limb
        return self._modular_function

    def modular_limb12(self, weights: np.ndarray, inputs: np.ndarray, modulus: int) -> np.ndarray:
        """Force the exact 12-bit-limb kernel for experiments."""
        w = np.ascontiguousarray(weights, dtype=np.int8)
        x = np.ascontiguousarray(inputs, dtype=np.uint32)
        _validate_shapes(w, x)
        if self._modular_limb12_function is None:
            return self.modular(w, x, modulus)
        output = np.empty((x.shape[0], w.shape[0]), dtype=np.uint32)
        self._modular_limb12_function(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            x.shape[0], w.shape[0], w.shape[1], int(modulus), self.threads,
        )
        return output

    def wrap32(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Evaluate exactly in Z/(2^32), used as a fast online-ring control."""
        w = np.ascontiguousarray(weights, dtype=np.int8)
        x = np.ascontiguousarray(inputs, dtype=np.uint32)
        _validate_shapes(w, x)
        if self._wrap32_function is None:
            raw = x.astype(np.int64) @ w.astype(np.int64).T
            return raw.astype(np.uint32)
        output = np.empty((x.shape[0], w.shape[0]), dtype=np.uint32)
        self._wrap32_function(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            x.shape[0],
            w.shape[0],
            w.shape[1],
            self.threads,
        )
        return output

    def clear(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        w = np.ascontiguousarray(weights, dtype=np.int8)
        x = np.ascontiguousarray(inputs, dtype=np.int8)
        _validate_shapes(w, x)
        if self._clear_function is None:
            return x.astype(np.int32) @ w.astype(np.int32).T
        output = np.empty((x.shape[0], w.shape[0]), dtype=np.int32)
        self._clear_function(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            x.shape[0],
            w.shape[0],
            w.shape[1],
            self.threads,
        )
        return output

    def _load(self, path: Path) -> None:
        self.library_path = path.resolve()
        self.lib = ctypes.CDLL(str(self.library_path))
        modular = getattr(self.lib, "masked_gemm_i8_u32_mod", None)
        if modular is None:
            modular = getattr(self.lib, "masked_gemm_i8_mod_u32", None)
        if modular is None:
            raise NativeKernelError(f"{path} does not export a modular GEMM kernel")
        modular.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        modular.restype = None
        self._modular_function = modular

        limb12 = getattr(self.lib, "masked_gemm_i8_u32_mod_limb12", None)
        if limb12 is not None:
            limb12.argtypes = list(modular.argtypes)
            limb12.restype = None
        self._modular_limb12_function = limb12

        wrap32 = getattr(self.lib, "masked_gemm_i8_u32_wrap", None)
        if wrap32 is not None:
            wrap32.argtypes = [
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            wrap32.restype = None
        self._wrap32_function = wrap32

        clear = getattr(self.lib, "clear_gemm_i8_i8_i32", None)
        if clear is not None:
            clear.argtypes = [
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            clear.restype = None
        self._clear_function = clear


def _validate_shapes(weights: np.ndarray, inputs: np.ndarray) -> None:
    if weights.ndim != 2 or inputs.ndim != 2 or weights.shape[1] != inputs.shape[1]:
        raise NativeKernelError("expected weights [out,in] and inputs [batch,in]")


def build_native_library(output: str | Path | None = None) -> Path:
    package_dir = Path(__file__).resolve().parent
    root = package_dir.parent
    sources = (
        root / "native" / "masked_gemm.cpp",
        package_dir / "native_src" / "masked_gemm.cpp",
    )
    source = next((candidate for candidate in sources if candidate.exists()), None)
    if source is None:
        raise NativeKernelError(f"missing native source; searched {sources}")
    if output is None:
        import hashlib

        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        cache = Path(
            os.environ.get(
                "PLLM_NATIVE_CACHE",
                Path.home() / ".cache" / "pllm" / "native",
            )
        )
        output_path = cache / digest / _library_name()
    else:
        output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with MaskedGEMM._build_lock:
        if output_path.exists() and output_path.stat().st_mtime >= source.stat().st_mtime:
            return output_path
        compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise NativeKernelError("no C++ compiler found")
        command = [
            compiler,
            "-O3",
            "-march=native",
            "-mavx2",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(source),
            "-o",
            str(output_path),
        ]
        with_openmp = command[:1] + ["-fopenmp"] + command[1:]
        try:
            subprocess.run(with_openmp, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(command, check=True, capture_output=True)
    return output_path


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libhe_openai_masked.dylib"
    if os.name == "nt":
        return "he_openai_masked.dll"
    return "libhe_openai_masked.so"


def main() -> None:
    path = build_native_library()
    print(path)


if __name__ == "__main__":
    main()
