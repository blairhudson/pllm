"""Bounded exact digit-lifted CPU GEMM research kernel.

This executor is benchmark-only. Live stage dispatch never selects it. Every
float32 product and possible partial sum is an integer strictly below 2**24;
the proof requires ordinary full-precision CPU SGEMM, not reduced-precision GPU
accumulation. Conversion and reconstruction are included in each operation.
"""
from __future__ import annotations

import numpy as np


class DigitGEMM:
    def __init__(self, weights, *, digit_bits=4):
        if (not isinstance(weights, np.ndarray) or weights.dtype != np.int8
                or weights.ndim != 2 or not weights.size or digit_bits not in (4, 8)):
            raise ValueError("digit GEMM requires a nonempty signed-i8 matrix and 4/8-bit digits")
        if weights.size > 128 << 20:
            raise ValueError("digit GEMM weight allocation exceeds 128 MiB")
        self.weights = weights.copy()
        self.weights.flags.writeable = False
        self.digit_bits = digit_bits
        self.block = ((1 << 24) - 1) // (128 * (1 << (digit_bits - 1)))

    def modular(self, values, modulus):
        if type(modulus) is not int or modulus not in (1 << 16, 1 << 24, 1 << 32):
            raise ValueError("digit GEMM supports exact u16/u24/u32 rings")
        bits = modulus.bit_length() - 1
        if (not isinstance(values, np.ndarray) or values.dtype != np.uint32
                or values.ndim != 2 or values.shape[1] != self.weights.shape[1]
                or not 1 <= values.shape[0] <= 4096
                or max(values.size, values.shape[0] * self.weights.shape[0]) > 4 << 20
                or np.any(values.astype(np.uint64) >= modulus)):
            raise ValueError("digit GEMM input exceeds its admitted ring or geometry")
        output = np.zeros((len(values), len(self.weights)), np.uint32)
        half, base = 1 << (self.digit_bits - 1), 1 << self.digit_bits
        for start in range(0, values.shape[1], self.block):
            end = min(start + self.block, values.shape[1])
            weights = self.weights[:, start:end].astype(np.float32)
            remaining = values[:, start:end].astype(np.int64)
            for shift in range(0, bits, self.digit_bits):
                digit = ((remaining + half) & (base - 1)) - half
                remaining = (remaining - digit) // base
                partial = digit.astype(np.float32) @ weights.T
                integer = partial.astype(np.int64).astype(np.uint32)
                np.add(output, integer << np.uint32(shift), out=output)
        if bits < 32:
            np.bitwise_and(output, np.uint32(modulus - 1), out=output)
        return output

    def wrap32(self, values):
        return self.modular(values, 1 << 32)

    def close(self):
        self.weights = None
