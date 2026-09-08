"""Exact integer GEMM on BFV coefficient arrays, using eight-bit limbs.

This is a backend kernel experiment, not a new wire format or a full HE backend.
For each public W4 matrix, it computes the same coefficient residues as SEAL's
ciphertext additions. It uses integer accumulation, never floating point.
"""
from __future__ import annotations
import numpy as np
import torch


def evaluate_coefficients(weight: np.ndarray, ciphertexts: np.ndarray, q: int, *, encrypted_zero: np.ndarray | None = None):
    if weight.ndim!=2 or ciphertexts.ndim!=2 or weight.shape[1]!=ciphertexts.shape[0]:
        raise ValueError('incompatible matrix dimensions')
    if weight.dtype!=np.int8 or ciphertexts.dtype!=np.uint64:
        raise TypeError('int8 weights and uint64 coefficients required')
    if not 2<=q<2**54 or np.any(ciphertexts>=q):
        raise ValueError('kernel supports canonical residues and q below 2**54')
    if np.any(weight < -7) or np.any(weight > 7):raise ValueError('W4 weights required')
    if weight.shape[1]*7*128>2**31-1:raise ValueError('int32 limb accumulator overflow')
    matrix=torch.from_numpy(np.ascontiguousarray(weight))
    correction=128*weight.astype(np.int64).sum(axis=1)[:,None]
    result=np.zeros((weight.shape[0],ciphertexts.shape[1]),dtype=np.int64)
    for shift in reversed(range(0,q.bit_length(),8)):
        # Convert unsigned digits to the signed range for exact int8 GEMM.
        digits=((ciphertexts>>shift)&255).astype(np.int16)
        signed=np.ascontiguousarray((digits-128).astype(np.int8))
        product=torch._int_mm(matrix,torch.from_numpy(signed)).numpy().astype(np.int64)
        # q < 2**54 ensures 256*result + product fits signed int64.
        result=(result*256+product+correction)%q
    result=result.astype(np.uint64)
    zero_rows=~np.any(weight,axis=1)
    if np.any(zero_rows):
        if encrypted_zero is None or encrypted_zero.shape!=(ciphertexts.shape[1],):
            raise ValueError('an encrypted zero is required for all-zero weight rows')
        if encrypted_zero.dtype!=np.uint64 or np.any(encrypted_zero>=q):
            raise ValueError('invalid encrypted-zero coefficient array')
        result[zero_rows]=encrypted_zero
    return result
