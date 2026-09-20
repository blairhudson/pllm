from __future__ import annotations

import secrets

import numpy as np
import pytest

from pllm import _native


def policy(max_attempts: int = 1_000) -> _native.FreivaldsPolicy:
    return _native.FreivaldsPolicy(
        40,
        max_attempts,
        1_000,
        10_000,
        10_000,
        10_000,
        1_000_000,
        64,
        secrets.token_bytes(32),
        True,
    )


def material(
    max_attempts: int = 1_000,
) -> tuple[
    _native.FreivaldsProjectionInventory,
    bytes,
    bytes,
    bytes,
    _native.FreivaldsPolicy,
]:
    weights = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.int8)
    root_seed = bytes([7]) * 32
    binding = b"model/body/stage/weight/shape/q10"
    material_id = bytes([9]) * 32
    execution_policy = policy(max_attempts)
    prepared = _native.prepare_freivalds(
        weights.tobytes(),
        2,
        3,
        4,
        root_seed,
        binding,
        material_id,
        10,
        1_000,
        execution_policy,
    )
    with pytest.raises(ValueError, match="preparation material cannot be claimed"):
        prepared.claim(root_seed, binding, 0, 1)
    imported = _native.import_freivalds(
        prepared.payload(),
        prepared.authentication_tag(root_seed),
        root_seed,
        prepared.inventory_rows,
        prepared.in_features,
        prepared.out_features,
        binding,
        material_id,
        10,
        1_000,
        prepared.max_row_l1,
        execution_policy,
    )
    return imported, root_seed, binding, material_id, execution_policy


def test_native_freivalds_verifies_runtime_oriented_batches_once() -> None:
    inventory, root_seed, binding, _, _ = material()
    verifier = inventory.claim(root_seed, binding, 1, 2)
    input_values = np.array([[2, -3, 5], [-1, 4, 2]], dtype=np.int32)
    output_values = np.array([[23, 53], [-3, -12]], dtype=np.int64)
    assert (
        np.frombuffer(
            verifier.verify(input_values.tobytes(), output_values.tobytes()),
            dtype="<i8",
        )
        .reshape(2, 2)
        .tolist()
        == output_values.tolist()
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        verifier.verify(input_values.tobytes(), output_values.tobytes())
    with pytest.raises(ValueError, match="already consumed"):
        inventory.claim(root_seed, binding, 1, 1)


def test_native_freivalds_rejects_corruption_binding_and_ranges() -> None:
    inventory, root_seed, binding, _, _ = material()
    with pytest.raises(ValueError, match="binding"):
        inventory.claim(root_seed, b"wrong", 0, 1)

    verifier = inventory.claim(root_seed, binding, 0, 1)
    input_values = np.array([2, -3, 5], dtype=np.int32)
    corrupted = np.array([23, 54], dtype=np.int64)
    with pytest.raises(ValueError, match="verification failed"):
        verifier.verify(input_values.tobytes(), corrupted.tobytes())

    inventory, root_seed, binding, _, _ = material()
    verifier = inventory.claim(root_seed, binding, 0, 1)
    with pytest.raises(ValueError, match="declared bound"):
        verifier.verify(
            np.array([20, 0, 0], dtype=np.int32).tobytes(),
            np.array([20, 80], dtype=np.int64).tobytes(),
        )


def test_native_freivalds_rejects_malformed_import_and_supports_cancel() -> None:
    inventory, root_seed, binding, material_id, execution_policy = material()
    inventory.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        inventory.claim(root_seed, binding, 0, 1)

    with pytest.raises(ValueError):
        _native.import_freivalds(
            b"\x00\x01",
            bytes(16),
            root_seed,
            1,
            3,
            2,
            binding,
            material_id,
            10,
            1_000,
            15,
            execution_policy,
        )


def test_native_freivalds_type_failure_burns_verifier() -> None:
    inventory, root_seed, binding, _, _ = material()
    verifier = inventory.claim(root_seed, binding, 0, 1)
    with pytest.raises(TypeError, match="input must be bytes"):
        verifier.verify("not-bytes", b"")
    assert verifier.cancel() is False


def test_native_freivalds_session_rejects_material_reimport() -> None:
    inventory, root_seed, binding, material_id, execution_policy = material()
    with pytest.raises(ValueError, match="already consumed"):
        _native.import_freivalds(
            inventory.payload(),
            inventory.authentication_tag(root_seed),
            root_seed,
            4,
            3,
            2,
            binding,
            material_id,
            10,
            1_000,
            15,
            execution_policy,
        )


def test_native_freivalds_policy_composes_declared_attempts() -> None:
    assert policy(1).checks == 2
    assert policy(1 << 22).checks == 2
    assert policy(1 << 23).checks == 3
    assert policy(1).conservative_failure_bits == 62
    with pytest.raises(ValueError):
        _native.FreivaldsPolicy(0, 1, 1, 1, 1, 1, 1, 1, bytes(32), True)
