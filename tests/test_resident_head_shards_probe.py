"""Installed-native controls for the investigation-only shard owner."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import weakref

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/probe_resident_head_shards.py"
SPEC = importlib.util.spec_from_file_location("resident_head_shards_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("choice", ["paged", "partial", "full"])
@pytest.mark.parametrize("compression", ["raw", "zlib"])
def test_complete_native_outputs_and_lookup(choice, compression):
    row = probe.run_case(choice, compression=compression)
    assert row["checked_full_output_elements"] == 201
    expected = {"paged": 0, "partial": 1056, "full": 2211}[choice]
    assert row["ownership"]["resident_i8_payload_bytes"] == expected


def make_source(tmp_path):
    from pllm.native import PagedGEMM
    weights = np.arange(51, dtype=np.int8).reshape(17, 3)
    path = tmp_path / "source.raw"
    path.write_bytes(weights.tobytes())
    return PagedGEMM.from_raw(path, probe.sha(weights.tobytes()), weights.shape, threads=1), weights


def test_caps_range_immutable_eviction_and_close(tmp_path):
    source, weights = make_source(tmp_path)
    owner = probe.ResidentHead(source, tmp_path, shard_rows=4, cap_bytes=13, workspace_bytes=48)
    kernel_ref = weakref.ref(owner._kernel)
    source.close()
    try:
        assert owner.resident_bytes == 12
        owner.promote(0)  # Idempotent, no second resident owner.
        with pytest.raises(ValueError, match="cap"):
            owner.promote(1)
        with pytest.raises(ValueError, match="range"):
            owner.evict(-1)
        with pytest.raises(ValueError):
            owner.promote(True)
        for ids in ([-1], [17], [], [True], [0.5]):
            with pytest.raises(ValueError):
                owner.gather_rows(ids)
        for ids in ([-1], [3], [], [0.5]):
            with pytest.raises(ValueError):
                owner.gather_columns(ids)
        view = owner._resident[0].weight_view()
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view.setflags(write=True)
        del view  # Escaped views would retain owners; public probe API returns copies only.
        output = owner.gather_rows([0])
        output[:] = 0
        np.testing.assert_array_equal(owner.gather_rows([0]), weights[:1])
        owner.evict(0)
        assert owner.resident_bytes == 0
        owner.promote(1)
        assert owner.resident_bytes == 12
        with pytest.raises(ValueError):
            owner.execute(np.ones((1, 2), np.int8))
        with pytest.raises(ValueError):
            owner.execute(np.ones((1, 3), np.int8), "modular", True)
        np.testing.assert_array_equal(owner.execute(np.ones((1, 3), np.int8)),
                                      weights.astype(np.int32).sum(1)[None, :])
    finally:
        fallbacks = [s.paged for s in owner._shards]
        owner.close()
        owner.close()
    assert owner.resident_bytes == 0
    assert owner._shards == ()
    assert owner._kernel is None
    assert kernel_ref() is None
    for action in (lambda: owner.gather_rows([0]), lambda: owner.promote(0),
                   lambda: owner.execute(np.ones((1, 3), np.int8)), lambda: owner.ownership):
        with pytest.raises(ValueError, match="closed"):
            action()
    for matrix in fallbacks:
        with pytest.raises(ValueError, match="closed"):
            matrix.gather_rows([0])


@pytest.mark.parametrize("kwargs", [
    {"cap_bytes": -1}, {"shard_rows": 0}, {"workspace_bytes": 0},
    {"selected": [0, 0]}, {"selected": [-1]}, {"selected": [5]},
    {"cap_bytes": True}, {"compression": "wrong"},
])
def test_reject_before_export(tmp_path, kwargs):
    source, _ = make_source(tmp_path)
    try:
        with pytest.raises(ValueError):
            probe.ResidentHead(source, tmp_path, **kwargs)
        assert list(tmp_path.glob("shard-*")) == []
    finally:
        source.close()


def test_actual_byte_authentication_and_private_source_isolation(tmp_path):
    from pllm.native import PagedGEMM
    weights = np.ones((16, 3), np.int8)
    path = tmp_path / "compressed.paged"
    digest = PagedGEMM.export(weights, path, page_rows=16, compression="zlib")
    matrix = PagedGEMM(path, digest, threads=1)
    try:
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with pytest.raises(ValueError):
            PagedGEMM(path, digest, threads=1)
        # Even a new artifact hash cannot bless invalid decoded numeric identity.
        with pytest.raises(ValueError):
            PagedGEMM(path, hashlib.sha256(data).hexdigest(), threads=1)
        np.testing.assert_array_equal(matrix.gather_rows([0, 15]), weights[[0, 15]])
    finally:
        matrix.close()


def test_failure_retires_all_constructed_snapshots(tmp_path, monkeypatch):
    from pllm.native import PagedGEMM
    source, _ = make_source(tmp_path)
    closed = []
    failed_owners = []
    kernel_refs = []
    original = PagedGEMM.close

    def close(matrix):
        closed.append(matrix)
        original(matrix)

    def fail_promotion(self, index):
        failed_owners.append(self)
        kernel_refs.append(weakref.ref(self._kernel))
        raise RuntimeError("injected promotion failure")

    monkeypatch.setattr(PagedGEMM, "close", close)
    monkeypatch.setattr(probe.ResidentHead, "promote", fail_promotion)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            probe.ResidentHead(source, tmp_path, shard_rows=4, cap_bytes=12, workspace_bytes=48)
        assert len(closed) == 5
        assert failed_owners[0]._kernel is None
        assert failed_owners[0]._shards == ()
        assert failed_owners[0]._resident == {}
        assert kernel_refs[0]() is None
        for matrix in closed:
            with pytest.raises(ValueError, match="closed"):
                matrix.gather_rows([0])
        # Borrowed original remains usable after candidate failure.
        assert source.shape == (17, 3)
    finally:
        source.close()


@pytest.mark.parametrize("rows", [probe.MAX_SHARD_OWNERS + 1, 8 << 20])
def test_shard_owner_preflight_before_native_or_source_allocation(tmp_path, monkeypatch, rows):
    from pllm.native import MaskedGEMM, PagedGEMM

    class GeometryOnlySource:
        shape = (rows, 1)  # Within byte bound, far beyond permitted owner count.

        @property
        def weight_digest(self):
            pytest.fail("digest access must follow owner-count refusal")

        def gather_rows(self, ids):
            pytest.fail("gather allocation before owner-count refusal")

    def forbidden(*args, **kwargs):
        pytest.fail("native executor/export allocation before owner-count refusal")

    monkeypatch.setattr(MaskedGEMM, "__init__", forbidden)
    monkeypatch.setattr(PagedGEMM, "export", forbidden)
    monkeypatch.setattr(PagedGEMM, "__init__", forbidden)
    with pytest.raises(ValueError, match="shard owner count"):
        probe.ResidentHead(GeometryOnlySource(), tmp_path, shard_rows=1, workspace_bytes=4)
    assert list(tmp_path.iterdir()) == []


def test_unguarded_fresh_process_path_removed(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(PATH), "--fresh-process"])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2
