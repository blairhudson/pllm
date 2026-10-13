"""Isolated exact algebra/storage tests; no native SDK or security claims."""
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "low_correlation_streaming", ROOT/"scripts/probe_low_correlation_streaming.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("bits", [1, 8, 32, 64])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_external_matches_independent_evaluation_and_inverse(tmp_path, bits, n):
    ring = probe.GR(bits)
    values = [((i*127+3) % ring.mod, (i*17+1) % ring.mod) for i in range(3**n)]
    source, output = tmp_path/"input", tmp_path/"output"
    source.write_bytes(b"".join(probe.WORD.pack(*x) for x in values))
    expected = probe.direct_transform(values, ring, n)
    counts = probe.external_transform(source, output, tmp_path, ring, n)
    assert list(probe.WORD.iter_unpack(output.read_bytes())) == expected
    assert probe.full_transform(expected, ring, n, inverse=True) == values
    assert counts["stage_element_reads"] == n*3**n
    assert counts["stage_element_writes"] == n*3**n
    assert counts["final_copy_io_bytes"] == 32*3**n
    assert probe.ole_reconstruction_control(ring, n) == 2*3**n


def test_transform_and_trace_bijections_preserve_uniform_algebra_domain():
    ring = probe.GR(1)
    elements = list(itertools.product(range(2), repeat=2))
    images = {tuple(probe.full_transform(v, ring, 1))
              for v in itertools.product(elements, repeat=3)}
    assert len(images) == 64
    ring = probe.GR(3)
    bases = ring.roots()[1:]
    trace_images = {tuple(ring.trace(ring.mul(b, x)) for b in bases)
                    for x in itertools.product(range(8), repeat=2)}
    assert trace_images == set(itertools.product(range(8), repeat=2))
    # Incorrect characteristic-two shortcut must fail at k>1.
    assert ring.sigma((1, 1)) != ring.mul((1, 1), (1, 1))


@pytest.mark.parametrize("bits", [0, 65, True, 1.5])
def test_malformed_ring(bits):
    with pytest.raises(ValueError):
        probe.GR(bits)


def test_storage_limits_and_no_overwrite(tmp_path):
    source, output = tmp_path/"input", tmp_path/"output"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="length"):
        probe.external_transform(source, output, tmp_path, probe.GR(8), 1)
    source.write_bytes(probe.WORD.pack(256, 0)*3)
    with pytest.raises(ValueError, match="bounds"):
        probe.external_transform(source, output, tmp_path, probe.GR(8), 1)
    assert not output.exists()
    source.write_bytes(probe.WORD.pack(1, 0)*3)
    output.write_bytes(b"immutable")
    with pytest.raises(ValueError, match="exists"):
        probe.external_transform(source, output, tmp_path, probe.GR(8), 1)
    assert output.read_bytes() == b"immutable"
    with pytest.raises(ValueError):
        probe.external_transform(source, source, tmp_path, probe.GR(8), 1)
    with pytest.raises(ValueError):
        probe.external_transform(source, tmp_path/"next", tmp_path, probe.GR(8), 6)


def test_resource_model_exact_bytes_and_window_sensitivity():
    low = probe.resource_model()
    medium = probe.resource_model(window_mib=64)
    size = 3**15
    assert low["parameters"]["c"] == 5
    assert low["parameters"]["t"] == 27
    assert low["gr_vector_bytes"] == 16*size
    assert low["secret_disk_payload_peak_bytes"] == 48*size
    assert low["public_precomputed_transform_disk_bytes"] == 64*size
    assert low["full_x_z_output_bytes"] == 32*size
    assert low["cross_term_transform_stage_io_bytes"] == 50*30*16*size
    assert low["cross_term_transformed_scan_read_bytes"] == 50*16*size == 11479125600
    assert low["final_z_and_output_consumption_io_bytes"] is None
    assert low["ledger_owner"] == "one party/client expansion; not combined two-party expansion"
    assert low["cross_term_transform_gr_multiplications"] == 50*45*size
    assert low["global_expansion_unrequested_scalar_oles"] == 2*size-1
    for key in ("secret_disk_payload_peak_bytes", "cross_term_transform_stage_io_bytes",
                "cross_term_transform_gr_multiplications"):
        assert low[key] == medium[key]
    assert medium["window_scalar_pairs"] == 8*low["window_scalar_pairs"]
    assert low["native_process_peak_rss_bytes"] is None
    assert low["encrypted_spill_aead_metadata_and_io_bytes"] is None
    assert low["complete_pcg_operation_count"] is None
    pages = (16*size+4095)//4096
    assert low["hypothetical_secret_padded_authenticated_pages_bytes"] == 3*pages*4124
    assert low["hypothetical_uncached_cross_term_transform_page_io_bytes"] == 50*30*size*4124
    assert low["table3_rounded_seed_log2_bits"] == "26.60"
    assert probe.resource_model(n=16)["table3_rounded_seed_log2_bits"] == "26.70"


def test_malformed_transform_and_canonical_coefficients():
    ring = probe.GR(8)
    for values in ([], [(0, 0)]*2, [(True, 0)]*3, [(0, -1)]*3, [[0, 0]]*3):
        with pytest.raises(ValueError):
            probe.full_transform(values, ring, 1)


@pytest.mark.parametrize("kwargs", [{"n": 14}, {"n": 17}, {"window_mib": 0},
                                   {"demand": 0}, {"demand": 2*3**15+1}])
def test_resource_model_rejects_unlisted_security_or_capacity(kwargs):
    with pytest.raises(ValueError):
        probe.resource_model(**kwargs)


def test_source_locks_detect_changes(tmp_path):
    path = next(iter(probe.LOCKS))
    target = tmp_path/path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong revision")
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.source_locks(tmp_path)


def test_locked_evidence_replays(tmp_path):
    recorded = json.loads((ROOT/"docs/evidence/research-low-correlation-streaming.json").read_text())
    assert probe.evidence(tmp_path) == recorded
    assert recorded["cryptographic_implementation"] is False
    assert recorded["dory_reference"]["whole_client_qualification"] is False
