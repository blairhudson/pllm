"""Independent bounded oracles for the source-only atlas screen."""
import hashlib
import json

import pytest

from scripts import probe_private_prefix_atlas as probe


@pytest.mark.parametrize("catalog", [1, 2, 3, 4, 7, 8])
def test_dense_uniform_single_server_privacy_and_exact_oracle(catalog):
    result = probe.toy_checks(catalog)
    assert result["exact_reconstructions"] == catalog * 2**catalog
    assert result["single_party_query_and_reply_view_histograms_equal"]


def test_colluding_query_pair_exposes_target():
    for target in range(5):
        for mask in range(32):
            assert mask ^ (mask ^ (1 << target)) == 1 << target


@pytest.mark.parametrize("size", [1, 65536, 65537, 2967627])
def test_padded_stripes_exact_reassembly(size):
    # No full checkpoint: largest fixture only one public synthetic capsule.
    data = (bytes(range(256)) * ((size + 255) // 256))[:size]
    layout = probe.layout(32, size)
    padded = data + bytes(layout["padded_capsule_bytes"] - size)
    stripes = [padded[i:i + 65536] for i in range(0, len(padded), 65536)]
    assert len(stripes) == layout["pages"]
    assert all(len(row) == 65536 for row in stripes)
    assert b"".join(stripes)[:size] == data


@pytest.mark.parametrize("params", [(True, 1), (0, 1), (1, 0), (8193, 65536), (1, 67108865)])
def test_native_and_capsule_policy_fail_before_allocations(params):
    with pytest.raises(ValueError):
        probe.layout(*params)


def test_complete_known_links_and_no_free_replica():
    row = probe.cost(32, 2967627, 16, 1, 16)
    assert row["layout"]["pages"] == 46
    assert row["layout"]["query_bytes_each"] == 167
    assert row["per_miss_bytes"] == 46 * (2 * 167 + 2 * (65536 + 65) + 4 * 64)
    assert row["dense_uniform_xor_alternative_per_miss_bytes"] == 46 * (
        2 * (65 + 4) + 2 * (65536 + 65) + 4 * 64)
    assert row["atlas_known_incremental_bytes"] == sum(row["links"].values())
    assert row["two_server_catalog_owned_bytes"] == 2 * 32 * 46 * 65536
    assert row["two_server_logical_scan_bytes"] == 16 * row["two_server_catalog_owned_bytes"]
    assert all(value is None for value in row["unknown"].values())
    assert row["existing_local_one_capsule_hit_incremental_bytes"] == 0


@pytest.mark.parametrize("catalog", [1, 2, 4, 32, 128])
def test_break_even_matches_independent_cohort_enumeration(catalog):
    threshold = probe.cold_client_threshold(catalog, 2967627)
    winners = [k for k in range(1, 101) if probe.cost(catalog, 2967627, k, 1, k)[
        "beats_local_full_catalog_known_bytes"]]
    if threshold is None:
        assert not winners
    else:
        assert threshold == winners[0]
        assert not probe.cost(catalog, 2967627, threshold - 1, 1, threshold - 1)[
            "beats_local_full_catalog_known_bytes"]


def test_long_horizon_full_catalog_is_stronger_than_download_every_time():
    row = probe.cost(32, 2967627, 16, 16, 256)
    assert row["atlas_known_incremental_bytes"] > row["local_full_catalog_cold_bytes"]
    assert row["atlas_known_incremental_bytes"] < row["download_all_each_request_bytes"]


def test_strict_horizon_limit_independent_enumeration():
    limit = probe.all_miss_horizon_limit(32, 2967627, 16)
    winners = [h for h in range(1, 100) if probe.cost(32, 2967627, 16, h, 16 * h)[
        "beats_local_full_catalog_known_bytes"]]
    assert limit == max(winners) == 13
    assert probe.all_miss_horizon_limit(1, 2967627, 16) == 0


def test_distribution_oracle_exhaustive_two_draws():
    probabilities = [0.75, 0.25]
    expected = sum(probabilities[a] * probabilities[b] *
                   (int(a == 0 or b == 0) + int(a == 1) + int(b == 1))
                   for a in range(2) for b in range(2))
    assert probe.expected_misses(probabilities, 2, 1) == expected
    assert probe.expected_misses([1.0], 99, 1) == 1
    assert probe.expected_misses([0.5, 0.5], 5, 0) == 5


@pytest.mark.parametrize("probabilities,horizon,capacity", [([], 1, 0), ([0.2], 1, 1),
                                                          ([1.0], 0, 1), ([1.0], 1, 2)])
def test_bad_distribution_rejects(probabilities, horizon, capacity):
    with pytest.raises(ValueError):
        probe.expected_misses(probabilities, horizon, capacity)


def test_replay_determinism_source_hashes_and_no_measurement_claims():
    report = probe.build_report()
    assert report == probe.build_report()
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((probe.ROOT / path).read_bytes()).hexdigest() == digest
    assert report["complete_cost_break_even"] is None
    assert "not DPF/AES" in report["toy"]["scope"]
    evidence = json.loads((probe.ROOT / "docs/evidence/research-private-prefix-atlas.json").read_text())
    assert evidence == report
