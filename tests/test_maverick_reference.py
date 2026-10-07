"""Retained source/shape evidence must not become an executable/privacy claim."""
import hashlib
import json
from pathlib import Path

from pllm.protocols import MaverickDelegatedLinear, ProtocolMethod


ROOT = Path(__file__).resolve().parents[1]


def test_retained_raa_reference_is_source_bound_and_not_a_decoder_claim(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.maverick_reference import CONFIGURATION

    evidence = json.loads((ROOT / 'docs/evidence/research-maverick-reference-qwen25.json').read_text())
    assert evidence['configuration'] == CONFIGURATION
    # Replay factories are current contracts; native/runtime source hashes are
    # historical provenance and may change without rewriting old observations.
    for path in ('benchmarks/research/maverick_reference.py',
                 'benchmarks/research/paper_baseline.py', 'benchmarks/research/common.py'):
        expected = evidence['source_sha256'][path]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert evidence['executable_sdk'] is False
    assert evidence['privacy_certified'] is False
    assert evidence['verification_soundness_certified'] is False
    gate = evidence['source_gate']
    assert CONFIGURATION['field_modulus'] < gate['appendix_b_cited_field_lower_bound_exclusive']
    assert gate['distance_failure_probability'] is None
    assert gate['babybear_meets_cited_field_range'] is False
    assert sum(case['compiled_stage_count'] for case in evidence['cases']) == 96
    for case in evidence['cases']:
        expected = 64 * case['rows'] * case['columns']
        assert case['privacy_matrix_bytes'] + case['verification_matrix_bytes'] == expected
        if case['status'] == 'reference_checks_passed':
            assert case['checked_outputs'] == case['samples'] * case['rows']
            assert case['forgeries_rejected'] == 3
            assert case['persistent_client_payload_bytes'] + case['max_live_query_payload_bytes'] <= CONFIGURATION['reference_payload_bound_bytes']
        else:
            assert case['status'] == 'rejected_before_preprocessing'
            assert expected > CONFIGURATION['reference_payload_bound_bytes']
            assert 'observations' not in case
    assert evidence['stages_rejected_before_preprocessing'] == 48
    # Historical SDK absence remains historical; the current bounded operator
    # is independently tested to reject executable decoder composition.
    assert isinstance(MaverickDelegatedLinear(), ProtocolMethod)


def test_sdk_costs_keep_source_shapes_and_security_storage_gates_separate():
    report = json.loads((ROOT / 'docs/evidence/research-coded-delegation-cost.json').read_text())
    for path in ('benchmarks/research/coded_delegation.py', 'benchmarks/research/paper_baseline.py',
                 'benchmarks/research/common.py'):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == report['source_sha256'][path]
    assert report['admitted_reference_stages'] == 48
    assert sum(c['compiled_stage_count'] for c in report['cases']) == 96
    assert report['checked_outputs'] == 50688
    assert report['fresh_processes'] == 9
    assert report['storage_lower_bound']['preprocessing_bytes_lower_bound'] == 22900899840
    assert report['scope']['privacy_certified'] is False
    assert report['scope']['distance_certified'] is False
    assert report['scope']['whole_model_executable'] is False
    assert report['scope']['checkpoint_quality'] is None
    for case in report['cases']:
        p = case['projection']
        assert p['field'] < 2**31
        assert p['privacy_matrix_bytes'] + p['verification_matrix_bytes'] == 64 * case['rows'] * case['columns']
        assert bool(case['reports']) == p['admitted']
        for r in case['reports']:
            assert r['resources']['public_table_bytes_by_party'][0] == p['persistent_client_payload_bytes']
            assert r['quality']['exact_integer_outputs'] == case['rows'] * case['queries']
            assert r['full_wire_bytes'] is None
