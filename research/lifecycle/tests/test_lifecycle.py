import json
from pathlib import Path


def test_executed_lifecycle_evidence():
    path=Path(__file__).resolve().parents[1]/'results/lifecycle.json'
    data=json.loads(path.read_text())
    assert data['exact_tokens'] and data['exact_final_logits']
    assert data['empty_initial_inventory'] and data['all_preparation_included']
    assert not data['server_process_has_he_secret']
    assert data['consumed_correlations']==data['audit']['used_correlations']
    assert data['consumed_correlations']>64*17
    assert 'scale' not in data['audit']['fields'] and 'token' not in data['audit']['fields']
    assert data['prepared_correlations']==data['consumed_correlations']+data['unused_correlations']
