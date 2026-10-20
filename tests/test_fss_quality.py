"""Quality routing/outcome controls only: explicit source, native-error and RPC doubles.

Tokenizer runs on inert in-memory JSON. No source tensors, installed FSS services,
FP32 model, Q12 oracle, benchmarks, downloads or qualification evidence are used.
"""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from pllm import Model
from pllm.configuration import Deployment, ExecutionBudget, Experiment
from pllm.profiles import FuseFssSourceCpu, resolve_runtime_composition
from pllm.runtime import reference_benchmark as quality, quality_outcomes as outcomes
from test_quality_outcomes import quality_controls, experiment as existing_experiment


PROFILES = ("fusefss.gpt2-complete-ring50-q12.v1", "fusefss.dense-rope-ring50-q12.v1",
            "fusefss.dense-rope-checked-ranges-ring50-q12.v1")


def experiment(profile=PROFILES[0], name="fss-quality"):
    return Experiment(name, FuseFssSourceCpu(Model.path('/unit-only-source'), numeric_profile=profile),
        Deployment.local(root='/unit-only-roles'),
        ExecutionBudget(requests=8,max_input_tokens=8,max_new_tokens=3))


@pytest.fixture
def fss_controls(quality_controls,monkeypatch):
    from tokenizers import Tokenizer,models,pre_tokenizers
    from pllm.runtime import benchmark_memory,fss_source,fss_transport,servers
    from pllm.runtime.client import ProtocolError

    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'seed':1,'boom':2,'pass':3,'reject':4},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    raw=tokenizer.to_str().encode()
    calls,admissions,role_calls,references,phases,policies=[],[],[],[],[],[]
    state=SimpleNamespace(error=None,token=4,mutation=None,
        final_admission={"admitted": True, "full_topology_admitted": True})

    def source(experiment):
        admissions.append(experiment)
        resolved=quality_controls.source.source
        contract={'schema_version':'pllm.fss.sdk-contract.v1','weights':'public',
            'configuration':experiment.configuration_digest(),'composition':experiment.pipeline.digest(),
            'input_rows':experiment.budget.max_input_tokens,'output_tokens':experiment.budget.max_new_tokens,
            'source_lock':resolved.source_lock_digest,'tokenizer':'c'*64,
            'vocabulary':5,'numeric_profile':experiment.pipeline.components['linear'].params['numeric_profile'],
            'runtime_kind':'fss','client_runtime':'compiled_fss_v1','ring_bits':50,'fractional_bits':12}
        if experiment.pipeline.components['linear'].params.get('capacity') == 'decoder':
            contract['capacity']='decoder'
        # Deliberately NO legacy source.source or source.tokenizer() compatibility.
        owner=SimpleNamespace(path=resolved.path,source_lock_digest=resolved.source_lock_digest,
            checkpoint_digest=resolved.checkpoint_digest,tokenizer_digest='c'*64,contract=contract,
            tokenizer_bytes=raw,tokenizer_descriptor={'bos_token_id':0,'eos_token_id':1,'add_bos_token':False})
        if state.mutation:
            state.mutation(owner)
        return owner

    @contextmanager
    def roles(experiment, **policy):
        quality_controls.lifecycle.append('roles_start')
        role_calls.append(experiment)
        policies.append(policy)
        names=('worker_a','worker_b','dealer') if resolve_runtime_composition(experiment.pipeline).client_runtime=='compiled_fss_v1' else ('worker_a','worker_b','worker_c')
        try:
            yield SimpleNamespace(worker_connections=lambda:dict.fromkeys(names,('http://127.0.0.1:1','fake-key')),
                memory_admission=state.final_admission)
        finally:
            quality_controls.lifecycle.append('roles_close')

    def execute(experiment,source,ids,outputs,connections,timeout,cancelled):
        assert set(connections)=={'worker_a','worker_b','dealer'}
        assert source.contract['configuration']==experiment.configuration_digest()
        assert outputs==1 and 0<timeout<=600 and callable(cancelled)
        calls.append((tuple(ids),outputs,experiment,source.contract['numeric_profile']))
        if state.error is not None:
            raise state.error
        if ids[0]==4:
            raise quality_controls.rejection('SECRET_INTERNAL_FSS_RANGE_SITE')
        if ids[0]==2:
            raise ProtocolError('numeric_domain_rejected SECRET_GENERIC_HTTP400',400)
        return SimpleNamespace(selected=(state.token,))

    original_reference=quality._reference_trajectories
    def reference(*args):
        references.append(args[2:4])
        return original_reference(*args)

    def execution_admission(experiment, **kwargs):
        phases.append((experiment, kwargs))
        return {"admitted": True, "full_topology_admitted": False, "admission_scope": "source_qualification_only"}

    monkeypatch.setattr(benchmark_memory,'execution_memory_admission',execution_admission)
    monkeypatch.setattr(fss_source,'resolve_client_source',source)
    monkeypatch.setattr(fss_transport,'execute_request',execute)
    monkeypatch.setattr(servers,'build_roles',roles)
    monkeypatch.setattr(quality,'_reference_trajectories',reference)
    return SimpleNamespace(calls=calls,admissions=admissions,roles=role_calls,references=references,
        state=state,shared=quality_controls,source=source,phases=phases,policies=policies)


def run(experiments,prompts,**kwargs):
    return quality.run_reference_benchmark(experiments,prompts,outcomes='record',
        cohort_key=b'f'*32,cohort_key_id='fss-unit-controls',**kwargs)


@pytest.mark.parametrize('profile',PROFILES)
def test_final_ids_v1_uses_exact_fss_identity_and_unknown_metrics(fss_controls,profile):
    candidate=experiment(profile)
    report=quality.run_reference_benchmark([candidate],['pass'],decode_steps=1)
    row,=report['candidates']
    assert report['schema_version']==quality.SCHEMA
    assert row['numeric_profile']==profile and row['fraction_bits']==12 and row['ring_bits']==50
    assert row['execution']=='three-role-offline-tape-final-token-full-prefix'
    assert row['candidate_logits_available'] is False
    assert row['diagnostic_requests']==2 and row['output_selections_per_request']==1
    assert [c[0] for c in fss_controls.calls]==[(3,),(3,4)]
    assert all(c[2] is candidate for c in fss_controls.calls)  # no altered budgets/profile
    for section in (row,row['same_token_decode']):
        assert section['top1_agreement']==1
        assert section['top_k_recall'] is section['max_abs_logit_error'] is None
    assert any('independent integer/Q12 correctness is not source-model' in s for s in report['limitations'])


def test_record_range_rejection_keeps_all_positions_and_no_partial_ranking(fss_controls):
    report=run([experiment()],['pass','reject','pass'])
    row,=report['candidates']
    assert row['coverage']=={'total':3,'accepted':2,'rejected':1,'failed':0}
    assert row['accepted_agreement']['top1_agreement']==1 and row['top1_agreement'] is None
    assert row['outcomes'][1]['code']=='numeric_domain_rejected'
    assert report['terminal_state']=='incomplete'
    assert outcomes.read_report(json.dumps(report))==report
    with pytest.raises(ValueError,match='partial'):
        outcomes.require_complete_for_ranking(report)
    serialized=json.dumps(report)
    assert not any(secret in serialized for secret in ('SECRET','"pass"','"reject"','"selected"','"logits"','unit-only-source'))


def test_checked_decoder_quality_scores_only_accepted_original_positions(fss_controls, monkeypatch):
    candidate = experiment(PROFILES[2]).with_params(pipeline__linear__capacity="decoder")
    values, original = [], quality.measure_reference_agreement
    def metric(*args, **kwargs):
        values.append(kwargs["selected_token"])
        return original(*args, **kwargs)
    monkeypatch.setattr(quality, "measure_reference_agreement", metric)
    report = run([candidate], ["pass", "reject", "pass"], memory_safety_margin_percent=40)
    row, = report["candidates"]
    assert row["numeric_profile"] == PROFILES[2]
    assert row["outcomes"] == [
        {"position": 0, "status": "accepted", "code": None, "attempted": True},
        {"position": 1, "status": "rejected", "code": "numeric_domain_rejected", "attempted": True},
        {"position": 2, "status": "accepted", "code": None, "attempted": True}]
    assert row["sample_count"] == row["accepted_agreement"]["sample_count"] == 2
    assert row["accepted_agreement"]["top1_agreement"] == 1 and row["top1_agreement"] is None
    assert values and set(values) == {4}  # No zero placeholder reaches the scorer.
    assert fss_controls.phases == [(candidate, {"backend": "native", "memory_safety_margin_percent": 40})]
    assert fss_controls.policies == [{"memory_safety_margin_percent": 40}]
    assert "SECRET_INTERNAL_FSS_RANGE_SITE" not in json.dumps(report)
    outcomes.validate_report(report)


@pytest.mark.parametrize("final", [None, {"admitted": True, "full_topology_admitted": False}])
def test_checked_decoder_quality_cannot_execute_on_qualification_only_admission(fss_controls, final):
    fss_controls.state.final_admission = final
    candidate = experiment(PROFILES[2]).with_params(pipeline__linear__capacity="decoder")
    report = run([candidate], ["pass"])
    assert not fss_controls.calls and not fss_controls.references
    assert report["candidates"][0]["outcomes"] == [
        {"position": 0, "status": "failed", "code": "admission_failed", "attempted": False}]


@pytest.mark.parametrize("returned", [PROFILES[1], None])
def test_checked_quality_rejects_missing_or_static_native_identity(fss_controls, returned):
    fss_controls.state.mutation = (lambda source: source.contract.pop("numeric_profile")) if returned is None else (
        lambda source: source.contract.update(numeric_profile=returned))
    with pytest.raises(outcomes.QualityIntegrityError):
        run([experiment(PROFILES[2])], ["pass"])
    assert not fss_controls.calls and not fss_controls.roles


def test_checked_quality_requires_native_decoder_capacity(fss_controls):
    fss_controls.state.mutation = lambda source: source.contract.pop("capacity")
    candidate = experiment(PROFILES[2]).with_params(pipeline__linear__capacity="decoder")
    with pytest.raises(outcomes.QualityIntegrityError):
        run([candidate], ["pass"])
    assert not fss_controls.roles


def test_frozen_same_token_prefixes_survive_range_rejection(fss_controls):
    report=run([experiment()],['reject','pass'],decode_steps=1,continuations=[(3,),(1,)])
    assert [c[0] for c in fss_controls.calls]==[(4,),(4,3),(3,),(3,1)]
    assert report['candidates'][0]['all_positions']['coverage']=={'total':4,'accepted':2,'rejected':2,'failed':0}
    assert report['cohort']['same_token_decode']['teacher_forcing']=='caller-frozen'
    assert fss_controls.references==[(((4,),(3,)),[(3,),(1,)])]
    outcomes.validate_report(report)


@pytest.mark.parametrize('order',['fss-first','puma-first'])
def test_shared_cohort_preserves_distinct_fss_and_puma_identities(fss_controls,order):
    fss=experiment()
    puma=existing_experiment(name='puma-quality')
    candidates=[fss,puma] if order=='fss-first' else [puma,fss]
    report=run(candidates,['pass'],decode_steps=1,continuations=[(1,)])
    rows={r['name']:r for r in report['candidates']}
    assert (rows[fss.name]['ring_bits'],rows[fss.name]['fraction_bits'])==(50,12)
    assert (rows[puma.name]['ring_bits'],rows[puma.name]['fraction_bits'])==(64,18)
    assert [c[0] for c in fss_controls.calls]==[(3,),(3,1)]
    assert fss_controls.shared.calls==[(3,),(3,1)]
    assert all(r['top_k_recall'] is r['max_abs_logit_error'] is None for r in rows.values())
    outcomes.validate_report(report)


def test_generic_http400_range_text_is_failure_never_rejection(fss_controls):
    report=run([experiment()],['pass','boom','pass'])
    row,=report['candidates']
    assert row['coverage']=={'total':3,'accepted':1,'rejected':0,'failed':2}
    assert row['outcomes'][1]['code']=='transport_failed'
    assert row['outcomes'][2]['attempted'] is False
    assert report['reference_status']=='unavailable' and not fss_controls.references
    assert row['top1_agreement'] is None and row['candidate_logits_available'] is False
    assert 'SECRET' not in json.dumps(report)
    outcomes.validate_report(report)


def test_capacity_admission_failure_records_unattempted_failure(fss_controls,monkeypatch):
    from pllm.runtime import benchmark_memory
    def capacity(*args,**kwargs): raise benchmark_memory.BenchmarkMemoryError('SECRET_CAPACITY')
    monkeypatch.setattr(benchmark_memory,'benchmark_memory',capacity)
    report=run([experiment()],['pass'])
    row,=report['candidates']
    assert row['outcomes']==[{'position':0,'status':'failed','code':'admission_failed','attempted':False}]
    assert row['fraction_bits']==12 and row['ring_bits']==50
    assert not fss_controls.calls and not fss_controls.roles and not fss_controls.references
    outcomes.validate_report(report)


def test_default_v1_typed_rejection_still_aborts(fss_controls):
    with pytest.raises(fss_controls.shared.rejection):
        quality.run_reference_benchmark([experiment()],['pass','reject','pass'])
    assert [c[0] for c in fss_controls.calls]==[(3,),(4,)]


def test_integer_correctness_does_not_manufacture_source_model_agreement(fss_controls):
    fss_controls.state.token=3  # valid accepted ID, unlike independent reference argmax=4
    report=run([experiment()],['pass'])
    row,=report['candidates']
    assert row['top1_agreement']==0 and row['coverage']['accepted']==1
    assert row['top_k_recall'] is row['max_abs_logit_error'] is None
    assert fss_controls.references
    outcomes.validate_report(report)


def test_complete_final_id_report_does_not_bypass_existing_search_metric_scope(fss_controls):
    from pllm.search import QualityLockedNetworkSearch,SearchError
    report=run([experiment()],['pass'])
    assert report['terminal_state']=='complete'
    with pytest.raises(SearchError):
        QualityLockedNetworkSearch(0,0,1).select(report,[])


def test_wrong_output_is_failed_not_a_quality_sample(fss_controls):
    fss_controls.state.token=True
    report=run([experiment()],['pass','pass'])
    row,=report['candidates']
    assert row['coverage']=={'total':2,'accepted':0,'rejected':0,'failed':2}
    assert report['reference_status']=='unavailable'
    outcomes.validate_report(report)


def test_cancellation_retires_role_context_and_sanitizes_report(fss_controls):
    fss_controls.state.error=KeyboardInterrupt('SECRET_CANCELLED')
    with pytest.raises(outcomes.QualityRunCancelled) as cancelled:
        run([experiment()],['pass','pass'])
    report=cancelled.value.report
    assert report['terminal_state']=='cancelled'
    assert [r['attempted'] for r in report['candidates'][0]['outcomes']]==[True,False]
    assert fss_controls.shared.lifecycle==['guard_start','roles_start','roles_close','guard_close']
    assert 'SECRET' not in json.dumps(report)
    outcomes.validate_report(report)


def test_fenced_second_fss_candidate_keeps_own_profile(fss_controls):
    report=run([experiment(),experiment(PROFILES[1],name='fss-second')],['boom'])
    row=report['candidates'][1]
    assert row['numeric_profile']==PROFILES[1] and row['candidate_logits_available'] is False
    assert row['ring_bits']==50 and row['outcomes'][0]['attempted'] is False
    assert len(fss_controls.roles)==1
    outcomes.validate_report(report)


@pytest.mark.parametrize('change',['profile','fraction','saved_budget','source_lock','reference_lock'])
def test_source_and_numeric_binding_fail_before_roles(fss_controls,change,monkeypatch):
    if change=='reference_lock':
        resolved=fss_controls.shared.source.source
        replacement=SimpleNamespace(**vars(resolved)); replacement.checkpoint_digest='f'*64
        monkeypatch.setattr(quality,'resolve_model',lambda model:replacement)
    else:
        def mutation(source):
            if change=='profile': source.contract['numeric_profile']=PROFILES[1]
            elif change=='fraction': source.contract['fractional_bits']=18
            elif change=='saved_budget': source.contract['input_rows']=1
            else: source.source_lock_digest='f'*64
        fss_controls.state.mutation=mutation
    with pytest.raises(outcomes.QualityIntegrityError):
        run([experiment()],['pass'])
    assert not fss_controls.calls and not fss_controls.roles


def test_source_capacity_or_missing_symbols_never_falls_back_to_generic_loader(fss_controls,monkeypatch):
    from pllm.runtime import fss_source
    def unavailable(experiment): raise fss_source.FssSourceError('installed native core lacks admission')
    def bypass(model): pytest.fail('generic metadata loader bypassed missing native admission')
    monkeypatch.setattr(fss_source,'resolve_client_source',unavailable)
    monkeypatch.setattr(quality,'resolve_model',bypass)
    with pytest.raises(fss_source.FssSourceError):
        run([experiment()],['pass'])
    assert not fss_controls.calls and not fss_controls.roles


def test_record_mode_requires_typed_final_acceptance_hook(fss_controls,monkeypatch):
    monkeypatch.delattr(outcomes._native,'NumericDomainRejected')
    with pytest.raises(quality.ReferenceBenchmarkError,match='typed native'):
        run([experiment()],['pass'])
    assert not fss_controls.admissions and not fss_controls.roles


@pytest.mark.parametrize('case',['requests','prefix','teacher','session_ledger'])
def test_quality_request_and_prefix_budgets_precede_role_launch(fss_controls,case):
    candidate=experiment()
    if case=='requests':
        candidate=candidate.with_params(budget__requests=1)
        call=lambda:quality.run_reference_benchmark([candidate],['pass'],decode_steps=1)
    elif case=='prefix':
        candidate=candidate.with_params(budget__max_input_tokens=1)
        call=lambda:run([candidate],['pass'],decode_steps=1,continuations=[(1,)])
    elif case=='teacher':
        call=lambda:run([candidate],['pass'],decode_steps=1,continuations=[(5,)])
    else:
        candidate=candidate.with_params(budget__requests=128,budget__max_new_tokens=4)
        call=lambda:quality.run_reference_benchmark([candidate],['pass']*17,decode_steps=3)
    with pytest.raises(quality.ReferenceBenchmarkError): call()
    assert not fss_controls.calls and not fss_controls.roles
