from copy import deepcopy
import ctypes.util
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
import jsonschema
from pllm_uplift.config import validate_config,ConfigurationError
from pllm_uplift.metrics import decode_tps,directed_bytes,sustainable_bound
from pllm_uplift.attacks import run_attacks
ROOT=Path(__file__).resolve().parents[1]
PAPERS=json.loads((ROOT/"research/top20.json").read_text())["papers"]

def config(name="single-evaluator-target"):
    return json.loads((ROOT/"configs"/(name+".json")).read_text())

@pytest.mark.parametrize("paper",PAPERS,ids=lambda p:p["id"])
def test_paper_card_and_scope(paper):
    text=(ROOT/paper["card"]).read_text()
    assert paper["primary_url"] in text
    assert "not a completed native reproduction" in text
    assert not paper["eligible_default"]
    assert paper["assurance_status"]=="not_reviewed"
    assert paper["source_lock"]["paper_sha256"] is None

@pytest.mark.parametrize("paper",PAPERS,ids=lambda p:p["id"])
def test_recipe_not_fake_run(paper):
    r=json.loads((ROOT/"research/recipes"/(paper["id"]+".json")).read_text())
    assert r["status"]=="not_executed"
    assert r["original_artifact_command"] is None
    assert r["pllm_reproduction_command"] is None
    assert r["gates"]==["acquire","specify","reference","native","assure","benchmark","document"]

@pytest.mark.parametrize("name",["operator-smoke","single-evaluator-target","qwen-archival-baseline","fss-comparison"])
def test_config_schema_and_contract(name):
    c=config(name)
    jsonschema.validate(c,json.loads((ROOT/"schemas/experiment.schema.json").read_text()))
    assert validate_config(c)["config_valid"]

@pytest.mark.parametrize("field,value",[("online_workers",2),("preparation_online",True),("client_model",True),("he",True)])
def test_no_silent_topology_change(field,value):
    c=config();c["policy"][field]=value
    with pytest.raises(ConfigurationError):validate_config(c)

@pytest.mark.parametrize("paper",["R07","R08","R13","R14","R19","R20"])
def test_forbidden_method_family(paper):
    c=config();c["implementation"]["method_ids"]=[paper]
    with pytest.raises(ConfigurationError):validate_config(c)

def test_target_config_not_executable():
    r=validate_config(config())
    assert not r["runnable_in_bundle_after_native_build"]
    assert any("E_COVERAGE" in e for e in r["execution_blockers"])
    assert not r["privacy_proven"]

def test_operator_can_build():
    assert validate_config(config("operator-smoke"))["runnable_in_bundle_after_native_build"]

def test_unknown_config_field_rejected():
    c=config();c["secret_override"]=True
    with pytest.raises(ConfigurationError):validate_config(c)

def test_bad_bool_not_integer():
    c=config();c["policy"]["online_workers"]=True
    with pytest.raises(ConfigurationError):validate_config(c)

@pytest.mark.parametrize("times,want",[([],None),([5],None),([0,0],None),([0,1000000000],1.0),([0,500000000,1000000000],2.0)])
def test_tps_defined_correctly(times,want):assert decode_tps(times)==want

def test_tps_rejects_time_reversal():
    with pytest.raises(ValueError):decode_tps([10,1])

def test_bytes_not_double_counted():
    event={"sender":"client","receiver":"inference","phase":"online","transfer_id":"1","bytes":32}
    assert directed_bytes([{**event,"observer":"client"},{**event,"observer":"inference"}])=={"client->inference/online":32}

def test_duplicate_sender_receipt_rejected():
    e={"sender":"client","receiver":"inference","phase":"online","transfer_id":"1","bytes":32,"observer":"client"}
    with pytest.raises(ValueError):directed_bytes([e,e])

def test_supply_bound():
    assert sustainable_bound(50,[{"production_per_second":1000,"consumption_per_useful_token":100}])==10

def test_unknown_cost_not_zero():
    schema=json.loads((ROOT/"schemas/measurement.schema.json").read_text())
    data={"schema_version":"pllm.measurement.v1","origin":"not_available","scope":"deployment","privacy_cohort":"test","numeric_cohort":"integer","value":None,"unit":"gpu_seconds","unavailable_reason":"no GPU"}
    jsonschema.validate(data,schema)

def test_security_true_is_not_evidence():
    schema=json.loads((ROOT/"schemas/assurance-result.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate({"secure":True},schema)

def test_scoped_result_accepted():
    d={"schema_version":"pllm.claim_result.v1","claim_id":"F01","outcome":"proved_in_model","scope":"scalar 8-bit ring","origin":"solver_executed","implementation_refinement":"not_established","evidence_paths":["formal/F01"]}
    jsonschema.validate(d,json.loads((ROOT/"schemas/assurance-result.schema.json").read_text()))

def test_attack_negative_controls():
    r=run_attacks();assert len(r["findings"])==6
    for f in r["findings"]:
        if "successes" in f:assert f["successes"]==f["cases"]
        else:assert f["mismatches"]==30720
    assert r["ideal_uniform_control"]["exact_equal_joint_views"]
    assert not r["production_runtime_attacked"]

def test_formal_scopes_and_expectations():
    if not ctypes.util.find_library("z3"):pytest.skip("native Z3 unavailable; not counted as formal success")
    from pllm_uplift.formal import run_formal
    result=run_formal(ROOT)
    assert len(result["checks"])==10
    assert all(c["expectation_met"] for c in result["checks"])
    assert sum(c["solver_result"]=="sat" for c in result["checks"])==3
    assert not result["rust_implementation_verified"]
    assert not result["whole_protocol_privacy_proven"]

def test_archival_integrity():
    m=json.loads((ROOT/"legacy/manifest.json").read_text())
    assert len(m["artifacts"])==10
    for a in m["artifacts"]:
        assert hashlib.sha256((ROOT/a["path"]).read_bytes()).hexdigest()==a["sha256"]
        assert a["validation"]=="archived_not_rerun_this_round"

def test_package_and_task_graph():
    spec=importlib.util.spec_from_file_location("check_package",ROOT/"tools/check_package.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    r=mod.check(ROOT);assert r["passed"];assert r["tasks"]==150

def test_repository_scan_does_not_read_secrets(tmp_path):
    spec=importlib.util.spec_from_file_location("inspect_repo",ROOT/"tools/inspect_repository.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    (tmp_path/"src").mkdir();(tmp_path/"src/main.rs").write_text("SECRET-CONTENT")
    (tmp_path/".env").write_text("secret");(tmp_path/"private_key.py").write_text("secret")
    r=mod.inspect(tmp_path);text=json.dumps(r)
    assert "main.rs" in text and "SECRET-CONTENT" not in text
    assert ".env" not in text and "private_key" not in text
    assert not r["contents_read"]

def test_cli_list():
    r=subprocess.run([sys.executable,"-m","pllm_uplift","research","list"],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert len(json.loads(r.stdout))==20
