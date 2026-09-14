"""Lightweight DX checks. Rust must independently revalidate at the trust boundary."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

class ConfigurationError(ValueError):
    pass

def load_config(path: str | Path) -> dict:
    data=json.loads(Path(path).read_text())
    validate_config(data)
    return data

def validate_config(data: dict) -> dict:
    def fail(message): raise ConfigurationError(message)
    allowed={"schema_version","id","profile","policy","workload","implementation","deployment"}
    if set(data)-allowed: fail("E_CONFIG: unknown top-level fields")
    for field in ["schema_version","id","profile","policy","workload","implementation"]:
        if field not in data: fail(f"E_CONFIG: missing {field}")
    if data["schema_version"]!="pllm.experiment.v2":fail("E_SCHEMA: unsupported version")
    profiles={"operator_smoke","archive_masked_linear","target_single_evaluator","compare_two_online_fss","compare_he"}
    if data["profile"] not in profiles:fail("E_CONFIG: unknown profile")
    p=data["policy"]
    keys={"online_workers","preparation_online","client_model","he","experimental","require_composition_review"}
    if set(p)!=keys:fail("E_CONFIG: incomplete/unknown privacy policy")
    if type(p["online_workers"]) is not int or not 1<=p["online_workers"]<=4:fail("E_PARTIES: invalid online worker count")
    if any(type(p[k]) is not bool for k in keys-{"online_workers"}):fail("E_CONFIG: policy flags must be boolean")
    if data["profile"]=="target_single_evaluator":
        if p["online_workers"]!=1:fail("E_PARTIES: target requires exactly one online worker")
        if p["preparation_online"]:fail("E_ONLINE_PREPARATION: target requires offline preparation")
        if p["client_model"]:fail("E_CLIENT_MODEL: target excludes client weights")
        if p["he"]:fail("E_HE: target excludes homomorphic encryption")
    if data["profile"]=="compare_two_online_fss" and p["online_workers"]!=2:fail("E_PARTIES: comparison requires two online workers")
    w=data["workload"]; wk={"kind","model_id","revision","numeric_manifest","prompt_tokens","generation_cap","dim","lanes","bits","tile","repetitions"}
    if set(w)-wk or w.get("kind") not in {"public_kernel","full_model"}:fail("E_WORKLOAD: invalid fields/kind")
    for k,lo,hi in [("dim",1,4096),("lanes",1,128),("bits",1,32),("tile",1,2**31),("repetitions",3,1000),("prompt_tokens",1,2**31),("generation_cap",1,2**31)]:
        if k in w and (type(w[k]) is not int or not lo<=w[k]<=hi):fail(f"E_WORKLOAD: invalid {k}")
    i=data["implementation"]
    if set(i)!={"native_method","method_ids"}:fail("E_IMPLEMENTATION: invalid implementation fields")
    if i["native_method"] not in {"public_modular_matrix","not_installed"}:fail("E_IMPLEMENTATION: unsupported native method")
    if not isinstance(i["method_ids"],list) or any(x not in {f"R{j:02}" for j in range(1,21)} for x in i["method_ids"]) or len(i["method_ids"])!=len(set(i["method_ids"])):fail("E_METHOD: invalid method identifiers")
    if data["profile"]=="target_single_evaluator":
        selected=set(i["method_ids"])
        if selected & {"R13","R14","R19"}:fail("E_PARTIES: selected source protocol requires two online workers")
        if selected & {"R07"}:fail("E_CLIENT_MODEL: source adaptation is the separate client-heavy baseline")
        if selected & {"R08"}:fail("E_ASSURANCE: Carnival variant is quarantined for attack review")
        if selected & {"R20"}:fail("E_HE: source ciphertext-compression protocol is a separate comparison; masked-ring adaptation needs its own descriptor")
    if "deployment" in data:
        d=data["deployment"]
        if set(d)-{"launcher","party_manifest","disconnect_preparation","max_prepared_bytes"}:fail("E_DEPLOYMENT: unknown field")
        if d.get("launcher") not in {"inprocess_fixture","local_processes","remote_agents"}:fail("E_DEPLOYMENT: invalid launcher")
        if data["profile"]=="target_single_evaluator" and d.get("disconnect_preparation") is not True:fail("E_ONLINE_PREPARATION: target requires disconnect test")
    reasons=[]
    if w["kind"]=="full_model":
        if not w.get("revision"): reasons.append("E_MODEL_LOCK: checkpoint revision must be locked")
        if not w.get("numeric_manifest"): reasons.append("E_NUMERIC_MANIFEST: frozen numerical graph is missing")
        reasons.append("E_COVERAGE: full-model runtime is not implemented by this staging bundle")
    elif i["native_method"]!="public_modular_matrix":reasons.append("E_METHOD_NOT_INSTALLED")
    if i["method_ids"]:reasons.append("E_METHOD_NOT_INSTALLED: top20 are reproduction targets, not installed protocol implementations")
    return {"config_valid":True,"runnable_in_bundle_after_native_build":not reasons,"execution_blockers":reasons,"privacy_proven":False}
