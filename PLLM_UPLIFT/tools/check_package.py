#!/usr/bin/env python3
"""Validate package metadata and preserved artifacts without executing old archives."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from pllm_uplift.config import validate_config

def check(root:Path=ROOT)->dict:
    errors=[];papers=json.loads((root/"research/top20.json").read_text())["papers"]
    ids={p["id"] for p in papers}
    if len(papers)!=20 or len(ids)!=20:errors.append("portfolio must have exactly twenty unique papers")
    for p in papers:
        if not (root/p["card"]).is_file():errors.append("missing card "+p["id"])
        if not (root/"research/recipes"/(p["id"]+".json")).is_file():errors.append("missing recipe "+p["id"])
        if any(x not in ids for x in p["depends_on"]):errors.append("unknown paper dependency "+p["id"])
        if p["eligible_default"]:errors.append("unreviewed portfolio method marked default "+p["id"])
    artifacts=json.loads((root/"legacy/manifest.json").read_text())["artifacts"]
    for a in artifacts:
        path=root/a["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=a["sha256"]:errors.append("archive mismatch "+a["path"])
    configs=[]
    for p in sorted((root/"configs").glob("*.json")):
        c=json.loads(p.read_text())
        if c.get("schema_version")=="pllm.experiment.v2":
            try:configs.append({"path":str(p.relative_to(root)),**validate_config(c)})
            except ValueError as e:errors.append(str(e))
    tasks=json.loads((root/"backlog/tasks.json").read_text())["tasks"]
    graph={t["id"]:t["depends_on"] for t in tasks};done=set();stack=set()
    def visit(n):
        if n in done:return
        if n in stack:raise ValueError("task dependency cycle at "+n)
        if n not in graph:raise ValueError("missing task dependency "+n)
        stack.add(n)
        for d in graph[n]:visit(d)
        stack.remove(n);done.add(n)
    try:
        for n in graph:visit(n)
    except ValueError as e:errors.append(str(e))
    return {"schema_version":"pllm.package_check.v1","papers":len(papers),"archived_artifacts":len(artifacts),"tasks":len(tasks),"configs":configs,"errors":errors,"passed":not errors,"native_compilation_checked":False}
if __name__=="__main__":
    report=check();print(json.dumps(report,indent=2));raise SystemExit(0 if report["passed"] else 1)
