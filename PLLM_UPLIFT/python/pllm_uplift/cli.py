from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .config import load_config, validate_config

def root_path(value: str | None) -> Path:
    root=Path(value).resolve() if value else Path.cwd()
    if not (root/"research/top20.json").is_file():raise ValueError("Run from the extracted package root or pass --root PATH.")
    return root

def emit(data: dict | list, output: str | None = None) -> None:
    text=json.dumps(data,indent=2,ensure_ascii=False)+"\n"
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    else:print(text,end="")

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="PLLM uplift staging tools; no production inference fallback")
    parser.add_argument("--root")
    sub=parser.add_subparsers(dest="command",required=True)
    research=sub.add_parser("research"); rs=research.add_subparsers(dest="action",required=True)
    rs.add_parser("list");show=rs.add_parser("show");show.add_argument("id")
    validate=sub.add_parser("validate");validate.add_argument("config")
    assure=sub.add_parser("assure");assure.add_argument("--output");assure.add_argument("--fixtures-only",action="store_true")
    bench=sub.add_parser("bench")
    for name,default in [("dim",64),("lanes",18),("repeats",7),("bits",24),("tile",32)]:bench.add_argument("--"+name,type=int,default=default)
    bench.add_argument("--output")
    a=parser.parse_args(argv)
    try:
        if a.command=="bench":
            from . import native_benchmark
            emit(native_benchmark(dim=a.dim,lanes=a.lanes,repeats=a.repeats,bits=a.bits,tile=a.tile),a.output)
        elif a.command=="validate": emit(validate_config(load_config(a.config)))
        elif a.command=="research":
            root=root_path(a.root);papers=json.loads((root/"research/top20.json").read_text())["papers"]
            if a.action=="list":emit([{k:p[k] for k in ["id","title","track","implementation_status"]} for p in papers])
            else:
                found=[p for p in papers if p["id"]==a.id]
                if not found:raise ValueError("Unknown research identifier")
                print((root/found[0]["card"]).read_text())
        elif a.command=="assure":
            from .attacks import run_attacks
            root=root_path(a.root)
            result={"schema_version":"pllm.assurance_campaign.v1","public_fixtures":run_attacks(),"production_runtime_attacked":False,"whole_protocol_privacy_proven":False}
            if not a.fixtures_only:
                from .formal import run_formal
                result["formal"]=run_formal(root)
            emit(result,a.output)
            if "formal" in result and not all(x["expectation_met"] for x in result["formal"]["checks"]):return 1
        return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print(f"error: {exc}",file=sys.stderr);return 2
