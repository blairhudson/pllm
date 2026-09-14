#!/usr/bin/env python3
"""Read-only, metadata-only inventory. No source contents, credentials or Git history."""
import argparse
import hashlib
import json
import os
from pathlib import Path

def inspect(root:Path)->dict:
    root=root.resolve()
    if not root.is_dir():raise ValueError("repository directory does not exist")
    skip={".git",".venv","venv","target","node_modules",".cache","__pycache__","runs","secrets","credentials"}
    names={"Cargo.toml","Cargo.lock","pyproject.toml","uv.lock","requirements.txt","README.md"}
    files=[];counts={};limit=10000;truncated=False
    for base,dirs,fs in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in skip and not (Path(base)/d).is_symlink() and not d.startswith("."))
        for name in sorted(fs):
            path=Path(base)/name
            if path.is_symlink() or name.startswith(".") or any(x in name.lower() for x in ["secret","credential","token","password","private_key"]):continue
            if path.suffix not in {".rs",".py",".toml"} and name not in names:continue
            rel=str(path.relative_to(root));counts[path.suffix]=counts.get(path.suffix,0)+1
            files.append({"path":rel,"bytes":path.stat().st_size,"kind":"dependency_manifest" if name in names else "source_metadata_only"})
            if len(files)>=limit:truncated=True;break
        if truncated:break
    return {"schema_version":"pllm.repository_inventory.v1","read_only":True,"contents_read":False,"repository_identity":"user_supplied_local_path_not_published","counts":counts,"files":files,"truncated":truncated,"missing_questions":["Which revision is currently deployed?","Which baseline numeric manifest is authoritative?","Which APIs/aliases are public?","Which current native/crypto dependencies are reviewed?"]}

def main():
    p=argparse.ArgumentParser();p.add_argument("repository",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=inspect(a.repository);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
