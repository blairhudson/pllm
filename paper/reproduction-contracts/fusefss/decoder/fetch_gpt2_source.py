"""Fetch bounded inert GPT-2 metadata bytes only; never fetch model weights.

This is a source-evidence refresh, not part of offline oracle validation.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
REPOSITORY = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
OUT = HERE / "gpt2-source"
MAX_METADATA_BYTES = 256 << 10
# Closed allowlist: no weights, tokenizer vocabulary/merge payload, model header
# range request, pickle, author code, redirected arbitrary repository path.
FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
         "special_tokens_map.json", "model.safetensors.index.json")


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PLLM-inert-metadata-lock/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_METADATA_BYTES:
            raise ValueError("metadata response exceeds byte limit")
        data = response.read(MAX_METADATA_BYTES + 1)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("metadata response exceeds byte limit")
        return data, response.geturl(), response.status


def capture():
    OUT.mkdir(exist_ok=True)
    api_url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}"
    api, final_url, status = get(api_url)
    info = json.loads(api)
    if info.get("sha") != REVISION:
        raise ValueError("repository revision mismatch")
    (OUT / "repository-metadata.json").write_bytes(api)
    records = [{"path": "repository-metadata.json", "requested_url": api_url,
                "resolved_url": final_url, "http_status": status, "bytes": len(api),
                "sha256": sha256(api).hexdigest()}]
    available = {s["rfilename"] for s in info.get("siblings", [])}
    for name in FILES:
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        try:
            data, final_url, status = get(url)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            records.append({"path": name, "requested_url": url, "http_status": 404,
                            "listed_in_repository": name in available, "downloaded": False})
            continue
        json.loads(data)  # Parse only inert JSON; never execute source bytes.
        (OUT / name).write_bytes(data)
        records.append({"path": name, "requested_url": url, "resolved_url": final_url,
                        "http_status": status, "listed_in_repository": name in available,
                        "bytes": len(data), "sha256": sha256(data).hexdigest()})
    manifest = {"schema": "pllm.fusefss.gpt2-public-metadata-lock.v1",
                "repository": REPOSITORY, "revision": REVISION,
                "claim": "public-source-metadata-only-not-evaluated-build-or-complete-checkpoint-lock",
                "weights_downloaded": False, "source_code_executed": False,
                "max_response_bytes": MAX_METADATA_BYTES,
                "excluded": ["weights", "safetensors-payload-or-header", "tokenizer-vocabulary-and-merges", "author-code"],
                "files": records}
    (OUT / "source-lock.json").write_bytes(canonical(manifest))
    return manifest


def verify():
    manifest = json.loads((OUT / "source-lock.json").read_bytes())
    if manifest["revision"] != REVISION or manifest["repository"] != REPOSITORY:
        raise ValueError("source identity mismatch")
    for record in manifest["files"]:
        name = record["path"]
        if name not in FILES + ("repository-metadata.json",):
            raise ValueError("source file outside closed allowlist")
        if record["http_status"] == 404:
            if (OUT / name).exists():
                raise ValueError("unexpected bytes for absent metadata")
            continue
        data = (OUT / name).read_bytes()
        if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("source bytes/hash mismatch")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = capture() if args.fetch else verify()
    print(json.dumps({"revision": manifest["revision"], "files": len(manifest["files"]),
                      "weights_downloaded": False}, sort_keys=True))
