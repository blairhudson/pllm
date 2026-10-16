"""Replay Paper 1's exact Listing 1 through installed SDK and local HTTP gateway.

Uses an already cached checkpoint, conservative native admission and the existing
host-pressure watchdog. This is functionality evidence, not a timing/RSS cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Explain why the sky is blue in one sentence."


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    (output / "driver.py").write_bytes(script.read_bytes())
    record = {"schema": "pllm.paper_listing_replay.v1", "status": "incomplete",
              "scope": "Cached public checkpoint functionality; no performance or memory-tier claim",
              "public_prompt": PROMPT, "argv": sys.argv, "driver_sha256": digest(script)}
    watchdog = None
    try:
        import pllm
        from pllm._benchmark_build import inspect
        from pllm.model_loader import resolve_model
        from pllm.runtime.benchmark_memory import MemoryWatchdog, benchmark_memory, require_admission
        from pllm.runtime.servers import build_roles
        from starlette.testclient import TestClient

        if Path(pllm.__file__).resolve().parent == ROOT / "python/pllm":
            raise RuntimeError("Use installed wheel with Python -I")
        text = (ROOT / "paper/manuscript.md").read_text()
        listings = re.findall(r"```\s*python\n(.*?)\n```", text, re.S)
        matches = [code for code in listings if '"prepared-qwen"' in code]
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one published prepared-qwen listing")
        listing = matches[0] + "\n"
        (output / "listing.py").write_text(listing)
        namespace = {}
        exec(compile(listing, "paper/manuscript.md:Listing1", "exec"), namespace)
        experiment = namespace["experiment"]
        source = resolve_model(experiment.pipeline.model)
        write(output / "experiment.json", experiment.to_spec())
        write(output / "build.json", inspect())
        record.update({"listing_sha256": digest(output / "listing.py"),
                       "manuscript_sha256_at_execution": digest(ROOT / "paper/manuscript.md"),
                       "source_lock_digest": source.source_lock_digest,
                       "checkpoint_digest": source.checkpoint_digest,
                       "installed_package": str(Path(pllm.__file__).resolve())})
        admission = benchmark_memory(experiment, backend="native")
        write(output / "admission.json", admission)
        require_admission(admission)
        if args.preflight_only:
            record["status"] = "admitted_not_executed"
            return
        roles = build_roles(experiment, log_dir=output / "role-logs")
        watchdog = MemoryWatchdog(admission, lambda _reason: roles.close())
        watchdog.start()
        with roles:
            with roles.client(bundle_cache_dir=output / "client-cache") as client:
                response = client.responses.create(input=PROMPT, max_output_tokens=8, temperature=0)
                audit = client.privacy_audit.to_dict()
                if (audit["online_steps"] <= 0 or audit["plaintext_prompt_bytes_sent"] != 0
                        or audit["plaintext_token_ids_sent"] != 0):
                    raise RuntimeError("SDK protected-transport functionality check failed")
                sdk_text = response.output_text
                if response.usage is None:
                    raise RuntimeError("SDK usage missing")
                record["sdk_usage"] = response.usage.to_dict()
            with TestClient(roles.gateway_app(local_api_key="public-paper-replay")) as gateway:
                reply = gateway.post("/v1/responses", headers={"authorization": "Bearer public-paper-replay"},
                    json={"model": response.model, "input": PROMPT,
                          "max_output_tokens": 8, "temperature": 0})
                if reply.status_code != 200:
                    raise RuntimeError(f"Gateway status {reply.status_code}")
                body = reply.json()
                gateway_text = "".join(content.get("text", "") for item in body["output"]
                                       for content in item.get("content", []))
                if gateway_text != sdk_text or body["usage"]["output_tokens"] <= 0:
                    raise RuntimeError("SDK/gateway output disagreement")
        watchdog.check()
        if watchdog.error or watchdog.maximum_swap_growth_bytes:
            raise RuntimeError("Host-pressure or swap-growth gate failed")
        record.update({"status": "completed", "model": response.model, "sdk_audit": audit,
                       "gateway_status": reply.status_code, "gateway_usage": body["usage"],
                       "sdk_gateway_text_equal": True,
                       "output_text_sha256": hashlib.sha256(sdk_text.encode()).hexdigest(),
                       "transport": "In-process ASGI HTTP gateway; separate local provider children over TCP"})
    except BaseException as error:
        record.update({"status": "rejected", "failure_type": type(error).__name__})
        raise
    finally:
        if watchdog is not None:
            watchdog.close()
            record["maximum_observed_swap_growth_bytes"] = watchdog.maximum_swap_growth_bytes
        record["files"] = {p.relative_to(output).as_posix(): digest(p)
                           for p in sorted(output.glob("*")) if p.is_file() and p.name != "result.json"}
        write(output / "result.json", record)


if __name__ == "__main__":
    main()
