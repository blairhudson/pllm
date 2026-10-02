"""Finite, shaped real-HTTP placement fixture for ordinary gateway/benchmark CLI.

Run: .venv/bin/python examples/benchmarks/network_scenarios.py --output docs/evidence/network-planner-benchmark-2026-10-01.json
Generated checkpoint/config files stay in an automatically removed directory.
Two HTTP provider apps each mount two installed role instances; four loopback
proxies apply actual byte-rate delays. This is controlled simulation, not WAN or
independent-operator evidence. No plaintext requests leave trusted client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.compiler import plan
from pllm.deployment import LinkObservation, NetworkSnapshot, NetworkSpec, PartyOffer, PartySpec, PartyTrust, discover
from pllm.model_loader import resolve_model
from pllm.modeling import lower_model
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.runtime.party import create_party_app
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.search import PlanningPolicy, PlanningRequest


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def running_http(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 60
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("fixture HTTP app failed to start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("fixture HTTP thread did not stop")


class ShapedProxy:
    """Count lengths only; never retain request/response bodies or credentials."""

    def __init__(self, upstream, upload_rate, download_rate):
        self.upstream, self.upload_rate, self.download_rate = upstream, upload_rate, download_rate
        self.rows = {}
        self.lock = threading.Lock()

        @asynccontextmanager
        async def lifespan(app):
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                app.state.http = client
                yield

        self.app = FastAPI(lifespan=lifespan)

        @self.app.api_route("/{path:path}", methods=["GET", "POST"])
        async def forward(path: str, request: Request):
            body = await request.body()
            role_call = path.startswith("roles/")
            category = ("online" if "/stages/" in path else
                        path.rsplit("/", 1)[-1] if path.startswith("v1/party/") else
                        "role-other" if role_call else "other")
            upload_delay = len(body) / self.upload_rate if role_call else 0
            sleep_started = time.monotonic()
            await asyncio.sleep(upload_delay)
            observed_upload = time.monotonic() - sleep_started if role_call else 0
            response = await request.app.state.http.request(
                request.method, self.upstream + "/" + path,
                content=body, headers={key: value for key, value in request.headers.items()
                                       if key.lower() not in {"host", "content-length"}},
            )
            download_delay = len(response.content) / self.download_rate if role_call else 0
            sleep_started = time.monotonic()
            await asyncio.sleep(download_delay)
            observed_download = time.monotonic() - sleep_started if role_call else 0
            with self.lock:
                row = self.rows.setdefault(category, {"requests": 0, "request_body_bytes": 0,
                    "response_body_bytes": 0, "controlled_transfer_delay_seconds": 0.0,
                    "requested_transfer_delay_seconds": 0.0,
                    "non_200_responses": 0})
                row["requests"] += 1
                row["request_body_bytes"] += len(body)
                row["response_body_bytes"] += len(response.content)
                row["controlled_transfer_delay_seconds"] += observed_upload + observed_download
                row["requested_transfer_delay_seconds"] += upload_delay + download_delay
                row["non_200_responses"] += response.status_code != 200
            return Response(response.content, status_code=response.status_code,
                            headers={key: value for key, value in response.headers.items()
                                     if key.lower() not in {"content-length", "transfer-encoding", "connection", "content-encoding"}})

    def snapshot(self):
        with self.lock:
            return {key: dict(value) for key, value in self.rows.items()}


def provider_group(children):
    @asynccontextmanager
    async def lifespan(app):
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            for child in children:
                await stack.enter_async_context(child.router.lifespan_context(child))
            yield

    app = FastAPI(lifespan=lifespan)
    for index, child in enumerate(children):
        prefix = f"/instance-{index}"

        async def isolated(scope, receive, send, child=child, prefix=prefix):
            inner = dict(scope)
            inner["path"] = scope["path"].removeprefix(prefix)
            inner["raw_path"] = inner["path"].encode()
            inner["root_path"] = ""
            await child(inner, receive, send)

        app.mount(prefix, isolated)
    return app


@contextmanager
def scenario_fixture(root: Path, *, fast_region="east", hidden_size=32):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = create_tiny_llama_checkpoint(root / "checkpoint", hidden_size=hidden_size,
        intermediate_size=hidden_size * 2, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=hidden_size // 4)
    model = Model.path(str(checkpoint), model_id="network-scenarios-tiny")
    budget = ExecutionBudget(1, 32, 2)
    installed = Experiment("installed", TwoOnlineOffsetCpu(model), Deployment.local(root="local://fixture"), budget)
    source = resolve_model(model).source_lock_digest
    model_plan = lower_model(json.loads((checkpoint / "config.json").read_bytes()), batch=1,
                              max_input_tokens=32, max_new_tokens=2)
    stamp = time.time_ns() // 1_000_000
    client = PartyOffer("client", "client", "c" * 64, stamp + 60_000, ("trusted_client",),
                        256 << 20, 64 << 20, 8, 0, ("*",), ("*",))
    names = ("a-east", "a-west", "b-east", "b-west")
    roots = tuple(PartyTrust(name, "operator-" + name, f"http://127.0.0.1:{free_port()}",
                            "PLLM_SCENARIO_" + name.upper().replace("-", "_")) for name in names)
    credentials = {item.credential_env: secrets.token_urlsafe(32) for item in roots}
    rates = {name: (20_000_000 if name.endswith(fast_region) else 200_000) for name in names}
    placeholders = tuple(PartyOffer(item.party_id, item.operator_id, str(index + 1) * 64,
        stamp + 60_000, ("public_linear_provider",), 256 << 20, 64 << 20, 2, 0, ("*",), ("*",))
        for index, item in enumerate(roots))
    links = tuple(LinkObservation(left, right, "controlled-proxy-rate", stamp, stamp + 60_000,
        rates[name], 0) for name in names for left, right in (("client", name), (name, "client")))
    network = NetworkSpec("shaped-fixture", NetworkSnapshot("shaped-fixture", (client, *placeholders),
        links, "controlled-loopback-fixture", stamp, stamp + 60_000), "http",
        ("client", *(item.operator_id for item in roots)), roots)
    specs = tuple(PartySpec(item.party_id, installed, source,
        ("worker_a",) if item.party_id.startswith("a-") else ("worker_b",),
        256 << 20, 64 << 20, max_sessions=2, offer_seconds=60, peer_party_ids=names) for item in roots)
    children = [create_party_app(spec, network, credentials=credentials) for spec in specs]
    proxies = []
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {**credentials, "XDG_CACHE_HOME": str(root / "cache")}))
        for group_index in range(2):
            origin = stack.enter_context(running_http(provider_group(children[group_index * 2:group_index * 2 + 2]), free_port()))
            for offset in range(2):
                index = group_index * 2 + offset
                proxy = ShapedProxy(origin + f"/instance-{offset}", rates[names[index]], rates[names[index]])
                stack.enter_context(running_http(proxy.app, int(roots[index].origin.rsplit(":", 1)[1])))
                proxies.append(proxy)
        snapshot = discover(network, credentials=credentials)
        candidate = replace(installed, name="selected-offset", deployment=Deployment.network(
            network_id=network.network_id, network_spec_digest=network.digest))
        request = PlanningRequest(model_plan, (candidate,), PlanningPolicy("client",
            time.time_ns() // 1_000_000, 4, 64, ("link_transfer_ms", "client_weight_bytes"),
            minimum_remote_mac_fraction=0.5, max_observation_age_ms=300_000), source_lock_digest=source)
        result = plan(request, snapshot=snapshot)
        if result.status != "feasible":
            raise RuntimeError("fixture has no admitted placement")
        paths = {}
        for name, record in (("network", network), ("request", request), ("plan", result), ("snapshot", snapshot)):
            paths[name] = root / (name + ".json")
            paths[name].write_bytes(record.canonical_bytes() + b"\n")
        paths["prompt"] = root / "private-prompt.txt"
        paths["prompt"].write_text("Hi", encoding="utf-8")
        yield {"network": network, "credentials": credentials, "request": request,
               "result": result, "paths": paths, "proxies": proxies, "children": children,
               "rates": rates, "source_lock_digest": source}
        if any(child.state.controller._reservations for child in children):
            raise RuntimeError("fixture capacity leaked")


def cli(arguments):
    completed = subprocess.run([sys.executable, "-m", "pllm", "--format", "json", *arguments],
                               capture_output=True, text=True, timeout=180)
    if completed.returncode:
        # Avoid exporting request content or credentials from a failure diagnostic.
        raise RuntimeError(f"fixture CLI failed with exit {completed.returncode}")
    return json.loads(completed.stdout)


def gateway_cli_probe(fixture, mode):
    """Launch ordinary CLI gateway, execute both API families, then reap child."""
    port = free_port()
    local_key = secrets.token_urlsafe(24)
    child = subprocess.Popen([sys.executable, "-m", "pllm", "gateway", "--" + mode,
        str(fixture["paths"][mode]), "--network", str(fixture["paths"]["network"]),
        "--port", str(port)], env={**os.environ, "PLLM_GATEWAY_API_KEY": local_key},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30,
                          headers={"authorization": f"Bearer {local_key}"}) as client:
            deadline = time.monotonic() + 30
            while True:
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("CLI gateway failed to start")
                try:
                    if client.get("/healthz").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.02)
            model = fixture["result"].experiment.resolve().model
            response = client.post("/v1/responses", json={"model": model,
                "input": fixture["paths"]["prompt"].read_text(), "max_output_tokens": 2,
                "temperature": 0}).raise_for_status().json()
            streamed = client.post("/v1/chat/completions", json={"model": model,
                "messages": [{"role": "user", "content": fixture["paths"]["prompt"].read_text()}],
                "max_tokens": 2, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}).raise_for_status()
            usages = [json.loads(line[6:])["usage"] for line in streamed.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"
                      and json.loads(line[6:]).get("usage")]
            zero = all(offer.active_sessions == 0 for offer in discover(fixture["network"]).offers)
            return {"mode": mode, "response_output_tokens": response["usage"]["output_tokens"],
                    "chat_stream_output_tokens": usages[-1]["completion_tokens"],
                    "stream_terminal_received": "data: [DONE]" in streamed.text,
                    "capacity_zero": zero}
    finally:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def run_scenarios(root):
    scenarios = []
    for region in ("east", "west"):
        with scenario_fixture(root / region, fast_region=region) as fixture:
            paths = fixture["paths"]
            commands = [
                ["--dry-run", "gateway", "--plan", str(paths["plan"]), "--network", str(paths["network"])],
                ["--dry-run", "gateway", "--request", str(paths["request"]), "--network", str(paths["network"])],
                ["--dry-run", "benchmark", "run", "--plan", str(paths["plan"]), "--network", str(paths["network"]),
                 "--prompt-file", str(paths["prompt"]), "--max-output-tokens", "2", "--temperature", "0"],
            ]
            for command in commands:
                cli(command)
            gateway_checks = [gateway_cli_probe(fixture, mode) for mode in ("plan", "request")]
            fixed = cli(["benchmark", "run", "--plan", str(paths["plan"]), "--network", str(paths["network"]),
                         "--prompt-file", str(paths["prompt"]), "--max-output-tokens", "2", "--temperature", "0"])
            before = [proxy.snapshot() for proxy in fixture["proxies"]]
            comparison = cli(["benchmark", "run", "--request", str(paths["request"]), "--network", str(paths["network"]),
                              "--compare-feasible", "--prompt-file", str(paths["prompt"]), "--max-output-tokens", "2",
                              "--temperature", "0", "--warmups", "1", "--repetitions", "3", "--capture-output-digest"])
            report = comparison["data"]["report"]
            proxy_deltas = {}
            for name, proxy, baseline in zip(fixture["rates"], fixture["proxies"], before, strict=True):
                proxy_deltas[name] = {category: {key: value - baseline.get(category, {}).get(key, 0)
                                     for key, value in row.items()} for category, row in proxy.snapshot().items()}
            online_delay = sum(rows.get("online", {}).get("controlled_transfer_delay_seconds", 0)
                               for rows in proxy_deltas.values())
            predicted_delay = sum(run["network_audit"]["prediction"]["link_transfer_ms"] / 1000
                for candidate in report["candidates"] for run in
                (*candidate["report"]["warmup_runs"], *candidate["report"]["runs"]))
            digests = {run["generation"]["output_text_digest"] for candidate in report["candidates"]
                       for run in candidate["report"]["runs"]}
            zero_capacity = all(offer.active_sessions == 0 for offer in discover(fixture["network"]).offers)
            scenarios.append({
                "name": region + "-fast", "shape": {"hidden_size": 32, "intermediate_size": 64,
                    "layers": 1, "heads": 4, "kv_heads": 2, "head_dim": 8},
                "source_lock_digest": fixture["source_lock_digest"], "numeric_contract": "W8A8",
                "provider_http_apps": 2, "installed_role_instances": 4, "shaping_proxies": 4,
                "rates_bytes_per_second": fixture["rates"], "offer_ttl_seconds": 60,
                "snapshot_ttl_seconds": 60, "snapshot_link_origin": "estimate",
                "link_observation_scope": "explicit controlled byte-rate settings; actual delays independently observed",
                "selected_roles": [{"role_id": row["role_id"], "party_id": row["party_id"]}
                                   for row in fixture["result"].native_placement["roles"]],
                "fixed_plan_report": fixed["data"]["report"], "comparison": report,
                "live_cli_gateways": gateway_checks,
                "proxy_body_ledger": proxy_deltas,
                "prediction_error": {"predicted_transfer_only_seconds": predicted_delay,
                    "observed_serialized_online_shaping_seconds": online_delay,
                    "difference_seconds": online_delay - predicted_delay,
                    "scope": "bounded arithmetic arrays vs observed monotonic proxy sleep intervals on actual online bodies; includes sleep scheduling error, excludes RTT and compute, not full latency error"},
                "checks": {"exact_public_output_digest_match": len(digests) == 1,
                           "capacity_zero_after_runs": zero_capacity, "cli_dry_runs_passed": True},
            })
    choices = [tuple((row["role_id"], row["party_id"]) for row in item["selected_roles"]) for item in scenarios]
    return {"schema": "pllm.network_planner_benchmark_evidence.v1", "scenarios": scenarios,
            "checks": {"different_directed_bandwidth_choices": choices[0] != choices[1]},
            "limitations": ["loopback controlled simulation; operator separation declarations not physical proof",
                "controller bytes counted once at proxy/client HTTP boundary; credentials/body contents never retained",
                "controller retries absent in successful fixture; failed-response counts observed separately",
                "full wire and aggregate whole-response CPU unmeasured; no full-cost ranking"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pllm-network-scenarios-") as temporary:
        evidence = run_scenarios(Path(temporary))
    data = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "checks": evidence["checks"]}))
    else:
        print(data)
