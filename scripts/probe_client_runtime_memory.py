"""Fresh-process client RSS through the ordinary supervised benchmark path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile

MODES = ("eager-resident", "lazy-resident", "lazy-paged")
TOKENIZER_MODES = ("paged-eager-tokenizer", "lazy-paged")
TOKENIZER_REUSE_MODES = ("lazy-paged", "paged-shared-tokenizer")
TOKENIZER_INDEX_MODES = ("paged-shared-tokenizer", "paged-indexed-tokenizer")


def experiment(mode, *, tiny=False, four_b=False, max_input_tokens=64):
    from pllm import Model
    if tiny:
        model = Model.tiny(model_id="isolated-client-memory")
    elif four_b:
        model = Model.hf("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c")
    else:
        model = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/benchmarks/client_memory.py"))
    return example["candidate"](mode, model,
        "paged" if mode in (*TOKENIZER_MODES, *TOKENIZER_REUSE_MODES, *TOKENIZER_INDEX_MODES) else "memory",
        max_input_tokens=max_input_tokens)


def reuse_one_bundle_tokenizer(factory):
    """Probe-only, one-response owner; never reuse across source bundle instances."""
    owner = None
    tokenizer = None

    def load(bundle):
        nonlocal owner, tokenizer
        if owner is None:
            tokenizer = factory(bundle)
            owner = bundle
        elif owner is not bundle:
            raise ValueError("tokenizer reuse probe cannot cross bundle identity")
        return tokenizer

    return load


def worker(args):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark
    profiles = {}
    owned_tokenizers = []
    if args.child == "paged-indexed-tokenizer":
        from client_offload_tokenizer import IndexedClientTokenizer
        from pllm.runtime.transformer_client import ClientBundle

        def indexed(bundle):
            tokenizer = IndexedClientTokenizer(bundle, args.index_root, args.index_contract_digest)
            owned_tokenizers.append(tokenizer)
            return tokenizer

        ClientBundle.tokenizer = indexed
    if args.profile_only or args.tokenizer_ablation or args.tokenizer_reuse_ablation or args.tokenizer_index_ablation:
        from pllm.runtime import client as client_module, model_binding, semantic_executor, transformer_client
        from pllm.runtime.benchmark_memory import process_memory

        def watch(owner, name):
            original = getattr(owner, name)
            label = f"{owner.__name__}.{name}"

            def measured(*values, **kwargs):
                before = process_memory()
                try:
                    return original(*values, **kwargs)
                finally:
                    after = process_memory()
                    row = profiles.setdefault(label, {"calls": 0, "peak_before_max": 0, "peak_after_max": 0,
                                                       "rss_after_max": 0, "new_highwater_max": 0})
                    row["calls"] += 1
                    if name == "tokenizer":
                        import traceback
                        row.setdefault("call_sites", []).append([
                            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                            for frame in traceback.extract_stack(limit=8)[:-1]
                        ])
                    row["peak_before_max"] = max(row["peak_before_max"], before["lifetime_peak_rss_bytes"] or 0)
                    row["peak_after_max"] = max(row["peak_after_max"], after["lifetime_peak_rss_bytes"] or 0)
                    row["rss_after_max"] = max(row["rss_after_max"], after["rss_bytes"])
                    if before["lifetime_peak_rss_bytes"] and after["lifetime_peak_rss_bytes"]:
                        row["new_highwater_max"] = max(row["new_highwater_max"],
                            after["lifetime_peak_rss_bytes"] - before["lifetime_peak_rss_bytes"])
            setattr(owner, name, measured)

        watch(client_module.RuntimeClient, "_load_artifact_bundle")
        watch(model_binding, "compile_runtime_model")
        watch(transformer_client.ClientBundle, "tokenizer")
        for name in ("prepare_ids", "forward_ids", "snapshot"):
            watch(semantic_executor.SemanticDecoderRuntime, name)
    if args.child in TOKENIZER_INDEX_MODES:
        from pllm.runtime.transformer_client import ClientBundle

        ClientBundle.tokenizer = reuse_one_bundle_tokenizer(ClientBundle.tokenizer)
    if args.child in {"eager-resident", "paged-eager-tokenizer"}:
        from pllm.runtime.transformer_client import MaskedTransformerClientRuntime
        runtime_constructor = MaskedTransformerClientRuntime.__init__

        def eager_tokenizer(self, *values, **kwargs):
            runtime_constructor(self, *values, **kwargs)
            self.tokenizer  # Restore the historical eager decoder allocation.

        MaskedTransformerClientRuntime.__init__ = eager_tokenizer
    if args.child == "eager-resident":
        # Test-local historical allocation control. The ordinary SDK remains lazy.
        from pllm.runtime import client
        from pllm.runtime.preparation_protocol import expand_preparation_mask, expand_output_mask
        original = client.PreparedStageRows

        def eager(*values, **kwargs):
            request = kwargs.get("request", values[0] if values else None)
            kwargs.update(input_mask=expand_preparation_mask(request),
                          output_mask=expand_output_mask(request))
            return original(*values, **kwargs)

        client.PreparedStageRows = eager
    try:
        result = run_loopback_benchmark(model="unused", model_id=None, tiny=False,
            experiment=experiment(args.child, tiny=args.tiny, four_b=args.four_b,
                                  max_input_tokens=args.max_input_tokens),
            prompt="Explain private inference in one sentence.", max_output_tokens=8,
            warmups=0, repetitions=1, timeout_seconds=300, temperature=0,
            capture_output_digest=True, _cohort_salt=bytes.fromhex(os.environ["PLLM_MEMORY_COHORT_SALT"]))
    finally:
        for tokenizer in owned_tokenizers:
            tokenizer.close()
    result["client_process_memory"]["isolated_candidate"] = True
    result["client_process_memory"]["allocation_control"] = args.child
    if profiles:
        result["memory_phase_diagnostic"] = {"scope": "nested process samples, non-additive", "phases": profiles}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--four-b", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, choices=range(16, 65), default=64,
                        metavar="16..64", help="public input bound shared by both candidates")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--profile-only", action="store_true", help="single paged diagnostic; no paired comparison")
    diagnostic.add_argument("--tokenizer-ablation", action="store_true", help="matched eager/lazy decoder-tokenizer pair")
    diagnostic.add_argument("--tokenizer-reuse-ablation", action="store_true",
        help="matched probe-only tokenizer reuse across one response's setup and execution")
    diagnostic.add_argument("--tokenizer-index-ablation", action="store_true",
        help="matched 4B shared-tokenizer/indexed-tokenizer complete-response probe")
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--index-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--index-contract-digest", help=argparse.SUPPRESS)
    parser.add_argument("--child", choices=(*MODES, "paged-eager-tokenizer", *TOKENIZER_INDEX_MODES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.tiny and args.four_b:
        parser.error("choose one model")
    if args.tokenizer_index_ablation and not args.four_b:
        parser.error("tokenizer index probe requires --four-b")
    if args.child:
        def interrupted(*_):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        worker(args)
        return
    from huggingface_hub.constants import HF_HUB_CACHE
    from pllm.metrics import benchmark_memory
    from pllm.runtime.benchmark_memory import BenchmarkMemoryError, require_admission
    records = {}
    index_metadata = None
    salt = os.urandom(32).hex()
    def checkpoint(document):
        archive.seek(0)
        json.dump(document, archive, indent=2, sort_keys=True)
        archive.truncate()
        archive.flush()
    with args.output.open("x") as archive, tempfile.TemporaryDirectory(
            prefix="pllm-client-runtime-memory-", dir=args.scratch_dir) as directory:
        checkpoint({"schema": "pllm.client_runtime_memory_probe.v1", "status": "running", "records": records})
        root = Path(directory)
        modes = (("lazy-paged",) if args.profile_only else
                  TOKENIZER_MODES if args.tokenizer_ablation else
                  TOKENIZER_REUSE_MODES if args.tokenizer_reuse_ablation else
                  TOKENIZER_INDEX_MODES if args.tokenizer_index_ablation else MODES)
        if args.tokenizer_index_ablation:
            from client_offload_tokenizer import file_digest
            from huggingface_hub import hf_hub_download
            from pllm.runtime.benchmark_memory import GiB, host_memory

            host = host_memory()
            if host.available - host.reserve < 2 * GiB:
                raise BenchmarkMemoryError("public tokenizer compiler needs 2 GiB admitted headroom")
            source = hf_hub_download("Qwen/Qwen3-4B", "tokenizer.json",
                revision="1cfa9a7208912126459214e8b04321603b3df60c", local_files_only=True)
            index_root = root / "public-tokenizer"
            compiler = Path(__file__).with_name("probe_client_offload.py")
            index_metadata = json.loads(subprocess.check_output([sys.executable, str(compiler),
                "--worker", "compile", "--source", source, "--artifact", str(index_root)], text=True, timeout=60))
            index_digest = file_digest(index_root / "contract.json")
            index_metadata["contract_sha256"] = index_digest
            index_metadata["probe_sha256"] = file_digest(Path(__file__))
            index_metadata["tokenizer_implementation_sha256"] = file_digest(Path(__file__).with_name("client_offload_tokenizer.py"))
            index_metadata["scope"] = "public offline compilation; excluded from client process, no artifact transport"
        for mode in modes:
            preflight = benchmark_memory(experiment(mode, tiny=args.tiny, four_b=args.four_b,
                max_input_tokens=args.max_input_tokens), backend="native")
            try:
                require_admission(preflight)
                # The parent remains live and eager control restores historical masks.
                extra = 256 << 20
                if mode == "eager-resident":
                    estimates = preflight["estimate"]
                    extra += estimates["components"]["legacy_client_mask_bytes"] * 5 // 4
                required = preflight["estimate"]["native_total_peak_bytes"] + extra
                if required > preflight["host"]["admission_budget_bytes"]:
                    raise BenchmarkMemoryError("isolated client probe exceeds physical headroom including control allocations")
            except BenchmarkMemoryError as error:
                checkpoint({"schema": "pllm.client_runtime_memory_probe.v1", "status": "blocked_admission",
                    "records": records, "mode": mode, "blocker": str(error), "preflight": preflight})
                raise
            cache = root / mode
            cache.mkdir()
            output = root / f"{mode}.json"
            env = dict(os.environ, XDG_CACHE_HOME=str(cache), HF_HUB_CACHE=HF_HUB_CACHE,
                       HUGGINGFACE_HUB_CACHE=HF_HUB_CACHE, HF_HUB_OFFLINE="1",
                       PLLM_MEMORY_COHORT_SALT=salt)
            command = [sys.executable, __file__, "--child", mode, "--output", str(output),
                       "--max-input-tokens", str(args.max_input_tokens)]
            if args.tiny:
                command.append("--tiny")
            if args.four_b:
                command.append("--four-b")
            if args.profile_only:
                command.append("--profile-only")
            if args.tokenizer_ablation:
                command.append("--tokenizer-ablation")
            if args.tokenizer_reuse_ablation:
                command.append("--tokenizer-reuse-ablation")
            if args.tokenizer_index_ablation:
                command.extend(["--tokenizer-index-ablation", "--index-root", str(index_root),
                                "--index-contract-digest", index_digest])
            print(f"running {mode}", flush=True)
            child = subprocess.Popen(command, env=env)
            try:
                if child.wait(timeout=420) != 0:
                    raise RuntimeError(f"{mode} failed")
            except BaseException as error:
                checkpoint({"schema": "pllm.client_runtime_memory_probe.v1",
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "records": records, "mode": mode, "error_type": type(error).__name__,
                    "blocker": str(error), "public_tokenizer_index": index_metadata})
                raise
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            records[mode] = json.loads(output.read_bytes())
            checkpoint({"schema": "pllm.client_runtime_memory_probe.v1", "status": "running", "records": records})
    # Same salt, public prompt, workload, numeric body and outputs in all children.
    controls = [record["runs"][0] for record in records.values()]
    result = {"schema": "pllm.client_runtime_memory_probe.v1", "status": "complete", "records": records,
              "comparison_available": len(controls) > 1,
              "checks_passed": all(record["checks"]["passed"] for record in records.values())}
    if index_metadata:
        result["public_tokenizer_index"] = index_metadata
    for key in (() if args.profile_only else ("model_fingerprint", "input_tokens", "output_tokens", "output_text_digest")):
        values = [(run["tokens"].get(key) if key.endswith("_tokens") else
                   run["generation"].get(key) if key == "output_text_digest" else run.get(key)) for run in controls]
        result[f"matched_{key}"] = len({json.dumps(value, sort_keys=True) for value in values}) == 1
        if values[0] is None:
            raise ValueError(f"benchmark did not report {key}")
    if not args.profile_only:
        result["matched_prompt_digest"] = len({row["configuration"]["prompt_digest"] for row in records.values()}) == 1
    if not all(value for key, value in result.items() if key.startswith("matched_") or key == "checks_passed"):
        raise ValueError("client memory cohort mismatch")
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({mode: row["client_process_memory"] for mode, row in records.items()}, indent=2))


if __name__ == "__main__":
    main()
