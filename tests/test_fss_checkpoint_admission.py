"""Opt-in installed-wheel checkpoint admission and raw-ID transport regression.

After ordinary wheel installation, retry 1+1 from the repository root first::

    env -u PYTHONPATH HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
        PLLM_RUN_REAL_QWEN25=1 uv run --no-sync python -m pytest \
        -o pythonpath= -q -rA -x \
        'tests/test_fss_checkpoint_admission.py::test_pinned_qwen_checked_tensor_source_admission[1]'

Only after that succeeds, retry 4+1 with the same guards::

    env -u PYTHONPATH HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
        PLLM_RUN_REAL_QWEN25=1 uv run --no-sync python -m pytest \
        -o pythonpath= -q -rA -x \
        'tests/test_fss_checkpoint_admission.py::test_pinned_qwen_checked_tensor_source_admission[4]'

Complete-topology admission has a separate opt-in. It imports one qualification
source, obtains ordinary native program pricing, retires that source and its
leases, then requires and reserves the full graph. No children, material tapes,
private execution or trained-request quality are exercised. Run 1+1 first::

    env -u PYTHONPATH HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
        PLLM_RUN_FSS_TOPOLOGY_ADMISSION=1 uv run --no-sync python -m pytest \
        -o pythonpath= -q -rA -x \
        'tests/test_fss_checkpoint_admission.py::test_pinned_qwen_checked_tensor_topology_admission[1]'

Only after that succeeds, use the same command with the final node ID ``[4]``.
Source-only success is not full-topology admission; full-graph rejection remains
a failure, never a skip or a retry with reduced caps or safety margins.

The separate ``[3]`` workload uses ``qwen_checked_tensor_minimal_smoke`` and fits
ordinary role-prefixed ``user: hi`` input. Its result cannot replace the original
4+1 workload's admission or execution result; all guards and ceilings stay fixed.

The explicit Hub cache keeps the cached source visible despite the ordinary
test fixture's isolated XDG_CACHE_HOME. This remains a conditional retry recipe,
not a claim that checkpoint admission has passed.

Uses the existing qwen_checked_tensor_smoke factory with only its input-row
budget varied. Source numeric/work admission is not tape/transport admission,
an accepted private request, serving qualification, or trained-model quality.

The checkpoint-backed raw-ID transport regression has independent per-node
execution guards: both PLLM_RUN_REAL_QWEN25=1 and
PLLM_RUN_FSS_RAW_TRANSPORT_ROWS=N are required for node [N], N in 1..4.
Admission opt-ins alone never launch it. Each node needs a separate execution
grant, the installed wheel, an explicit existing HF_HUB_CACHE, and unchanged
local-only checkpoint pins and native caps; no downloads or fallback occur.
It sends exactly N raw tokenizer IDs and requests one selected ID, not ChatML
or Responses input. This is not ordinary benchmark acceptance, whole-model
quality, or security qualification. Context closure and completion receipts
are not independent physical-retirement, funding, or security proofs.
"""

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading

import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_REAL_QWEN25") != "1",
    reason="opt in to the pinned cached real Qwen2.5 checkpoint",
)
@pytest.mark.parametrize("initial_rows", [1, 4])
def test_pinned_qwen_checked_tensor_source_admission(initial_rows, monkeypatch):
    """Require the complete native numeric/resource walk, not serving qualification."""
    with _qualified_qwen_source(initial_rows, monkeypatch):
        pass


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_FSS_TOPOLOGY_ADMISSION") != "1",
    reason="opt in to pinned Qwen native-priced complete-topology admission",
)
@pytest.mark.parametrize("initial_rows", [1, 3, 4])
def test_pinned_qwen_checked_tensor_topology_admission(initial_rows, monkeypatch):
    """Require full graph capacity, without starting children or making material."""
    _complete_topology_admission(_qualified_qwen_source(initial_rows, monkeypatch))


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_REAL_QWEN25") != "1",
    reason="opt in to the pinned cached real Qwen2.5 checkpoint",
)
@pytest.mark.parametrize("initial_rows", [1, 2, 3, 4])
def test_pinned_qwen_checked_tensor_raw_transport(initial_rows, monkeypatch):
    if os.environ.get("PLLM_RUN_FSS_RAW_TRANSPORT_ROWS") != str(initial_rows):
        pytest.skip(
            f"opt in separately to raw-ID execution for this node with "
            f"PLLM_RUN_FSS_RAW_TRANSPORT_ROWS={initial_rows}"
        )

    cache = os.environ.get("HF_HUB_CACHE")
    assert cache and Path(cache).is_dir(), "HF_HUB_CACHE must name an explicit existing cache directory"

    import tokenizers

    from pllm.runtime.benchmark_memory import (
        BenchmarkMemoryError,
        MemoryWatchdog,
        require_admission,
    )
    from pllm.runtime.fss_transport import execute_request, FssResult
    from pllm.runtime.servers import build_roles

    with _qualified_qwen_source(initial_rows, monkeypatch) as (experiment, loaded, check_memory):
        source = loaded.source
        expected_pins = dict(source.snapshot_pins)
        tokenizer = tokenizers.Tokenizer.from_buffer(source.tokenizer_bytes)
        encoded = tokenizer.encode("hello", add_special_tokens=False).ids
        assert encoded
        ids = tuple((encoded * initial_rows)[:initial_rows])
        assert len(ids) == initial_rows
        assert all(type(token) is int and 0 <= token < source.contract["vocabulary"] for token in ids)
        check_memory()
    assert loaded._closed
    del loaded

    report = _source_qualification_report(experiment)
    cancelled = threading.Event()
    watchdog = MemoryWatchdog(report, lambda _reason: cancelled.set())

    def check_execution_memory():
        if watchdog.error is not None:
            raise BenchmarkMemoryError(watchdog.error)
        watchdog.check()
        if watchdog.error is not None:
            raise BenchmarkMemoryError(watchdog.error)
        assert not cancelled.is_set(), "checkpoint-backed raw-ID transport cancelled"

    with ExitStack() as stack:
        stack.callback(watchdog.close)
        watchdog.start()
        topology = stack.enter_context(build_roles(experiment))
        admission = topology.memory_admission
        assert type(admission) is dict and admission.get("full_topology_admitted") is True
        require_admission(admission)
        qualification = admission["source_qualification"]
        assert qualification["native_program_priced"] is True
        assert qualification["owner_retired"] is True
        assert qualification["completed"] is True
        assert admission["estimate"]["fss_admission"]["source_pins"] == expected_pins
        check_execution_memory()
        result = execute_request(
            experiment, source, ids, 1, topology.worker_connections(), 600.0, cancelled.is_set,
        )
        check_execution_memory()
        assert isinstance(result, FssResult)
        assert len(result.selected) == 1
        assert type(result.selected[0]) is int
        assert 0 <= result.selected[0] < source.contract["vocabulary"]
    assert topology.closed


@contextmanager
def _qualified_qwen_source(initial_rows, monkeypatch):
    """One guarded native owner shared by independent source/graph admission tests."""
    import pllm
    from pllm.deployment.resources import reserve_incremental_disk, reserve_incremental_memory
    from pllm.runtime.benchmark_memory import (
        BenchmarkMemoryError,
        MemoryWatchdog,
    )
    from pllm.runtime.fss_source import load_role_source, resolve_client_source

    from examples.papers.fusefss import (
        qwen_checked_tensor_minimal_smoke,
        qwen_checked_tensor_smoke,
    )

    # Default pytest pythonpath shadows the wheel. Reject mixed Python/native
    # imports rather than skipping an outdated or missing installed implementation.
    checkout_python = Path(__file__).resolve().parents[1] / "python"
    installed_package = Path(pllm.__file__).resolve().parent
    assert not installed_package.is_relative_to(checkout_python), (
        "Run with the rebuilt installed wheel, PYTHONPATH unset and -o pythonpath="
    )
    for name, module in tuple(sys.modules.items()):
        if name.startswith("pllm.") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(installed_package), name

    experiment = qwen_checked_tensor_smoke().with_params(
        budget__max_input_tokens=initial_rows,
    )
    if initial_rows == 3:
        assert experiment == qwen_checked_tensor_minimal_smoke()
    model = experiment.pipeline.model
    assert model.kind == "huggingface"
    assert model.source == "Qwen/Qwen2.5-0.5B-Instruct"
    assert model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert model.local_files_only is True
    assert experiment.budget.requests == experiment.budget.max_new_tokens == 1
    profile = "fusefss.dense-rope-checked-ranges-ring50-q12.v1"
    assert experiment.pipeline.components["linear"].params["numeric_profile"] == profile

    # Use the ordinary runtime's source-only qualification phase: the unchanged
    # whole-topology memory estimate/margin, one source disk owner, no tapes.
    # A later native-priced complete-topology gate is still required for roles.
    report = _source_qualification_report(experiment)
    cancelled = threading.Event()
    monitor = MemoryWatchdog(report, lambda _reason: cancelled.set())

    def check_memory():
        if monitor.error is not None:
            raise BenchmarkMemoryError(monitor.error)
        monitor.check()

    try:
        monitor.start()
        source = resolve_client_source(experiment)
        check_memory()
        # Verify immutable original-source pins before decoding checkpoint values.
        assert source.source_lock_digest == (
            "383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29"
        )
        assert source.checkpoint_digest == (
            "f69322057253c4fb853c10e73da5d84111262e0fb1edba33933b9e5b8caf59fd"
        )
        pins = dict(source.snapshot_pins)
        contract_bytes = source.runtime_contract(experiment, initial_rows, 1)
        contract = json.loads(contract_bytes)
        assert contract["configuration"] == experiment.configuration_digest()
        assert contract["composition"] == experiment.pipeline.digest()
        assert contract["source_lock"] == pins["source_lock"]
        assert contract["tokenizer"] == pins["tokenizer"]
        assert contract["source_config_sha256"] == hashlib.sha256(source.config_bytes).hexdigest()
        assert pins["config"] == hashlib.sha256(source.config_source_bytes).hexdigest()
        assert contract["weights"] == "public"
        assert contract["capacity"] == "decoder"
        assert contract["numeric_profile"] == profile
        assert contract["material_delivery"] == "batch-v1"
        assert contract["online_schedule"] == "tensor-v1"
        assert contract["input_rows"] == contract["maximum_sequence"] == initial_rows
        assert contract["output_tokens"] == 1
        assert contract["selection_prefix_rows"] == [initial_rows]
        assert contract["width"] == 896
        assert contract["vocabulary"] == 151936
        assert contract["ring_bits"] == 50
        assert contract["fractional_bits"] == 12
        config = json.loads(source.config_bytes)
        assert config["num_hidden_layers"] == 24
        assert config["num_attention_heads"] == 14
        assert config["num_key_value_heads"] == 2
        assert config["intermediate_size"] == 4864
        assert contract["state_policy"] == "complete_prefix_recomputation"
        assert contract["persistent_kv"] is False
        # Metadata-only source resolution must not masquerade as value admission.
        assert contract["admission"]["numeric"] == "not_performed"
        assert contract["admission"]["program_resources"] is None

        # Native exceptions retain their exact text; native public kernel/range
        # diagnostics remain in pytest's captured stderr if admission fails.
        monkeypatch.setenv("PLLM_FSS_ADMISSION_DEBUG", "1")

        # Reserve the exact native contract price on the native tempfile device,
        # retaining production qualification's 2 GiB floor at this live recheck.
        # Bind only this local owner's backing, as fss_worker does. The ordinary
        # role owner price covers import/compile; create no tapes.
        with reserve_incremental_memory(report["estimate"]["incremental_owner_bounds"]["per_role"]):
            with reserve_incremental_disk(
                contract["source_disk_bytes"], directory=Path(tempfile.gettempdir()),
                minimum_free_bytes=2 << 30,
            ) as source_lease:
                loaded = None
                try:
                    check_memory()
                    try:
                        loaded = load_role_source(
                            experiment,
                            expected_pins=pins,
                            input_rows=initial_rows,
                            outputs=1,
                            cancelled=cancelled.is_set,
                            source=source,
                        )
                        if hasattr(loaded.native_program, "disk_backing"):
                            source_lease.bind_backing(loaded.native_program.disk_backing)
                        resources = json.loads(loaded.native_program.metadata())
                    except ValueError as error:
                        check_memory()
                        # Preserve native public context/stage/ledger verbatim.
                        # Rejection is a test failure, never successful evidence.
                        pytest.fail(
                            f"Pinned FuseFSS source admission rejected "
                            f"(initial_rows={initial_rows}, selections=1, profile={profile}, "
                            f"source_pins={pins}): {error}",
                            pytrace=False,
                        )
                    check_memory()
                    assert loaded.contract_bytes == contract_bytes
                    assert loaded.source.snapshot_pins == pins
                    manifest = loaded.manifest
                    assert manifest["contract"] == hashlib.sha256(contract_bytes).hexdigest()
                    assert manifest["profile"] == profile
                    assert manifest["initial"] == initial_rows
                    assert manifest["selections"] == 1
                    assert manifest["vocabulary"] == 151936
                    assert resources["schema"] == "pllm.fss.program_resources.v1"
                    assert resources["scope"] == "bounded-native-program-before-role-transport"
                    assert resources["capacity"] == "decoder"
                    assert resources["numeric_profile"] == profile
                    assert resources["online_schedule"] == "tensor-v1"
                    assert resources["initial_tokens"] == initial_rows
                    assert resources["selections"] == 1
                    assert resources["vocabulary"] == 151936
                    assert resources["program_digest"] == manifest["program"]
                    assert resources["source_digest"] == manifest["source"]
                    assert resources["original_input_digest"] == manifest["input_digest"]
                    assert 0 < resources["source_disk_bytes"] <= contract["source_disk_bytes"]
                    assert resources["primitive_requests_per_party"] > 0
                    # Existing decoder cumulative ceilings and unchanged 64 MiB
                    # live native owner ceiling; never pin historical totals.
                    ceilings = {
                        "kernels": 1 << 22,
                        "work": 1 << 38,
                        "material_bytes": 32 << 30,
                        "frame_bytes": 8 << 30,
                        "public_copy_bytes": 32 << 30,
                        "metadata_bytes": 64 << 20,
                        "live_array_bytes_per_party": 64 << 20,
                        "peak_native_bytes": 64 << 20,
                        "combined_reservation_bytes": 64 << 20,
                    }
                    for field, ceiling in ceilings.items():
                        assert type(resources[field]) is int
                        assert 0 < resources[field] <= ceiling, (field, resources[field], ceiling)
                    assert resources["combined_reservation_bytes"] == (
                        3 * resources["metadata_bytes"]
                        + 2 * resources["live_array_bytes_per_party"]
                        + resources["peak_native_bytes"]
                    )
                    yield experiment, loaded, check_memory
                finally:
                    # Stop ledger observations before closing native files; keep
                    # the full pending charge until native ownership is retired.
                    try:
                        source_lease.clear_backing()
                    finally:
                        if loaded is not None:
                            loaded.close()
        check_memory()
    finally:
        monitor.close()


def _complete_topology_admission(qualification):
    """Mirror LocalTopology qualification/pricing/reservation, stopping before children."""
    from pllm.deployment.resources import reserve_incremental_disk, reserve_incremental_memory
    from pllm.runtime.benchmark_memory import (
        admit_memory,
        benchmark_memory,
        host_memory,
        require_admission,
    )

    with qualification as (experiment, loaded, check_memory):
        # Only this ordinary API may produce the native program/transport price.
        # It checks a live LoadedFssSource, original pins/identity, full saved
        # envelope and manifest/resource agreement; no JSON price is supplied.
        priced = benchmark_memory(experiment, backend="native", fss_imported=loaded)
        estimate = priced["estimate"]
        admission = estimate["fss_admission"]
        assert admission["source_pins"] == loaded.source.snapshot_pins
        assert admission["metadata_only"] is False
        assert admission["numeric_admission"] == "native_imported_program"
        assert admission["pricing_scope"] == "saved_full_envelope; not_request_admission"
        assert admission["qualification_owner_lifetime"] == (
            "caller_owned; retire_before_full_topology_source_imports"
        )
        assert admission["complete_program_resources"] == json.loads(loaded.native_program.metadata())
        assert admission["transport_admission"]["scope"] == "program-pricing-not-request-admission"
        assert admission["disk_price_scope"] == "native transport admission after import"
        assert admission["serving_qualified"] is False
        assert estimate["max_input_tokens"] == experiment.budget.max_input_tokens
        assert estimate["max_output_tokens"] == experiment.budget.max_new_tokens
        check_memory()
        # As in LocalTopology, this live-owner observation is not the final
        # admission: qualification files can temporarily consume graph headroom.

    # Context retirement must finish before the final complete-graph capacity check.
    # Keep the normal policy/margin, resampling actual native tempfile storage.
    assert loaded._closed
    directory = Path(tempfile.gettempdir())
    host = replace(host_memory(), disk_free=int(shutil.disk_usage(directory).free))
    report = admit_memory(estimate, host, backend="native")
    report.update(
        admission_scope="complete_topology_before_children",
        full_topology_admitted=report["admitted"],
        reservation_scope="process_local_logical_memory_and_disk; not_filesystem_or_cross_process_quota",
    )
    # Ordinary report stays in pytest output even when the unchanged gate rejects.
    print(json.dumps(report, sort_keys=True))
    require_admission(report)
    # Same logical complete-graph leases as LocalTopology. No source reimport,
    # tape allocation, material generation, role creation or execution follows.
    with ExitStack() as graph:
        required = report["candidates"]["native"]
        graph.enter_context(reserve_incremental_memory(required["required_host_bytes"]))
        disk = required["required_disk_bytes"] - (2 << 30)
        if disk:
            graph.enter_context(reserve_incremental_disk(
                disk, directory=directory, minimum_free_bytes=2 << 30,
            ))
    return report


def _source_qualification_report(experiment):
    """Use the production next-phase gate, never admit a serving graph here."""
    from pllm.runtime.benchmark_memory import (
        BenchmarkMemoryError,
        execution_memory_admission,
        require_admission,
    )

    report = execution_memory_admission(experiment, backend="native")
    require_admission(report)
    if (
        report.get("admission_scope") != "source_qualification_only"
        or report.get("full_topology_admitted") is not False
    ):
        raise BenchmarkMemoryError("expected source-only qualification admission")
    return report
