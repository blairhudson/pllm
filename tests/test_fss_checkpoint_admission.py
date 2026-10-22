"""Opt-in installed-wheel checkpoint admission; no roles or private execution.

After ordinary wheel installation, retry both row counts from the repository root::

    env -u PYTHONPATH HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
        PLLM_RUN_REAL_QWEN25=1 uv run --no-sync pytest \
        -o pythonpath= -q -rA tests/test_fss_checkpoint_admission.py

The explicit Hub cache keeps the cached source visible despite the ordinary
test fixture's isolated XDG_CACHE_HOME. This remains a conditional retry recipe,
not a claim that checkpoint admission has passed.

Uses the existing qwen_checked_tensor_smoke factory with only its input-row
budget varied. Source numeric/work admission is not tape/transport admission,
an accepted private request, serving qualification, or trained-model quality.
"""

import hashlib
import json
import os
from pathlib import Path
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
    import pllm
    from pllm.deployment.resources import reserve_incremental_disk, reserve_incremental_memory
    from pllm.runtime.benchmark_memory import (
        BenchmarkMemoryError,
        MemoryWatchdog,
        benchmark_memory,
        require_admission,
    )
    from pllm.runtime.fss_source import load_role_source, resolve_client_source

    from examples.papers.fusefss import qwen_checked_tensor_smoke

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
    model = experiment.pipeline.model
    assert model.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert model.local_files_only is True
    assert experiment.budget.requests == experiment.budget.max_new_tokens == 1
    profile = "fusefss.dense-rope-checked-ranges-ring50-q12.v1"
    assert experiment.pipeline.components["linear"].params["numeric_profile"] == profile

    # Preserve the ordinary whole-topology gate and its default allocation
    # margin, even though only one source owner will actually be imported.
    report = benchmark_memory(experiment, backend="native")
    require_admission(report)
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

        # Match fss_worker: reserve the exact native contract price on the native
        # tempfile device, then bind only this locally imported owner's backing.
        # The ordinary role owner price covers import/compile; create no tapes.
        with reserve_incremental_memory(report["estimate"]["incremental_owner_bounds"]["per_role"]):
            with reserve_incremental_disk(
                contract["source_disk_bytes"], directory=Path(tempfile.gettempdir())
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
