"""Frozen two-Experiment head-codec comparison; never imports old observations.

Certificate pin integration is deliberately narrow. Publication/delivery is not
measured by the response target and therefore cannot support cold-network claims.
"""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import statistics
import struct
import zlib

from pllm import Experiment
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario

from . import private_boundary_decode as screen
from . import private_boundary_decode_confirmation as confirmation
from . import private_boundary_byte_dot as kernel
from . import private_boundary_tiers as tiers
from .sota_combinations import Rejected

CONTROL = "parallel-uniform-residues"
CANDIDATE = "parallel-row-residues"
NAMES = (CONTROL, CANDIDATE)
METRICS = kernel.METRICS
ROLES = kernel.ROLES
preflight = tiers.preflight
_require = screen._require
CERTIFICATE_FIELD = "head_certificate"  # Observed SDK contract; parent must review integration.
ROOT = Path(__file__).resolve().parents[2]
# Complete repository-local import closure. Installed SDK/dependencies remain
# separately committed by BuildRef; historical observations are not dependencies.
VERIFIER_SOURCES = (
    "benchmarks/__init__.py", "benchmarks/research/__init__.py",
    "benchmarks/research/private_boundary_row_codec.py",
    "benchmarks/research/private_boundary_byte_dot.py",
    "benchmarks/research/private_boundary_decode.py",
    "benchmarks/research/private_boundary_decode_confirmation.py",
    "benchmarks/research/private_boundary_tiers.py",
    "benchmarks/research/sota_combinations.py",
    "benchmarks/research/historical_evidence.py",
    "examples/prepared_sdk.py",
    "scripts/benchmark_sota_combinations.py",
    "scripts/benchmark_private_boundary_row_codec.py",
    "scripts/summarize_private_boundary_row_codec.py",
)
SDK_SOURCES = ("runtime/private_boundary_certificate.py", "runtime/private_boundary_head_codec.py",
    "runtime/private_boundary_transport.py", "runtime/benchmark_response_audit.py",
    "runtime/benchmark_response_target.py",
    "runtime/private_boundary_protocol.py", "runtime/private_boundary_loader.py",
    "runtime/model_binding.py", "runtime/quantization.py", "_benchmark_worker.py",
    "_benchmark_workload.py", "_benchmark_contract.py", "runtime/benchmark_context.py",
    "runtime/topology_accounting.py", "runtime/dashboard.py", "deployment/execution.py",
    "runtime/transformer_client.py")
VERIFIER_SOURCES += tuple("sdk/" + name for name in SDK_SOURCES)
SOURCE_FIELDS = ("source_digest", "checkpoint_digest", "model_config_digest", "head_weight_digest",
    "head_scales_digest", "lookup_digest", "vocabulary", "hidden", "activation_bits", "weight_bits",
    "signed_output_bound")
CERT_NUMERIC = "signed-i8-row-major-bias-free/symmetric-qmax127/int-wrap32/f32-separate-multiplies/v1"


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def bounded_read(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    _require(0 < len(data) <= limit, "row_codec_input_size_invalid")
    return data


def bounded_json(path, limit):
    try:
        value = json.loads(bounded_read(path, limit))
    except (ValueError, RecursionError) as exc:
        raise Rejected("row_codec_input_json_invalid") from exc
    _require(type(value) is dict, "row_codec_input_json_invalid")
    return value


def hashes(paths):
    result = {}
    for path in paths:
        with Path(path).open("rb") as stream:
            result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def public_prompt_context(prompt):
    """Match dashboard /api/run string preparation before SDK rendering.

    The SDK currently inlines strip() at that route; validate_context is shared.
    Raw file bytes still go to benchmark stdin and its salted input commitment.
    """
    from pllm.runtime.benchmark_context import validate_context
    return validate_context(prompt.strip())


def receipt_record(document, plan):
    """Canonical public record bindings, not signatures or private-input proofs."""
    from pllm._benchmark_contract import digest
    return {"schema": "pllm.private_boundary_row_codec_receipt.v1", "status": "completed",
        "post_run_checks": "passed", "cohort": document.get("cohort"),
        "plan_digest": digest(plan), "raw_report_digest": digest(document),
        "public_prompt_digest": digest(plan.get("public_prompt"))}


def terminal_receipt(document, plan, sources):
    """Driver-only issuance point: never issue on interruption or source drift."""
    _require(verifier_sources() | hashes(sources) == plan["frozen_sources"],
             "sources_changed_during_run")
    _require(document.get("complete") is True, "row_codec_incomplete_cohort")
    return receipt_record(document, plan)


def verify_receipt(document, plan, receipt):
    from pllm._benchmark_contract import digest
    _require(type(receipt) is dict and digest(receipt) == digest(receipt_record(document, plan)),
             "row_codec_terminal_receipt_missing_or_invalid")


def verifier_sources():
    import pllm
    sdk = Path(pllm.__file__).parent
    paths = {name: sdk / name[4:] if name.startswith("sdk/") else ROOT / name for name in VERIFIER_SOURCES}
    return {name: hashes([path])[str(path)] for name, path in paths.items()}


def validate_source_manifest(manifest):
    _require(type(manifest) is dict and 0 < len(manifest) <= 64 and all(
        type(path) is str and 0 < len(path) <= 4096 and type(value) is str
        and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        for path, value in manifest.items()), "row_codec_source_manifest_invalid")


def verify_verifier_sources(manifest):
    validate_source_manifest(manifest)
    _require(set(manifest) == set(VERIFIER_SOURCES), "row_codec_verifier_inventory_changed")
    _require(manifest == verifier_sources(), "row_codec_replay_verifier_changed")


def validate_pair_specs(control, candidate, *, max_output_tokens, tiny=False):
    """Exact structural allowlist: codec plus explicit certificate pin, nothing else."""
    cap = 2 if tiny else max_output_tokens
    _require(max_output_tokens in (8, 64) and (not tiny or max_output_tokens == 8),
             "row_codec_workload_invalid")
    left, right = deepcopy(control), deepcopy(candidate)
    for spec, encoding in ((left, "uniform_residues"), (right, "row_residues")):
        _require(spec["schema"] == "pllm.experiment.v2" and spec["budget"] == {
            "max_input_tokens": 128 if tiny else 150, "max_new_tokens": cap, "requests": 1},
            "row_codec_public_bounds_changed")
        boundary = spec["pipeline"]["components"]["boundary"]
        _require(boundary["component"] == "pllm/private-lookup-full-head/v1"
            and boundary["params"].get("head_dispatch") == "parallel"
            and boundary["params"].get("head_output_encoding") == encoding,
            "row_codec_boundary_changed")
    a = left["pipeline"]["components"]["boundary"]["params"]
    b = right["pipeline"]["components"]["boundary"]["params"]
    _require(CERTIFICATE_FIELD not in a and CERTIFICATE_FIELD in b, "row_codec_certificate_pin_missing")
    pin = b.pop(CERTIFICATE_FIELD)
    _require(isinstance(pin, dict) and set(pin) == {"path", "digest", "size_bytes"}
        and isinstance(pin["path"], str) and 0 < len(pin["path"]) <= 4096 and Path(pin["path"]).is_absolute()
        and type(pin["size_bytes"]) is int and 13 <= pin["size_bytes"] <= 1 << 20
        and isinstance(pin["digest"], str) and len(pin["digest"]) == 64
        and all(c in "0123456789abcdef" for c in pin["digest"]), "row_codec_certificate_pin_invalid")
    b["head_output_encoding"] = "uniform_residues"
    _require(left == right, "row_codec_unapproved_experiment_difference")
    return deepcopy(pin)


def matched_suite(*, control, candidate, build, max_output_tokens=8, tiny=False):
    validate_pair_specs(control.to_spec(), candidate.to_spec(), max_output_tokens=max_output_tokens, tiny=tiny)
    cap = 2 if tiny else max_output_tokens
    return BenchmarkSuite([BenchmarkCase(name, BenchmarkTarget.response(exp, backend="native",
        scenario=Scenario.cold(max_output_tokens=cap), wan=WanConditions(), state_audit=True), build)
        for name, exp in zip(NAMES, (control, candidate))],
        Sampling(processes=3, order="balanced", threads=1, repetitions=1, warmups=0,
            timeout_seconds=3600, memory_bytes=8 << 30), "private-boundary-row-codec-v1")


def certificate_accounting(pin, declaration, *, read_artifact=True):
    """Account actual artifact size, not fictitious network observations.

    Missing owners are a declared setup scenario, not discovered cache state.
    A local read proves file bytes only. No publisher traffic is inferred.
    """
    _require(isinstance(declaration, dict) and set(declaration) == {
        "missing_owners", "delivery_locations", "derivation"}, "row_codec_certificate_declaration_invalid")
    owners, locations = declaration["missing_owners"], declaration["delivery_locations"]
    _require(isinstance(owners, list) and len(owners) == len(set(owners))
        and set(owners) <= {"client", "boundary_a", "boundary_b"}
        and isinstance(locations, dict) and set(locations) == {"client", "boundary_a", "boundary_b"}
        and all(isinstance(value, str) and bool(value) for value in locations.values())
        and isinstance(declaration["derivation"], str) and bool(declaration["derivation"]),
        "row_codec_certificate_delivery_invalid")
    if read_artifact:
        path = Path(pin["path"])
        _require(type(pin["size_bytes"]) is int and 13 <= pin["size_bytes"] <= 1 << 20,
                 "row_codec_certificate_size_invalid")
        payload = bounded_read(path, pin["size_bytes"])
        _require(len(payload) == pin["size_bytes"], "row_codec_certificate_size_mismatch")
        _require(hashlib.sha256(payload).hexdigest() == pin["digest"],
                 "row_codec_certificate_digest_mismatch")
    else:
        payload = None
    return {"artifact": deepcopy(pin), "artifact_bytes": pin["size_bytes"],
        "artifact_base64": base64.b64encode(payload).decode("ascii") if payload is not None else None,
        "artifact_bytes_verified": read_artifact, "delivery_locations": deepcopy(locations),
        "declared_missing_owners": list(owners), "extra_setup_fixed_byte_budget": pin["size_bytes"] * len(owners),
        "extra_setup_fixed_byte_budget_origin": "scenario arithmetic; not network measurement",
        "publisher_distribution_network_bytes": None, "publisher_distribution_seconds": None,
        "derivation": declaration["derivation"], "derivation_cpu_seconds": None,
        "cold_network_complete": False, "sota_claim_admissible": False,
        "integration_status": "raw source/layout/frame telemetry required by cohort audit"}


def certificate_evidence(certificate, pin):
    """Authenticate archived public bytes before bounded metadata/layout parsing."""
    encoded = certificate["artifact_base64"]
    _require(type(encoded) is str and len(encoded) <= 4 * (((1 << 20) + 2) // 3),
             "row_codec_certificate_archive_invalid")
    payload = base64.b64decode(encoded, validate=True)
    _require(len(payload) == pin["size_bytes"] and hashlib.sha256(payload).hexdigest() == pin["digest"],
             "row_codec_certificate_archive_mismatch")
    _require(len(payload) >= 13, "row_codec_certificate_envelope_invalid")
    magic, size = struct.unpack_from("<8sI", payload)
    _require(magic == b"PLLMBRC1" and 0 < size <= 16384 and 12 + size < len(payload),
             "row_codec_certificate_envelope_invalid")
    raw = payload[12:12 + size]
    # The certificate schema admits only three shallow objects, no arrays.
    depth = objects = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 93):
            raise Rejected("row_codec_certificate_metadata_invalid")
        elif byte == 123:
            depth += 1
            objects += 1
            _require(depth <= 2 and objects <= 3, "row_codec_certificate_metadata_invalid")
        elif byte == 125:
            depth -= 1

    def unique(pairs):
        result = dict(pairs)
        _require(len(result) == len(pairs), "row_codec_certificate_metadata_invalid")
        return result

    metadata = json.loads(raw, object_pairs_hook=unique)
    _require(type(metadata) is dict and set(metadata) == {"schema", "source", "numeric", "layout",
        "widths_digest", "bounds_digest", "implementation"}
        and metadata["schema"] == "pllm.private_boundary.row_certificate.v1"
        and metadata["numeric"] == CERT_NUMERIC and metadata["layout"] == "zlib-u8-minimal-signed-widths-v1"
        and type(metadata["source"]) is dict and set(metadata["source"]) == set(SOURCE_FIELDS)
        and _sha(metadata["widths_digest"]) and _sha(metadata["bounds_digest"]),
        "row_codec_certificate_metadata_invalid")
    source = metadata["source"]
    _require(all(_sha(source[key]) for key in SOURCE_FIELDS if key.endswith("digest"))
        and all(type(source[key]) is int for key in SOURCE_FIELDS if not key.endswith("digest"))
        and 0 < source["vocabulary"] <= 1 << 20 and 0 < source["hidden"] <= 1 << 20
        and source["activation_bits"] == source["weight_bits"] == 8
        and 0 < source["signed_output_bound"] < 1 << 31, "row_codec_certificate_source_invalid")
    implementation = metadata["implementation"]
    _require(type(implementation) is dict and set(implementation) == {"pllm._native",
        "pllm.runtime.private_boundary_loader", "pllm.runtime.quantization",
        "pllm.runtime.private_boundary_certificate"} and all(_sha(value) for value in implementation.values()),
        "row_codec_certificate_publisher_invalid")
    decoder = zlib.decompressobj()
    widths = decoder.decompress(payload[12 + size:], source["vocabulary"] + 1)
    _require(len(widths) == source["vocabulary"] and decoder.eof and not decoder.unused_data
        and not decoder.unconsumed_tail and all(1 <= value <= 32 for value in widths)
        and hashlib.sha256(widths).hexdigest() == metadata["widths_digest"]
        and max(widths) <= (2 * source["hidden"] * 128 * 127).bit_length(),
        "row_codec_certificate_widths_invalid")
    return metadata, widths


def validate_native_commitments(value):
    # Caller-approved independent native metadata, never copied from the worker
    # whose response is under audit. Needed because snapshot has no separate
    # authoritative native-body -> descriptor/schedule binding.
    fields = {"pipeline_digest", "composition_digest", "schedule_digest", "descriptor_digest", "numeric_body_fingerprint"}
    _require(type(value) is dict and set(value) == set(NAMES) and all(type(arm) is dict
        and set(arm) == fields and all(_sha(v) for v in arm.values()) for arm in value.values()),
        "row_codec_native_commitments_missing_or_invalid")


def _head_evidence(raw, binding, exp, report, approved, pin, native):
    from pllm._benchmark_contract import digest
    from pllm.runtime.private_boundary_transport import BoundaryDescriptor
    _require(type(raw) is dict and set(raw) == {"schema", "descriptor", "descriptor_digest", "layout", "roles"}
        and raw["schema"] == "pllm.private_boundary_head_accounting.v1", "row_codec_head_telemetry_invalid")
    descriptor = BoundaryDescriptor.from_dict(raw["descriptor"])
    desc = descriptor.to_dict()
    _require(desc == raw["descriptor"] and descriptor.digest == raw["descriptor_digest"]
        and native["pipeline_digest"] == exp.pipeline.digest()
        and all(native[key] == desc[key] for key in ("composition_digest", "schedule_digest"))
        and native["descriptor_digest"] == raw["descriptor_digest"]
        and native["numeric_body_fingerprint"] == binding["numeric_body_fingerprint"],
        "row_codec_native_descriptor_binding_mismatch")
    boundary = exp.pipeline.components["boundary"].get_params()
    encoding = boundary["head_output_encoding"]
    _require(desc["source_digest"] == report["configuration"]["source_lock_digest"]
        and desc["head_dispatch"] == boundary["head_dispatch"] == "parallel"
        and descriptor.head_kernel == boundary.get("head_kernel", "existing")
        and desc["head_output_encoding"] == encoding
        and desc["activation_bits"] == desc["weight_bits"] == 8
        and desc["head_stage_id"] == "lm_head" and desc["ring"] == "u32" and desc["wire_bits"] == 32
        and desc["modulus"] == 1 << 32 and desc["leaf_bits"] == 128,
        "row_codec_head_source_numeric_changed")
    metadata, widths = approved
    _require({key: desc[key] for key in SOURCE_FIELDS} == metadata["source"],
             "row_codec_head_certificate_source_mismatch")
    layout = raw["layout"]
    if encoding == "uniform_residues":
        widths = bytes([(2 * desc["hidden"] * 128 * 127).bit_length()]) * desc["vocabulary"]
        certificate = None
        _require(descriptor.head_certificate_digest == "", "row_codec_control_certificate_changed")
    else:
        _require(descriptor.head_certificate_digest == pin["digest"], "row_codec_head_certificate_pin_mismatch")
        certificate = {"digest": pin["digest"], "size_bytes": pin["size_bytes"],
            "metadata": {**metadata, "artifact_digest": pin["digest"],
                "artifact_size_bytes": pin["size_bytes"]}}
    bits = sum(widths)
    expected = {"widths_digest": hashlib.sha256(widths).hexdigest(),
        "width_counts": {str(width): widths.count(width) for width in sorted(set(widths))},
        "total_bits": bits, "payload_bytes_per_role": (bits + 7) // 8,
        "subframe_header_bytes": struct.calcsize("<8sBB32s32s"), "certificate": certificate}
    # Canonical JSON equality also distinguishes booleans from numeric counters.
    _require(digest(layout) == digest(expected), "row_codec_head_layout_mismatch")
    roles = raw["roles"]
    _require(type(roles) is dict and set(roles) == {"boundary_a", "boundary_b"}, "row_codec_head_owners_missing")
    result = {"outer_body_bytes": 0, "subframe_bytes": 0, "header_bytes": 0, "payload_bytes": 0,
        "envelope_bytes": 0, "descriptor_digest": descriptor.digest, "layout": expected}
    selections = len(binding["selections"])
    privacy = report["runs"][0]["privacy"]
    _require(privacy["private_boundary_head_calls"] == selections, "row_codec_head_selection_count_mismatch")
    for role, counters in roles.items():
        _require(type(counters) is dict and set(counters) == {"replies", "outer_body_bytes", "subframe_bytes",
            "header_bytes", "payload_bytes"} and all(type(v) is int and v >= 0 for v in counters.values())
            and counters["replies"] == selections
            and counters["payload_bytes"] == selections * expected["payload_bytes_per_role"]
            and counters["header_bytes"] == selections * expected["subframe_header_bytes"]
            and counters["subframe_bytes"] == counters["payload_bytes"] + counters["header_bytes"]
            and counters["outer_body_bytes"] >= counters["subframe_bytes"]
            and counters["outer_body_bytes"] <= privacy[f"role_link.{role}.online_download_bytes"],
            "row_codec_head_frame_partition_mismatch")
        for key in ("outer_body_bytes", "subframe_bytes", "header_bytes", "payload_bytes"):
            result[key] += counters[key]
    _require(result["outer_body_bytes"] == privacy["private_boundary_head_download_bytes"],
             "row_codec_head_raw_body_mismatch")
    result["envelope_bytes"] = result["outer_body_bytes"] - result["subframe_bytes"]
    result["roles"] = deepcopy(roles)
    return result


def _state_identity(bindings, identity):
    # Codec changes numeric-body fingerprint. Every full-state commitment remains exact.
    state = deepcopy(bindings)
    for binding in state:
        binding.pop("numeric_body_fingerprint")
    common = {key: value for key, value in identity.items() if key != "body"}
    return state, common


def _workload_witness(row, case, report):
    """Replay public identity, never guess the unrevealed input/salt preimage.

    Independent reconstruction intentionally does not call the worker's binding
    builder/finalizer. Hashes establish record consistency, not authentication.
    """
    from pllm._benchmark_contract import digest
    from pllm._benchmark_workload import WORKLOAD_SCHEMA, MEASUREMENT_SCHEMA
    from pllm.runtime.topology_accounting import ACCOUNTING_SCHEMA, accounting_coverage_identity
    execution = row["execution"]
    detail, comparison = execution["details"], execution["comparison"]
    params, config = case.target.to_spec()["parameters"], report["configuration"]
    request, = report["runs"]
    public = digest({"tokens": [request["tokens"]], "scenario": params["scenario"],
        "backend": config["provider_backend"], "wan": config["wan_emulation_digest"],
        "sampling": config["sampling"]})
    binding = detail.get("workload_binding")
    _require(type(binding) is dict and set(binding) == {"schema", "public_workload_digest",
        "private_inputs_commitment", "prompt_commitment", "fixture_digest"}
        and binding["schema"] == WORKLOAD_SCHEMA and all(_sha(value) for key, value in binding.items()
            if key != "schema"), "row_codec_workload_binding_missing_or_invalid")
    _require(config["provider_backend"] == params["backend"] and config["sampling"] == request["sampling"]
        and binding["public_workload_digest"] == public,
        "row_codec_public_workload_digest_mismatch")
    _require(binding["prompt_commitment"] == config["prompt_digest"]
        and binding["fixture_digest"] == execution["fixture"]["digest"] == config["source_lock_digest"],
        "row_codec_workload_prompt_or_fixture_mismatch")
    _require(row["workload_digest"] == comparison["workload_digest"] == digest(binding),
        "row_codec_raw_workload_digest_mismatch")
    target_measurement = digest({"schema": report["schema_version"], "scenario": params["scenario"],
        "cached_checkpoint": True})
    coverage = accounting_coverage_identity(report.get("topology_accounting"))
    if coverage not in {ACCOUNTING_SCHEMA, "historical_coverage_not_recorded"}:
        target_measurement = digest({"response": target_measurement, "body_counter_coverage": coverage})
    target_measurement = digest({"response": target_measurement, "state_audit": "pllm.response_state_audit.v1"})
    expected_measurement = {"schema": MEASUREMENT_SCHEMA, "workload_binding_schema": WORKLOAD_SCHEMA,
        "target_measurement_digest": target_measurement}
    _require(detail.get("workload_measurement") == expected_measurement
        and comparison["measurement_digest"] == digest(expected_measurement),
        "row_codec_workload_measurement_version_mismatch")
    return binding


def _reconcile(bodies):
    paired = []
    for rep in range(3):
        a, b = bodies[CONTROL, rep], bodies[CANDIDATE, rep]
        _require(all(a[key] == b[key] for key in ("prepared", "fixed_work", "boundary_fixed")),
                 "row_codec_fixed_work_changed")
        left, right = a["stage_bodies"], b["stage_bodies"]
        _require(set(left) == set(right) and all(
            {k: v for k, v in left[stage].items() if k != "inference->client"}
            == {k: v for k, v in right[stage].items() if k != "inference->client"} for stage in left),
            "row_codec_fixed_stage_work_changed")
        bound = a["reply_metadata_delta_bound_bytes"]
        _require(b["reply_metadata_delta_bound_bytes"] == bound and sum(
            abs(right[stage]["inference->client"] - left[stage]["inference->client"])
            for stage in left) <= bound, "row_codec_reply_metadata_bound_exceeded")
        total, head, control, prepared = (b[key] - a[key] for key in
            ("total", "head_reply", "controls", "prepared_reply"))
        codec = {key + "_delta_bytes": b["head_codec"][key] - a["head_codec"][key]
            for key in ("payload_bytes", "header_bytes", "envelope_bytes")}
        _require(head == sum(codec.values()), "row_codec_head_arm_delta_mismatch")
        paired.append({"replicate": rep, "covered_body_delta_bytes": total,
            "head_reply_delta_bytes": head, "lifecycle_control_delta_bytes": control,
            "prepared_stage_reply_delta_bytes": prepared,
            "prepared_stage_reply_metadata_compatibility_bound_bytes": bound,
            "unattributed_bootstrap_and_metadata_delta_bytes": total - head - control - prepared,
            "head_codec_partition": codec})
    return paired


def audit(document, *, max_output_tokens=8, tiny=False, certificate=None, native_commitments=None,
          source_manifest=None):
    """Read-only, single invocation audit. No historical evidence or repaired totals."""
    try:
        from pllm.metrics import BenchmarkRun
        BenchmarkRun(document)
        suite = BenchmarkSuite.from_spec(document["suite"])
        _require(tuple(c.name for c in suite.cases) == NAMES, "row_codec_cells_changed")
        experiments = [Experiment.from_spec(c.target.to_spec()["parameters"]["model"]["experiment"])
            for c in suite.cases]
        pin = validate_pair_specs(experiments[0].to_spec(), experiments[1].to_spec(),
            max_output_tokens=max_output_tokens, tiny=tiny)
        _require(certificate is not None, "row_codec_frozen_certificate_missing")
        approved = certificate_evidence(certificate, pin)
        validate_native_commitments(native_commitments)
        source_manifest = verifier_sources() if source_manifest is None else source_manifest
        verify_verifier_sources(source_manifest)
        expected = matched_suite(control=experiments[0], candidate=experiments[1], build=suite.cases[0].build,
            max_output_tokens=max_output_tokens, tiny=tiny)
        _require(suite.to_spec() == expected.to_spec() and suite.cases[0].build.expected_digest,
                 "row_codec_suite_contract_changed")
        _require(document["complete"] is True and len(document["results"]) == 6, "row_codec_incomplete_cohort")
        cohort = document["cohort"]
        _require(isinstance(cohort, str) and len(cohort) == 64 and all(c in "0123456789abcdef" for c in cohort),
                 "row_codec_private_cohort_missing")
        cases = dict(zip(NAMES, suite.cases))
        samples, bodies, works, cpus, identities, seen = {}, {}, {}, {}, {}, set()
        state = identity = environment = metric_contracts = None
        private_inputs_commitment = None
        arm_contracts = {}
        checked = 0
        cap = 2 if tiny else max_output_tokens
        # Match actual balanced SDK schedule; do not pretend it is AB/BA/AB.
        schedule = [(CONTROL, 0), (CANDIDATE, 0), (CANDIDATE, 1), (CONTROL, 1),
                    (CANDIDATE, 2), (CONTROL, 2)]
        for row, scheduled in zip(document["results"], schedule):
            execution = row["execution"]
            name, rep = execution["case"]["name"], execution["replicate"]
            _require((name, rep) == scheduled and (name, rep) not in seen and row["status"] == "completed"
                and execution["case"] == cases[name].to_spec() and execution["cohort"] == cohort
                and execution["suite_digest"] == suite.digest and execution["sampling"] == suite.sampling.to_spec()
                and execution["build"]["digest"] == suite.cases[0].build.expected_digest
                and execution["isolation"]["fresh_process"] is True
                and execution["isolation"]["controller_included"] is False, "row_codec_changed_or_mixed_cell")
            seen.add((name, rep))
            _require(all(execution["build"]["attributes"]["sdk_sources"].get(path)
                == source_manifest["sdk/" + path] for path in SDK_SOURCES), "row_codec_sdk_verifier_build_mismatch")
            report, metrics, bindings, current, work = confirmation._state_and_memory(row, cap, tiny)
            _require(report["runs"][0]["model_fingerprint"] == current["body"],
                     "row_codec_raw_numeric_body_mismatch")
            confirmation._checked_measurements(row)
            normalized, common_identity = _state_identity(bindings, current)
            _require((state is None or state == normalized) and (identity is None or identity == common_identity),
                     "row_codec_source_history_logits_kv_rng_output_mismatch")
            state, identity = normalized, common_identity
            checked += sum(s["checked_values"] for s in bindings[0]["selections"])
            screen._wire(report, metrics, True)
            exp = experiments[NAMES.index(name)]
            _require(report["experiment"]["pipeline_digest"] == exp.pipeline.digest()
                and set(report["configuration"]["roles"]) == ROLES, "row_codec_executed_pipeline_changed")
            common, arm = confirmation._contracts(row, cases[name], exp, cap, work)
            workload_binding = _workload_witness(row, cases[name], report)
            _require(private_inputs_commitment is None
                or private_inputs_commitment == workload_binding["private_inputs_commitment"],
                "row_codec_private_inputs_commitment_changed")
            private_inputs_commitment = workload_binding["private_inputs_commitment"]
            _require(environment is None or common["environment"] == environment, "row_codec_environment_changed")
            _require(name not in arm_contracts or arm_contracts[name] == arm, "row_codec_arm_numeric_contract_changed")
            _require(metric_contracts is None or metric_contracts == arm["metrics"], "row_codec_metric_contract_changed")
            environment, metric_contracts, arm_contracts[name] = common["environment"], arm["metrics"], arm
            confirmation._material(report)
            # EOS-aware version also accepts cap runs and preserves unused material.
            body = confirmation._eos_body(report, metrics, True,
                chunk_rows=exp.pipeline.components["linear"].get_params()["prefill_chunk_rows"])
            body["head_codec"] = _head_evidence(execution["details"]["private_boundary_head"],
                bindings[0], exp, report, approved, pin, native_commitments[name])
            _require(body["boundary_fixed"]["private_boundary_head_calls"] == work["sampled_selections"]
                and body["boundary_fixed"]["private_boundary_lookup_rows"]
                == current["tokens"]["input_tokens"] + work["executed_decode_steps"], "row_codec_executed_work_changed")
            metrics["topology.cold_cpu"], cpus[name, rep] = kernel._cold_cpu(report)
            bodies[name, rep], samples[name, rep] = body, metrics
            works[name, rep] = work | body["fixed_work"]
            identities[name, rep] = {"numeric_body_fingerprint": current["body"],
                "comparison": common["comparison"], "build": execution["build"],
                "fixture": execution["fixture"], "environment": common["environment"]}
            identities[name, rep]["workload_binding"] = deepcopy(workload_binding)
        groups = {}
        for name in NAMES:
            rows = [samples[name, rep] for rep in range(3)]
            peak = max(row["process.peak_rss"] for row in rows)
            groups[name] = {"samples": rows, "maximum_client_peak_bytes": peak,
                "observed_tiers": {tier: peak <= ceiling for tier, ceiling in tiers.TIERS.items()},
                "medians": {metric: statistics.median(row[metric] for row in rows)
                    if all(row[metric] is not None for row in rows) else None for metric in METRICS},
                "executed_work": [works[name, rep] for rep in range(3)],
                "cold_first_cpu_seconds_by_role": [cpus[name, rep] for rep in range(3)],
                "head_codec": [bodies[name, rep]["head_codec"] for rep in range(3)],
                "identities": [identities[name, rep] for rep in range(3)]}
        return {"schema": "pllm.private_boundary_row_codec.v1", "complete": True, "tiny": tiny,
            "cohort": cohort, "clients": 6, "replicates": 3, "output_cap": cap, "identity": identity,
            "workload_binding": {"schema": workload_binding["schema"],
                "private_inputs_commitment": private_inputs_commitment,
                "scope": "public record consistency and opaque cohort equality, not worker authentication or proof of private inputs"},
            "checked_values": checked, "all_logits_active_kv_positions_selections_rng_and_outputs_match": True,
            "state_commitment": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
            "maximum_observed_swap_growth_bytes": 0, "groups": groups,
            "comparisons": {CANDIDATE: {CONTROL: kernel._compare(groups[CONTROL], groups[CANDIDATE])}},
            "body_reconciliation": {CANDIDATE: _reconcile(bodies)},
            "codec_source_certification_verified": True, "cold_network_complete": False, "sota_claim_admissible": False,
            "publisher_implementation": approved[0]["implementation"],
            "native_head_commitments": deepcopy(native_commitments),
            "scope": {"request": f"source-present cold {'128' if tiny else '150'}+{cap}; natural EOS",
                "client": "whole fresh SDK/supervisor/dashboard/relay/audit worker lifetime; idle launcher excluded",
                "wire": "shared100/40 native directed TCP streams, not physical wire",
                "cpu": "aggregate five-owner CPU through first response, not cleanup",
                "serializer": "head/control/metadata deltas attributed separately; residual retained, never assumed zero",
                "limits": "trusted publisher assertion, not cryptographic quantization proof; publisher distribution/derivation unmeasured"}}
    except Rejected:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError, zlib.error) as exc:
        raise Rejected("row_codec_missing_or_invalid_evidence") from exc
