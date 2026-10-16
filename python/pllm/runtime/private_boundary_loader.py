"""Head-only source import for the explicitly selected coupled boundary.

Source hashing streams checkpoint files; semantic preflight reads tensor headers.
Only the canonical tied head is quantized/compiled. No decoder, token-table
transpose, client tensor array, or full-table interleaving is materialized here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Callable

from pllm.configuration import Experiment
from pllm.model_loader import ResolvedModel, resolve_model
from pllm.modeling import DecoderRuntimeSchedule, ModelPlan, lower_model
from pllm.roles import PrivateLookupAndFullHead

from .private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, MAX_HEADS, MAX_LOOKUP_ROWS,
    ROLES, canonical, digest, integer,
)
from .safetensors_store import SafeTensorStore
from .semantic_tensors import preflight_semantic_checkpoint


@dataclass(frozen=True, slots=True)
class LoadedPrivateBoundary:
    source: ResolvedModel
    plan: ModelPlan
    schedule: DecoderRuntimeSchedule
    head: Any
    tokenizer: dict[str, Any]
    config: dict[str, Any]


def load_private_boundary(
    experiment: Experiment, *, cache_dir: str | Path | None = None,
    _admit_import: Callable | None = None,
) -> LoadedPrivateBoundary:
    """Resolve original source lock and import one semantically declared head.

    This is source admission, not decoder-session admission. The worker factory
    still requires an independently validated descriptor and admission callback.
    """
    import json

    from pllm.profiles import resolve_runtime_composition

    from .models import StageSpec
    from .transformer_engine import MaskedTransformerEngine

    if _admit_import is not None:
        # Metadata-only publication preflight must not materialize Model.tiny()
        # or download an absent source before its tensor import is reserved.
        model = experiment.pipeline.model
        if (model.kind not in {"huggingface", "safetensors", "vllm", "mlx", "mlx-lm"}
                or (not Path(model.source).expanduser().is_dir() and not model.local_files_only)):
            raise BoundaryError("publication requires an existing local or explicitly offline cached checkpoint")
    experiment.resolve()
    options = resolve_runtime_composition(experiment.pipeline)
    if options is None or not getattr(options, "private_boundary", False):
        raise BoundaryError("private boundary requires explicit coupled-boundary selection")
    if experiment.deployment.kind == "network":
        raise BoundaryError("UNSUPPORTED_NETWORK_GRAPH: private boundary supports local children only")
    if (options.weight_bits, options.activation_bits) != (8, 8):
        raise BoundaryError("private boundary source import initially requires W8A8")
    source = resolve_model(experiment.pipeline.model, cache_dir=cache_dir)
    if source.path is None or source.source_lock_digest is None or source.checkpoint_digest is None:
        raise BoundaryError("private boundary requires a locked local checkpoint")
    raw_config = json.loads((source.path / "config.json").read_text(encoding="utf-8"))
    budget = experiment.budget
    plan = lower_model(raw_config, batch=1, max_input_tokens=budget.max_input_tokens,
                       max_new_tokens=budget.max_new_tokens)
    schedule = plan.runtime_schedule(experiment.pipeline)
    if type(schedule.to_dict().get("private_boundary")) is not dict:
        raise BoundaryError("native private boundary contract is absent")
    store = SafeTensorStore(source.path)
    preflight_semantic_checkpoint(plan, store)
    config = raw_config.get("text_config", raw_config)
    if not source.manifest.tied_embeddings or config.get("tie_word_embeddings") is not True:
        raise BoundaryError("private boundary requires a tied token/head checkpoint")
    head_spec = None
    for phase in ("prefill", "decode"):
        operations = plan.to_dict()[phase]["operations"]
        by_id = {row["id"]: row for row in operations}
        tokens = [row for row in operations if row["operator"] == "token_lookup"]
        heads = [row for row in operations if row["operator"] == "output_head"]
        if len(tokens) != 1 or len(heads) != 1:
            raise BoundaryError("private boundary requires one dense token table and full head")
        token, head = tokens[0], heads[0]
        attrs = head["attributes"]
        if (attrs.get("weight") != token["attributes"].get("weight")
                or attrs.get("bias") is not None or len(head["inputs"]) != 1
                or head["inputs"][0] not in by_id):
            raise BoundaryError("private boundary token/head are not a canonical bias-free tied matrix")
        hidden = by_id[head["inputs"][0]]["output_shape"][-1]
        vocabulary = head["output_shape"][-1]
        if (hidden != source.manifest.hidden_size or vocabulary != source.manifest.vocab_size
                or token["output_shape"][-1] != hidden):
            raise BoundaryError("private boundary token/head geometry differs")
        spec = StageSpec(id="lm_head", op="lm_head", in_features=hidden,
                         out_features=vocabulary, weight_keys=(attrs["weight"],),
                         role="lm_head", weight_bits=8, activation_bits=8)
        if head_spec is not None and spec != head_spec:
            raise BoundaryError("private boundary head changes between decoder phases")
        head_spec = spec
    if head_spec is None:
        raise BoundaryError("private boundary has no admitted head")
    manifest = replace(source.manifest, metadata={**source.manifest.metadata,
                                               "stage_origin": "semantic_schedule_v1"})
    # The publisher acquires an outer-lifetime reservation here, after all
    # source/semantic/header checks and before importer/native or tensor owners.
    # The callback must keep its owner alive through the subsequent publication.
    if _admit_import is not None:
        _admit_import(source, head_spec, store)
    kernels = experiment.pipeline.components["kernels"]
    importer = MaskedTransformerEngine(threads=kernels.params.get("threads", 1),
                                      weight_bits=8, activation_bits=8)
    # Deliberately never call importer.load(): that loads a decoder graph.
    try:
        head = importer._load_stage(store, head_spec, manifest)
        if head.bias is not None or head.compiled_weight is None or head.compiled_weight._matrix is None:
            raise BoundaryError("private boundary requires an immutable native bias-free head")
        tokenizer = importer._load_tokenizer_descriptor(source.path, manifest, dict(config))
    finally:
        store.clear_cache()
    return LoadedPrivateBoundary(source, plan, schedule, head, tokenizer, raw_config)


ContractFactory = Callable[
    [LoadedPrivateBoundary, Experiment],
    tuple[BoundaryDescriptor, Callable[[BoundaryDescriptor, BoundaryAdmission], BoundaryAdmission]],
]


def build_private_boundary_contract(loaded: LoadedPrivateBoundary, experiment: Experiment):
    """Derive descriptor and tighter request admission from native source schedule.

    Native lowering admits the decoder structure. This bridge checks its exact
    boundary contract against imported head/source, never against client claims.
    """
    import numpy as np
    from pllm import _native

    plan = loaded.plan.to_dict()
    schedule = loaded.schedule.to_dict()
    contract = schedule.get("private_boundary")
    if type(contract) is not dict:
        raise BoundaryError("native private boundary contract is absent")
    boundary = experiment.pipeline.components.get("boundary")
    try:
        if boundary is None or boundary.component != PrivateLookupAndFullHead.descriptor.component:
            raise ValueError("private boundary component is absent")
        boundary_options = PrivateLookupAndFullHead.from_params(boundary.params).to_spec()["params"]
    except ValueError as exc:
        raise BoundaryError("private boundary options are unsupported") from exc
    max_input = integer(plan["prefill"]["query_sequence"], 1, MAX_LOOKUP_ROWS, "plan input bound")
    # The prefill head produces token one. Only heads 2..N append decode rows.
    max_key = integer(plan["decode"]["maximum_key_sequence"],
                      max_input, MAX_LOOKUP_ROWS, "plan key bound")
    max_new = integer(max_key - max_input + 1,
                      1, MAX_HEADS, "plan generation bound")
    if (max_input != experiment.budget.max_input_tokens
            or max_new != experiment.budget.max_new_tokens
            or plan["prefill"]["maximum_key_sequence"] != max_input
            or plan["decode"]["query_sequence"] != 1
            or schedule.get("model_plan_digest") != loaded.plan.digest
            or schedule.get("model_config_digest") != plan["config_digest"]
            or loaded.schedule.composition_digest != experiment.pipeline.digest()):
        raise BoundaryError("native private boundary plan or composition differs")
    head = loaded.head
    hidden, vocabulary = head.spec.in_features, head.spec.out_features
    if boundary_options.get("head_kernel") == "byte_dot" and (
            hidden > 131071 or vocabulary * hidden > 256 << 20):
        raise BoundaryError("private boundary byte-dot geometry exceeds native bounds")
    theoretical_bound = hidden * 127 * 127
    ring_bits = next((bits for bits in (16, 24, 32)
                      if theoretical_bound < 1 << (bits - 1)), None)
    if ring_bits is None or not 0 <= head.signed_output_bound <= theoretical_bound:
        raise BoundaryError("native private boundary signed head bound differs")
    phases = {}
    for phase in ("prefill", "decode"):
        operations = plan[phase]["operations"]
        lookup = [row for row in operations if row["operator"] == "token_lookup"]
        heads = [row for row in operations if row["operator"] == "output_head"]
        if len(lookup) != 1 or len(heads) != 1:
            raise BoundaryError("native private boundary operation ownership differs")
        lookup, output = lookup[0], heads[0]
        if (lookup["attributes"].get("weight") != head.spec.weight_keys[0]
                or output["attributes"].get("weight") != head.spec.weight_keys[0]
                or lookup["output_shape"][-1] != hidden
                or output["output_shape"][-1] != vocabulary):
            raise BoundaryError("native private boundary head identity differs")
        head_rows = 1 if len(output["output_shape"]) == 2 else output["output_shape"][1]
        if head_rows != 1:
            raise BoundaryError("private boundary network path requires final-row head only")
        phases[phase] = {
            "lookup_operation": lookup["id"], "head_operation": output["id"],
            "lookup_rows": max_input if phase == "prefill" else 1, "head_rows": 1,
        }
    expected = {
        "schema_version": "pllm.private_boundary.v1", "model_plan_digest": loaded.plan.digest,
        "model_config_digest": plan["config_digest"],
        "composition_digest": loaded.schedule.composition_digest,
        "weight_id": head.spec.weight_keys[0], "vocabulary": vocabulary, "hidden_width": hidden,
        "owners": list(ROLES), "lookup_protocol": "packed128_xor_dpf_v2",
        "head_protocol": "two_party_additive_full_head_v1", "weight_bits": 8,
        "activation_bits": 8, "head_ring_bits": ring_bits,
        "head_signed_output_bound": theoretical_bound,
        "embedding_numeric": "tied_per_token_row_f32_v1",
        "head_numeric": "ordinary_symmetric_per_row_integer_then_f32_v1",
        "lookup_batch_bound": 64, "lookup_query_bound": max_key,
        "matrix_bytes": vocabulary * hidden, "scale_bytes": vocabulary * 4,
        "lookup_scratch_bytes": vocabulary * 64, "client_weight_bytes": 0, **phases,
        **boundary_options,
    }
    if canonical(contract) != canonical(expected):
        raise BoundaryError("native private boundary contract differs from imported source/head")
    if contract["lookup_query_bound"] > MAX_LOOKUP_ROWS:
        raise BoundaryError("native private boundary lookup bound exceeds protocol")
    matrix = head.compiled_weight
    weight_digest = hashlib.sha256(memoryview(matrix.weight_view()).cast("B")).hexdigest()
    if weight_digest != head.weight_digest:
        raise BoundaryError("native private boundary head weight identity differs")
    scales = np.asarray(head.weight_scales)
    if (scales.dtype != np.float32 or scales.shape != (vocabulary,)
            or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
        raise BoundaryError("native private boundary head scales differ")
    scale_bytes = scales.astype("<f4", copy=False).tobytes()
    constructor = getattr(_native.PrivatePageServer, "from_matrix", None)
    if constructor is None:
        raise BoundaryError("native shared-matrix private lookup API unavailable")
    lookup_root = constructor(matrix._matrix, scale_bytes, 0)
    lookup_digest, lookup_vocabulary, record_bytes = lookup_root.descriptor()
    del lookup_root
    if (lookup_vocabulary, record_bytes) != (vocabulary, hidden + 4):
        raise BoundaryError("native private lookup geometry differs")
    descriptor = BoundaryDescriptor(
        source_digest=loaded.source.source_lock_digest,
        checkpoint_digest=loaded.source.checkpoint_digest,
        model_config_digest=plan["config_digest"], composition_digest=loaded.schedule.composition_digest,
        schedule_digest=loaded.schedule.digest, head_weight_digest=weight_digest,
        head_scales_digest=hashlib.sha256(scale_bytes).hexdigest(), lookup_digest=lookup_digest.hex(),
        vocabulary=vocabulary, hidden=hidden, signed_output_bound=head.signed_output_bound,
        head_dispatch=contract.get("head_dispatch", "serial"),
        head_output_encoding=contract.get("head_output_encoding", "u32"),
        head_kernel=contract.get("head_kernel", "existing"),
        head_certificate_digest=contract.get("head_certificate", {}).get("digest", ""),
    )
    if descriptor.head_output_encoding == "row_residues":
        from .private_boundary_certificate import import_certificate
        import_certificate(boundary_options["head_certificate"], descriptor, weights=matrix.weight_view())
    plan_digest = loaded.plan.digest

    def admit(actual, requested):
        if (type(actual) is not BoundaryDescriptor or actual != descriptor
                or type(requested) is not BoundaryAdmission
                or requested.descriptor_digest != descriptor.digest
                or requested.plan_digest != plan_digest):
            raise BoundaryError("private boundary source/plan/head admission differs")
        heads = integer(requested.head_calls, 1, max_new, "head calls")
        rows = integer(requested.lookup_rows, 1, min(MAX_LOOKUP_ROWS, contract["lookup_query_bound"]),
                       "lookup rows")
        input_rows = integer(rows - heads + 1, 1, max_input, "input rows")
        # Independently derive the regular final-row-only execution demand.
        return BoundaryAdmission(descriptor.digest, plan_digest, input_rows + heads - 1, heads)

    return descriptor, admit


def create_private_boundary_role_app(
    experiment: Experiment, role_id: str, api_key: str, peer_credential_digest: str, *,
    contract_factory: ContractFactory | None = None, cache_dir: str | Path | None = None,
):
    """Join source import to worker only through a trusted compiled-contract bridge.

    Default bridge derives actual source/head commitments from native schedule.
    Request-declared digests are never admission evidence.
    """
    if role_id not in ROLES:
        raise BoundaryError("unknown private boundary role")
    if type(api_key) is not str or len(api_key) < 16:
        raise BoundaryError("boundary own credential is invalid")
    digest(peer_credential_digest, "peer credential digest")
    if hashlib.sha256(api_key.encode()).hexdigest() == peer_credential_digest:
        raise BoundaryError("boundary credentials must be distinct")
    loaded = load_private_boundary(experiment, cache_dir=cache_dir)
    expected, trusted_admit = build_private_boundary_contract(loaded, experiment)
    descriptor, admit = ((expected, trusted_admit) if contract_factory is None
                         else contract_factory(loaded, experiment))
    if type(descriptor) is not BoundaryDescriptor or descriptor != expected or not callable(admit):
        raise BoundaryError("compiled private boundary differs from imported source")
    from .private_boundary_worker import create_private_boundary_worker_app

    app = create_private_boundary_worker_app(
        descriptor=descriptor, head=loaded.head.compiled_weight,
        head_scales=loaded.head.weight_scales, role_id=role_id,
        api_key=api_key, peer_credential_digest=peer_credential_digest,
        admit=lambda actual, requested: admit(actual, trusted_admit(actual, requested)),
        head_certificate=experiment.pipeline.components["boundary"].params.get("head_certificate"),
    )
    app.state.private_boundary_source = loaded
    return app
