"""Bounded research-only signed provider artifact; no production protocol changes.

Publisher verifies generated BF16 bytes. Cold reader trusts an independently
configured publisher key and binding; it cannot revalidate absent BF16 bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

MAGIC = b"PLLM-RESEARCH-PROVIDER-1\0"
MAX_BYTES = 1 << 20
DOMAIN = b"pllm.research.provider-artifact.v1\0"
STAGES = (("token_lookup", 7, 4), ("layers.0.qkv", 6, 4),
          ("layers.0.down", 4, 6), ("lm_head", 7, 4))
ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def test_key():
    """Public deterministic fixture key: never a deployable trust root."""
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"public fixture key only").digest())


def public_key():
    return test_key().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def generated_source():
    """BF16 exactly representable values; complete source bytes, not real model."""
    tensors, layout, raw = {}, [], bytearray()
    for index, (sid, rows, cols) in enumerate(STAGES):
        value = ((np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
                  + index * 11) % 23 - 11) / np.float32(8)
        bf16 = (value.view(np.uint32) >> 16).astype("<u2").tobytes()
        offset = len(raw)
        raw.extend(bf16)
        tensors[sid] = (np.frombuffer(bf16, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
        layout.append({"id": sid, "shape": [rows, cols], "offset": offset,
                       "bytes": len(bf16), "dtype": "BF16", "transpose": False})
    files = {"weights.bf16": bytes(raw), "config.json": canonical({"model": "tiny-research", "hidden": 4}),
             "tokenizer.json": canonical({"tokens": ["<pad>", "a", "b", "c", "d", "e", "f"]}),
             "norm.f32": np.array([1, 1.25, .75, 1], dtype="<f4").tobytes(),
             "bias.f32": np.array([0, .125, -.125, 0, .25, -.25], dtype="<f4").tobytes()}
    witness = {"schema": "research.source-witness.v1", "repo": "generated-not-hub",
               "revision": "fixture-v1", "layout": layout,
               "files": [{"path": name, "size": len(data), "sha256": sha(data)}
                         for name, data in sorted(files.items())]}
    return files, tensors, witness


def verify_source(files, witness):
    expected = witness["files"]
    if set(files) != {row["path"] for row in expected}:
        raise ValueError("source file set")
    for row in expected:
        data = files[row["path"]]
        if len(data) != row["size"] or sha(data) != row["sha256"]:
            raise ValueError("source content")


def runtime_identity():
    # Installed APIs only. No upstream runtime imports, builds, or installs.
    import pllm._native as native
    import pllm.runtime.native as kernels
    import pllm.runtime.quantization as quantization
    return {"native": sha(Path(native.__file__).read_bytes()),
            "native_wrapper": sha(Path(kernels.__file__).read_bytes()),
            "quantizer": sha(Path(quantization.__file__).read_bytes())}


def seal(manifest, payload, key=None):
    header = canonical(manifest)
    signature = (key or test_key()).sign(DOMAIN + header)
    return MAGIC + struct.pack("<I", len(header)) + header + signature + payload


def split(blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError("artifact bytes-like type")
    # Buffer byte count, not len(view): typed/multidimensional views may report
    # element counts, and subclasses may override __len__. Reject before copy.
    with memoryview(blob) as view:
        if view.nbytes > MAX_BYTES or view.nbytes < len(MAGIC) + 4:
            raise ValueError("artifact framing")
    blob = bytes(blob)
    if len(blob) > MAX_BYTES or not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 4:
        raise ValueError("artifact framing")
    length = struct.unpack_from("<I", blob, len(MAGIC))[0]
    start = len(MAGIC) + 4
    end = start + length
    if length > MAX_BYTES or end + 64 > len(blob):
        raise ValueError("artifact truncation")
    header = blob[start:end]
    manifest = json.loads(header)
    if canonical(manifest) != header:
        raise ValueError("noncanonical manifest")
    return manifest, header, blob[end:end + 64], blob[end + 64:]


def publish(role):
    from pllm.runtime.quantization import quantize_weight_per_row
    if role not in {"inference", "preparation"}:
        raise ValueError("role")
    files, tensors, witness = generated_source()
    verify_source(files, witness)
    stages, objects, payload = [], [], bytearray()

    def add(name, data, dtype, shape):
        objects.append({"name": name, "offset": len(payload), "bytes": len(data),
                        "sha256": sha(data), "dtype": dtype, "shape": shape})
        payload.extend(data)

    for sid, rows, cols in STAGES:
        retained = role == "inference" or sid not in {"token_lookup", "lm_head"}
        q = quantize_weight_per_row(tensors[sid], bits=8)
        weight, scales = q.values.tobytes(), q.scales.astype("<f4").tobytes()
        stages.append({"id": sid, "shape": [rows, cols], "weight_bits": 8,
                       "activation_bits": 8, "retained": retained,
                       "weight_digest": sha(weight), "scales_digest": sha(scales),
                       "source": witness["layout"][len(stages)]})
        if retained:
            add(sid + "/weight", weight, "i1", [rows, cols])
            add(sid + "/scales", scales, "<f4", [rows])
    # Exact metadata witnesses charged to both roles. Local tensors and tokenizer
    # are delivered to inference, which publishes the public client bundle.
    if role == "inference":
        for name in ("config.json", "tokenizer.json", "norm.f32", "bias.f32"):
            add(name, files[name], "bytes", [len(files[name])])
    binding = {"role": role, "source_witness_digest": sha(canonical(witness)),
               "model_plan": sha(canonical([s[0] for s in STAGES])),
               "composition": "research-w8a8-client-boundaries-v1",
               "compiler": "installed-per-row-quantizer/no-portable-machine-code",
               "execution": runtime_identity()}
    manifest = {"schema": "research.provider-artifact.v1", "binding": binding,
                "source_witness": witness, "stages": stages, "objects": objects,
                "payload_bytes": len(payload)}
    return seal(manifest, bytes(payload)), binding


class Imported:
    def __init__(self, objects, stages):
        from pllm.runtime.native import MaskedGEMM
        self.objects, self.stages = objects, stages
        self.kernel, self.matrices = None, {}
        self.closed = False
        try:
            self.kernel = MaskedGEMM(threads=1, build=False)
            if not self.kernel.native:
                raise ValueError("installed native executor required")
            for stage in stages:
                if stage["retained"]:
                    self.matrices[stage["id"]] = self.kernel.compile(
                        np.frombuffer(objects[stage["id"] + "/weight"], np.int8).reshape(stage["shape"]))
        except BaseException:
            self.close()
            raise

    def execute(self, sid, inputs):
        if self.closed:
            raise ValueError("artifact retired")
        return self.matrices[sid].clear(inputs)

    def close(self):
        self.closed = True
        self.matrices.clear()
        self.objects.clear()
        # MaskedGEMM has no close API: release its executor owner after matrices.
        self.kernel = None
        self.stages = []


def import_artifact(blob, trusted_key, expected_binding):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if expected_binding.get("role") not in {"inference", "preparation"}:
        raise ValueError("unsupported expected role")
    manifest, header, signature, payload = split(blob)
    Ed25519PublicKey.from_public_bytes(trusted_key).verify(signature, DOMAIN + header)
    if set(manifest) != {"schema", "binding", "source_witness", "stages", "objects", "payload_bytes"}:
        raise ValueError("manifest fields")
    if manifest["schema"] != "research.provider-artifact.v1" or manifest["binding"] != expected_binding:
        raise ValueError("binding mismatch")
    if sha(canonical(manifest["source_witness"])) != expected_binding["source_witness_digest"]:
        raise ValueError("source witness mismatch")
    if expected_binding["execution"] != runtime_identity():
        raise ValueError("runtime mismatch")
    if len(payload) != manifest["payload_bytes"]:
        raise ValueError("payload length")
    stages = manifest["stages"]
    if [s["id"] for s in stages] != [s[0] for s in STAGES]:
        raise ValueError("stage order")
    required, dimensions = [], {}
    for stage, (sid, rows, cols), source in zip(stages, STAGES, manifest["source_witness"]["layout"], strict=True):
        retained = expected_binding["role"] == "inference" or sid not in {"token_lookup", "lm_head"}
        if (stage["shape"] != [rows, cols] or stage["source"] != source or
                stage["weight_bits"] != 8 or stage["activation_bits"] != 8 or stage["retained"] != retained):
            raise ValueError("stage metadata")
        if retained:
            required.extend((sid + "/weight", sid + "/scales"))
            dimensions[sid + "/weight"] = ("i1", [rows, cols], rows * cols)
            dimensions[sid + "/scales"] = ("<f4", [rows], rows * 4)
    if expected_binding["role"] == "inference":
        required.extend(("config.json", "tokenizer.json", "norm.f32", "bias.f32"))
    if [o["name"] for o in manifest["objects"]] != required:
        raise ValueError("object order or ownership")
    objects, offset = {}, 0
    for obj in manifest["objects"]:
        if type(obj["bytes"]) is not int or obj["bytes"] < 0 or obj["offset"] != offset:
            raise ValueError("object extent")
        if obj["name"] in dimensions and (obj["dtype"], obj["shape"], obj["bytes"]) != dimensions[obj["name"]]:
            raise ValueError("object tensor metadata")
        end = offset + obj["bytes"]
        data = payload[offset:end]
        if len(data) != obj["bytes"] or sha(data) != obj["sha256"]:
            raise ValueError("object digest")
        objects[obj["name"]] = data
        offset = end
    if offset != len(payload):
        raise ValueError("unclaimed payload")
    for stage in stages:
        if stage["retained"]:
            if sha(objects[stage["id"] + "/weight"]) != stage["weight_digest"] or sha(objects[stage["id"] + "/scales"]) != stage["scales_digest"]:
                raise ValueError("stage digest")
            scales = np.frombuffer(objects[stage["id"] + "/scales"], "<f4")
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("stage scales")
    if expected_binding["role"] == "inference":
        witness_files = {r["path"]: r for r in manifest["source_witness"]["files"]}
        for name in required[-4:]:
            if sha(objects[name]) != witness_files[name]["sha256"] or len(objects[name]) != witness_files[name]["size"]:
                raise ValueError("local source content")
    return Imported(objects, stages)


def run():
    from pllm.runtime.quantization import quantize_weight_per_row, quantize_activation_per_row
    files, tensors, witness = generated_source()
    samples = []
    for role in ("inference", "preparation"):
        blob, binding = publish(role)
        loaded = import_artifact(blob, public_key(), binding)
        manifest, header, _, payload = split(blob)
        outputs = {}
        for stage in manifest["stages"]:
            if not stage["retained"]:
                continue
            sid = stage["id"]
            q = quantize_weight_per_row(tensors[sid], bits=8)
            assert loaded.objects[sid + "/weight"] == q.values.tobytes()
            assert loaded.objects[sid + "/scales"] == q.scales.astype("<f4").tobytes()
            cols = stage["shape"][1]
            x = quantize_activation_per_row(np.arange(3 * cols, dtype=np.float32).reshape(3, cols) / 7 - 1, bits=8)
            actual = loaded.execute(sid, x.values)
            expected = x.values.astype(np.int64) @ q.values.astype(np.int64).T
            np.testing.assert_array_equal(actual, expected)
            outputs[sid] = sha(actual.astype("<i8").tobytes())
        loaded.close()
        samples.append({"role": role, "artifact_sha256": sha(blob), "artifact_bytes": len(blob),
                        "manifest_bytes": len(header), "framing_and_signature_bytes": len(MAGIC) + 4 + 64,
                        "payload_bytes": len(payload), "object_count": len(manifest["objects"]),
                        "retained_stage_ids": list(outputs), "integer_outputs_sha256": outputs,
                        "binding": binding})
    sources = ["python/pllm/model_loader.py", "python/pllm/runtime/party.py",
               "python/pllm/runtime/servers.py", "python/pllm/runtime/transformer_engine.py",
               "python/pllm/runtime/bundle_artifacts.py", "python/pllm/runtime/model_binding.py",
               "python/pllm/runtime/execution_capsule.py", "scripts/probe_cold_provider_artifacts.py",
               "python/pllm/runtime/safetensors_store.py", "python/pllm/runtime/benchmark_memory.py",
               "tests/test_cold_provider_artifacts.py", "docs/research/cold-provider-artifacts-reference.md"]
    return {"schema": "pllm.cold-provider-artifacts-synthetic.v1", "synthetic": True,
            "source_bytes": sum(map(len, files.values())), "source_witness_bytes": len(canonical(witness)),
            "samples": samples, "both_role_delivery_bytes": sum(s["artifact_bytes"] for s in samples),
            "trust_bootstrap": {"public_key_bytes_once": len(public_key()),
                               "external_expected_bindings_bytes": sum(len(canonical(s["binding"])) for s in samples),
                               "scope": "out-of-band fixture trust configuration; no production attestation chain"},
            "delivery_plus_fixture_trust_bytes": sum(s["artifact_bytes"] + len(canonical(s["binding"]))
                                                      for s in samples) + len(public_key()),
            "publisher_source_verification_bytes": sum(map(len, files.values())),
            "parity": "exact W8A8 values/scales and independent int64 dot products",
            "source_assurance": "publisher attestation; cold importer does not rehash absent BF16",
            "array_bound_bytes": 1 << 20, "timing_claim": None,
            "checks": ["exact_weight_and_scale_bytes", "independent_integer_outputs", "complete_payload_bytes",
                       "publisher_signature_and_external_binding", "cold_import_no_source_or_quantization",
                       "bytes_like_preflight_before_copy", "unsupported_expected_role_rejected",
                       "installed_only_executor_and_success_failure_owner_retirement",
                       "tests_cover_tamper_mismatch_truncation_order_source_substitution_mutation_retirement",
                       "dishonest_trusted_publisher_not_detected_without_derivation_proof"],
            "replay_locks": {name: sha((ROOT / name).read_bytes()) for name in sources}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
