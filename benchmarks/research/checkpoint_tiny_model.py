"""Bounded, checkpoint-specific extractive ranker. No tensor imports at import time.

This is a helper, never a standalone language model. Selected source sentences
remain verbatim and in order; target logits/KV are a changed-input contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = "pllm.checkpoint_tiny_model.v1"
FEATURES = 264
PARAMETER_LIMIT = 1_000_000_000
DATA_REVISION = "240e165ab706d95bd4323653bb92421f446208cf"
DATA_BLOB = "e9a3f913ad1468ebe105b891334ca7b0bc0e2510"
DATA_URL = f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{DATA_REVISION}/dataset/dev-v1.1.json"
PILOT = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    "files": {
        "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    },
}
# Declarative source records, not executable family dispatch. Missing tensor locks
# deliberately block training admission, even when checkpoint runtime exists.
SOURCES = {
    "Dense Qwen2 (Qwen2.5-0.5B-Instruct)": PILOT,
    "Dense Qwen3 (Qwen3-0.6B)": {
        "model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
    "SmolLM2-135M-Instruct (bounded dense bias-free decoder)": {
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "files": {"config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
                  "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"}},
    "Qwen3.5-4B text decoder": {
        "model": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"},
    "Phi-4-mini-instruct": {
        "model": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083"},
    "Gemma 4 E2B text decoder": {
        "model": "google/gemma-4-E2B-it", "revision_prefix_only": "3e22461f"},
    "Gemma 4 E4B text decoder": {
        "model": "google/gemma-4-E4B-it", "revision_prefix_only": "ee0ef602"},
}
POLICY = {
    "split": "sha256(article title), first 8 hex modulo 10: 0..5 train, 6..7 dev, 8..9 heldout",
    "counts": {"train": 8, "dev": 2, "heldout": 4},
    "max_input_tokens": 512, "max_output_tokens": 24, "max_answer_tokens": 16,
    "epochs": 80, "learning_rate": 0.05, "seed": 6104,
    "candidate_policy": "head-half,tail-half,query-overlap-half; sorted original sentence indices",
    "fallback": "once if no EOS or stripped output is not a nonempty substring of retained context",
    "gates": {"minimum_qa_em": 0.75, "max_em_drop_vs_full": 0.0,
              "minimum_stress_em": 1.0, "max_projected_arithmetic_ratio": 0.8},
}

# Checkpoint-level evidence extends adapter-level summaries without editing the
# shared registry or treating one adapter row as every supported public source.
CHECKPOINT_EXTENSIONS = ({
    "checkpoint": "Qwen/Qwen3-4B", "adapter": "pllm.qwen3.v1",
    "source_lock": {"model": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                    "source_lock_digest": "308249a17bc5540401ce8ee373e24232036de08601b13e215fbb312943b8fdf1",
                    "checkpoint_digest": "75a83dc4d370f01ae8e62e060c02d2a16316ef7638f9dfbc3b920be9882bc586"},
    "config_reference": "https://huggingface.co/Qwen/Qwen3-4B/raw/1cfa9a7208912126459214e8b04321603b3df60c/config.json",
    "config_sha256": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "config": {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False,
               "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
               "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560,
               "initializer_range": 0.02, "intermediate_size": 9728, "max_position_embeddings": 40960,
               "max_window_layers": 36, "model_type": "qwen3", "num_attention_heads": 32,
               "num_hidden_layers": 36, "num_key_value_heads": 8, "rms_norm_eps": 1e-6,
               "rope_scaling": None, "rope_theta": 1000000, "sliding_window": None,
               "tie_word_embeddings": True, "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
               "use_cache": True, "use_sliding_window": False, "vocab_size": 151936},
    "requires": ["gqa", "rotary", "rmsnorm", "gated-silu", "kv-cache", "head-width", "qk-norm"],
    "runtime_evidence": "Pinned real-checkpoint native SDK responses, resident/paged client memory and indexed-tokenizer controls; output-text agreement, not full-logit/KV or representative quality. Separate float32-reference quality probe memory-rejected.",
    "references": ["docs/research/qwen3-client-memory.md", "docs/research/qwen3-sdk-artifacts.md",
                   "docs/evidence/research-qwen3-client-memory.json",
                   "docs/evidence/research-qwen3-client-memory-publisher.json",
                   "docs/evidence/research-qwen3-client-memory-quality.json",
                   "docs/evidence/research-qwen3-client-memory-configurations.json",
                   "docs/evidence/research-qwen3-sdk-artifacts.json"],
    "lock_reference": "docs/evidence/research-qwen3-client-memory-publisher.json",
    "quality_reference": "docs/evidence/research-qwen3-client-memory-quality.json",
},)

# Completed parent-controlled run, not regenerated by mechanics/source snapshots.
# Literal result facts avoid circular hashes of this source or its own reports.
COMPLETED_PILOTS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "status": "trained_but_failed_heldout_screen", "role": "extractive_helper_not_standalone",
        "helper_parameters": 264, "heldout_qa_correct": 1, "heldout_qa_count": 4,
        "heldout_copy_order_correct": 2, "heldout_copy_order_count": 2,
        "heldout_full_projected_bytes": 2_662_871_040,
        "heldout_learned_projected_bytes": 2_039_872_512,
        "heldout_fallback_inclusive_projected_bytes": 3_525_417_984,
        "heldout_fallback_ratio": 1.3239161532959554, "screen_pass": False,
        "maximum_swap_growth_bytes": 0, "serving_qualified": False,
        "report_sha256": "f431bf5b83c06ca65ee446ec7e709f020b0e457aeac69993b4b5c58587bc3bcd",
        "artifact_sha256": "29392cbbf430f5a072586cdd68cf414ea915c0fd4055a81e396231037c6a051e",
        "manifest_sha256": "740ca7839ba611223c7f9a5f784628a7f8f4bd765b8901df321ceb7006cfae0b",
        "dependency_provenance": "decoder_probe_support.py omitted from original manifest; parent verified unchanged HEAD/after-launch snapshot hash, not a retroactive manifest lock",
        "decoder_dependency_sha256": "6b102bd674403a431e898ef1d6d411e74a663f063555e021ac56a1159a8a0d3e",
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_parameters(components):
    """All runtime parameter owners, not only trainable components; aliases once."""
    if not components or any(type(n) is not int or n < 0 for n in components.values()):
        raise ValueError("complete runtime parameter inventory required")
    count = sum(components.values())
    if not 0 < count < PARAMETER_LIMIT:
        raise ValueError("runtime parameters must be strictly <1e9")
    return count


def split_for(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "dev" if bucket < 8 else "heldout"


def sentences(context):
    # Keep punctuation and whitespace, so extraction is source-exact.
    spans = re.findall(r".*?(?:[.!?](?:\s+|$)|$)", context, flags=re.S)
    return [span for span in spans if span]


def words(text):
    return re.findall(r"\w+", text.casefold())


def candidates(row):
    spans = sentences(row["context"])
    if not 2 <= len(spans) <= 16:
        raise ValueError("need 2..16 source sentences")
    keep = (len(spans) + 1) // 2
    query = set(words(row["question"]))
    lexical = sorted(range(len(spans)), key=lambda i: (-len(query & set(words(spans[i]))), i))[:keep]
    masks = [list(range(keep)), list(range(len(spans) - keep, len(spans))), sorted(lexical)]
    return [{"indices": mask, "text": "".join(spans[i] for i in mask)} for mask in masks]


def features(row, candidate):
    query = words(row["question"])
    text = words(candidate["text"])
    all_words = words(row["context"])
    q, t = set(query), set(text)
    n = len(sentences(row["context"]))
    indices = candidate["indices"]
    vector = np.zeros(FEATURES, dtype=np.float64)
    vector[:8] = (1, len(q & t) / max(1, len(q)), len(text) / max(1, len(all_words)),
                  indices[0] / n, indices[-1] / n, len(indices) / n,
                  sum(any(c.isdigit() for c in w) for w in text) / max(1, len(text)),
                  sum(w in t for w in query) / max(1, len(query)))
    # Ordered query bigrams and exact query/context intersections. No answer input.
    signals = ["match:" + w for w in sorted(q & t)]
    signals += ["bigram:" + a + ":" + b for a, b in zip(text, text[1:]) if a in q or b in q]
    for signal in signals:
        raw = hashlib.sha256(signal.encode()).digest()
        vector[8 + int.from_bytes(raw[:2], "little") % 256] += (1 if raw[2] & 1 else -1) / max(1, len(signals))
    return vector


@dataclass
class Ranker:
    weights: np.ndarray
    source_binding: str
    training_split_digest: str

    @property
    def parameter_inventory(self):
        # No backbone, vocabulary embedding or head exists in this helper.
        return {"ranker_including_bias": int(self.weights.size), "frozen_backbone": 0,
                "vocabulary_embedding": 0, "vocabulary_head": 0}

    def choose(self, row):
        runtime_parameters(self.parameter_inventory)
        choices = candidates(row)
        return choices[int(np.argmax([features(row, c) @ self.weights for c in choices]))]

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        runtime_parameters(self.parameter_inventory)
        document = {"schema": SCHEMA, "role": "extractive_helper_not_standalone",
                    "source_binding": self.source_binding, "training_split_digest": self.training_split_digest,
                    "parameters": self.parameter_inventory, "weights": self.weights.tolist()}
        document["artifact_digest"] = digest(document)
        path.write_text(json.dumps(document, indent=2) + "\n")

    @classmethod
    def load(cls, path, source_binding):
        doc = json.loads(Path(path).read_text())
        identity = doc.pop("artifact_digest")
        if digest(doc) != identity or doc["source_binding"] != source_binding or doc["schema"] != SCHEMA:
            raise ValueError("artifact/source binding mismatch")
        weights = np.asarray(doc["weights"], dtype=np.float64)
        if weights.shape != (FEATURES,) or not np.isfinite(weights).all():
            raise ValueError("invalid ranker weights")
        model = cls(weights, source_binding, doc["training_split_digest"])
        if model.parameter_inventory != doc["parameters"]:
            raise ValueError("incomplete parameter inventory")
        runtime_parameters(model.parameter_inventory)
        return model


def fit(train_rows, teacher_losses, source_binding, *, epochs=80):
    """Pairwise logistic ranking against frozen checkpoint answer NLL. Train only."""
    if not train_rows or any(row["split"] != "train" for row in train_rows):
        raise ValueError("only frozen train split may fit")
    if not 1 <= epochs <= 80 or len(train_rows) != len(teacher_losses):
        raise ValueError("bounded training rows/epochs required")
    differences, labels = [], []
    for row, losses in zip(train_rows, teacher_losses, strict=True):
        if len(losses) != 3 or not np.isfinite(losses).all():
            raise ValueError("three finite teacher losses required")
        vectors = [features(row, c) for c in candidates(row)]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            if abs(losses[i] - losses[j]) > 1e-9:
                differences.append(vectors[i] - vectors[j])
                labels.append(float(losses[i] < losses[j]))
    weights = np.zeros(FEATURES, dtype=np.float64)
    if differences:
        x, y = np.asarray(differences), np.asarray(labels)
        for _ in range(epochs):
            predictions = 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))
            weights -= 0.05 * (x.T @ (predictions - y) / len(y) + 0.001 * weights)
    model = Ranker(weights, source_binding, digest(train_rows))
    runtime_parameters(model.parameter_inventory)
    return model


def stress_rows(split):
    # Public procedural controls; never evidence of natural-language usefulness.
    code = hashlib.sha256(("unseen-entity-" + split).encode()).hexdigest()[:14]
    routes = {"train": ("Oslo", "Lima", "Kyoto"), "dev": ("Perth", "Bern", "Accra"),
              "heldout": ("Riga", "Quito", "Dover")}
    first, second, third = routes[split]
    rows = [
        ("rare_copy", f"The first parcel code is ZX-{code}. The second parcel code is QJ-{code[::-1]}. ",
         "Copy the first parcel code exactly.", f"ZX-{code}"),
        ("order", f"The ordered route starts at {first}. It then visits {second}. It ends at {third}. ",
         "Which city is visited second?", second),
    ]
    return [{"id": f"stress-{split}-{kind}", "group": f"stress-{split}", "split": split,
             "kind": kind, "context": context, "question": question, "answers": [answer]}
            for kind, context, question, answer in rows]


def prepare_data(path):
    raw = Path(path).read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != DATA_BLOB or len(raw) != 4_854_279:
        raise ValueError("public SQuAD source lock mismatch")
    source = json.loads(raw)
    pools = {key: [] for key in POLICY["counts"]}
    for article in source["data"]:
        split = split_for(article["title"])
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not 2 <= len(sentences(context)) <= 16 or not 120 <= len(context) <= 850:
                continue
            for qa in paragraph["qas"]:
                answers = list(dict.fromkeys(answer["text"] for answer in qa["answers"]))
                if not answers or len(words(answers[0])) > 8:
                    continue
                pools[split].append({"id": qa["id"], "group": article["title"], "split": split,
                                     "kind": "public_qa", "context": context,
                                     "question": qa["question"], "answers": answers})
    rows = []
    for split, count in POLICY["counts"].items():
        seen = set()
        for row in sorted(pools[split], key=lambda row: digest(row["id"])):
            key = digest(row["context"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(seen) == count:
                break
        if len(seen) != count:
            raise ValueError("insufficient bounded public examples")
        rows.extend(stress_rows(split))
    for left in ("train", "dev", "heldout"):
        for right in ("train", "dev", "heldout"):
            if left == right:
                continue
            if {r["context"] for r in rows if r["split"] == left} & {r["context"] for r in rows if r["split"] == right}:
                raise ValueError("cross-split context duplication")
    return {"url": DATA_URL, "git_blob": blob, "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": rows, "split_digest": digest(rows), "policy": POLICY}


def prompt(row, context):
    return "Use only this context. Answer the question with the shortest exact span; no explanation.\nContext:\n" + context + "\nQuestion: " + row["question"] + "\nAnswer:"


def exact_match(text, answers):
    def normalize(value):
        value = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", value.lower()))
        return " ".join(value.split())
    return any(normalize(text) == normalize(answer) for answer in answers)


def needs_fallback(text, eos, retained):
    return not eos or not text.strip() or text.strip() not in retained


def projected_cost(stages, executed_rows, reserved_rows):
    if not 0 < executed_rows <= reserved_rows:
        raise ValueError("invalid execution/preparation rows")
    return {
        "client_to_inference": sum(executed_rows * s["inputs"] * s["bits"] // 8 for s in stages),
        "inference_to_client": sum(executed_rows * s["outputs"] * s["bits"] // 8 for s in stages),
        "preparation_to_inference": sum(reserved_rows * s["outputs"] * s["bits"] // 8 for s in stages),
    }


def support_table(root):
    """Discover actual support cells; never branch execution by family name."""
    path = Path(root) / "docs/data/model-compatibility.json"
    rows = []
    for adapter in json.loads(path.read_text())["adapters"]:
        real = adapter.get("evidence", {}).get("checkpoint") == "real-local"
        fixture = adapter.get("fixture")
        source = SOURCES.get(adapter["name"])
        rows.append({"checkpoint": adapter["name"], "adapter": adapter["adapter"],
                     "real_checkpoint_runtime": real,
                     "tiny_artifact_status": "untrained" if real else "blocked_no_real_checkpoint_runtime",
                     "source_fixture": fixture, "config": json.loads((Path(root) / fixture).read_text()) if fixture else adapter.get("config"),
                     "config_sha256": file_hash(Path(root) / fixture) if fixture else None,
                     "runtime_evidence": adapter["runtime_evidence"],
                     "requires": adapter["requires"], "source_lock": source,
                     "source_lock_status": "tensor_hashes_known_tokenizer_lock_pending" if source and source.get("files") else "incomplete_training_source_lock",
                     "training_admission": "pilot_only_after_preflight" if source == PILOT else "blocked_unimplemented_per_checkpoint_pilot"})
        pilot = COMPLETED_PILOTS.get(source["model"]) if source else None
        if pilot is not None:
            rows[-1].update(tiny_artifact_status=pilot["status"], completed_pilot=pilot,
                            training_admission="completed_failed_screen_no_rerun_authorized")
    for extension in CHECKPOINT_EXTENSIONS:
        row = dict(extension)
        lock = json.loads((Path(root) / row["lock_reference"]).read_text())
        quality = json.loads((Path(root) / row["quality_reference"]).read_text())
        if any(lock[key] != row["source_lock"][key] for key in ("model", "revision", "source_lock_digest")):
            raise ValueError("checkpoint extension source evidence differs")
        if quality["checkpoint_digest"] != row["source_lock"]["checkpoint_digest"]:
            raise ValueError("checkpoint extension checkpoint evidence differs")
        row.update(real_checkpoint_runtime=True, tiny_artifact_status="untrained",
                   source_fixture=None, source_lock_status="retained_runtime_lock_training_files_not_frozen",
                   training_admission="blocked_unimplemented_per_checkpoint_pilot",
                   quality_status=quality["status"],
                   evidence_sha256={name: file_hash(Path(root) / name) for name in row["references"]})
        rows.append(row)
    return {"capability_source_sha256": file_hash(path), "rows": rows,
            "inventory_scope": "adapter summary plus explicit checkpoint evidence; not an exhaustive family/source catalog"}


def mechanics_report(root):
    """Tiny analytic-label fit, never checkpoint or useful-model evidence."""
    train = stress_rows("train")
    losses = [[0.0 if row["answers"][0] in c["text"] else 5.0 for c in candidates(row)] for row in train]
    model = fit(train, losses, "SYNTHETIC-NOT-A-CHECKPOINT")
    tests = stress_rows("heldout")
    return {"schema": SCHEMA, "status": "synthetic_mechanics_only_no_new_checkpoint_training",
            "real_tensor_loads": 0, "teacher": "analytic answer-retention fixture, not a language model",
            "parameters": model.parameter_inventory, "strict_parameter_check": runtime_parameters(model.parameter_inventory),
            "policy": POLICY, "train_digest": digest(train), "heldout_digest": digest(tests),
            "weights_digest": digest(model.weights.tolist()),
            "fresh_heldout_retention": [{"id": row["id"], "kind": row["kind"],
                                          "answer_retained": row["answers"][0] in model.choose(row)["text"]} for row in tests],
            "support": support_table(root),
             "code": {name: file_hash(Path(root) / name) for name in (
                 "scripts/train_checkpoint_tiny_model.py", "benchmarks/research/checkpoint_tiny_model.py",
                 "tests/test_checkpoint_tiny_model.py", "scripts/decoder_probe_support.py",
                 "scripts/frontier_probe_support.py")},
             "limitations": ["This mechanics invocation executes no public-checkpoint tensors or training; separate parent pilot trained and failed heldout",
                            "No language usefulness, serving byte reduction, latency or Low/Medium qualification",
                            "Cold all-directed-wire/setup/fallback accounting remains a separate serving gate"]}
