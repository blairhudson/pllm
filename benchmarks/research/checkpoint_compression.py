"""Frozen, local-clear checkpoint-specific neural extraction experiment.

No target adaptation, pretrained helper, private serving, or usefulness claim.
All neural owners are explicit; NumPy implements both forward and reverse math.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from benchmarks.research.checkpoint_tiny_model import (
    DATA_BLOB, DATA_URL, digest, exact_match, file_hash, needs_fallback,
    prepare_data as old_prepare_data, prompt, runtime_parameters, sentences, words,
)

SCHEMA = "pllm.checkpoint_compression.v1"
PRIOR_MANIFEST = "docs/evidence/checkpoint-tiny-model-pilot-manifest-v1.json"
PRIOR_MANIFEST_SHA256 = "740ca7839ba611223c7f9a5f784628a7f8f4bd765b8901df321ceb7006cfae0b"
POLICY = {
    "seed": 610162, "dimension": 32, "hidden": 64,
    "epochs": 16, "learning_rate": 0.003, "adam_beta1": 0.9,
    "adam_beta2": 0.999, "adam_epsilon": 1e-8, "gradient_norm": 1.0,
    "teacher_temperature": 1.0, "l2": 0.0001,
    "counts": {"train": 192, "admission": 64, "heldout": 128},
    "controls_per_kind": {"train": 16, "admission": 8, "heldout": 16},
    "control_kinds": ["copy", "rare", "order", "long_context"],
    "qa_min_em": 0.75, "qa_min_wilson_lower": 0.60,
    "control_min_em": 0.875, "max_quality_drop": 0.02,
    "learned_min_em_gain_over_lexical": 0.02,
    "max_fallback_fraction": 0.10, "max_fallbacks_per_chunk": 2,
    "chunk_items": 8, "timeout_seconds": 900,
    "max_attempts_per_item": 2, "max_child_attempts": 138,
    "baseline_early_stop": "quality_only_optimistic_all_remaining_success_v1",
    "post_reserve_ram_bytes": 4 << 30, "disk_free_bytes": 3 << 30,
    "max_input_tokens": 512, "max_output_tokens": 24,
    "max_sentence_bytes": 2048, "max_question_bytes": 512,
    "max_context_bytes": 6144,
    "future_measured_first_request_traffic_ratio": 0.8,
}


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json(path, document, *, exclusive=False):
    """Durable bounded publication, including directory entry; never partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(document, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    if len(raw) > 16 << 20:
        raise ValueError("bounded artifact exceeds 16MiB")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            fsync_directory(path.parent)


def split_for(title):
    bucket = int(digest(["checkpoint-compression-v1", title])[:8], 16) % 10
    return "train" if bucket < 6 else "admission" if bucket < 8 else "heldout"


def controls(split, count):
    rows = []
    for i in range(count):
        code = digest(["checkpoint-compression-controls-v1", split, i])[:16]
        for kind in POLICY["control_kinds"]:
            a, b, c = ("K" + code[:5], "V" + code[5:10], "Z" + code[10:])
            if kind == "order":
                context = f"The route starts at {a}. It visits {b} second. It ends at {c}. "
                question, answer = "Which stop is visited second?", b
            else:
                answer = ("PK-" if kind == "copy" else "Ω-ß-") + code
                context = f"Parcel one has code {answer}. Parcel two has code XX-{code[::-1]}. "
                if kind == "long_context":
                    # Long relative to this 512-token screen, not a 32K-context claim.
                    context = "".join(f"Record {j} concerns ordinary blue paper in storage. " for j in range(9)) + context
                    context += "".join(f"Note {j} describes plain red folders on shelves. " for j in range(9))
                question = "Copy parcel one's code exactly."
            rows.append({"id": f"control-{split}-{kind}-{i}", "group": f"control-{split}-{kind}-{i}",
                         "split": split, "kind": kind, "context": context,
                         "question": question, "answers": [answer]})
    return rows


def prepare_data(path):
    # Re-derive the old pilot, exclude its ENTIRE articles, not merely its QA IDs.
    old = old_prepare_data(path)
    archived = Path(__file__).resolve().parents[2] / PRIOR_MANIFEST
    if file_hash(archived) != PRIOR_MANIFEST_SHA256:
        raise ValueError("original pilot exclusion manifest changed")
    if old["rows"] != json.loads(archived.read_text())["data"]["rows"]:
        raise ValueError("prior pilot re-derivation differs from frozen original")
    raw = Path(path).read_bytes()
    if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != DATA_BLOB:
        raise ValueError("public corpus identity")
    excluded = sorted({r["group"] for r in old["rows"] if r["kind"] == "public_qa"})
    pools = {s: [] for s in POLICY["counts"]}
    for article in json.loads(raw)["data"]:
        if article["title"] in excluded:
            continue
        split = split_for(article["title"])
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not 120 <= len(context) <= 1400 or not 3 <= len(sentences(context)) <= 24:
                continue
            if any(len(s.encode()) > POLICY["max_sentence_bytes"] for s in sentences(context)):
                continue
            for qa in paragraph["qas"]:
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                if not answers or any(len(words(a)) > 8 for a in answers):
                    continue
                if len(qa["question"].encode()) > POLICY["max_question_bytes"]:
                    continue
                pools[split].append({"id": qa["id"], "group": article["title"], "split": split,
                    "kind": "public_qa", "context": context, "question": qa["question"], "answers": answers})
    rows, seen = [], set()
    for split, count in POLICY["counts"].items():
        selected = []
        for row in sorted(pools[split], key=lambda r: digest(["fresh-v1", r["id"]])):
            key = digest(row["context"])
            if key not in seen:
                seen.add(key)
                selected.append(row)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f"insufficient fresh public cohort: {split}")
        rows.extend(selected + controls(split, POLICY["controls_per_kind"][split]))
    assert_split_integrity(rows, excluded)
    return {"url": DATA_URL, "git_blob": DATA_BLOB, "sha256": file_hash(path),
            "license": "SQuAD v1.1 CC BY-SA 4.0; Rajpurkar et al.; procedural controls public recipe",
            "prior_pilot_digest": old["split_digest"], "excluded_articles": excluded,
            "prior_manifest_sha256": PRIOR_MANIFEST_SHA256,
            "rows": rows, "split_digest": digest(rows)}


def assert_split_integrity(rows, excluded=()):
    ids, contexts, groups = set(), {}, {}
    for row in rows:
        if row["id"] in ids or row["group"] in excluded:
            raise ValueError("duplicate ID or prior-pilot article overlap")
        ids.add(row["id"])
        for table, key in ((contexts, digest(row["context"])), (groups, row["group"])):
            if key in table and table[key] != row["split"]:
                raise ValueError("cross-split article/context overlap")
            table[key] = row["split"]


def masks(row):
    n = len(sentences(row["context"]))
    if not 2 <= n <= 24:
        raise ValueError("sentence bound")
    k = (n + 1) // 2
    query = set(words(row["question"]))
    lexical = sorted(sorted(range(n), key=lambda i: (-len(query & set(words(sentences(row['context'])[i]))), i))[:k])
    choices = [list(range(i, i + k)) for i in range(n - k + 1)] + [lexical]
    return list(dict.fromkeys(tuple(x) for x in choices))


def lexical_mask(row):
    n = len(sentences(row["context"]))
    query = set(words(row["question"]))
    return sorted(sorted(range(n), key=lambda i: (-len(query & set(words(sentences(row['context'])[i]))), i))[:(n + 1) // 2])


def extract(row, mask):
    spans = sentences(row["context"])
    if list(mask) != sorted(set(mask)) or not mask or min(mask) < 0 or max(mask) >= len(spans):
        raise ValueError("invalid source mask")
    return "".join(spans[i] for i in mask)


class Encoder:
    """Trainable UTF-8 3-byte CNN + query-conditioned contextual sentence MLP.

    Frozen learned parameters: zero. No target embedding/head/backbone is used.
    Byte IDs 1..256; padding 0. Previous/next sentence mean and source positions
    supply cross-sentence context. All retained text remains source-exact.
    """
    def __init__(self, binding, train_digest="", weights=None):
        self.binding, self.train_digest = binding, train_digest
        d, h = POLICY["dimension"], POLICY["hidden"]
        rng = np.random.default_rng(POLICY["seed"])
        shapes = {"embedding": (257, d), "conv": (3 * d, d), "conv_bias": (d,),
                  "context_head": (4 * d + 2, h), "head_bias": (h,), "output": (h,)}
        self.weights = {k: rng.normal(0, 0.1, s) for k, s in shapes.items()} if weights is None else weights
        if set(self.weights) != set(shapes) or any(self.weights[k].shape != s or
                not np.isfinite(self.weights[k]).all() for k, s in shapes.items()):
            raise ValueError("neural shape/nonfinite inventory")
        runtime_parameters(self.inventory)
        if sum(self.inventory.values()) != 19_776 or any(self.inventory[k] for k in
                ("frozen_backbone", "frozen_embedding", "target_vocabulary_head")):
            raise ValueError("entire helper must be 19,776 parameters with no hidden backbone")

    @property
    def inventory(self):
        return {**{k: int(v.size) for k, v in self.weights.items()}, "frozen_backbone": 0,
                "frozen_embedding": 0, "target_vocabulary_head": 0}

    def _encode(self, text, bound):
        raw = text.encode("utf-8")
        if not 0 < len(raw) <= bound:
            raise ValueError("byte bound; no silent truncation")
        ids = np.pad(np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1, (1, 1))
        windows = np.stack([ids[:-2], ids[1:-1], ids[2:]], axis=1)
        x = self.weights["embedding"][windows].reshape(len(raw), -1)
        y = np.tanh(x @ self.weights["conv"] + self.weights["conv_bias"])
        return y.mean(axis=0), (windows, x, y)

    def forward(self, row):
        if len(row["context"].encode()) > POLICY["max_context_bytes"]:
            raise ValueError("context byte bound")
        encoded = [self._encode(s, POLICY["max_sentence_bytes"]) for s in sentences(row["context"])]
        q, qc = self._encode(row["question"], POLICY["max_question_bytes"])
        s = np.stack([v for v, _ in encoded])
        n = len(s)
        if not 2 <= n <= 24:
            raise ValueError("sentence bound")
        neighbor = (np.vstack([np.zeros_like(s[:1]), s[:-1]]) + np.vstack([s[1:], np.zeros_like(s[:1])])) / 2
        x = np.column_stack([s, np.broadcast_to(q, s.shape), s * q, neighbor,
                             np.arange(n) / n, (np.arange(n) + 1) / n])
        z = np.tanh(x @ self.weights["context_head"] + self.weights["head_bias"])
        return z @ self.weights["output"], (encoded, qc, s, q, x, z)

    def backward(self, cache, dscores):
        encoded, qc, s, q, x, z = cache
        w = self.weights
        g = {k: np.zeros_like(v) for k, v in w.items()}
        g["output"] = z.T @ dscores
        dz = dscores[:, None] * w["output"] * (1 - z * z)
        g["context_head"], g["head_bias"] = x.T @ dz, dz.sum(axis=0)
        dx = dz @ w["context_head"].T
        d = s.shape[1]
        ds = dx[:, :d] + dx[:, 2*d:3*d] * q
        dq = (dx[:, d:2*d] + dx[:, 2*d:3*d] * s).sum(axis=0)
        dn = dx[:, 3*d:4*d] / 2
        ds[:-1] += dn[1:]
        ds[1:] += dn[:-1]
        for grad, (_, ec) in zip(ds, encoded, strict=True):
            self._back_encode(ec, grad, g)
        self._back_encode(qc, dq, g)
        return g

    def _back_encode(self, cache, grad, g):
        ids, x, y = cache
        dy = grad[None, :] * (1 - y*y) / len(y)
        g["conv"] += x.T @ dy
        g["conv_bias"] += dy.sum(axis=0)
        de = (dy @ self.weights["conv"].T).reshape(len(y), 3, -1)
        np.add.at(g["embedding"], ids, de)

    def choose(self, row):
        scores, _ = self.forward(row)
        k = (len(scores) + 1) // 2
        return sorted(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k])

    def save(self, path):
        doc = {"schema": SCHEMA, "binding": self.binding, "train_digest": self.train_digest,
               "inventory": self.inventory, "weights": {k: v.tolist() for k, v in self.weights.items()}}
        doc["digest"] = digest(doc)
        publish_json(path, doc, exclusive=True)

    @classmethod
    def load(cls, path, binding):
        doc = json.loads(Path(path).read_text())
        identity = doc.pop("digest")
        if doc["schema"] != SCHEMA or digest(doc) != identity or doc["binding"] != binding:
            raise ValueError("artifact/source binding")
        model = cls(binding, doc["train_digest"], {k: np.asarray(v) for k, v in doc["weights"].items()})
        if model.inventory != doc["inventory"]:
            raise ValueError("inventory mismatch")
        return model


def softmax(x):
    y = np.exp(x - np.max(x))
    return y / y.sum()


def objective(model, row, losses):
    choices = masks(row)
    if len(losses) != len(choices) or not np.isfinite(losses).all():
        raise ValueError("finite teacher losses for every frozen candidate required")
    scores, cache = model.forward(row)
    matrix = np.zeros((len(choices), len(scores)))
    for i, mask in enumerate(choices):
        matrix[i, list(mask)] = 1 / len(mask)
    target = softmax(-np.asarray(losses) / POLICY["teacher_temperature"])
    probability = softmax(matrix @ scores)
    loss = -float(target @ np.log(probability + 1e-300))
    return loss, model.backward(cache, matrix.T @ (probability - target))


def fit(rows, losses, binding, checkpoint=None):
    if not rows or any(r["split"] != "train" for r in rows) or len(rows) != len(losses):
        raise ValueError("frozen train split only")
    model = Encoder(binding, digest(rows))
    moment = {k: np.zeros_like(v) for k, v in model.weights.items()}
    variance = {k: np.zeros_like(v) for k, v in model.weights.items()}
    step = 0
    for epoch in range(POLICY["epochs"]):
        total = 0.0
        for row, target in zip(rows, losses, strict=True):
            loss, gradients = objective(model, row, target)
            total += loss
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
            step += 1
            for key, w in model.weights.items():
                g = gradients[key] * min(1, POLICY["gradient_norm"] / max(norm, 1e-30)) + POLICY["l2"] * w
                moment[key] = 0.9 * moment[key] + 0.1 * g
                variance[key] = 0.999 * variance[key] + 0.001 * g*g
                w -= POLICY["learning_rate"] * (moment[key] / (1 - 0.9**step)) / (np.sqrt(variance[key] / (1 - 0.999**step)) + 1e-8)
        if checkpoint:
            checkpoint(epoch + 1, total / len(rows))
    return model


def correct(row, result):
    return bool(result["eos"] and (exact_match(result["text"], row["answers"]) if row["kind"] == "public_qa"
                else result["text"].strip() in row["answers"]))


def wilson_lower(successes, count):
    if not count:
        return 0.0
    p, z = successes / count, 1.96
    return (p + z*z/(2*count) - z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1 + z*z/count)


def quality(rows, results):
    metrics = {}
    for kind in ["public_qa", *POLICY["control_kinds"]]:
        selected = [r for r in rows if r["kind"] == kind]
        wins = sum(correct(r, results[r["id"]]) for r in selected)
        metrics[kind] = {"count": len(selected), "correct": wins, "em": wins / len(selected) if selected else 0,
                         "wilson_lower": wilson_lower(wins, len(selected))}
    return metrics


def baseline_gate(rows, results, split="admission"):
    expected = POLICY["counts"][split] + 4 * POLICY["controls_per_kind"][split]
    if (len(rows) != expected or len({r["id"] for r in rows}) != expected
            or any(r["split"] != split for r in rows) or set(results) != {r["id"] for r in rows}):
        return {"admitted": False, "reason": "missing/incompatible baseline cohort"}
    metrics = quality(rows, results)
    qa = metrics["public_qa"]
    admitted = (qa["count"] == POLICY["counts"][split] and qa["em"] >= POLICY["qa_min_em"]
                and qa["wilson_lower"] >= POLICY["qa_min_wilson_lower"]
                and all(metrics[k]["count"] == POLICY["controls_per_kind"][split]
                        and metrics[k]["em"] >= POLICY["control_min_em"] for k in POLICY["control_kinds"]))
    return {"admitted": admitted, "metrics": metrics, "reason": "useful original baseline" if admitted else "weak original baseline; training blocked"}


def optimistic_baseline(rows, results):
    """Quality-only stopping rule. Unknown/missing work is optimistic success, not error."""
    by_id = {r["id"]: r for r in rows}
    if set(results) - set(by_id):
        raise ValueError("baseline results outside frozen cohort")
    failures, impossible = {}, []
    for kind in ["public_qa", *POLICY["control_kinds"]]:
        selected = [r for r in rows if r["kind"] == kind]
        count = POLICY["counts"]["admission"] if kind == "public_qa" else POLICY["controls_per_kind"]["admission"]
        if len(selected) != count:
            raise ValueError("complete frozen admission denominator required")
        errors = sum(not correct(r, results[r["id"]]) for r in selected if r["id"] in results)
        failures[kind] = errors
        wins = count - errors
        threshold = POLICY["qa_min_em"] if kind == "public_qa" else POLICY["control_min_em"]
        if wins / count < threshold or (kind == "public_qa" and wilson_lower(wins, count) < POLICY["qa_min_wilson_lower"]):
            impossible.append(kind)
    return {"baseline_rejected": bool(impossible), "rule": POLICY["baseline_early_stop"],
            "valid_completed_quality_errors": failures, "impossible_kinds": impossible,
            "reason": "optimistic all-remaining-success cannot pass" if impossible else "quality still mathematically attainable"}
