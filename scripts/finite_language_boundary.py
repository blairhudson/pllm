"""Bounded public byte-language compiler for research head specialization.

All tokenizer segmentations are admitted, not merely canonical encodings of each
answer. This is an explicit restricted-output contract, not a serving component.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


MAX_TOKENIZER_BYTES = 64 << 20
MAX_VOCABULARY = 262144
MAX_LANGUAGE_BYTES = 4096
MAX_EDGES = 65536


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def byte_alphabet():
    original = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [i for i in range(256) if i not in original]
    return {chr(code): byte for byte, code in
            zip(original + missing, original + list(range(256, 256 + len(missing))), strict=True)}


def tokenizer_pieces(raw, vocabulary, eos):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_TOKENIZER_BYTES:
        raise ValueError("bounded original tokenizer bytes required")
    if (type(vocabulary) is not int or not 1 <= vocabulary <= MAX_VOCABULARY
            or type(eos) is not int or not 0 <= eos < vocabulary):
        raise ValueError("invalid vocabulary or EOS")
    config = json.loads(raw)
    model = config.get("model", {})
    if (model.get("type") != "BPE" or model.get("dropout") or model.get("unk_token")
            or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix")
            or model.get("byte_fallback") or model.get("ignore_merges")
            or (config.get("decoder") or {}).get("type") != "ByteLevel"):
        raise ValueError("unsupported context-independent byte decoder")
    vocab = model.get("vocab")
    added = config.get("added_tokens", [])
    if (type(vocab) is not dict or not 1 <= len(vocab) <= MAX_VOCABULARY
            or type(added) is not list or len(added) > 256):
        raise ValueError("bounded tokenizer vocabulary required")
    inverse, pieces, ids = byte_alphabet(), {}, set()
    for token, index in vocab.items():
        if (type(index) is not int or not 0 <= index < vocabulary or index in ids
                or type(token) is not str or not 1 <= len(token) <= MAX_LANGUAGE_BYTES):
            raise ValueError("invalid or duplicate vocabulary entry")
        try:
            piece = bytes(inverse[c] for c in token)
        except KeyError as error:
            raise ValueError("non-byte-level token") from error
        ids.add(index)
        pieces[index] = piece
    special = set()
    for token in added:
        index = token.get("id")
        if (type(index) is not int or not 0 <= index < vocabulary or index in ids
                or type(token.get("special")) is not bool or type(token.get("content")) is not str
                or not 1 <= len(token["content"]) <= MAX_LANGUAGE_BYTES
                or any(token.get(k, False) for k in ("single_word", "lstrip", "rstrip", "normalized"))):
            raise ValueError("unsupported added-token semantics")
        ids.add(index)
        if token["special"]:
            special.add(index)
        else:
            try:
                pieces[index] = bytes(inverse[c] for c in token["content"])
            except KeyError as error:
                raise ValueError("non-byte-level added token") from error
    if eos not in special:
        raise ValueError("EOS must be a declared distinct special token")
    return pieces


@dataclass(frozen=True, slots=True)
class FiniteLanguage:
    tokenizer_sha256: str
    vocabulary: int
    eos: int
    words: tuple[bytes, ...]
    prefixes: tuple[bytes, ...]
    transitions: tuple[tuple[tuple[int, int], ...], ...]
    minimum_selections: tuple[int, ...]

    @property
    def union(self):
        return tuple(sorted({token for edges in self.transitions for token, _ in edges}))

    def allowed(self, state, remaining):
        if (type(state) is not int or not 0 <= state < len(self.prefixes)
                or type(remaining) is not int or not 1 <= remaining <= MAX_LANGUAGE_BYTES + 1):
            raise ValueError("invalid state or remaining selection budget")
        result = tuple(token for token, next_state in self.transitions[state]
                       if next_state == -1 or remaining > self.minimum_selections[next_state])
        if not result:
            raise ValueError("selection budget cannot complete language with EOS")
        return result

    def advance(self, state, token, remaining):
        if type(token) is not int or token not in self.allowed(state, remaining):
            raise ValueError("token outside public language or completion budget")
        return dict(self.transitions[state])[token]

    def to_spec(self):
        return {"schema": "pllm.research.finite_byte_language.v1",
                "visibility": "public", "tokenizer_sha256": self.tokenizer_sha256,
                "vocabulary": self.vocabulary, "eos": self.eos,
                "words_utf8_hex": [x.hex() for x in self.words],
                "prefixes_hex": [x.hex() for x in self.prefixes],
                "transitions": [[list(edge) for edge in row] for row in self.transitions],
                "minimum_selections": list(self.minimum_selections),
                "union": list(self.union), "selection_budget_includes_eos": True}

    @property
    def digest(self):
        return hashlib.sha256(canonical(self.to_spec())).hexdigest()


def compile_language(raw, words, *, vocabulary, eos, visibility="public"):
    if visibility != "public":
        raise ValueError("finite language must be explicitly public")
    if (type(words) not in (tuple, list) or not 1 <= len(words) <= 32
            or any(type(word) is not str or not word for word in words)):
        raise ValueError("1..32 nonempty public strings required")
    words = tuple(sorted(word.encode("utf-8") for word in words))
    if (len(set(words)) != len(words) or sum(map(len, words)) > MAX_LANGUAGE_BYTES
            or max(map(len, words)) > 256):
        raise ValueError("duplicate or overlarge finite language")
    pieces = tokenizer_pieces(raw, vocabulary, eos)
    by_piece = {}
    for index, piece in pieces.items():
        by_piece.setdefault(piece, []).append(index)
    # Prefixes of UTF-8 byte strings may themselves end inside a code point.
    prefixes = tuple(sorted({word[:i] for word in words for i in range(len(word) + 1)},
                            key=lambda p: (len(p), p)))
    positions = {prefix: i for i, prefix in enumerate(prefixes)}
    transitions, edge_count = [], 0
    terminals = set(words)
    for prefix in prefixes:
        edges = {eos: -1} if prefix in terminals else {}
        for word in words:
            if not word.startswith(prefix):
                continue
            for end in range(len(prefix) + 1, len(word) + 1):
                for token in by_piece.get(word[len(prefix):end], ()):
                    edges[token] = positions[word[:end]]
        edge_count += len(edges)
        if edge_count > MAX_EDGES:
            raise ValueError("finite language transition budget exceeded")
        transitions.append(tuple(sorted(edges.items())))
    complete = set()
    for state in reversed(range(len(prefixes))):
        transitions[state] = tuple((token, dest) for token, dest in transitions[state]
                                   if dest == -1 or dest in complete)
        if transitions[state]:
            complete.add(state)
    reachable = {0}
    for state, row in enumerate(transitions):
        if state in reachable:
            reachable.update(next_state for _, next_state in row if next_state != -1)
    if any(positions[word] not in reachable for word in words):
        raise ValueError("public string cannot be represented by tokenizer")
    # Discard unreachable byte boundaries (some tokenizers lack single-byte tokens).
    selected = sorted(reachable)
    remap = {old: new for new, old in enumerate(selected)}
    transitions = tuple(tuple((token, -1 if dest == -1 else remap[dest])
                              for token, dest in transitions[old]) for old in selected)
    minimum = [MAX_LANGUAGE_BYTES + 2] * len(selected)
    for state in reversed(range(len(selected))):
        minimum[state] = min((1 if dest == -1 else minimum[dest] + 1
                              for _, dest in transitions[state]), default=MAX_LANGUAGE_BYTES + 2)
    # Every retained edge must be on a complete accepting path.
    if any(v > MAX_LANGUAGE_BYTES + 1 for v in minimum):
        raise ValueError("tokenizer leaves a reachable dead end")
    return FiniteLanguage(hashlib.sha256(raw).hexdigest(), vocabulary, eos, words,
                          tuple(prefixes[i] for i in selected), transitions, tuple(minimum))


def dense_restricted(logits, token_ids, language, state, remaining):
    """Restore original vocabulary shape before invoking the unchanged SDK sampler."""
    import numpy as np

    ids = tuple(token_ids)
    value = np.asarray(logits)
    if (ids != language.union or value.dtype != np.float32 or value.shape != (len(ids),)
            or not np.all(np.isfinite(value))):
        raise ValueError("sliced logits differ from finite-language identity")
    allowed = set(language.allowed(state, remaining))
    full = np.full(language.vocabulary, -np.inf, dtype=np.float32)
    for index, token in enumerate(ids):
        if token in allowed:
            full[token] = value[index]
    return full
