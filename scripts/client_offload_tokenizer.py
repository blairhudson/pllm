"""Bounded research BPE index: public compilation, entirely client-local queries.

The caller must trust the compiler's artifact digest, not a digest supplied by an
untrusted serving peer. This does not replace the installed tokenizer contract.
SQLite is a native, bounded-page reference backend, not a proposed fast BPE kernel.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_public_tokenizer(source: Path, root: Path) -> dict:
    if source.stat().st_size > 64 << 20:
        raise ValueError("tokenizer source exceeds 64 MiB")
    config = json.loads(source.read_bytes())
    model = config["model"]
    if (model["type"] != "BPE" or model.get("dropout") or model.get("unk_token")
            or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix")
            or model.get("byte_fallback") or model.get("ignore_merges")
            or config.get("truncation") or config.get("padding")):
        raise ValueError("unsupported BPE contract")
    if len(model["vocab"]) > 200_000 or len(model["merges"]) > 500_000:
        raise ValueError("public BPE index exceeds capacity")
    post_processor = config.get("post_processor")
    if post_processor and post_processor.get("type") != "ByteLevel":
        raise ValueError("unsupported token-inserting post-processor")
    added = config["added_tokens"]
    if any(t["single_word"] or t["lstrip"] or t["rstrip"]
           or t["normalized"] for t in added):
        raise ValueError("unsupported added-token matching contract")
    root.mkdir(parents=True, exist_ok=False)
    path = root / "bpe.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE vocab(token TEXT PRIMARY KEY, id INTEGER UNIQUE) WITHOUT ROWID")
        db.execute("CREATE TABLE merges(a TEXT, b TEXT, rank INTEGER, PRIMARY KEY(a,b)) WITHOUT ROWID")
        db.executemany("INSERT INTO vocab VALUES(?,?)", model["vocab"].items())
        db.executemany("INSERT INTO merges VALUES(?,?,?)",
                       ((a, b, rank) for rank, (a, b) in enumerate(model["merges"])))
    light = dict(config, model=dict(model, vocab={}, merges=[]))
    contract = {
        "schema": "pllm.public_bpe_index_probe.v1", "source_sha256": file_digest(source),
        "database_sha256": file_digest(path), "light_tokenizer": light,
        "vocabulary_rows": len(model["vocab"]), "merge_rows": len(model["merges"]),
        "vocabulary_size": len(model["vocab"]) + sum(t["content"] not in model["vocab"] for t in added),
    }
    (root / "contract.json").write_text(json.dumps(contract, ensure_ascii=False))
    return {k: v for k, v in contract.items() if k != "light_tokenizer"} | {
        "artifact_bytes": sum(p.stat().st_size for p in root.iterdir()),
        "source_bytes": source.stat().st_size,
    }


class IndexedTokenizer:
    """Probe-only deterministic BPE, bounded caches and no remote lookup API."""

    def __init__(self, root: Path, expected_contract_digest: str, *, expected_source_digest: str | None = None):
        from tokenizers import Tokenizer

        if sqlite3.threadsafety != 3:
            raise ValueError("response-owned index requires serialized SQLite")
        contract_path = root / "contract.json"
        if contract_path.stat().st_size > 64 << 10:
            raise ValueError("tokenizer contract exceeds capacity")
        if file_digest(contract_path) != expected_contract_digest:
            raise ValueError("untrusted tokenizer contract")
        contract = json.loads(contract_path.read_bytes())
        if contract["schema"] != "pllm.public_bpe_index_probe.v1":
            raise ValueError("unsupported tokenizer index schema")
        if expected_source_digest is not None and contract["source_sha256"] != expected_source_digest:
            raise ValueError("tokenizer index source mismatch")
        self.vocab_size = contract["vocabulary_size"]
        path = root / "bpe.sqlite"
        if path.stat().st_size > 64 << 20 or file_digest(path) != contract["database_sha256"]:
            raise ValueError("tokenizer database identity mismatch")
        self.light = Tokenizer.from_str(json.dumps(contract["light_tokenizer"]))
        added = contract["light_tokenizer"]["added_tokens"]
        self.added = {t["content"]: t["id"] for t in added}
        self.added_ids = {v: k for k, v in self.added.items()}
        self.special_ids = {t["id"] for t in added if t["special"]}
        self.pattern = (re.compile("(" + "|".join(re.escape(t) for t in
                        sorted(self.added, key=len, reverse=True)) + ")")
                        if self.added else None)
        self.db = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1",
                                  uri=True, check_same_thread=False)
        self.db.execute("PRAGMA cache_size=-512")
        self.db.execute("PRAGMA mmap_size=0")
        self.rank = lru_cache(maxsize=2048)(self._rank)
        self.token_id = lru_cache(maxsize=2048)(self._token_id)

    def _rank(self, left: str, right: str) -> int:
        row = self.db.execute("SELECT rank FROM merges WHERE a=? AND b=?", (left, right)).fetchone()
        return row[0] if row else 1 << 60

    def _token_id(self, token: str) -> int:
        row = self.db.execute("SELECT id FROM vocab WHERE token=?", (token,)).fetchone()
        if row is None:
            raise ValueError("BPE symbol absent from committed vocabulary")
        return row[0]

    def encode(self, text: str) -> list[int]:
        if len(text.encode()) > 4096:
            raise ValueError("research tokenizer input exceeds 4096 bytes")
        result = []
        for segment in self.pattern.split(text) if self.pattern else [text]:
            if not segment:
                continue
            if segment in self.added:
                result.append(self.added[segment])
                continue
            if self.light.normalizer:
                segment = self.light.normalizer.normalize_str(segment)
            for token, _ in self.light.pre_tokenizer.pre_tokenize_str(segment):
                symbols = list(token)
                if len(symbols) > 1024:
                    raise ValueError("research BPE piece exceeds 1024 symbols")
                while len(symbols) > 1:
                    rank, index = min((self.rank(a, b), i)
                                      for i, (a, b) in enumerate(zip(symbols, symbols[1:])))
                    if rank == 1 << 60:
                        break
                    symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
                result.extend(self.token_id(symbol) for symbol in symbols)
        return result

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if len(ids) > 4096:
            raise ValueError("research decode exceeds 4096 tokens")
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_ids:
                continue
            if token_id in self.added_ids:
                tokens.append(self.added_ids[token_id])
            else:
                row = self.db.execute("SELECT token FROM vocab WHERE id=?", (token_id,)).fetchone()
                if row is None:
                    raise ValueError("token ID absent from committed vocabulary")
                tokens.append(row[0])
        return self.light.decoder.decode(tokens)

    def close(self):
        self.rank.cache_clear()
        self.token_id.cache_clear()
        self.db.close()


class IndexedClientTokenizer:
    """Probe-only SDK adapter, bound to the original bundle's tokenizer bytes."""

    name = "tokenizer-json"

    def __init__(self, bundle, root: Path, expected_contract_digest: str):
        descriptor = bundle.tokenizer_descriptor
        if descriptor.get("kind", descriptor.get("type")) != "tokenizer_json":
            raise ValueError("indexed tokenizer requires a tokenizer.json bundle")
        self.inner = IndexedTokenizer(root, expected_contract_digest,
            expected_source_digest=hashlib.sha256(bytes(descriptor["model"])).hexdigest())
        self.bos_token_id = int(bundle.cfg["bos_token_id"])
        self.eos_token_id = int(bundle.cfg["eos_token_id"])
        self.vocab_size = self.inner.vocab_size

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        values = self.inner.encode(text)
        return ([self.bos_token_id] + values) if add_bos else values

    def decode(self, tokens: list[int]) -> str:
        return self.inner.decode(tokens, skip_special_tokens=True)

    def close(self):
        self.inner.close()
