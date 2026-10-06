"""Bounded byte-level BPE, immutable private database snapshots, local queries."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import threading

from .tokenizer import Tokenizer as TokenizerContract

SCHEMA = "pllm.public_bpe_index.v1"
MAX_DATABASE = 64 << 20
MAX_CONTRACT = 64 << 10


def _read(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("public tokenizer artifact exceeds capacity")
    return data


def compile_index(source, directory):
    from pllm.tokenization import IndexedTokenizer
    from tokenizers import Tokenizer

    raw = _read(source, MAX_DATABASE)
    config = json.loads(raw)
    Tokenizer.from_str(raw.decode())  # Validate using the original implementation.
    model = config["model"]
    if (model["type"] != "BPE" or model.get("dropout") or model.get("unk_token")
            or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix")
            or model.get("byte_fallback") or model.get("ignore_merges")
            or config.get("truncation") or config.get("padding")):
        raise ValueError("unsupported indexed BPE contract")
    if len(model["vocab"]) > 200_000 or len(model["merges"]) > 500_000:
        raise ValueError("public BPE index exceeds capacity")
    pre = config.get("pre_tokenizer") or {}
    last = (pre.get("pretokenizers") or [{}])[-1] if pre.get("type") == "Sequence" else pre
    if last.get("type") != "ByteLevel" or last.get("add_prefix_space"):
        raise ValueError("indexed tokenizer requires byte-level pieces without prefix insertion")
    if (config.get("decoder") or {}).get("type") != "ByteLevel":
        raise ValueError("indexed tokenizer requires the byte-level decoder")
    if config.get("post_processor") and config["post_processor"].get("type") != "ByteLevel":
        raise ValueError("unsupported token-inserting post-processor")
    added = config["added_tokens"]
    if len(added) > 256 or any(t["single_word"] or t["lstrip"] or t["rstrip"] or t["normalized"] for t in added):
        raise ValueError("unsupported added-token matching contract")
    light = dict(config, model=dict(model, vocab={}, merges=[]))
    root = Path(directory).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(root)
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        stage = Path(temporary) / "index"
        stage.mkdir()
        database = stage / "bpe.sqlite"
        with sqlite3.connect(database) as db:
            db.execute("CREATE TABLE vocab(token TEXT PRIMARY KEY, id INTEGER UNIQUE) WITHOUT ROWID")
            db.execute("CREATE TABLE merges(a TEXT, b TEXT, rank INTEGER, PRIMARY KEY(a,b)) WITHOUT ROWID")
            db.executemany("INSERT INTO vocab VALUES(?,?)", model["vocab"].items())
            pairs = (pair.split(" ") if isinstance(pair, str) else pair for pair in model["merges"])
            db.executemany("INSERT INTO merges VALUES(?,?,?)", ((a, b, rank) for rank, (a, b) in enumerate(pairs)))
        db.close()
        if database.stat().st_size > MAX_DATABASE:
            raise ValueError("public BPE database exceeds capacity")
        with database.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        contract = {"schema": SCHEMA, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "database_sha256": digest, "light_tokenizer": light,
            "vocabulary_size": len(model["vocab"]) + sum(t["content"] not in model["vocab"] for t in added)}
        payload = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > MAX_CONTRACT:
            raise ValueError("public tokenizer contract exceeds capacity")
        (stage / "contract.json").write_bytes(payload)
        stage.rename(root)
    return IndexedTokenizer(str(root), digest=hashlib.sha256(payload).hexdigest())


class IndexedClientTokenizer(TokenizerContract):
    name = "tokenizer-json"

    def __init__(self, bundle, selection):
        from tokenizers import Tokenizer

        self._lock = threading.RLock()
        self._temporary = None
        self.db = None
        self._closed = False
        descriptor = bundle.tokenizer_descriptor
        if descriptor.get("kind", descriptor.get("type")) != "tokenizer_json" or sqlite3.threadsafety != 3:
            raise ValueError("indexed tokenizer requires tokenizer.json and serialized SQLite")
        root = Path(selection.params["path"])
        payload = _read(root / "contract.json", MAX_CONTRACT)
        if hashlib.sha256(payload).hexdigest() != selection.params["digest"]:
            raise ValueError("untrusted tokenizer contract")
        contract = json.loads(payload)
        if (contract["schema"] != SCHEMA
            or contract["source_sha256"] != hashlib.sha256(bytes(descriptor["model"])).hexdigest()):
            raise ValueError("tokenizer index source mismatch")
        self.light = Tokenizer.from_str(json.dumps(contract["light_tokenizer"]))
        added = contract["light_tokenizer"]["added_tokens"]
        self.added = {t["content"]: t["id"] for t in added}
        self.added_ids = {v: k for k, v in self.added.items()}
        self.special_ids = {t["id"] for t in added if t["special"]}
        self.pattern = (re.compile("(" + "|".join(re.escape(t) for t in sorted(self.added, key=len, reverse=True)) + ")")
                        if self.added else None)
        self.vocab_size = contract["vocabulary_size"]
        self.bos_token_id = int(bundle.cfg["bos_token_id"])
        self.eos_token_id = int(bundle.cfg["eos_token_id"])
        self._temporary = tempfile.TemporaryDirectory(prefix="pllm-tokenizer-")
        try:
            path = Path(self._temporary.name) / "bpe.sqlite"
            hasher, count = hashlib.sha256(), 0
            with (root / "bpe.sqlite").open("rb") as source, path.open("xb") as target:
                while chunk := source.read(128 << 10):
                    count += len(chunk)
                    if count > MAX_DATABASE:
                        raise ValueError("tokenizer database exceeds capacity")
                    hasher.update(chunk)
                    target.write(chunk)
            if hasher.hexdigest() != contract["database_sha256"]:
                raise ValueError("tokenizer database identity mismatch")
            self.artifact_bytes = len(payload) + count
            self.db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
            self.db.execute("PRAGMA trusted_schema=OFF")
            self.db.execute("PRAGMA cache_size=-512")
            self.db.execute("PRAGMA mmap_size=0")
            self.rank = lru_cache(maxsize=2048)(self._rank)
            self.token_id = lru_cache(maxsize=2048)(self._token_id)
        except BaseException:
            self.close()
            raise

    def _rank(self, left, right):
        row = self.db.execute("SELECT rank FROM merges WHERE a=? AND b=?", (left, right)).fetchone()
        return row[0] if row else 1 << 60

    def _token_id(self, token):
        row = self.db.execute("SELECT id FROM vocab WHERE token=?", (token,)).fetchone()
        if row is None:
            raise ValueError("BPE symbol absent from committed vocabulary")
        return row[0]

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        with self._lock:
            if self._closed or len(text.encode()) > 4096:
                raise ValueError("closed tokenizer or input exceeds 4096 UTF-8 bytes")
            result = [self.bos_token_id] if add_bos else []
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
                        raise ValueError("BPE piece exceeds 1024 symbols")
                    while len(symbols) > 1:
                        rank, index = min((self.rank(a, b), i) for i, (a, b) in enumerate(zip(symbols, symbols[1:])))
                        if rank == 1 << 60:
                            break
                        symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
                    result.extend(self.token_id(symbol) for symbol in symbols)
            return result

    def decode(self, tokens: list[int]) -> str:
        with self._lock:
            if self._closed or len(tokens) > 4096:
                raise ValueError("closed tokenizer or decode exceeds 4096 tokens")
            pieces = []
            for token in tokens:
                if token in self.special_ids:
                    continue
                if token in self.added_ids:
                    pieces.append(self.added_ids[token])
                else:
                    row = self.db.execute("SELECT token FROM vocab WHERE id=?", (int(token),)).fetchone()
                    if row is not None:
                        pieces.append(row[0])
            return self.light.decoder.decode(pieces)

    def close(self):
        with self._lock:
            self._closed = True
            for name in ("rank", "token_id"):
                if hasattr(self, name):
                    getattr(self, name).cache_clear()
            if self.db is not None:
                self.db.close()
                self.db = None
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
