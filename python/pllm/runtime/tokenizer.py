from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol


class Tokenizer(Protocol):
    name: str
    vocab_size: int
    bos_token_id: int
    eos_token_id: int

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class ByteTokenizer:
    """Portable client-side tokenizer for protocol and compatibility tests.

    IDs 0..255 are bytes; 256 is BOS and 257 is EOS. Invalid UTF-8 is replaced
    during decoding so streamed partial byte sequences never crash the client.
    """

    name: str = "pllm-byte-v1"
    vocab_size: int = 258
    bos_token_id: int = 256
    eos_token_id: int = 257

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        return ([self.bos_token_id] + tokens) if add_bos else tokens

    def decode(self, tokens: list[int]) -> str:
        data = bytes(token for token in tokens if 0 <= token <= 255)
        return data.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class AlphabetTokenizer:
    """Small exact tokenizer used by the masked bigram HE integration."""

    alphabet: str
    name: str = "pllm-alphabet-v1"

    def __post_init__(self) -> None:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet contains duplicate characters")
        if len(self.alphabet) < 2:
            raise ValueError("alphabet is too small")

    @property
    def bos_token_id(self) -> int:
        return len(self.alphabet)

    @property
    def eos_token_id(self) -> int:
        return len(self.alphabet) + 1

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 2

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        mapping = {char: index for index, char in enumerate(self.alphabet)}
        fallback = mapping.get(" ", 0)
        values = [mapping.get(char, fallback) for char in text]
        return ([self.bos_token_id] + values) if add_bos else values

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.alphabet[token] for token in tokens if 0 <= token < len(self.alphabet))


DEFAULT_ALPHABET = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n:-_/()[]{}'\""


def render_responses_input(body: dict[str, Any]) -> str:
    """Canonical client-side rendering of Responses input and function tools.

    The server receives only tokenized/encrypted output from this function. The
    rendering is deliberately deterministic so previous-response caching can
    key on its hash without storing plaintext server-side.
    """

    chunks: list[str] = []
    instructions = body.get("instructions")
    if instructions:
        chunks.append(f"<|developer|>\n{_content_text(instructions)}\n")

    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        chunks.append(f"<|user|>\n{raw_input}\n")
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                chunks.append(str(item))
                continue
            role = item.get("role", "user")
            content = _content_text(item.get("content", ""))
            chunks.append(f"<|{role}|>\n{content}\n")

    tools = body.get("tools") or []
    if tools:
        chunks.append("<|tools|>\n")
        chunks.append(json.dumps(tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        chunks.append("\n")

    chunks.append("<|assistant|>\n")
    return "".join(chunks)


def prompt_fingerprint(rendered: str) -> str:
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in {"input_text", "output_text", "text"}:
                    parts.append(str(item.get("text", "")))
                elif kind == "function_call_output":
                    parts.append(str(item.get("output", "")))
                elif kind in {"input_image", "input_file", "input_audio"}:
                    # Binary and URL media must be handled by a modality plugin.
                    parts.append(f"<{kind}>")
                else:
                    parts.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)

@dataclass(frozen=True, slots=True)
class PieceTokenizer:
    """Tiny piece vocabulary for the executable HE reference model."""

    pieces: tuple[str, ...]
    name: str = "pllm-piece-v1"

    @property
    def bos_token_id(self) -> int:
        return len(self.pieces)

    @property
    def eos_token_id(self) -> int:
        return len(self.pieces) + 1

    @property
    def vocab_size(self) -> int:
        return len(self.pieces) + 2

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        # The reference model derives its initial state from a prompt digest;
        # this method exists only for usage accounting and round-trip tests.
        values = [index for index, piece in enumerate(self.pieces) if piece.strip() in text]
        return ([self.bos_token_id] + values) if add_bos else values

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.pieces[token] for token in tokens if 0 <= token < len(self.pieces))
