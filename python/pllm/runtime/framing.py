from __future__ import annotations

import asyncio
import socket
import struct
from dataclasses import dataclass
from typing import Any

import msgpack

MAGIC = b"HER1"
PREFIX = struct.Struct("!4sIQ")
DEFAULT_MAX_HEADER = 1 << 20
DEFAULT_MAX_PAYLOAD = 1 << 31


class FrameError(ValueError):
    pass


@dataclass(slots=True)
class Frame:
    header: dict[str, Any]
    payload: bytes = b""


def encode_frame(header: dict[str, Any], payload: bytes = b"") -> bytes:
    if not isinstance(header, dict):
        raise FrameError("header must be a map")
    if not isinstance(payload, bytes):
        raise FrameError("payload must be bytes")
    packed = msgpack.packb(header, use_bin_type=True)
    return PREFIX.pack(MAGIC, len(packed), len(payload)) + packed + payload


def decode_frame(data: bytes, *, max_header: int = DEFAULT_MAX_HEADER, max_payload: int = DEFAULT_MAX_PAYLOAD) -> Frame:
    if len(data) < PREFIX.size:
        raise FrameError("truncated frame prefix")
    magic, header_len, payload_len = PREFIX.unpack_from(data)
    if magic != MAGIC:
        raise FrameError("invalid frame magic")
    if header_len > max_header or payload_len > max_payload:
        raise FrameError("frame exceeds configured size limit")
    expected = PREFIX.size + header_len + payload_len
    if len(data) != expected:
        raise FrameError(f"frame length mismatch: expected {expected}, got {len(data)}")
    try:
        header = msgpack.unpackb(data[PREFIX.size:PREFIX.size + header_len], raw=False, strict_map_key=True)
    except Exception as exc:
        raise FrameError("invalid MessagePack header") from exc
    if not isinstance(header, dict):
        raise FrameError("decoded header must be a map")
    return Frame(header, data[PREFIX.size + header_len:])


async def read_frame(reader: asyncio.StreamReader, *, max_header: int = DEFAULT_MAX_HEADER, max_payload: int = DEFAULT_MAX_PAYLOAD) -> Frame:
    prefix = await reader.readexactly(PREFIX.size)
    magic, header_len, payload_len = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise FrameError("invalid frame magic")
    if header_len > max_header or payload_len > max_payload:
        raise FrameError("frame exceeds configured size limit")
    header_blob = await reader.readexactly(header_len)
    payload = await reader.readexactly(payload_len)
    try:
        header = msgpack.unpackb(header_blob, raw=False, strict_map_key=True)
    except Exception as exc:
        raise FrameError("invalid MessagePack header") from exc
    if not isinstance(header, dict):
        raise FrameError("decoded header must be a map")
    return Frame(header, payload)


async def write_frame(writer: asyncio.StreamWriter, header: dict[str, Any], payload: bytes = b"") -> None:
    writer.write(encode_frame(header, payload))
    await writer.drain()


def recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket, *, max_header: int = DEFAULT_MAX_HEADER, max_payload: int = DEFAULT_MAX_PAYLOAD) -> Frame:
    prefix = recv_exact(sock, PREFIX.size)
    magic, header_len, payload_len = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise FrameError("invalid frame magic")
    if header_len > max_header or payload_len > max_payload:
        raise FrameError("frame exceeds configured size limit")
    header_blob = recv_exact(sock, header_len)
    payload = recv_exact(sock, payload_len)
    header = msgpack.unpackb(header_blob, raw=False, strict_map_key=True)
    if not isinstance(header, dict):
        raise FrameError("decoded header must be a map")
    return Frame(header, payload)
