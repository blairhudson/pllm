"""Research-only bounded plaintext TCP MLP conveyor; never a private SDK channel.

No checkpoint loading. --measure uses serial guarded fresh workers; default screen
only exercises tiny generated correctness, real TCP, cancellation and wire faults.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import secrets
import struct
import sys
import tempfile
import time
import traceback
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_mlp_conveyor import Contract, Lease, local_product, project, public_order

# Fixed envelope: no arbitrary JSON metadata or data-dependent length decisions.
HEADER = struct.Struct("!4sBB16s32sIBHHHII")
CAP = 16 << 20
CHUNK = 8192
MODES = ("serial", "conveyor", "full")
SOURCES = ("scripts/probe_mlp_conveyor_transport.py", "scripts/probe_mlp_conveyor.py",
           "tests/test_mlp_conveyor_transport.py", "scripts/frontier_probe_support.py",
           "python/pllm/runtime/native_wan.py", "python/pllm/runtime/benchmark_memory.py",
           "crates/pllm-compiler/src/mlp_tiling.rs")


def fixture(c):
    rng = np.random.default_rng(4101 + c.rows)
    x = rng.normal(size=(c.rows, c.hidden)).astype(np.float32)
    g = rng.integers(-5, 6, size=(2 * c.intermediate, c.hidden), dtype=np.int8)
    d = rng.integers(-5, 6, size=(c.hidden, c.intermediate), dtype=np.int8)
    return x, g, d


def descriptors(c):
    return [(stage, tile, start, min(c.chunk, end - start))
            for stage, tile in public_order(c)
            for begin, end in [c.tiles()[tile]] for start in range(begin, end, c.chunk)]


def geometry(rows, hidden, intermediate, tile, mode):
    if mode not in MODES:
        raise ValueError("unknown mode")
    if (hidden, intermediate) not in ((8, 13), (896, 4864)):
        raise ValueError("fixture admits tiny8/13 or synthetic Qwen896/4864 only")
    # Full is a distinct untiled stage control, with identical chunk16 transport.
    if mode == "full":
        if rows > 256:
            # Contract caps tiled rows at 256; full288 remains separate validated
            # descriptor geometry, not a forged production tiling selection.
            c = Contract(rows, hidden=hidden, intermediate=intermediate, tile=256, depth=1)
            return FullGeometry(c)
        tile = rows
    return Contract(rows, hidden=hidden, intermediate=intermediate,
                    tile=tile, depth=2 if mode == "conveyor" else 1)


class FullGeometry:
    """Untiled control only; never fed to native tiling admission."""
    def __init__(self, c):
        self.__dict__.update(asdict(c))
        self.tile = c.rows

    def tiles(self):
        return [(0, self.rows)]


def binding_digest(c):
    spec = vars(c) | {"numeric": "probe_mlp_conveyor.project+local_product:v1",
                      "wire": "network-header+little-endian-f32:v1"}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).digest()


def envelope(c, session, sequence, desc, reply, body, binding=None):
    stage, tile, start, count = desc
    width = ((2 * c.intermediate if stage == "G" else c.hidden) if reply else
             (c.hidden if stage == "G" else c.intermediate))
    if len(body) != count * width * 4 or len(body) > CAP:
        raise ValueError("body differs from public geometry")
    return HEADER.pack(b"MLPC", 1, int(reply), session, binding or binding_digest(c), sequence,
                       ord(stage), tile, start, count, width, len(body)) + body


def validate_header(raw, c, session, sequence, desc, reply, binding=None):
    stage, tile, start, count = desc
    width = ((2 * c.intermediate if stage == "G" else c.hidden) if reply else
             (c.hidden if stage == "G" else c.intermediate))
    expected = (b"MLPC", 1, int(reply), session, binding or binding_digest(c), sequence,
                ord(stage), tile, start, count, width, count * width * 4)
    if HEADER.unpack(raw) != expected or expected[-1] > CAP:
        raise ValueError("frame/session/binding/sequence/stage/row/width mismatch")
    return expected[-1]


def compiler_binding(c):
    """Public native eligibility only; does not bind fixture kernels to SDK numeric."""
    from pllm import Model, _native, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.schedulers import MlpPrefillTiling
    config = {"model_type": "qwen2", "hidden_size": c.hidden,
              "intermediate_size": c.intermediate, "num_hidden_layers": 1,
              "num_attention_heads": 2 if c.hidden == 8 else 14,
              "num_key_value_heads": 2, "vocab_size": 32,
              "max_position_embeddings": 512, "rms_norm_eps": 1e-6,
              "rope_theta": 1000000.0, "hidden_act": "silu", "tie_word_embeddings": False}
    tile = min(c.tile, 256)
    plan = lower_model(config, batch=1, max_input_tokens=c.rows, max_new_tokens=8)
    pipeline = MaskedLinearCpu(Model.tiny(), scheduler=MlpPrefillTiling(tile_rows=tile))
    raw = _native.decoder_mlp_tiling(plan.canonical_bytes(), pipeline.canonical_bytes(), tile)
    contract = json.loads(raw)
    matches = [r for r in contract["regions"] if
               (r["hidden_width"], r["intermediate_width"]) == (c.hidden, c.intermediate)]
    if len(matches) != 1:
        raise ValueError("native compiler did not admit generated MLP geometry")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "contract": contract,
            "scope": "generated public region eligibility only; fixture arithmetic is not SDK numeric"}


async def run_case(rows=17, hidden=8, intermediate=13, tile=7, mode="conveyor", *,
                   paced=False, cancel_after_stage=None, fault=None, timeout=30,
                   compiler_contract_sha256=None, kernel_fail_sequence=1):
    c = geometry(rows, hidden, intermediate, tile, mode)
    ds = descriptors(c)
    if fault not in (None, "session", "sequence", "row", "truncate", "provider", "kernel",
                     "request_session", "request_sequence", "request_row"):
        raise ValueError("unknown fault")
    x, gate, down = fixture(c)
    # Complete output oracle exists independently of candidate stage partition.
    scalar = hidden <= 8 and intermediate <= 13
    both = project(x, gate, scalar=scalar)
    expected = project(local_product(both[:, :intermediate], both[:, intermediate:]),
                       down, scalar=scalar)
    del both
    result = np.empty_like(x)  # Original full output is never elided.
    residual = x.copy()  # Explicit retained original residual owner.
    session = secrets.token_bytes(16)
    wire_binding = binding_digest(c)
    if compiler_contract_sha256 is not None:
        wire_binding = hashlib.sha256(wire_binding + bytes.fromhex(compiler_contract_sha256)).digest()
    client_lease, server_lease = Lease(rows), Lease(rows)
    ledger = {name: 0 for name in ("client_tx", "server_rx", "server_tx", "client_rx")}
    parts = {name: {"header": 0, "body": 0} for name in ledger}
    events = []
    slots = {}
    references = []
    provider_writers = set()
    handlers = set()
    native_jobs = set()
    errors = []
    queue = asyncio.Queue(maxsize=1)
    credits = asyncio.Semaphore(2)
    failed = asyncio.get_running_loop().create_future()
    state = {"inflight": 0, "peak_frame_credits": 0, "peak_tile_owners": 0,
             "peak_retained_gate_product_bytes": 0,
             "peak_request_queue": 0, "provider_cpu_seconds": 0.0,
             "provider_calls": 0, "reply_frames": 0, "request_frames": 0,
             "submitted_frames": 0, "completed_stages": 0, "canceled": False}
    pacer = None
    if paced:
        from pllm.deployment import WanConditions
        from pllm.runtime.native_wan import SharedPacer
        pacer = SharedPacer(WanConditions(download_mbps=100., upload_mbps=40.),
                            ("client", "provider"))

    def fail(error):
        errors.append(type(error).__name__ + ": " + str(error))
        if not failed.done():
            failed.set_result(errors[-1])

    async def exact(reader, n, direction, part):
        try:
            value = await reader.readexactly(n)
        except asyncio.IncompleteReadError as error:
            ledger[direction] += len(error.partial)
            parts[direction][part] += len(error.partial)
            raise
        ledger[direction] += len(value)
        parts[direction][part] += len(value)
        return value

    async def write(writer, data, direction):
        source, dest = (("client", "provider") if direction == "client_tx" else
                        ("provider", "client"))
        for offset in range(0, len(data), CHUNK):
            chunk = data[offset:offset + CHUNK]
            if pacer:
                await pacer.reserve(source, dest, len(chunk))
            writer.write(chunk)
            ledger[direction] += len(chunk)
            header = max(0, min(len(chunk), HEADER.size - offset))
            parts[direction]["header"] += header
            parts[direction]["body"] += len(chunk) - header
            await writer.drain()

    def kernel(value, weight, sequence):
        cpu = time.thread_time()
        if fault == "kernel" and sequence == kernel_fail_sequence:
            raise RuntimeError("injected executing kernel failure")
        output = project(value, weight)
        return output, time.thread_time() - cpu

    async def provider(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        provider_writers.add(writer)
        writer.transport.set_write_buffer_limits(high=2 * CHUNK, low=CHUNK)
        job = None
        try:
            for seq, desc in enumerate(ds):
                raw = await exact(reader, HEADER.size, "server_rx", "header")
                n = validate_header(raw, c, session, seq, desc, False, wire_binding)
                payload = await exact(reader, n, "server_rx", "body")
                stage, index, start, count = desc
                server_lease.claim(stage, start, start + count, seq)
                state["request_frames"] += 1
                if fault == "provider" and seq == 1:
                    raise RuntimeError("injected provider failure after claim")
                width = hidden if stage == "G" else intermediate
                value = np.frombuffer(payload, dtype="<f4").reshape(count, width)
                job = asyncio.create_task(asyncio.to_thread(kernel, value, gate if stage == "G" else down, seq))
                native_jobs.add(job)
                output, cpu = await asyncio.shield(job)
                native_jobs.discard(job)
                job = None
                state["provider_cpu_seconds"] += cpu
                state["provider_calls"] += 1
                body = output.astype("<f4", copy=False).tobytes()
                packet = envelope(c, session, seq, desc, True, body, wire_binding)
                if seq == 1 and fault in ("session", "sequence", "row"):
                    fields = list(HEADER.unpack(packet[:HEADER.size]))
                    field = {"session": 3, "sequence": 5, "row": 8}[fault]
                    fields[field] = b"x" * 16 if fault == "session" else fields[field] + 1
                    packet = HEADER.pack(*fields) + body
                if seq == 1 and fault == "truncate":
                    await write(writer, packet[:-1], "server_tx")
                    return
                events.append({"event": "provider_reply_start", "sequence": seq,
                               "seconds": time.perf_counter() - started})
                await write(writer, packet, "server_tx")
                events.append({"event": "provider_reply_end", "sequence": seq,
                               "seconds": time.perf_counter() - started})
                # Retire previous request/kernel/reply before another receive.
                del value, payload, output, body, packet
            # Connection remains open until client explicitly retires its attempt.
            await reader.read(1)
        except (ConnectionError, asyncio.IncompleteReadError) as error:
            if not state["canceled"] and not failed.done():
                fail(error)
        except Exception as error:
            fail(error)
        finally:
            try:
                if job is not None:
                    # gather consumes a failed kernel too. Never let re-awaiting
                    # its exception bypass mandatory retirement below.
                    outcomes = await asyncio.gather(job, return_exceptions=True)
                    for outcome in outcomes:
                        if isinstance(outcome, BaseException):
                            if not errors:
                                fail(outcome)
                            # Failed task tracebacks otherwise retain kernel input
                            # views (including down products) via reference cycles.
                            traceback.clear_frames(outcome.__traceback__)
                            outcome.__traceback__ = None
            finally:
                if job is not None:
                    native_jobs.discard(job)
                try:
                    server_lease.close()
                except Exception as error:
                    fail(error)
                finally:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except ConnectionError:
                        pass
                    except Exception as error:
                        fail(error)
                    finally:
                        provider_writers.discard(writer)
                        handlers.discard(task)

    started, cpu_started = time.perf_counter(), time.process_time()
    server = await asyncio.start_server(provider, "127.0.0.1", 0, limit=2 * CHUNK)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1], limit=2 * CHUNK)
    except BaseException:
        server.close()
        await server.wait_closed()
        for peer in tuple(provider_writers):
            peer.close()
        if handlers:
            await asyncio.gather(*tuple(handlers), return_exceptions=True)
        raise
    writer.transport.set_write_buffer_limits(high=2 * CHUNK, low=CHUNK)

    async def sender():
        try:
            while True:
                seq, desc, value = await queue.get()
                try:
                    await credits.acquire()
                    state["inflight"] += 1
                    state["peak_frame_credits"] = max(state["peak_frame_credits"], state["inflight"])
                    stage, _, start, count = desc
                    client_lease.claim(stage, start, start + count, seq)
                    packet = envelope(c, session, seq, desc, False,
                                      value.astype("<f4", copy=False).tobytes(), wire_binding)
                    if seq == 1 and fault and fault.startswith("request_"):
                        fields = list(HEADER.unpack(packet[:HEADER.size]))
                        field = {"request_session": 3, "request_sequence": 5, "request_row": 8}[fault]
                        fields[field] = b"x" * 16 if field == 3 else fields[field] + 1
                        packet = HEADER.pack(*fields) + packet[HEADER.size:]
                    events.append({"event": "client_send_start", "sequence": seq,
                                   "stage": stage, "row": start,
                                   "seconds": time.perf_counter() - started})
                    await write(writer, packet, "client_tx")
                    events.append({"event": "client_send_end", "sequence": seq,
                                   "seconds": time.perf_counter() - started})
                    del packet, value
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            fail(error)

    async def receiver():
        try:
            for seq, desc in enumerate(ds):
                raw = await exact(reader, HEADER.size, "client_rx", "header")
                n = validate_header(raw, c, session, seq, desc, True, wire_binding)
                payload = await exact(reader, n, "client_rx", "body")
                stage, index, start, count = desc
                slot = slots[index]
                begin, end = c.tiles()[index]
                width = 2 * intermediate if stage == "G" else hidden
                value = np.frombuffer(payload, dtype="<f4").reshape(count, width)
                target = slot["gate"] if stage == "G" else result[begin:end]
                target[start - begin:start - begin + count] = value
                if start + count == end:
                    slot[stage].set_result(None)
                state["reply_frames"] += 1
                state["inflight"] -= 1
                credits.release()
                events.append({"event": "client_reply_done", "sequence": seq,
                               "seconds": time.perf_counter() - started})
                del value, payload, target, slot
        except asyncio.CancelledError:
            raise
        except Exception as error:
            fail(error)

    send_task, receive_task = asyncio.create_task(sender()), asyncio.create_task(receiver())
    seq = 0

    def tile_bytes():
        size = sum(value.nbytes for slot in slots.values() for key, value in slot.items()
                   if key in ("gate", "product"))
        state["peak_retained_gate_product_bytes"] = max(state["peak_retained_gate_product_bytes"], size)

    async def submit(stage, index):
        nonlocal seq
        begin, end = c.tiles()[index]
        if stage == "G":
            if len(slots) >= c.depth:
                raise AssertionError("tile credit exceeded")
            slots[index] = {"gate": np.empty((end - begin, 2 * intermediate), np.float32),
                            "G": asyncio.get_running_loop().create_future(),
                            "D": asyncio.get_running_loop().create_future()}
            references.append(weakref.ref(slots[index]["gate"]))
            state["peak_tile_owners"] = max(state["peak_tile_owners"], len(slots))
            tile_bytes()
        value = x[begin:end] if stage == "G" else slots[index]["product"]
        for start in range(begin, end, c.chunk):
            count = min(c.chunk, end - start)
            desc = (stage, index, start, count)
            if desc != ds[seq]:
                raise AssertionError("schedule is not public descriptor order")
            await queue.put((seq, desc, value[start - begin:start - begin + count]))
            state["peak_request_queue"] = max(state["peak_request_queue"], queue.qsize())
            state["submitted_frames"] += 1
            seq += 1

    async def complete(stage, index):
        await slots[index][stage]
        state["completed_stages"] += 1
        if state["completed_stages"] == cancel_after_stage:
            state["canceled"] = True
            raise asyncio.CancelledError("public test cancellation cut")

    async def conveyor():
        try:
            await submit("G", 0)
            for index in range(len(c.tiles())):
                if c.depth == 2 and index + 1 < len(c.tiles()):
                    await submit("G", index + 1)
                await complete("G", index)
                both = slots[index].pop("gate")
                product = local_product(both[:, :intermediate], both[:, intermediate:])
                references.append(weakref.ref(product))
                slots[index]["product"] = product
                tile_bytes()
                del both, product
                await submit("D", index)
                await complete("D", index)
                slots.pop(index)
                if c.depth == 1 and index + 1 < len(c.tiles()):
                    await submit("G", index + 1)
        except asyncio.CancelledError:
            if not state["canceled"]:
                raise
            # Drain every submitted accepted request, then retire all unused rows.
            await queue.join()
            while state["reply_frames"] < state["submitted_frames"]:
                await asyncio.sleep(0.001)

    coordinator = asyncio.create_task(conveyor())
    success = False
    try:
        done, _ = await asyncio.wait((coordinator, failed), timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if not done:
            fail(TimeoutError("bounded transport deadline"))
        elif failed not in done:
            await coordinator
            success = not state["canceled"]
    except Exception as error:
        fail(error)
    finally:
        for task in (coordinator, send_task, receive_task):
            task.cancel()
        # Cancellation while draining must not be caught as a fresh soft cancel.
        await asyncio.gather(coordinator, send_task, receive_task, return_exceptions=True)
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass
        server.close()
        await server.wait_closed()
        for peer in tuple(provider_writers):
            peer.close()
        if handlers:
            await asyncio.gather(*tuple(handlers), return_exceptions=True)
        if native_jobs:
            await asyncio.gather(*tuple(native_jobs), return_exceptions=True)
        client_burned = client_lease.close()
        server_burned = server_lease.close()
        # Explicitly clear queued ndarray views after interrupted sends.
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
        slots.clear()
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    exact_bits = bool(np.array_equal(result.view(np.uint32), expected.view(np.uint32))) if success else None
    if success and not exact_bits:
        raise AssertionError("complete output bits changed")
    if any(ref() is not None for ref in references):
        raise AssertionError("gate/product owner retained after terminal cleanup")
    tx_body = rows * (hidden + intermediate) * 4
    rx_body = rows * (2 * intermediate + hidden) * 4
    if success and (ledger["client_tx"], ledger["client_rx"]) != (
            tx_body + len(ds) * HEADER.size, rx_body + len(ds) * HEADER.size):
        raise AssertionError("wire bytes differ from closed-form oracle")
    reconciled = ledger["client_tx"] == ledger["server_rx"] and ledger["server_tx"] == ledger["client_rx"]
    if (success or state["canceled"]) and not reconciled:
        raise AssertionError("drained transport bodies did not reconcile")
    replay = False
    try:
        client_lease.claim("G", 0, 1, 0)
    except ValueError:
        replay = True
    output_hash = hashlib.sha256(result.tobytes()).hexdigest() if success else None
    oracle_hash = hashlib.sha256(expected.tobytes()).hexdigest()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    intervals = {}
    for event in events:
        intervals[event["event"], event["sequence"]] = event["seconds"]
    overlap = 0.0
    for seq, (stage, index, _, _) in enumerate(ds):
        if stage != "G" or index == 0:
            continue
        for prior, (other, tile_index, _, _) in enumerate(ds[:seq]):
            if other != "G" or tile_index != index - 1:
                continue
            keys = (("client_send_start", seq), ("client_send_end", seq),
                    ("provider_reply_start", prior), ("provider_reply_end", prior))
            if all(key in intervals for key in keys):
                a, b, p, q = [intervals[key] for key in keys]
                overlap += max(0., min(b, q) - max(a, p))
    return {"scope": "generated plaintext single-process loopback TCP operator fixture",
            "geometry": vars(c), "mode": mode, "success": success, "exact_bits": exact_bits,
            "injection": {"fault": fault, "cancel_after_stage": cancel_after_stage,
                          "kernel_fail_sequence": kernel_fail_sequence if fault == "kernel" else None},
            "wire_binding_sha256": wire_binding.hex(),
            "compiler_contract_sha256": compiler_contract_sha256,
            "scalar_integer_oracle": scalar, "canceled": state["canceled"], "errors": errors,
            "session_replay_rejected": replay, "terminal_product_owners": 0,
            "weakref_retirement": True, "ledger": ledger, "wire_parts": parts,
            "wire_reconciled": reconciled,
            "unread_request_bytes": ledger["client_tx"] - ledger["server_rx"],
            "unread_reply_bytes": ledger["server_tx"] - ledger["client_rx"],
            "client_claimed_rows": len(client_lease.claimed), "client_burned_rows": client_burned,
            "provider_claimed_rows": len(server_lease.claimed), "provider_burned_rows": server_burned,
            "frames_expected": len(ds), "request_body_bytes_expected": tx_body,
            "reply_body_bytes_expected": rx_body, "header_bytes_per_frame": HEADER.size,
            "region_seconds": elapsed, "region_process_cpu_seconds": cpu,
            "region_nonprovider_cpu_seconds": max(0., cpu - state["provider_cpu_seconds"]),
            "next_gate_upload_vs_previous_gate_reply_write_overlap_seconds": overlap,
            "overlap_scope": "sum of next-tile G send-write vs previous-tile G reply-write interval intersections only; zero excludes neither request prepositioning nor compute overlap and establishes no timing causality",
            "whole_fixture_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "retained_source_output_residual_bytes": x.nbytes + result.nbytes + residual.nbytes,
            "retained_oracle_output_bytes": expected.nbytes,
            "generated_provider_weight_bytes": gate.nbytes + down.nbytes,
            "output_sha256": output_hash,
            "oracle_output_sha256": oracle_hash,
            "terminal_resources": {"provider_handlers": len(handlers),
                                   "provider_writers": len(provider_writers),
                                   "native_jobs": len(native_jobs), "tile_slots": len(slots),
                                   "queued_requests": queue.qsize(),
                                   "client_lease_closed": client_lease.closed,
                                   "provider_lease_closed": server_lease.closed},
            "remote_fixture_body_mac_fraction": 1.0 if success else None,
            "sdk_remote_body_mac_fraction": None, "state": state, "events": events,
            "pacer": pacer.snapshot() if pacer else None,
            "unknown_owners": ["SDK KV/state and other operators", "real correlations/corrections/inventory",
                               "real checkpoint and preparation", "separate process topology and relay",
                               "kernel socket buffers / IP ACKs and retransmissions"],
            "whole_client_admitted": False, "timing_privacy": "unpadded plaintext fixture; not private deployment"}


def source_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def reserve_output(path, report):
    """Exclusive reservation precedes admission/work. Existing files never touched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint_output(path, report):
    """Atomic replacement of this attempt's reservation, never an unrelated file."""
    current = json.loads(path.read_text())
    if current.get("attempt_id") != report["attempt_id"]:
        raise RuntimeError("output reservation ownership changed")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=path.name + ".", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist rename too, where directory fsync is supported.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def guard_observation(guard, host, sample):
    def host_spec(value):
        return asdict(value) | {"reserve_bytes": value.reserve}
    observation = {"initial_host": host_spec(host),
                   "guard_error": guard.error,
                   "minimum_available_bytes": guard.minimum_available_bytes,
                   "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                   "scope": "cumulative shared guard observations, not per-arm allocation/RSS"}
    try:
        observation["current_host"] = host_spec(sample())
    except Exception as error:
        observation["sample_error"] = str(error)[:4096]
    return observation


def validate_accepted_case(case, payload, hashes, build, bound, prior):
    """No accepted status until worker identity, full oracle and wire reconcile."""
    c = geometry(**payload)
    count = len(descriptors(c))
    def require(condition, reason):
        if not condition:
            raise ValueError("case validation: " + reason)
    require(case["source_sha256"] == hashes and source_hashes() == hashes, "source identity")
    require(case["build"] == build, "installed build identity")
    require(case["geometry"] == vars(c) and case["mode"] == payload["mode"], "public case geometry")
    require(case["success"] is True and case["exact_bits"] is True
            and not case["canceled"] and not case["errors"], "complete exact successful attempt")
    output_hash = case["output_sha256"]
    require(isinstance(output_hash, str) and len(output_hash) == 64
            and all(ch in "0123456789abcdef" for ch in output_hash), "output SHA-256 format")
    require(case["oracle_output_sha256"] == output_hash, "full-output oracle hash")
    require(all(other["output_sha256"] == output_hash for other in prior
                if (other["geometry"]["rows"], other["geometry"]["hidden"], other["geometry"]["intermediate"]) ==
                (c.rows, c.hidden, c.intermediate)), "cross-control output hash")
    require(case["compiler_eligibility"] == bound
            and canonical_hash(bound["contract"]) == bound["sha256"]
            and case["compiler_contract_sha256"] == bound["sha256"], "native compiler binding")
    binding = hashlib.sha256(binding_digest(c) + bytes.fromhex(bound["sha256"])).hexdigest()
    require(case["wire_binding_sha256"] == binding, "wire numeric/compiler binding")
    bodies = {"client_tx": c.rows * (c.hidden + c.intermediate) * 4,
              "client_rx": c.rows * (2 * c.intermediate + c.hidden) * 4}
    bodies["server_rx"], bodies["server_tx"] = bodies["client_tx"], bodies["client_rx"]
    for direction, size in bodies.items():
        require(case["wire_parts"][direction] == {"body": size, "header": count * HEADER.size}
                and case["ledger"][direction] == size + count * HEADER.size, "measured directed bodies/headers")
    require(case["frames_expected"] == count and case["header_bytes_per_frame"] == HEADER.size
            and case["request_body_bytes_expected"] == bodies["client_tx"]
            and case["reply_body_bytes_expected"] == bodies["client_rx"], "independent wire oracle")
    require(case["wire_reconciled"] is True and case["unread_request_bytes"] == case["unread_reply_bytes"] == 0,
            "wire reconciliation")
    require(all(case["state"][key] == count for key in ("request_frames", "reply_frames", "submitted_frames")),
            "measured frame counts")
    sends = [(e["sequence"], e["stage"], e["row"]) for e in case["events"]
             if e["event"] == "client_send_start"]
    require(sends == [(seq, stage, row) for seq, (stage, _, row, _) in enumerate(descriptors(c))],
            "deterministic public send schedule")
    for event_name in ("client_send_end", "provider_reply_start", "provider_reply_end", "client_reply_done"):
        require([e["sequence"] for e in case["events"] if e["event"] == event_name] == list(range(count)),
                "complete ordered frame trace")
    require(all(math.isfinite(e["seconds"]) and e["seconds"] >= 0 for e in case["events"]),
            "finite frame timestamps")
    for peer in ("client", "provider"):
        require(case[peer + "_claimed_rows"] == 2 * c.rows and case[peer + "_burned_rows"] == 0,
                "one-use row conservation")
    resources = case["terminal_resources"]
    require(all(resources[key] == 0 for key in ("provider_handlers", "provider_writers", "native_jobs",
                                               "tile_slots", "queued_requests"))
            and resources["client_lease_closed"] and resources["provider_lease_closed"], "mandatory retirement")
    require(case["weakref_retirement"] and case["session_replay_rejected"]
            and case["terminal_product_owners"] == 0, "array/replay retirement")
    require(0 < case["state"]["peak_tile_owners"] <= c.depth
            and 0 < case["state"]["peak_frame_credits"] <= 2
            and case["state"]["peak_request_queue"] <= 1, "owner bounds")
    pacer = case["pacer"]
    require(pacer["directed_stream_bytes"] == {"client->provider": case["ledger"]["client_tx"],
                                                "provider->client": case["ledger"]["server_tx"]}
            and pacer["total_stream_bytes"] == sum(case["ledger"][key] for key in ("client_tx", "server_tx")),
            "shared pacer directed bytes")
    require(pacer["enforced"] is True and pacer["role_parties"] == {"client": "client", "provider": "provider"}
            and pacer["conditions"] == {"schema": "pllm.wan_conditions.v1", "download_mbps": 100.0,
                                       "upload_mbps": 40.0, "parties": [], "role_parties": []},
            "shared100/40 pacing contract")


def measure_cohort(output, payloads, hashes, *, workers=None, sample=None, capture_build=None,
                   compile_binding=compiler_binding):
    """Durable serial driver; dependencies injectable for tiny failure regressions."""
    report = {"schema": "pllm.mlp_conveyor_transport.v2", "attempt_id": secrets.token_hex(16),
              "status": "incomplete", "phase": "reserved", "source_sha256": hashes,
              "build": None, "build_status": "not_captured", "measured_cohort": True,
              "serving_claim": False, "cases": [], "attempts": [], "guard_observations": [],
              "plan": payloads, "bounds": {"worker_timeout_seconds": 240,
                  "transport_timeout_seconds": 180, "planned_workers": len(payloads),
                  "worker_wait_ceiling_seconds": len(payloads) * 240, "failure_message_max_chars": 4096},
              "admission_memory_bytes": 2 << 30, "failure": None}
    reserve_output(output, report)
    guard = host = None
    try:
        if workers is None:
            from frontier_probe_support import fresh_workers
            workers = fresh_workers
        if sample is None:
            from pllm.runtime.benchmark_memory import host_memory
            sample = host_memory
        if capture_build is None:
            from pllm.metrics import BuildRef
            capture_build = lambda: BuildRef.capture().to_spec()
        report["build"] = capture_build()
        report["build_status"] = "captured_before_admission"
        report["phase"] = "admission"
        checkpoint_output(output, report)
        with workers(Path(__file__).resolve(), memory_bytes=2 << 30, timeout=240) as (run, guard, host):
            if run.build != report["build"] or source_hashes() != hashes:
                raise RuntimeError("build/source changed before cohort execution")
            report["build_status"] = "guarded"
            for index, payload in enumerate(payloads):
                guard.check()
                report["guard_observations"].append(guard_observation(guard, host, sample))
                if guard.error or source_hashes() != hashes or capture_build() != report["build"]:
                    raise RuntimeError("guard/source/build changed before arm")
                report["phase"] = "running"
                attempt = {"index": index, "payload": payload, "status": "running"}
                report["attempts"].append(attempt)
                checkpoint_output(output, report)  # Durable bounds/identity before execution.
                bound = compile_binding(geometry(**payload))
                case = run(payload)
                guard.check()
                report["guard_observations"].append(guard_observation(guard, host, sample))
                if guard.error or capture_build() != report["build"]:
                    raise RuntimeError("guard/build changed after arm")
                try:
                    validate_accepted_case(case, payload, hashes, report["build"], bound, report["cases"])
                except Exception:
                    attempt["status"] = "rejected"
                    attempt["returned_case"] = case
                    raise
                case["acceptance_status"] = "accepted"
                report["cases"].append(case)
                attempt["status"] = "accepted"
                checkpoint_output(output, report)
            report["phase"] = "final_guard"
            guard.check()
        if capture_build() != report["build"] or source_hashes() != hashes:
            raise RuntimeError("build/source changed at cohort close")
        report["cohort_exact_and_reconciled"] = True
        report["controls_equal_outputs"] = True
        report["status"], report["phase"] = "complete", "closed"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)[:4096],
                             "phase": report["phase"], "accepted_arms": len(report["cases"])}
        if report["attempts"] and report["attempts"][-1]["status"] == "running":
            report["attempts"][-1]["status"] = "failed"
    finally:
        if guard is not None:
            report["guard_observations"].append(guard_observation(guard, host, sample))
        checkpoint_output(output, report)
    return report


def screen():
    cases = [asyncio.run(run_case(mode=mode)) for mode in MODES]
    cases += [asyncio.run(run_case(cancel_after_stage=cut)) for cut in (1, 2, 3)]
    cases += [asyncio.run(run_case(fault=fault)) for fault in
              ("session", "sequence", "row", "truncate", "provider", "kernel",
               "request_session", "request_sequence", "request_row")]
    cases.append(asyncio.run(run_case(fault="kernel", kernel_fail_sequence=2)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rows", type=int, choices=(150, 288), nargs="+", default=[150, 288])
    args = parser.parse_args()
    if args.worker:
        from pllm.metrics import BuildRef
        hashes, build = source_hashes(), BuildRef.capture().to_spec()
        payload = json.load(sys.stdin)
        c = geometry(**payload)
        bound = compiler_binding(c)
        result = asyncio.run(run_case(**payload, paced=True, timeout=180,
                                     compiler_contract_sha256=bound["sha256"]))
        result["compiler_eligibility"] = bound
        if source_hashes() != hashes or BuildRef.capture().to_spec() != build:
            raise RuntimeError("worker source/build changed")
        result["source_sha256"], result["build"] = hashes, build
        print(json.dumps(result, allow_nan=False))
        return
    if args.output is None or args.output.exists():
        parser.error("require fresh --output")
    hashes = source_hashes()
    if args.source_lock and json.loads(args.source_lock.read_text())["source_sha256"] != hashes:
        parser.error("source lock differs")
    result = {"schema": "pllm.mlp_conveyor_transport.v1", "source_sha256": hashes,
              "measured_cohort": args.measure, "serving_claim": False}
    if args.measure:
        if len(args.rows) > 2 or len(set(args.rows)) != len(args.rows):
            parser.error("require one or two distinct public row shapes")
        payloads = [{"rows": rows, "hidden": 896, "intermediate": 4864, "tile": 64, "mode": mode}
                    for rows in args.rows for mode in MODES]
        result = measure_cohort(args.output, payloads, hashes)
        if result["status"] != "complete":
            raise SystemExit(1)
        return
    else:
        result["schema"] = "pllm.mlp_conveyor_transport.v2"
        result["cases"] = screen()
        try:
            result["tiny_compiler_eligibility"] = compiler_binding(geometry(17, 8, 13, 7, "conveyor"))
            result["cases"].append(asyncio.run(run_case(
                compiler_contract_sha256=result["tiny_compiler_eligibility"]["sha256"])))
        except ImportError as error:
            result["tiny_compiler_eligibility"] = {"unavailable": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
