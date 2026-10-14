"""Small installed-native controls; no checkpoint, builds, or timing cohort."""
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_column_streaming_linear.py"
SPEC = importlib.util.spec_from_file_location("column_streaming_linear_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class ColumnStreamingTests(unittest.TestCase):
    def assert_burned(self, provider):
        self.assertEqual(provider.state, "terminal")
        self.assertEqual(len(provider.book.burned), provider.contract.rows)
        self.assertIsNone(provider.book.correction)
        self.assertIsNone(provider.correction)
        self.assertFalse(np.any(provider.accumulator))
        with self.assertRaises(probe.StreamError):
            provider.book.claim(provider.contract)

    def test_exact_rings_and_uneven_blocks(self):
        for bits in (16, 24, 32):
            for block in (1, 4, 9):
                with self.subTest(bits=bits, block=block):
                    w = np.array([[3] * 9, [-3] * 9, [1, -1, 0] * 3], dtype=np.int8)
                    client, provider, expected, _ = probe.fixture(2, 9, 3, block, bits, w=w)
                    result, metrics = probe.execute(client, provider)
                    np.testing.assert_array_equal(result, expected)
                    self.assertEqual(metrics["native_launches"], (9 + block - 1) // block)
                    self.assertEqual(provider.correction_applications, 1)
                    self.assertEqual(provider.book.claims, 1)
                    self.assert_burned(provider)

    def test_no_prefix_and_replay(self):
        client, provider, expected, _ = probe.fixture()
        frames = list(client.frames())
        for frame in frames[:-1]:
            self.assertIsNone(provider.receive(frame))
            if provider.launches == 1:
                self.assertLess(provider.offset, provider.contract.columns)
                self.assertEqual(provider.correction_applications, 0)
        response = provider.receive(frames[-1])
        np.testing.assert_array_equal(client.finish(response), expected)
        with self.assertRaises(probe.StreamError):
            client.finish(response)
        with self.assertRaises(probe.StreamError):
            provider.receive(frames[-1])
        self.assertEqual(provider.correction_applications, 1)
        self.assert_burned(provider)

    def test_reorder_duplicate_truncate_tamper(self):
        for attack in ("reorder", "duplicate", "truncate", "tamper", "payload-short", "wrong-offset"):
            with self.subTest(attack=attack):
                client, provider, _, _ = probe.fixture()
                frames = list(client.frames())
                provider.receive(frames[0])
                if attack == "reorder":
                    bad = frames[2]
                elif attack == "duplicate":
                    provider.receive(frames[1])
                    bad = frames[1]
                elif attack == "truncate":
                    bad = frames[1][:-1]
                elif attack == "tamper":
                    raw = bytearray(frames[1])
                    raw[-34] ^= 1
                    bad = bytes(raw)
                else:
                    kind, seq, start, end, payload = probe.open_frame(client.key, client.contract, frames[1])
                    bad = probe.seal(client.key, client.contract, kind, seq,
                        start + (attack == "wrong-offset"), end,
                        payload[:-1] if attack == "payload-short" else payload)
                with self.assertRaises(probe.StreamError):
                    provider.receive(bad)
                self.assert_burned(provider)
                self.assertEqual(provider.correction_applications, 0)

    def test_incomplete_and_cancel(self):
        for cut in ("before-start", "eof", "finish", "cancel", "local-cancel"):
            with self.subTest(cut=cut):
                client, provider, _, _ = probe.fixture()
                frames = list(client.frames())
                if cut != "before-start":
                    provider.receive(frames[0])
                    provider.receive(frames[1])
                if cut in ("before-start", "eof"):
                    with self.assertRaises(probe.StreamError):
                        provider.eof()
                elif cut == "finish":
                    c = client.contract
                    bad = probe.seal(client.key, c, "finish", provider.sequence, c.columns, c.columns)
                    with self.assertRaises(probe.StreamError):
                        provider.receive(bad)
                elif cut == "cancel":
                    provider.receive(probe.seal(client.key, client.contract, "cancel", provider.sequence,
                        provider.offset, provider.offset))
                else:
                    provider.cancel()
                self.assert_burned(provider)

    def test_namespace_binding(self):
        for field in ("session", "stage", "tickets", "bits", "block", "weight_digest", "widths"):
            client, provider, _, _ = probe.fixture()
            c = client.contract
            changes = {"session": "different-session", "stage": "other-stage",
                       "tickets": tuple("0" * 31 + str(i) for i in range(c.rows)),
                       "bits": 32, "block": 5, "weight_digest": "0" * 64,
                       "widths": bytes(v - 1 for v in c.widths)}
            other = replace(c, **{field: changes[field]})
            with self.subTest(field=field), self.assertRaises(probe.StreamError):
                provider.receive(probe.seal(client.key, other, "start", 0, 0, 0))
            self.assert_burned(provider)

    def test_ring_overflow_and_signed_extremes(self):
        # Native partials exceed 2^32 routinely because masked inputs use full
        # ring; reduction must not use floating point or signed int32 sums.
        w = np.array([[-128] * 17, [127] * 17, [0] * 17], dtype=np.int8)
        x = np.array([[127] * 17, [-127] * 17], dtype=np.int8)
        for bits in (24, 32):
            client, provider, expected, _ = probe.fixture(2, 17, 3, 3, bits, w, x)
            client.r.fill((1 << bits) - 1)
            # Replace correction to match deliberate maximal mask.
            kernel = probe.MaskedGEMM(threads=1).compile(w)
            provider.book.correction = probe._native.prepared_pack_correction(
                kernel.wrap32(client.r).tobytes(), client.s.tobytes(), client.contract.widths, 2)
            result, _ = probe.execute(client, provider)
            np.testing.assert_array_equal(result, expected)
        with self.assertRaises(probe.StreamError):
            probe.fixture(2, 17, 3, 3, 16, w, x)  # Output signed bound needs >16 bits.

    def test_resource_and_activation_controls(self):
        client, _, _, _ = probe.fixture()
        for changes in ({"bits": 64}, {"rows": 33}, {"columns": 1025}, {"block": 0},
                        {"session": "x" * 65}, {"tickets": (client.contract.tickets[0],) * 3}):
            with self.subTest(changes=changes), self.assertRaises(probe.StreamError):
                replace(client.contract, **changes)
        client.x[0, 0] = -128
        with self.assertRaises(probe.StreamError):
            probe.Client(client.contract, client.x, client.r, client.s, client.key)

    def test_bad_result_auth_burns_client(self):
        client, provider, _, _ = probe.fixture()
        response = None
        for frame in client.frames():
            response = provider.receive(frame)
        bad = bytearray(response)
        bad[-1] ^= 1
        with self.assertRaises(probe.StreamError):
            client.finish(bytes(bad))
        self.assertTrue(client.closed)
        with self.assertRaises(probe.StreamError):
            list(client.frames())

    def test_transport_failure_burns_both_owners(self):
        client, provider, _, _ = probe.fixture()
        with self.assertRaises(probe.StreamError):
            probe.execute(client, provider, "unknown")
        self.assertTrue(client.closed)
        self.assertFalse(np.any(client.r))
        self.assertFalse(np.any(client.s))
        self.assert_burned(provider)

    def test_small_actual_socket_transport_correctness(self):
        client, provider, expected, _ = probe.fixture(2, 5, 2, 2)
        result, metrics = probe.execute(client, provider, "socketpair")
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(len(metrics["upload_frame_bytes"]), 5)
        self.assert_burned(provider)

    def test_schedule_accounts_final_upload_and_no_prefix_download(self):
        metrics = {"upload_frame_bytes": [100, 1000, 1000, 100], "download_frame_bytes": 50,
                   "partial_native_and_accumulate_seconds": [0.01, 0.01]}
        # uploads [0.0001,0.0011,0.0021,0.0022]; delay=.02;
        # compute ends .0311,.0411; finish reaches .0222.
        self.assertAlmostEqual(probe.modeled_schedule(metrics, 8, 8, 20, 1), 0.06115)


if __name__ == "__main__":
    unittest.main()
