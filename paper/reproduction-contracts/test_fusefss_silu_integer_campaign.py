"""Normal stdlib tests: public arithmetic model, never native/protected code.

Run: python3 -I -B paper/reproduction-contracts/test_fusefss_silu_integer_campaign.py -v
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fusefss_silu_integer_campaign", ROOT / "examples/papers/fusefss_silu_integer_campaign.py"
)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class SiluIntegerCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "target").mkdir(exist_ok=True)
        cls.frozen, cls.source = CAMPAIGN.load_tables()

    def test_complete_campaign(self):
        report = CAMPAIGN.run_campaign()
        self.assertEqual(report["valid_cases"], 131392)
        self.assertEqual(report["central_cases"], 131136)
        self.assertEqual(report["endpoint_cases"], 256)
        self.assertEqual(report["failures"], 0)
        self.assertEqual(report["request_models"], 32)
        self.assertEqual((report["accepted_models"], report["rejected_models"]), (4, 28))

    def test_known_complete_tuples(self):
        n = CAMPAIGN.RING
        expected = {
            -65: (-2, 0, 2, 1, 2, n-63),
            -64: (-1, 0, 1, 1, 1, n-31),
            -1: (-1, 0, 1, 1, 1, n-31),
            0: (0, 1, 0, 1, 0, 0),
            63: (0, 1, 0, 1, 0, 63),
            64: (1, 1, 1, 1, 1, 33),
            -65473: (-1024, 0, 1024, 0, 1023, 0),
            -65472: (-1023, 0, 1023, 1, 1023, 0),
            65535: (1023, 1, 1023, 1, 1023, 65535),
            65536: (1024, 1, 1024, 0, 1023, 65536),
            -(1 << 37): (-(1 << 31), 0, 1 << 31, 0, 1023, 0),
        }
        for x, result in expected.items():
            with self.subTest(x=x):
                self.assertEqual(CAMPAIGN.oracle(x, self.frozen), result)
                self.assertEqual(CAMPAIGN.pipeline(x % n, self.source), result)

    def test_oracle_kills_arithmetic_mutations(self):
        # Each mutant preserves enough output behavior to require full tuple
        # comparison, notably B at A=1024 where T[1023] is already zero.
        trunc_zero = list(CAMPAIGN.oracle(-65, self.frozen))
        trunc_zero[0] = -(65 // 64)
        abs_before_floor = list(CAMPAIGN.oracle(-1, self.frozen))
        abs_before_floor[2] = abs(-1) // 64
        reduced_branch = list(CAMPAIGN.oracle(64, self.frozen))
        reduced_branch[-1] = (1 - self.frozen[1]) % CAMPAIGN.RING
        wrong_clamp_boundary = list(CAMPAIGN.oracle(65536, self.frozen))
        wrong_clamp_boundary[3] = 1  # <=1024 instead of <1024
        original_pipeline = CAMPAIGN.pipeline
        for x, mutant in ((-65, trunc_zero), (-1, abs_before_floor),
                          (64, reduced_branch), (65536, wrong_clamp_boundary)):

            def mutated_pipeline(word, table):
                if CAMPAIGN.signed(word, 50) == x:
                    return tuple(mutant)
                return original_pipeline(word, table)

            with self.subTest(x=x), patch.object(CAMPAIGN, "pipeline", side_effect=mutated_pipeline):
                # Pipeline mutation cannot infect the independent equation oracle.
                self.assertNotEqual(CAMPAIGN.pipeline(x % CAMPAIGN.RING, self.source),
                                    CAMPAIGN.oracle(x, self.frozen))
                with self.assertRaisesRegex(ValueError, "tuple mismatch"):
                    CAMPAIGN.run_campaign()

    def test_invalid_is_zero_before_pipeline_and_false_stays_sticky(self):
        for x in CAMPAIGN.INVALID_X:
            with self.subTest(x=x), patch.object(CAMPAIGN, "pipeline", wraps=CAMPAIGN.pipeline) as call:
                valid, safe, sticky, result, _ = CAMPAIGN.checked_lane(x % CAMPAIGN.RING, self.source)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[:2], (0, self.source))
                self.assertEqual(tuple(call.call_args.args[2]), CAMPAIGN.PUBLIC_MODEL_SCHEDULE)
                self.assertEqual((valid, safe, sticky), (False, 0, False))
                self.assertEqual(result, CAMPAIGN.oracle(0, self.frozen))
        self.assertFalse(CAMPAIGN.checked_lane(64, self.source, False)[2])

    def test_campaign_detects_missing_executed_schedule_step(self):
        original_pipeline = CAMPAIGN.pipeline

        def missing_step(word, table, trace=None):
            result = original_pipeline(word, table, trace)
            if trace is not None:
                trace.remove("lookup")
            return result

        with patch.object(CAMPAIGN, "pipeline", side_effect=missing_step):
            with self.assertRaisesRegex(ValueError, "schedule mismatch"):
                CAMPAIGN.run_campaign()

    def test_noncanonical_words_fail_before_pipeline(self):
        for word in (-1, CAMPAIGN.RING, True, 1.0):
            with self.subTest(word=word), patch.object(CAMPAIGN, "pipeline") as call:
                with self.assertRaises(ValueError):
                    CAMPAIGN.checked_lane(word, self.source)
                call.assert_not_called()

    def test_missing_or_changed_fixture_and_changed_rust_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            fixture, values = Path(directory) / "fixture.json", Path(directory) / "values.rs"
            with self.assertRaises(FileNotFoundError):
                CAMPAIGN.load_tables(fixture, CAMPAIGN.VALUES)
            fixture.write_bytes(CAMPAIGN.FIXTURE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "fixture hash mismatch"):
                CAMPAIGN.load_tables(fixture, CAMPAIGN.VALUES)
            original = CAMPAIGN.VALUES.read_text()
            before, after = original.split("static SILU:", 1)
            values.write_text(before + "static SILU:" + after.replace("0,", "1,", 1))
            with self.assertRaisesRegex(ValueError, "table mismatch at 0"):
                CAMPAIGN.load_tables(CAMPAIGN.FIXTURE, values)


if __name__ == "__main__":
    unittest.main()
