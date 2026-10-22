"""Check the historical figure against retained ordinary benchmark reports."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import statistics
import struct
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "paper_figures", ROOT / "docs/scripts/render-paper-figures.py",
)
assert SPEC is not None and SPEC.loader is not None
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


class LowMemoryFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = json.loads(FIGURES.HISTORICAL_RESULTS.read_text())

    def test_every_point_pairs_the_same_original_replicate_and_metric_windows(self) -> None:
        for cohort in FIGURES.low_memory_cohorts(self.record):
            report = json.loads((ROOT / cohort["report"]).read_text())
            retained = next(c for c in self.record["cohorts"] if c["id"] == cohort["id"])
            self.assertEqual(report["cohort"], retained["privateCohortCommitment"])
            self.assertEqual(len(cohort["points"]), 9)
            for point in cohort["points"]:
                with self.subTest(cohort=cohort["id"], point=point):
                    original, = (result for result in report["results"]
                                 if result["execution"]["case"]["name"] == point["recipe"]
                                 and result["execution"]["replicate"] == point["replicate"])
                    self.assertEqual(original["status"], "completed")
                    row = next(row for row in retained["rows"]
                               if row["recipe"] == point["recipe"])
                    self.assertEqual(original["execution"]["build"]["digest"], row["buildDigest"])
                    metrics = {metric["id"]: metric for metric in original["metrics"]}
                    for metric, field, window in (
                        ("process.peak_rss", "rss_bytes", "process_birth_through_target"),
                        ("network.all_link_stream_bytes", "tcp_bytes", "relay_start_through_scenario"),
                    ):
                        self.assertEqual(metrics[metric]["unit"], "bytes")
                        self.assertEqual(metrics[metric]["window"], window)
                        self.assertEqual(point[field], metrics[metric]["value"])
                        self.assertIs(type(point[field]), int)

    def test_exact_maxima_and_medians_match_retained_table_cohorts(self) -> None:
        expected = (
            ((245579776, 478654277), (238518272, 367346669), (235012096, 365220717)),
            ((229441536, 536296308), (240812032, 454464360), (241893376, 445960681)),
            ((246530048, 613152300), (231440384, 570621257), (241762304, 553613865)),
        )
        for cohort, summaries in zip(FIGURES.low_memory_cohorts(self.record), expected):
            for (recipe, _, _, _), (peak, median_tcp) in zip(FIGURES.LOW_MEMORY_ARMS, summaries):
                points = [p for p in cohort["points"] if p["recipe"] == recipe]
                self.assertEqual(max(p["rss_bytes"] for p in points), peak)
                self.assertEqual(statistics.median(p["tcp_bytes"] for p in points), median_tcp)

    def test_cohort_selection_excludes_other_150_plus_8_trial_and_is_order_independent(self) -> None:
        modified = copy.deepcopy(self.record)
        # The earlier eight-arm 150+8 trial is not a replicate of the later one.
        modified["cohorts"][0]["rows"] = []
        modified["cohorts"].reverse()
        for cohort in modified["cohorts"]:
            cohort["rows"].reverse()
        extracted = FIGURES.low_memory_cohorts(modified)
        self.assertEqual(extracted, FIGURES.low_memory_cohorts(self.record))
        self.assertEqual([c["id"] for c in extracted], list(FIGURES.LOW_MEMORY_COHORT_IDS))
        self.assertEqual([(c["inputs"], c["outputs"], c["tier"]) for c in extracted],
                         [(150, 8, "confirmation"), (150, 32, "screen"), (150, 64, "screen")])

    def test_caption_describes_shared_build_and_distinct_private_salts(self) -> None:
        retained = [c for c in self.record["cohorts"]
                    if c["id"] in FIGURES.LOW_MEMORY_COHORT_IDS]
        self.assertEqual(len({row["buildDigest"] for c in retained for row in c["rows"]}), 1)
        self.assertEqual(len({c["privateCohortCommitment"] for c in retained}), 3)
        manuscript = (ROOT / "paper/low-memory.md").read_text()
        self.assertIn("same recorded build, different private salts", manuscript)
        self.assertEqual(manuscript.count("](figures/low-memory-history.png)"), 1)

    def test_missing_or_unpaired_samples_are_not_synthesized(self) -> None:
        for mutation in ("missing", "unpaired", "duplicate", "rounded"):
            with self.subTest(mutation=mutation):
                modified = copy.deepcopy(self.record)
                cohort = next(c for c in modified["cohorts"]
                              if c["id"] == FIGURES.LOW_MEMORY_COHORT_IDS[0])
                row = cohort["rows"][0]
                if mutation == "missing":
                    cohort["rows"].pop(0)
                elif mutation == "unpaired":
                    row["metrics"]["process.peak_rss"]["samples"].pop()
                elif mutation == "duplicate":
                    row["replicateIds"][1] = row["replicateIds"][0]
                else:
                    row["metrics"]["process.peak_rss"]["samples"][0] = 224.13
                with self.assertRaises(ValueError):
                    FIGURES.low_memory_cohorts(modified)

    def test_svg_has_separate_panels_exact_coordinates_and_unjittered_points(self) -> None:
        svg = ET.fromstring(FIGURES.low_memory_history())
        panels = svg.findall('.//{*}g[@data-cohort]')
        self.assertEqual([p.attrib["data-cohort"] for p in panels],
                         list(FIGURES.LOW_MEMORY_COHORT_IDS))
        ceiling = self.record["memorySensitivity"]["tiersBytes"]["low"]
        for index, (panel, cohort) in enumerate(zip(panels, FIGURES.low_memory_cohorts(self.record))):
            threshold, = panel.findall('{*}g[@class="memory-ceiling"]')
            self.assertEqual(int(threshold.attrib["data-bytes"]), ceiling)
            self.assertEqual(float(threshold.attrib["data-x"]),
                             145 + 570 * (ceiling / 2**20 - 200) / 64)
            markers = panel.findall('{*}g[@class="replicate"]')
            self.assertEqual(len(markers), 9)
            for marker, point in zip(markers, cohort["points"]):
                self.assertEqual(int(marker.attrib["data-rss-bytes"]), point["rss_bytes"])
                self.assertEqual(int(marker.attrib["data-tcp-bytes"]), point["tcp_bytes"])
                # Prepared circles expose exact unjittered coordinates directly.
                if point["recipe"] == "maintained":
                    circle = marker.find("{*}circle")
                    self.assertEqual(float(circle.attrib["cx"]),
                                     145 + 570 * (point["rss_bytes"] / 2**20 - 200) / 64)
                    self.assertEqual(float(circle.attrib["cy"]),
                                     275 + index * 310 + 205 * (650 - point["tcp_bytes"] / 10**6) / 310)
        text = " ".join(svg.itertext())
        self.assertIn("Axes truncated; identical scales", text)
        self.assertIn("overlaps not jittered", text)
        self.assertIn("source-present cold clients", text)

    @contextmanager
    def rendered_history(self):
        magick = shutil.which("magick") or shutil.which("convert")
        fontconfig = shutil.which("fc-match")
        if not magick or not fontconfig:
            self.skipTest("ImageMagick and fontconfig required for raster rendering")
        font = subprocess.check_output([fontconfig, "-f", "%{file}", "sans-serif"], text=True).strip()
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="low-memory-figure-") as temporary:
            source = Path(temporary) / "figure.svg"
            png = source.with_suffix(".png")
            source.write_text(FIGURES.low_memory_history())
            FIGURES.render_png(source, png, magick, font)
            yield magick, png

    def test_png_export_is_opaque_8_bit_truecolor_and_nonblank(self) -> None:
        with self.rendered_history() as (magick, png):
            data = png.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(data[12:16], b"IHDR")
            width, height, depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", data[16:29])
            self.assertEqual(width, 1200)
            self.assertGreater(height, 0)
            self.assertEqual(depth, 8)
            self.assertEqual(color_type, 2, "PDF figures must use RGB without an alpha soft mask")
            self.assertNotIn(b"tRNS", self.png_chunk_types(data))
            extrema = subprocess.check_output(
                [magick, str(png), "-format", "%[fx:minima] %[fx:maxima] %[fx:p{0,0}.r]", "info:"],
                text=True,
            )
            minimum, maximum, corner = map(float, extrema.split())
            self.assertLess(minimum, 0.1, "Figure lost its dark marks and text")
            self.assertEqual(maximum, 1.0)
            self.assertEqual(corner, 1.0, "Background must be white")

    @staticmethod
    def png_chunk_types(data: bytes) -> list[bytes]:
        chunks = []
        position = 8
        while position < len(data):
            length, = struct.unpack(">I", data[position:position + 4])
            chunks.append(data[position + 4:position + 8])
            position += 12 + length
        return chunks

    def test_existing_rasterizer_keeps_prepared_markers_and_dashed_ceiling_visible(self) -> None:
        with self.rendered_history() as (magick, png):
            scale = 1200 / 780

            def red_channel(x: float, y: float) -> float:
                return float(subprocess.check_output(
                    [magick, str(png), "-format", f"%[fx:p{{{round(x * scale)},{round(y * scale)}}}.r]", "info:"],
                    text=True,
                ))

            point = FIGURES.low_memory_cohorts(self.record)[0]["points"][0]
            x = 145 + 570 * (point["rss_bytes"] / 2**20 - 200) / 64
            y = 275 + 205 * (650 - point["tcp_bytes"] / 10**6) / 310
            self.assertLess(red_channel(x - 7, y), 0.3, "Prepared marker ring disappeared")
            self.assertGreater(red_channel(x, y), 0.9, "Prepared marker must remain hollow")
            ceiling_x = 145 + 570 * (256 - 200) / 64
            self.assertLess(red_channel(ceiling_x + 1, 278), 0.3, "256 MiB line disappeared")
            self.assertGreater(red_channel(ceiling_x + 1, 286), 0.6, "Dashed line became solid")


if __name__ == "__main__":
    unittest.main()
