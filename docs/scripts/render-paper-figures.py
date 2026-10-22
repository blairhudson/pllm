"""Build untracked paper diagrams and charts from source and pinned evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "build" / "papers" / "figures"
EVIDENCE = ROOT / "docs" / "evidence" / "current-runtime-2026-09-11.json"
HISTORICAL_RESULTS = ROOT / "paper" / "artifacts" / "historical-results-v1.json"
LOW_MEMORY_COHORT_IDS = (
    "qwen25-private-boundary-decode-confirmation-150-8",
    "qwen25-private-boundary-decode-long-output-150-32",
    "qwen25-private-boundary-decode-long-output-150-64",
)

INK = "#19343c"
MUTED = "#4c6570"
TEAL = "#087e79"
TEAL_LIGHT = "#d8efeb"
PANEL = "#f2f7f6"
LINE = "#b9d4cf"
LOW_MEMORY_ARMS = (
    ("maintained", "Maintained prepared", INK, "circle"),
    ("coupled-serial-u32", "Coupled serial / u32", "#a55324", "square"),
    ("coupled-parallel-uniform_residues", "Coupled parallel / residue", TEAL, "triangle"),
)


def text(x: int, y: int, value: str, size: int = 20, *, color: str = INK,
         weight: int = 400, anchor: str = "start") -> str:
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def rect(x: int, y: int, width: int, height: int, *, fill: str = PANEL,
         stroke: str = LINE, radius: int = 12) -> str:
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')


def arrow(x1: int, y: int, x2: int, *, color: str = TEAL) -> str:
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 10}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
            f'<polygon points="{x2 - 11},{y - 8} {x2},{y} {x2 - 11},{y + 8}" '
            f'fill="{color}"/>')


def return_arrow(x1: int, y: int, x2: int) -> str:
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 + 10}" y2="{y}" '
            f'stroke="{TEAL}" stroke-width="3"/>'
            f'<polygon points="{x2 + 11},{y - 8} {x2},{y} {x2 + 11},{y + 8}" '
            f'fill="{TEAL}"/>')


def svg(width: int, height: int, parts: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<g>'
            + "".join(parts) + "</g></svg>\n")


def mechanics() -> str:
    parts = [text(34, 37, "One response · distinct trust and execution roles", 26, weight=700)]
    for x, name, detail in [
        (34, "CLIENT", "Plaintext, state, masks"),
        (445, "PREPARATION", "Public body + offline root seed"),
        (856, "INFERENCE", "Public body + one-use rows"),
    ]:
        parts += [rect(x, 58, 310, 91), text(x + 16, 95, name, 24, color=TEAL, weight=700),
                  text(x + 16, 126, detail, 19)]
    parts += [text(34, 184, "OFFLINE", 20, color=TEAL, weight=700),
              rect(34, 201, 311, 51, fill=TEAL_LIGHT),
              rect(445, 201, 311, 51, fill=TEAL_LIGHT),
              rect(856, 201, 310, 51, fill=TEAL_LIGHT),
              text(190, 234, "Seed each stage", 20, anchor="middle"),
              text(600, 234, "Compute W·r − s", 20, anchor="middle"),
              text(1011, 234, "Seal one-use rows", 20, anchor="middle"),
              arrow(350, 226, 438), arrow(761, 226, 849),
              text(34, 291, "ONLINE", 20, color=TEAL, weight=700),
              rect(34, 305, 520, 52), rect(649, 305, 517, 52),
              text(294, 338, "Ticket + masked activation x − r", 20, anchor="middle"),
              text(908, 338, "Return Wx − s · Client adds s", 20, anchor="middle"),
              arrow(556, 317, 645), return_arrow(645, 344, 556),
              text(34, 383, "Preparation idle online · both services must follow protocol and not collude", 18,
                   color=MUTED)]
    return svg(1200, 397, parts)


def research_loop() -> str:
    steps = [
        ("PROPOSE", "Named objective", "Pinned control"),
        ("IMPLEMENT", "Bounded method", "Composable plan"),
        ("CHECK", "Exact behavior", "Explicit contracts"),
        ("MEASURE", "Matched cohort", "All relevant costs"),
        ("LEARN", "Retain or reject", "Next hypothesis"),
    ]
    parts = [text(27, 33, "One experiment surface for researchers and AI agents", 25,
                  weight=700)]
    for index, (label, detail, boundary) in enumerate(steps):
        x = 27 + index * 234
        parts += [rect(x, 51, 208, 100, fill=TEAL_LIGHT if index == 4 else PANEL),
                  text(x + 14, 79, label, 20, color=TEAL, weight=700),
                  text(x + 14, 110, detail, 18),
                  text(x + 14, 136, boundary, 17, color=MUTED)]
        if index < 4:
            parts.append(arrow(x + 209, 102, x + 233))
    return svg(1200, 166, parts)


def baseline() -> str:
    record = json.loads(EVIDENCE.read_text())
    if (record["model"]["id"] != "Qwen/Qwen2.5-0.5B-Instruct"
            or not record["implementation"]["git_revision"].startswith("277d19f")
            or record["method"]["measured_repetitions_per_context"] != 3
            or record["method"]["measured_runs"] != 9):
        raise ValueError("Pinned baseline chart requires the recorded Qwen2.5 cohort")
    parts = [text(30, 38, "Historical full-response latency · Qwen2.5-0.5B", 26, weight=700)]
    x0, span, maximum = 359, 704, 14
    for tick in (0, 5, 10, 14):
        x = x0 + span * tick // maximum
        parts += [f'<line x1="{x}" y1="75" x2="{x}" y2="274" '
                  f'stroke="{LINE}" stroke-width="1"/>',
                  text(x, 299, f"{tick} s", 17, color=MUTED, anchor="middle")]
    if [item["context_tokens"] for item in record["results"]] != [30, 63, 255]:
        raise ValueError("Baseline chart requires the recorded token cohorts")
    for index, result in enumerate(record["results"]):
        y = 98 + index * 76
        outputs = set(result["output_tokens"])
        if len(outputs) != 1 or len(result["full_seconds"]) != 3:
            raise ValueError("Baseline chart requires exact output-token cohorts")
        label = f'{result["context_tokens"]} in / {next(iter(outputs))} out'
        median = result["median"]["full_seconds"]
        if abs(median - statistics.median(result["full_seconds"])) > 1e-9:
            raise ValueError("Baseline chart median differs from its run records")
        width = round(span * median / maximum)
        parts += [text(30, y + 27, label, 21), rect(x0, y, span, 37, fill=PANEL,
                  stroke=PANEL, radius=7), rect(x0, y, width, 37, fill=TEAL,
                  stroke=TEAL, radius=7),
                  text(x0 + width + 13, y + 27, f"{median:.3f} s", 21, weight=700)]
    parts += [text(30, 338, "Apple M5 loopback · three warm runs/cohort · earlier source revision 277d19f",
                   18, color=MUTED),
              text(30, 366, "Different token counts; no competitor, price, energy, WAN or GPU measurement",
                   18, color=MUTED)]
    return svg(1200, 384, parts)


def low_memory_cohorts(record: dict) -> list[dict]:
    """Extract paired observations, never pooled summaries or scenario-window RSS."""
    cohorts = []
    for identifier in LOW_MEMORY_COHORT_IDS:
        cohort, = (item for item in record["cohorts"] if item["id"] == identifier)
        points = []
        for recipe, _, _, _ in LOW_MEMORY_ARMS:
            row, = (item for item in cohort["rows"] if item["recipe"] == recipe)
            rss = row["metrics"]["process.peak_rss"]
            tcp = row["metrics"]["network.all_link_stream_bytes"]
            replicates = row["replicateIds"]
            if (rss["unit"] != "bytes" or tcp["unit"] != "bytes"
                    or len(replicates) != len(set(replicates))
                    or len(replicates) != row["sampleCount"]
                    or len(replicates) != len(rss["samples"])
                    or len(replicates) != len(tcp["samples"])):
                raise ValueError(f"Unpaired byte observations for {identifier}/{recipe}")
            for replicate, memory, traffic in zip(replicates, rss["samples"], tcp["samples"]):
                if any(type(value) is not int or value < 0 for value in (memory, traffic)):
                    raise ValueError("Figure requires unrounded nonnegative byte counts")
                points.append({"recipe": recipe, "replicate": replicate,
                               "rss_bytes": memory, "tcp_bytes": traffic})
        cohorts.append({"id": identifier, "report": cohort["report"],
                        "inputs": cohort["identity"]["inputs"],
                        "outputs": cohort["identity"]["outputs"],
                        "tier": cohort["evidenceTier"], "points": points})
    return cohorts


def point_marker(x: float, y: float, color: str, shape: str) -> str:
    if shape == "circle":
        # ImageMagick's built-in SVG reader can ignore stroke attributes. Nested
        # filled circles preserve the hollow marker in that ordinary PNG path.
        return (f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/>'
                f'<circle cx="{x}" cy="{y}" r="5" fill="white"/>')
    if shape == "square":
        return f'<rect x="{x - 8}" y="{y - 8}" width="16" height="16" fill="{color}"/>'
    return (f'<polygon points="{x},{y - 10} {x - 10},{y + 8} {x + 10},{y + 8}" '
            f'fill="{color}"/>')


def low_memory_history() -> str:
    record = json.loads(HISTORICAL_RESULTS.read_text())
    cohorts = low_memory_cohorts(record)
    ceiling_bytes = record["memorySensitivity"]["tiersBytes"]["low"]
    parts = [text(40, 42, "Historical memory and all-link TCP traffic", 30, weight=700)]
    for index, (_, label, color, shape) in enumerate(LOW_MEMORY_ARMS):
        y = 82 + index * 37
        parts += [point_marker(56, y - 8, color, shape), text(82, y, label, 28)]
    parts += [text(40, 194, "One point/client; overlaps not jittered.", 26, color=MUTED)]
    # Identical, explicitly truncated axes; keep byte counts exact until plotting.
    x0, width, height = 145, 570, 205

    def memory_x(value: int) -> float:
        return x0 + width * (value / 2**20 - 200) / 64

    for index, cohort in enumerate(cohorts):
        top = 220 + index * 310
        y0 = top + 55
        parts += [f'<g data-cohort="{cohort["id"]}">',
                  f'<title>{escape(cohort["report"])}</title>',
                  text(110, top + 15,
                       f'({"abc"[index]}) {cohort["inputs"]}+{cohort["outputs"]} · '
                       f'{cohort["tier"]}', 30, weight=700),
                  text(x0, y0 - 8, "TCP MB (decimal)", 24, color=MUTED)]
        for tick in (350, 450, 550, 650):
            y = y0 + height * (650 - tick) / 310
            parts += [f'<rect x="{x0}" y="{y}" width="{width}" height="1" fill="{LINE}"/>',
                      text(x0 - 16, y + 8, str(tick), 24, anchor="end", color=MUTED)]
        for tick in (208, 224, 240, 256):
            x = memory_x(tick * 2**20)
            parts += [f'<rect x="{x}" y="{y0}" width="1" height="{height}" fill="{LINE}"/>',
                      text(x, y0 + height + 29, str(tick), 24, anchor="middle", color=MUTED)]
        ceiling_x = memory_x(ceiling_bytes)
        parts.append(f'<g class="memory-ceiling" data-bytes="{ceiling_bytes}" data-x="{ceiling_x}">')
        # Explicit filled dash segments also survive the built-in SVG reader.
        for offset in range(0, height, 14):
            parts.append(f'<rect x="{ceiling_x}" y="{y0 + offset}" width="2" '
                         f'height="{min(8, height - offset)}" fill="{INK}"/>')
        parts += ['</g>', text(ceiling_x, y0 - 8, f"{ceiling_bytes / 2**20:g} MiB", 24,
                              anchor="middle", color=MUTED)]
        for point in cohort["points"]:
            _, label, color, shape = next(arm for arm in LOW_MEMORY_ARMS
                                         if arm[0] == point["recipe"])
            memory, traffic = point["rss_bytes"], point["tcp_bytes"]
            if not (200 * 2**20 <= memory <= 264 * 2**20
                    and 340 * 10**6 <= traffic <= 650 * 10**6):
                raise ValueError("Historical point outside the declared figure axes")
            x = memory_x(memory)
            y = y0 + height * (650 - traffic / 10**6) / 310
            parts += [f'<g class="replicate" data-recipe="{point["recipe"]}" '
                      f'data-replicate="{point["replicate"]}" data-rss-bytes="{memory}" '
                      f'data-tcp-bytes="{traffic}">',
                      f'<title>{escape(label)}; replicate {point["replicate"]}; '
                      f'RSS {memory} bytes; TCP {traffic} bytes</title>',
                      point_marker(x, y, color, shape), '</g>']
        parts.append('</g>')
    parts += [text(430, 1176, "Complete-client lifetime peak RSS (MiB)", 28, anchor="middle"),
              text(40, 1217, "Axes truncated; identical scales in each panel.", 26, color=MUTED),
              text(40, 1252, "CPU loopback · source-present cold clients", 26, color=MUTED)]
    return svg(780, 1273, parts)


def render_png(source: Path, target: Path, magick: str, font_path: str) -> None:
    # Opaque 8-bit RGB avoids 16-bit RGBA soft masks that can render invisible
    # when embedded by the PDF engine. Keep the existing raster size and fonts.
    subprocess.run([
        magick, "-font", font_path, "-background", "white", "-density", "144", str(source),
        "-resize", "1200x", "-background", "white", "-alpha", "remove", "-alpha", "off",
        "-depth", "8", "-define", "png:color-type=2", "-strip", str(target),
    ], check=True)


def main() -> None:
    generators = {
        "mechanics": mechanics, "research-loop": research_loop,
        "qwen-baseline": baseline, "low-memory-history": low_memory_history,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=generators,
                        help="Render only the figures needed by the selected paper")
    names = parser.parse_args().only or generators
    magick = shutil.which("magick") or shutil.which("convert")
    fontconfig = shutil.which("fc-match")
    if magick is None or fontconfig is None:
        raise SystemExit("ImageMagick and fontconfig required to regenerate paper figures")
    font_path = subprocess.check_output(
        [fontconfig, "-f", "%{file}", "sans-serif"], text=True,
    ).strip()
    if not Path(font_path).is_file():
        raise SystemExit("fontconfig did not resolve a usable sans-serif font")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in names:
        contents = generators[name]()
        source = FIGURES / f"{name}.svg"
        target = FIGURES / f"{name}.png"
        source.write_text(contents)
        render_png(source, target, magick, font_path)
        print(f"Rendered {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
