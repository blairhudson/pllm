"""Rebuild paper charts from checked protocol diagrams and pinned evidence.

Requires ImageMagick for SVG-to-PNG rasterization. The PNG outputs are checked in
so ordinary paper and website builds require only Pandoc and a TeX engine.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
EVIDENCE = ROOT / "docs" / "evidence" / "current-runtime-2026-09-11.json"

INK = "#19343c"
MUTED = "#4c6570"
TEAL = "#087e79"
TEAL_LIGHT = "#d8efeb"
PANEL = "#f2f7f6"
LINE = "#b9d4cf"


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
        ("LOCK SOURCE", "Pinned paper/source", "provenance only"),
        ("IMPLEMENT", "Typed local method", "bounded semantics"),
        ("COMPOSE", "Valid immutable plan", "reject gaps"),
        ("MEASURE", "Matched cohort", "retain failures"),
        ("SELECT", "Eligible candidate", "deploy after gates"),
    ]
    parts = [text(27, 38, "Research-to-capability loop · evidence grows, claims stay scoped", 26,
                  weight=700)]
    for index, (label, detail, boundary) in enumerate(steps):
        x = 27 + index * 234
        parts += [rect(x, 71, 208, 149, fill=TEAL_LIGHT if index == 4 else PANEL),
                  text(x + 14, 107, label, 20, color=TEAL, weight=700),
                  text(x + 14, 143, detail, 18),
                  text(x + 14, 184, boundary, 17, color=MUTED)]
        if index < 4:
            parts.append(arrow(x + 209, 145, x + 233))
    parts += [text(27, 261, "Checks: function · privacy contract · quality · trust · numeric policy",
                   19, color=INK, weight=600),
              text(27, 291, "No automatic paper-to-code translation or measured SOTA winner yet",
                   19, color=MUTED)]
    return svg(1200, 312, parts)


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


def main() -> None:
    magick = shutil.which("magick")
    fontconfig = shutil.which("fc-match")
    if magick is None or fontconfig is None:
        raise SystemExit("ImageMagick and fontconfig required to regenerate paper figures")
    font_path = subprocess.check_output(
        [fontconfig, "-f", "%{file}", "sans-serif"], text=True,
    ).strip()
    if not Path(font_path).is_file():
        raise SystemExit("fontconfig did not resolve a usable sans-serif font")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, contents in [
        ("mechanics", mechanics()), ("research-loop", research_loop()),
        ("qwen-baseline", baseline()),
    ]:
        source = FIGURES / f"{name}.svg"
        target = FIGURES / f"{name}.png"
        source.write_text(contents)
        subprocess.run([magick, "-font", font_path, "-background", "white", "-density", "144", str(source),
                        "-resize", "1200x", "-strip", str(target)], check=True)
        print(f"Rendered {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
