#!/usr/bin/env python3
"""Revalidate the 168-cell BD tableau for the v33 strict-deblock decode.

The bitstream is unchanged from v32; only the decoder's gas-deblock trigger
moved. This recomputes the 48 AUREA frontier points with the current build
and the BD-rate of every (image x metric x JPEG profile) cell against the
stored optimized-JPEG anchors of the v32 report. All 168 cells must stay
negative.

    .venv-showcase\\Scripts\\python.exe scripts\\revalidate_v33_frontier.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_jpeg_frontier import METRICS, aurea_point, load_rgb, quality_values
from benchmark_v13_corrections import bd_rate

CORPUS = Path(r"D:\x267\Aurea\datasets\originals")
POINTS = Path(r"D:\x267\docs\data\points.csv")
AUREA_CLI = Path(r"D:\Aurea-main\target\release\aurea.exe")
WORK = Path(r"D:\Aurea-main\target\v33_frontier_lab")
QUALITIES = (20, 40, 60, 80)
ANCHORS = ("jpeg_420_opt", "jpeg_444_opt")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    with POINTS.open(newline="", encoding="utf-8") as stream:
        stored = list(csv.DictReader(stream))
    images = sorted({r["image"] for r in stored})

    red = []
    total = 0
    for image_name in images:
        src = CORPUS / f"{image_name}.png"
        _, reference = load_rgb(src)
        rows = [
            aurea_point(
                src, reference, image_name, q, AUREA_CLI, WORK, 600.0, dict(os.environ)
            )
            for q in QUALITIES
        ]
        line = [image_name]
        for anchor_name in ANCHORS:
            anchor = [
                r for r in stored if r["image"] == image_name and r["codec"] == anchor_name
            ]
            for metric in METRICS:
                bd = bd_rate(
                    [float(r["bytes"]) for r in anchor],
                    quality_values(anchor, metric),
                    [float(r["bytes"]) for r in rows],
                    quality_values(rows, metric),
                )
                total += 1
                if bd >= 0:
                    red.append((image_name, anchor_name, metric, bd))
                line.append(f"{metric.split('_')[-1][:4]}={bd:+.2f}")
        print("  ".join(line), flush=True)

    print(f"\ncellules: {total}  rouges: {len(red)}")
    for image_name, anchor_name, metric, bd in red:
        print(f"  ROUGE {image_name} {anchor_name} {metric}: {bd:+.3f}%")
    print("TABLEAU TOUT VERT" if not red else "REGRESSION")
    return 0 if not red else 1


if __name__ == "__main__":
    sys.exit(main())
