#!/usr/bin/env python3
"""Modulator-stacking torture harness (cycle 39).

The codec multiplies several step modulations per block (foveal, Turing
multiscale, Weber bright compensation / chroma breathing, EOB field);
each factor is log-neutral by construction, but nothing bounds their
*product*. This harness encodes a battery of synthetic patterns built to
provoke destructive stacking, reads the encoder's AUREA_MOD_STATS probe,
and compares the observed product envelope against the natural-image
corpus.

    .venv-showcase\\Scripts\\python.exe scripts\\torture_mires.py
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
AUREA = ROOT / "target" / "release" / "aurea.exe"
OUT = ROOT / "target" / "mires"
SRC = OUT / "src"
WORK = OUT / "work"
ORIGINALS = Path(r"D:\x267\Aurea\datasets\originals")

SIZE = 768
QUALITIES = (20, 40, 75)
CORPUS_QUALITIES = (20, 75)
PHI = (1 + 5 ** 0.5) / 2

STATS_RE = re.compile(
    r"\[AUREA_MOD_STATS\] ch=(\d) n=(\d+) ac_step=([\d.]+) "
    r"prod_min=([\d.]+) p01=([\d.]+) gmean=([\d.]+) p99=([\d.]+) prod_max=([\d.]+) "
    r"out_phi=(\d+) out_phi2=(\d+) floor_hits=(\d+)"
)


# ---------------------------------------------------------------------------
# Pattern generators — each returns HxWx3 uint8
# ---------------------------------------------------------------------------

def _grid():
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    cx = cy = (SIZE - 1) / 2
    r = np.hypot(x - cx, y - cy)
    return x, y, r


def zone_plate():
    _, _, r = _grid()
    kmax = math.pi  # Nyquist at the rim
    v = 0.5 + 0.5 * np.cos(kmax * r * r / SIZE)
    g = (v * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def zone_plate_chroma():
    _, _, r = _grid()
    v = np.cos(math.pi * r * r / SIZE)
    base = np.full((SIZE, SIZE), 128.0)
    red = np.clip(base + 90 * v, 0, 255).astype(np.uint8)
    blue = np.clip(base - 90 * v, 0, 255).astype(np.uint8)
    green = np.full((SIZE, SIZE), 128, dtype=np.uint8)
    return np.dstack([red, green, blue])


def ramp_h():
    x, _, _ = _grid()
    g = (x / (SIZE - 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def ramp_dark():
    x, _, _ = _grid()
    g = (x / (SIZE - 1) * 64).astype(np.uint8)
    return np.dstack([g, g, g])


def checker(px: int):
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    g = (((x // px) + (y // px)) % 2 * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def noise_binary():
    rng = np.random.default_rng(0x267)
    g = (rng.random((SIZE, SIZE)) > 0.5).astype(np.uint8) * 255
    return np.dstack([g, g, g])


def noise_color():
    rng = np.random.default_rng(0x267)
    return rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)


def siemens_star():
    x, y, r = _grid()
    theta = np.arctan2(y - (SIZE - 1) / 2, x - (SIZE - 1) / 2)
    v = 0.5 + 0.5 * np.sign(np.cos(72 * theta))
    v[r < 8] = 0.5
    g = (v * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def sweep_h():
    x, _, _ = _grid()
    # chirp: DC on the left, Nyquist on the right
    phase = math.pi * x * x / (2 * SIZE)
    v = 0.5 + 0.5 * np.cos(phase)
    g = (v * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def solid_mid():
    return np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)


def disks_dark():
    _, _, _ = _grid()
    rng = np.random.default_rng(0x267)
    img = np.full((SIZE, SIZE), 12.0)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    for _ in range(140):
        cx, cy = rng.random(2) * SIZE
        rad = 4 + rng.random() * 14
        bright = 120 + rng.random() * 135
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        img = np.maximum(img, np.where(d2 < rad * rad, bright, 0.0))
    g = img.astype(np.uint8)
    return np.dstack([g, g, g])


MIRES = {
    "zone_plate": zone_plate,
    "zone_plate_chroma": zone_plate_chroma,
    "ramp_h": ramp_h,
    "ramp_dark": ramp_dark,
    "checker_1px": lambda: checker(1),
    "checker_2px": lambda: checker(2),
    "noise_binary": noise_binary,
    "noise_color": noise_color,
    "siemens_star": siemens_star,
    "sweep_h": sweep_h,
    "solid_mid": solid_mid,
    "disks_dark": disks_dark,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(src_png: Path, quality: int, tag: str):
    aur = WORK / f"{tag}_q{quality}.aur"
    dec = WORK / f"{tag}_q{quality}.png"
    env = dict(os.environ, AUREA_MOD_STATS="1")
    enc = subprocess.run(
        [str(AUREA), "encode", str(src_png), str(aur), "-q", str(quality)],
        capture_output=True, text=True, env=env,
    )
    if enc.returncode != 0:
        return {"tag": tag, "q": quality, "error": f"encode rc={enc.returncode}"}
    stats = {}
    for m in STATS_RE.finditer(enc.stderr + enc.stdout):
        ch = int(m.group(1))
        stats[ch] = {
            "n": int(m.group(2)),
            "prod_min": float(m.group(4)),
            "gmean": float(m.group(6)),
            "prod_max": float(m.group(8)),
            "out_phi": int(m.group(9)),
            "out_phi2": int(m.group(10)),
            "floor_hits": int(m.group(11)),
        }
    dcd = subprocess.run(
        [str(AUREA), "decode", str(aur), str(dec)], capture_output=True, text=True
    )
    if dcd.returncode != 0:
        return {"tag": tag, "q": quality, "error": f"decode rc={dcd.returncode}"}
    ref = np.asarray(Image.open(src_png).convert("RGB"), dtype=np.float64)
    out = np.asarray(Image.open(dec).convert("RGB"), dtype=np.float64)
    err = np.abs(ref - out)
    mse = float(np.mean((ref - out) ** 2))
    psnr = 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
    return {
        "tag": tag,
        "q": quality,
        "bytes": aur.stat().st_size,
        "psnr": psnr,
        "max_err": float(err.max()),
        "p999_err": float(np.quantile(err, 0.999)),
        "stats": stats,
    }


def fmt_stats(stats):
    if not stats:
        return "—"
    parts = []
    for ch in sorted(stats):
        s = stats[ch]
        flag = ""
        if s["floor_hits"]:
            flag += f" FLOOR×{s['floor_hits']}"
        if s["out_phi2"]:
            flag += f" !phi2x{s['out_phi2']}"
        parts.append(f"ch{ch}[{s['prod_min']:.2f},{s['prod_max']:.2f}]{flag}")
    return " ".join(parts)


def main() -> int:
    for d in (SRC, WORK):
        d.mkdir(parents=True, exist_ok=True)

    print("generating mires...", flush=True)
    for name, gen in MIRES.items():
        p = SRC / f"{name}.png"
        if not p.exists():
            Image.fromarray(gen()).save(p)

    results = []
    print("running mires...", flush=True)
    for name in MIRES:
        for q in QUALITIES:
            r = run_case(SRC / f"{name}.png", q, name)
            results.append(r)
            print(f"  {name} q{q}: "
                  + (r.get("error") or f"{r['psnr']:.2f} dB, max_err={r['max_err']:.0f}, {fmt_stats(r['stats'])}"),
                  flush=True)

    print("running corpus envelope...", flush=True)
    corpus = []
    for src in sorted(ORIGINALS.glob("*.png")):
        for q in CORPUS_QUALITIES:
            r = run_case(src, q, f"corpus_{src.stem}")
            corpus.append(r)
            print(f"  {src.stem} q{q}: {fmt_stats(r.get('stats', {}))}", flush=True)

    # Corpus envelope of the stacking product, all channels together.
    def envelope(rows):
        lo, hi, phi2, floor = math.inf, -math.inf, 0, 0
        for r in rows:
            for s in r.get("stats", {}).values():
                lo = min(lo, s["prod_min"])
                hi = max(hi, s["prod_max"])
                phi2 += s["out_phi2"]
                floor += s["floor_hits"]
        return lo, hi, phi2, floor

    c_lo, c_hi, c_phi2, c_floor = envelope(corpus)
    m_lo, m_hi, m_phi2, m_floor = envelope(results)

    lines = [
        "# Mires de torture — empilement des modulateurs (cycle 39)",
        "",
        f"Sonde : `AUREA_MOD_STATS=1` — produit par bloc foveal × turing × "
        f"(bright|breathing) × champ EOB, exactement comme `quant_block` le multiplie.",
        "",
        f"## Enveloppe du produit",
        "",
        f"| | min | max | hors [φ⁻²,φ²] | clamps plancher |",
        f"|---|---:|---:|---:|---:|",
        f"| Corpus 12 originals (q20+q75) | {c_lo:.3f} | {c_hi:.3f} | {c_phi2} | {c_floor} |",
        f"| Mires ({len(MIRES)} × q{'/'.join(map(str, QUALITIES))}) | {m_lo:.3f} | {m_hi:.3f} | {m_phi2} | {m_floor} |",
        "",
        "## Détail mires",
        "",
        "| Mire | q | octets | PSNR (dB) | err max | err p99.9 | produit par canal |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['tag']} | {r['q']} | — | ERREUR {r['error']} | | | |")
            continue
        lines.append(
            f"| {r['tag']} | {r['q']} | {r['bytes']} | {r['psnr']:.2f} | "
            f"{r['max_err']:.0f} | {r['p999_err']:.1f} | {fmt_stats(r['stats'])} |"
        )
    lines += [
        "",
        "## Corpus (référence)",
        "",
        "| Image | q | PSNR (dB) | produit par canal |",
        "|---|---:|---:|---|",
    ]
    for r in corpus:
        lines.append(
            f"| {r['tag'].removeprefix('corpus_')} | {r['q']} | {r['psnr']:.2f} | {fmt_stats(r['stats'])} |"
        )

    verdicts = []
    if m_floor:
        verdicts.append(f"ROUGE : {m_floor} blocs clampes au plancher 0.1 sur les mires.")
    if m_phi2:
        verdicts.append(f"ROUGE : {m_phi2} blocs hors [φ⁻², φ²] sur les mires.")
    if m_lo < c_lo / 1.15 or m_hi > c_hi * 1.15:
        verdicts.append(
            f"JAUNE : l'enveloppe mires [{m_lo:.3f}, {m_hi:.3f}] déborde "
            f"l'enveloppe corpus [{c_lo:.3f}, {c_hi:.3f}] de plus de 15 %."
        )
    bad_q75 = [r for r in results if r.get("q") == 75 and r.get("psnr", 99) < 25]
    if bad_q75:
        verdicts.append(
            "ROUGE qualité : PSNR < 25 dB à q75 sur "
            + ", ".join(r["tag"] for r in bad_q75) + "."
        )
    if not verdicts:
        verdicts.append(
            "VERT : aucun empilement destructif — produit borné, aucun clamp, "
            "aucune sortie de bande φ², pas d'effondrement qualité à q75."
        )
    lines += ["", "## Verdict", ""] + [f"- {v}" for v in verdicts]

    report = OUT / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(["", "=== VERDICT ==="] + verdicts))
    print(f"rapport: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
