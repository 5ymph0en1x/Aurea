"""
AUREA DiffView — Pixel-level image comparison tool.
Supports .aur, .jpg, .png files.
Black/white diff map: white pixel = difference above threshold.
"""
import subprocess, tempfile, os
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import TKinterModernThemes as TKMT

AUREA_CLI = str(Path(__file__).parent.parent / "target" / "release" / "aurea.exe")


def load_image(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".aur":
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        subprocess.run([AUREA_CLI, "decode", path, tmp.name], capture_output=True)
        img = np.array(Image.open(tmp.name).convert("RGB"))
        os.unlink(tmp.name)
        return img
    return np.array(Image.open(path).convert("RGB"))


class DiffView(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__("AUREA DiffView", "park", "dark")

        self.img_a = None
        self.img_b = None
        self.diff_photo = None
        self.zoom = 1.0
        self.threshold_var = tk.IntVar(value=0)
        self.amplify_var = tk.BooleanVar(value=False)
        self.path_a = ""
        self.path_b = ""

        self._build_ui()
        self.threshold_var.trace_add("write", lambda *_: self._recompute())
        self.root.geometry("1400x820")
        self.run()

    def _build_ui(self):
        # Row 0: file pickers + options
        top = self.addFrame("Files", row=0, col=0, padx=5, pady=5)
        top.Button("Image A (original)", self._browse_a)
        top.Button("Image B (compressed)", self._browse_b)
        top.Checkbutton("Amplify x10", self.amplify_var, self._recompute)

        # Row 1: threshold slider
        ctrl = self.addFrame("Threshold", row=1, col=0, padx=5, pady=2)
        ctrl.Label("Threshold (0=any diff, 30=big diffs only):")
        ctrl.Scale(0, 30, self.threshold_var)

        # Row 2: status bar (plain tkinter label in a TKMT frame)
        sf = self.addFrame("Info", row=2, col=0, padx=5, pady=2)
        self.status_text = tk.StringVar(value="Load two images to compare")
        lbl = tk.Label(sf.master, textvariable=self.status_text,
                       font=("Consolas", 10), fg="#e0c060", bg="#1e1e1e",
                       anchor="w")
        lbl.grid(row=0, column=0, sticky="ew", padx=10)

        # Row 3: canvas
        cf = self.addFrame("Diff", row=3, col=0, padx=5, pady=5)
        self.canvas = tk.Canvas(cf.master, bg="#111111",
                                highlightthickness=0, height=550)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-1>", self._on_click)

    def _browse_a(self):
        p = filedialog.askopenfilename(
            title="Image A (original)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.aur")])
        if p:
            self.path_a = p
            self.img_a = load_image(p)
            self.status_text.set(
                f"A: {Path(p).name} ({self.img_a.shape[1]}x{self.img_a.shape[0]})")
            self._recompute()

    def _browse_b(self):
        p = filedialog.askopenfilename(
            title="Image B (compressed)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.aur")])
        if p:
            self.path_b = p
            self.img_b = load_image(p)
            self.status_text.set(
                f"B: {Path(p).name} ({self.img_b.shape[1]}x{self.img_b.shape[0]})")
            self._recompute()

    def _recompute(self, *_):
        if self.img_a is None or self.img_b is None:
            return
        ha, wa = self.img_a.shape[:2]
        hb, wb = self.img_b.shape[:2]
        if ha != hb or wa != wb:
            self.status_text.set(f"Size mismatch: A={wa}x{ha}, B={wb}x{hb}")
            return

        a = self.img_a.astype(np.float64)
        b = self.img_b.astype(np.float64)
        t = self.threshold_var.get()

        diff_rgb = np.abs(a - b)
        diff_max = np.max(diff_rgb, axis=2)
        mask = diff_max > t

        n_diff = int(np.count_nonzero(mask))
        pct = 100.0 * n_diff / (ha * wa)
        mean_e = float(np.mean(diff_max[mask])) if n_diff > 0 else 0.0
        max_e = float(np.max(diff_max))
        mse = float(np.mean((a - b) ** 2))
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))

        na = Path(self.path_a).name
        nb = Path(self.path_b).name
        self.status_text.set(
            f"{na} vs {nb} | {n_diff:,} px ({pct:.1f}%) | "
            f"mean={mean_e:.1f} max={max_e:.0f} | "
            f"PSNR={psnr:.2f} dB | t={t}")

        if self.amplify_var.get():
            amp = np.clip(diff_max * 10, 0, 255).astype(np.uint8)
            out = np.stack([amp, amp, amp], axis=2)
        else:
            out = np.zeros((ha, wa, 3), dtype=np.uint8)
            out[mask] = 255

        self._show(out)

    def _show(self, arr):
        h, w = arr.shape[:2]
        zh = max(1, int(h * self.zoom))
        zw = max(1, int(w * self.zoom))
        pil = Image.fromarray(arr)
        if self.zoom != 1.0:
            pil = pil.resize((zw, zh), Image.NEAREST)
        self.diff_photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.diff_photo)
        self.canvas.config(scrollregion=(0, 0, zw, zh))

    def _on_scroll(self, event):
        if event.delta > 0:
            self.zoom = min(8.0, self.zoom * 1.25)
        else:
            self.zoom = max(0.1, self.zoom / 1.25)
        self._recompute()

    def _on_click(self, event):
        if self.img_a is None or self.img_b is None:
            return
        x = int(event.x / self.zoom)
        y = int(event.y / self.zoom)
        h, w = self.img_a.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            pa = self.img_a[y, x]
            pb = self.img_b[y, x]
            d = np.abs(pa.astype(int) - pb.astype(int))
            self.status_text.set(
                f"({x},{y}): A=({pa[0]},{pa[1]},{pa[2]}) "
                f"B=({pb[0]},{pb[1]},{pb[2]}) "
                f"diff=({d[0]},{d[1]},{d[2]})")


if __name__ == "__main__":
    DiffView()
