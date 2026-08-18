#!/usr/bin/env python3
"""Build the AUREA codec MSI installer (x64, per-machine).

Pure-Python authoring via `msilib` (present in Python <= 3.12) — no WiX, no
admin rights needed to *build*. The package installs:

  [ProgramFiles64]\\AUREA\\bin\\aurea.exe, aurea-viewer.exe, aurea_wic.dll
  [ProgramFiles64]\\AUREA\\presets\\*.cmd + PRESETS.txt
  [ProgramFiles64]\\AUREA\\docs\\README.txt, LICENSE.txt

plus: the WIC decoder registration (Windows Photo Viewer, Explorer
thumbnails, Paint and WPF all open .aur natively), the `aurea.PhotoViewer`
ProgId as .aur default, a cascading "Convertir en AUR" context menu on
every image, `bin` on the system PATH, a Start-Menu shortcut, and a clean
uninstall entry.

    D:\\Aurea-main\\.venv-showcase\\Scripts\\python.exe scripts/build_msi.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import msilib
from msilib import (
    Binary,
    CAB,
    Dialog,
    Directory,
    Feature,
    add_data,
    add_tables,
    schema,
    sequence,
)

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "target" / "msi_staging"
OUTDIR = ROOT / "target" / "msi"
VERSION = "33.0.0"
PRODUCT_NAME = "AUREA Codec"
MANUFACTURER = "Symphoenix"
# Stable across releases — this is what lets a future 33.x upgrade replace us.
UPGRADE_CODE = "{C7A2B9D4-90F1-4B7E-9E3A-AA0C51F2A1B4}"

# Photo Viewer drop target — REQUIRED for prev/next arrows: without it the
# shell launches the viewer as a bare command line, with no folder context.
PHOTOVIEWER_DROPTARGET = "{FFE2A43C-56B9-4BF5-9A79-CC6D4285608A}"
# Photo property handler — the viewer's browse list only includes files
# whose extension has a photo property handler (same as .webp/.heic).
PHOTO_PROPERTY_HANDLER = "{a38b883c-1682-497e-97b0-0a3a9e801682}"
TIFF_CONTAINER = "{163bcc30-e2e9-4f0b-961d-a3e9fdb788a3}"

# WIC decoder COM identity (must match src/wic/src/lib.rs).
WIC_CLSID = "{A08EA2A4-32F0-4D42-9E3B-1A76A2A7E132}"
WIC_VENDOR = "{A08EA2A4-32F0-4D42-9E3B-1A76A2A7E100}"
WIC_CONTAINER = "{B54DDAB2-8C51-4B2A-BFCA-4A2A21F3D4C2}"
CATID_WIC_DECODERS = "{7ED96837-96F0-4812-B211-F13C24117ED3}"
PF_24BPP_BGR = "{6FDDC324-4E03-4BFE-B185-3D77768DC90C}"

GOLD = (232, 168, 56)
DARK = (17, 17, 17)

PRESETS = {
    "aurea-web.cmd": ("Web (qualite 40)", "encode", "40", ".aur"),
    "aurea-balanced.cmd": ("Equilibre (qualite 60)", "encode", "60", ".aur"),
    "aurea-archive.cmd": ("Archive (qualite 80)", "encode", "80", ".aur"),
    "aurea-quasi-lossless.cmd": ("Quasi sans perte (qualite 95)", "encode", "95", ".aur"),
    "aurea-decode.cmd": ("Decodage complet", "decode", None, ".png"),
    "aurea-preview.cmd": ("Apercu progressif (2 couches)", "preview", None, ".png"),
}

# (registry key under AUREA.Convert\shell, preset cmd, label) — key names
# are alphabetical, which fixes the menu order.
CONVERT_MENU = [
    ("cmd1web", "aurea-web.cmd", "Web (qualite 40)"),
    ("cmd2bal", "aurea-balanced.cmd", "Equilibre (qualite 60)"),
    ("cmd3arc", "aurea-archive.cmd", "Archive (qualite 80)"),
    ("cmd4los", "aurea-quasi-lossless.cmd", "Quasi sans perte (qualite 95)"),
]


def make_icon(path: Path) -> None:
    """A golden Phi on a dark rounded square, multi-size .ico."""
    from PIL import Image, ImageDraw

    base = 256
    img = Image.new("RGBA", (base, base), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([8, 8, base - 8, base - 8], radius=48, fill=DARK + (255,))
    gold = GOLD + (255,)
    cx, cy = base // 2, base // 2
    rx, ry = 62, 78
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=gold, width=18)
    draw.line([cx, 40, cx, base - 40], fill=gold, width=18)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def _phi(draw, cx, cy, rx, ry, width, color):
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=color, width=width)
    draw.line([cx, cy - int(ry * 1.35), cx, cy + int(ry * 1.35)], fill=color, width=width)


def make_banner(path: Path) -> None:
    """Top strip for the progress dialogs: dark left-to-right fade, golden
    Phi and wordmark."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 500, 60
    img = Image.new("RGB", (w, h), DARK)
    draw = ImageDraw.Draw(img)
    for x in range(w):  # subtle horizontal fade toward warm dark
        t = x / w
        col = tuple(int(DARK[i] + t * 18) for i in range(3))
        draw.line([(x, 0), (x, h)], fill=col)
    _phi(draw, 36, h // 2, 13, 17, 4, GOLD)
    try:
        font = ImageFont.truetype("segoeuib.ttf", 24)
        small = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((64, 14), "AUREA", font=font, fill=GOLD)
    draw.text((66, 41), "codec image  -  format AUR2 v33", font=small, fill=(180, 170, 150))
    draw.line([(0, h - 2), (w, h - 2)], fill=GOLD, width=2)
    img.save(path, "BMP")


def make_side(path: Path) -> None:
    """Left panel for welcome/exit dialogs: dark column, large Phi,
    vertical wordmark and golden rule."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 190, 380
    img = Image.new("RGB", (w, h), DARK)
    draw = ImageDraw.Draw(img)
    for y in range(h):  # vertical fade: pure dark up top, warm at the base
        t = y / h
        col = tuple(int(DARK[i] + t * 22) for i in range(3))
        draw.line([(0, y), (w, y)], fill=col)
    _phi(draw, w // 2, 120, 44, 56, 9, GOLD)
    dim = tuple(int(c * 0.35) for c in GOLD)
    _phi(draw, w // 2 - 62, 250, 20, 26, 4, dim)
    _phi(draw, w // 2 + 62, 288, 14, 18, 3, dim)
    try:
        font = ImageFont.truetype("segoeuib.ttf", 30)
        small = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((w // 2 - 51, 300), "AUREA", font=font, fill=GOLD)
    draw.text((w // 2 - 62, 340), "l'image au nombre d'or", font=small, fill=(180, 170, 150))
    draw.line([(w - 2, 0), (w - 2, h)], fill=GOLD, width=2)
    img.save(path, "BMP")


def preset_body(label: str, kind: str, quality: str | None) -> str:
    lines = [
        "@echo off",
        f"rem AUREA -- preset {label}",
        'if "%~1"=="" (',
        f"  echo Usage: %~n0 fichier [sortie]",
        "  exit /b 1",
        ")",
        'set "OUT=%~2"',
    ]
    if kind == "encode":
        lines += [
            'if "%OUT%"=="" set "OUT=%~dpn1.aur"',
            f'"%~dp0..\\bin\\aurea.exe" encode "%~1" "%OUT%" -q {quality}',
        ]
    elif kind == "decode":
        lines += [
            'if "%OUT%"=="" set "OUT=%~dpn1.png"',
            '"%~dp0..\\bin\\aurea.exe" decode "%~1" "%OUT%"',
        ]
    else:  # preview
        lines += [
            'if "%OUT%"=="" set "OUT=%~dpn1.apercu.png"',
            '"%~dp0..\\bin\\aurea.exe" decode "%~1" "%OUT%" --layers 2',
        ]
    lines.append("exit /b %errorlevel%")
    return "\r\n".join(lines) + "\r\n"


def build_staging() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    for sub in ("bin", "presets", "docs"):
        (STAGING / sub).mkdir(parents=True)

    release = ROOT / "target" / "release"
    shutil.copy2(release / "aurea.exe", STAGING / "bin" / "aurea.exe")
    shutil.copy2(release / "aurea-viewer.exe", STAGING / "bin" / "aurea-viewer.exe")
    shutil.copy2(release / "aurea_wic.dll", STAGING / "bin" / "aurea_wic.dll")

    for name, (label, kind, quality, _ext) in PRESETS.items():
        (STAGING / "presets" / name).write_text(preset_body(label, kind, quality), encoding="ascii")
    (STAGING / "presets" / "PRESETS.txt").write_text(
        "AUREA -- presets d'encodage\r\n"
        "===========================\r\n\r\n"
        "Chaque preset s'utilise en glissant une image dessus, en ligne de\r\n"
        "commande (`aurea-web mon_image.png [sortie.aur]`) ou par clic droit\r\n"
        "sur une image > Convertir en AUR.\r\n\r\n"
        + "".join(
            f"  {name:<26} {label}\r\n" for name, (label, _k, _q, _e) in PRESETS.items()
        )
        + "\r\nLe dossier bin est ajoute au PATH : `aurea encode|decode` est\r\n"
        "disponible partout. Les fichiers .aur s'ouvrent d'un double-clic\r\n"
        "dans la Visionneuse de photos Windows.\r\n",
        encoding="ascii",
    )

    readme = f"""AUREA Codec {VERSION} (format AUR2 v33)
=====================================

AUREA est un codec d'image experimental qui, sur son corpus de reference
de 12 images natives, bat libjpeg-turbo optimise (profils 4:2:0 et 4:4:4)
sur les 168 cellules BD-rate du protocole (PSNR RGB/Y/Cb/Cr, SSIM-Y,
CIEDE2000 moyen et p95) ainsi qu'en LPIPS, DISTS et PSNR sur les 24
paires a debit apparie de sa vitrine -- a environ 5x le temps de calcul
de JPEG.

Integration Windows
-------------------
Le decodeur WIC (aurea_wic.dll) rend le format natif pour Windows :
la Visionneuse de photos Windows ouvre les .aur d'un double-clic,
l'Explorateur affiche leurs miniatures, Paint et WPF les lisent.
Un clic droit sur n'importe quelle image propose "Convertir en AUR".

Utilisation
-----------
  aurea encode image.png image.aur -q 60
  aurea decode image.aur image.png
  aurea decode image.aur apercu.png --layers 2   (apercu progressif)
  aurea-viewer image.aur

Le dossier `presets` contient des raccourcis d'encodage (voir
PRESETS.txt). Le dossier `bin` est dans le PATH systeme apres
installation.

Desinstallation : Parametres Windows > Applications > AUREA Codec.
"""
    (STAGING / "docs" / "README.txt").write_text(readme, encoding="ascii")
    license_src = ROOT / "LICENSE"
    if license_src.exists():
        shutil.copy2(license_src, STAGING / "docs" / "LICENSE.txt")
    else:
        (STAGING / "docs" / "LICENSE.txt").write_text("Voir le depot source.\r\n")

    make_icon(STAGING / "aurea.ico")
    shutil.copy2(STAGING / "aurea.ico", STAGING / "bin" / "aurea.ico")
    make_banner(STAGING / "banner.bmp")
    make_side(STAGING / "side.bmp")


def add_ui(db) -> None:
    """Installer wizard (welcome → progress → exit, plus error and cancel
    handling) — msilib ships the sequences but not a single dialog, so a
    double-click otherwise shows nothing at all. Welcome/exit dialogs get
    a golden side panel, progress dialogs a banner strip."""
    x = y = 50
    w, h = 370, 300
    side_w = 152  # dialog units reserved for the side bitmap
    tx = side_w + 14  # left edge of text next to the side panel
    tw = w - tx - 12
    title = "[ProductName]"
    modal = 3  # visible | modal
    modeless = 1
    gold = 56 * 65536 + 168 * 256 + 232  # COLORREF 0x00bbggrr

    add_data(
        db,
        "Property",
        [
            ("DefaultUIFont", "DlgFont8"),
            ("ErrorDialog", "ErrorDlg"),
            ("Progress1", "Installation"),
            ("Progress2", "installe"),
        ],
    )
    add_data(
        db,
        "TextStyle",
        [
            ("DlgFont8", "Tahoma", 9, None, 0),
            ("DlgFontBold10", "Tahoma", 10, None, 1),
            ("TitleGold", "Tahoma", 13, gold, 1),
        ],
    )
    add_data(
        db,
        "Binary",
        [
            ("sidebmp", Binary(str(STAGING / "side.bmp"))),
            ("bannerbmp", Binary(str(STAGING / "banner.bmp"))),
        ],
    )
    add_data(
        db,
        "InstallUISequence",
        [
            ("PrepareDlg", None, 140),
            ("WelcomeDlg", "NOT Installed", 1230),
            ("ProgressDlg", None, 1280),
        ],
    )

    def finish_dialog(name: str, headline: str, text: str, event: str) -> None:
        d = Dialog(db, name, x, y, w, h, modal, title, "Finish", "Finish", "Finish")
        d.bitmap("Side", 0, 0, side_w, h, "sidebmp")
        d.text("Title", tx, 18, tw, 40, 0x30003, r"{\TitleGold}" + headline)
        d.text("Description", tx, 66, tw, 190, 0x30003, text)
        d.control("BottomLine", "Line", 0, h - 36, w, 0, 1, None, None, None, None)
        c = d.pushbutton("Finish", w - 84, h - 27, 70, 17, 3, "Fermer", None)
        c.event("EndDialog", event)

    finish_dialog(
        "ExitDialog",
        "Installation terminee",
        "AUREA Codec est installe.\n\n"
        "Les fichiers .aur s'ouvrent d'un double-clic dans la Visionneuse "
        "de photos Windows et leurs miniatures apparaissent dans "
        "l'Explorateur.\n\n"
        "Un clic droit sur une image propose desormais \"Convertir en "
        "AUR\".\n\n"
        "Le dossier bin est dans le PATH systeme : ouvrez un nouveau "
        "terminal pour utiliser `aurea`.",
        "Return",
    )
    finish_dialog(
        "UserExit",
        "Installation annulee",
        "L'installation d'AUREA Codec a ete annulee. Aucun changement n'a "
        "ete effectue.",
        "Exit",
    )
    finish_dialog(
        "FatalError",
        "Echec de l'installation",
        "L'installation d'AUREA Codec a echoue.\n\nRelancez le fichier .msi "
        "et consultez l'erreur affichee, ou installez en ligne de commande "
        "avec journal : msiexec /i AUREA-Codec.msi /l*v aurea-install.log",
        "Exit",
    )

    prep = Dialog(db, "PrepareDlg", x, y, w, h, modeless, title, "Cancel", "Cancel", "Cancel")
    prep.bitmap("Banner", 0, 0, w, 44, "bannerbmp")
    prep.text("Title", 15, 56, w - 30, 25, 0x30003, r"{\TitleGold}Preparation...")
    prep.text("Description", 15, 90, w - 30, 40, 0x30003, "Un instant.")
    prep.control("BottomLine", "Line", 0, h - 36, w, 0, 1, None, None, None, None)
    c = prep.pushbutton("Cancel", w - 84, h - 27, 70, 17, 3, "Annuler", None)
    c.event("SpawnDialog", "CancelDlg")

    welcome = Dialog(db, "WelcomeDlg", x, y, w, h, modal, title, "Install", "Install", "Cancel")
    welcome.bitmap("Side", 0, 0, side_w, h, "sidebmp")
    welcome.text("Title", tx, 16, tw, 42, 0x30003, r"{\TitleGold}AUREA Codec [ProductVersion]")
    welcome.text(
        "Description",
        tx,
        62,
        tw,
        192,
        0x30003,
        "Ce programme installe le codec d'image AUREA (format AUR2 v33) :\n\n"
        "  - ouverture des .aur dans la Visionneuse de photos Windows\n"
        "  - miniatures .aur dans l'Explorateur (decodeur WIC)\n"
        "  - clic droit sur une image : \"Convertir en AUR\"\n"
        "  - aurea.exe et AUREA Viewer, ajoutes au PATH systeme\n"
        "  - presets d'encodage et dossier AUREA du menu Demarrer\n\n"
        "Cliquez sur Installer pour continuer (une elevation "
        "administrateur sera demandee), ou sur Annuler pour quitter.",
    )
    welcome.control("BottomLine", "Line", 0, h - 36, w, 0, 1, None, None, None, None)
    b = welcome.pushbutton("Install", w - 162, h - 27, 70, 17, 3, "Installer", "Cancel")
    b.event("EndDialog", "Return")
    b = welcome.pushbutton("Cancel", w - 84, h - 27, 70, 17, 3, "Annuler", "Install")
    b.event("SpawnDialog", "CancelDlg")

    progress = Dialog(db, "ProgressDlg", x, y, w, h, modeless, title, "Cancel", "Cancel", "Cancel")
    progress.bitmap("Banner", 0, 0, w, 44, "bannerbmp")
    progress.text("Title", 15, 56, w - 30, 25, 0x30003, r"{\TitleGold}Installation d'AUREA Codec")
    progress.text("Description", 15, 92, w - 30, 20, 0x30003, "Installation en cours...")
    progress.text("StatusText", 15, 118, w - 30, 20, 3, None).mapping("ActionText", "Text")
    c = progress.control(
        "ProgressBar", "ProgressBar", 15, 148, w - 30, 12, 65537, None, "Progres", None, None
    )
    c.mapping("SetProgress", "Progress")
    progress.control("BottomLine", "Line", 0, h - 36, w, 0, 1, None, None, None, None)
    c = progress.pushbutton("Cancel", w - 84, h - 27, 70, 17, 3, "Annuler", None)
    c.event("SpawnDialog", "CancelDlg")

    cancel = Dialog(db, "CancelDlg", 50, 80, 260, 85, modal, title, "No", "No", "No")
    cancel.text(
        "Question", 15, 15, 230, 30, 0x30003, "Voulez-vous vraiment annuler l'installation ?"
    )
    c = cancel.pushbutton("Yes", 50, 57, 56, 17, 3, "Oui", "No")
    c.event("EndDialog", "Exit")
    c = cancel.pushbutton("No", 150, 57, 56, 17, 3, "Non", "Yes")
    c.event("EndDialog", "Return")

    error = Dialog(db, "ErrorDlg", 50, 30, 330, 101, 65543, title, "ErrorText", None, None)
    error.text("ErrorText", 20, 15, 290, 60, 3, "")
    for label, name, position in [
        ("Non", "N", 120),
        ("Oui", "Y", 240),
        ("Abandonner", "A", 0),
        ("Annuler", "C", 42),
        ("Ignorer", "I", 81),
        ("OK", "O", 159),
        ("Reessayer", "R", 198),
    ]:
        c = error.pushbutton(name, position, 72, 56, 17, 3, label, None)
        c.event("EndDialog", f"Error{ {'N':'No','Y':'Yes','A':'Abort','C':'Cancel','I':'Ignore','O':'Ok','R':'Retry'}[name] }")


def wic_registry_rows(component: str) -> list[tuple]:
    """HKLM (root -1 + ALLUSERS=1) rows registering the WIC decoder: COM
    server, decoder metadata, magic pattern, category membership."""
    clsid_key = rf"Software\Classes\CLSID\{WIC_CLSID}"
    rows = [
        ("R_wic0", -1, clsid_key, None, "Decodeur AUREA", component),
        ("R_wic1", -1, clsid_key, "FriendlyName", "Decodeur AUREA", component),
        ("R_wic2", -1, clsid_key, "Author", MANUFACTURER, component),
        ("R_wic3", -1, clsid_key, "Description", "Decodeur d'images AUREA (AUR2)", component),
        ("R_wic4", -1, clsid_key, "Vendor", WIC_VENDOR, component),
        ("R_wic5", -1, clsid_key, "Version", VERSION, component),
        ("R_wic6", -1, clsid_key, "SpecVersion", "1.0.0.0", component),
        ("R_wic7", -1, clsid_key, "ContainerFormat", WIC_CONTAINER, component),
        ("R_wic8", -1, clsid_key, "FileExtensions", ".aur", component),
        ("R_wic9", -1, clsid_key, "MimeTypes", "image/aurea", component),
        ("R_wic10", -1, clsid_key, "SupportsAnimation", "#0", component),
        ("R_wic11", -1, clsid_key, "SupportChromakey", "#0", component),
        ("R_wic12", -1, clsid_key, "SupportLossless", "#0", component),
        ("R_wic13", -1, clsid_key, "SupportMultiframe", "#0", component),
        ("R_wic14", -1, clsid_key + r"\InprocServer32", None, "[BINDIR]aurea_wic.dll", component),
        ("R_wic15", -1, clsid_key + r"\InprocServer32", "ThreadingModel", "Both", component),
        ("R_wic16", -1, clsid_key + rf"\Formats\{PF_24BPP_BGR}", None, None, component),
        ("R_wic17", -1, clsid_key + r"\Patterns\0", "Position", "#0", component),
        ("R_wic18", -1, clsid_key + r"\Patterns\0", "Length", "#4", component),
        ("R_wic19", -1, clsid_key + r"\Patterns\0", "Pattern", "#x41555232", component),
        ("R_wic20", -1, clsid_key + r"\Patterns\0", "Mask", "#xFFFFFFFF", component),
        (
            "R_wic21",
            -1,
            rf"Software\Classes\CLSID\{CATID_WIC_DECODERS}\Instance\{WIC_CLSID}",
            "CLSID",
            WIC_CLSID,
            component,
        ),
        (
            "R_wic22",
            -1,
            rf"Software\Classes\CLSID\{CATID_WIC_DECODERS}\Instance\{WIC_CLSID}",
            "FriendlyName",
            "Decodeur AUREA",
            component,
        ),
    ]
    return rows


def assoc_registry_rows(component: str) -> list[tuple]:
    """.aur file type: Photo-Viewer ProgId (default), legacy viewer as
    secondary, Explorer thumbnails, image context menu."""
    # [System64Folder], not [SystemFolder]: in an x64 package the latter is
    # SysWOW64 (works only because rundll32 relaunches its x64 twin).
    photoviewer_cmd = (
        r"[System64Folder]rundll32.exe "
        r'"[ProgramFiles64Folder]Windows Photo Viewer\PhotoViewer.dll", '
        r"ImageView_Fullscreen %1"
    )
    pv = r"Software\Classes\aurea.PhotoViewer"
    rows = [
        # Extension: Photo Viewer is the default handler; matches the
        # UserChoice ProgId Windows already stored for this machine.
        ("R_ext", -1, r"Software\Classes\.aur", None, "aurea.PhotoViewer", component),
        ("R_ext_ct", -1, r"Software\Classes\.aur", "Content Type", "image/aurea", component),
        ("R_ext_pt", -1, r"Software\Classes\.aur", "PerceivedType", "image", component),
        ("R_owp1", -1, r"Software\Classes\.aur\OpenWithProgids", "aurea.PhotoViewer", "", component),
        ("R_owp2", -1, r"Software\Classes\.aur\OpenWithProgids", "AUREA.Image", "", component),
        # Explorer thumbnails: the stock photo thumbnail provider renders
        # through WIC, so pointing the .aur ShellEx at it lights them up.
        (
            "R_thumb",
            -1,
            r"Software\Classes\.aur\ShellEx\{E357FCCD-A995-4576-B01F-234630154E96}",
            None,
            "{C7657C4A-9F68-40FA-A4DF-96BC08EB3551}",
            component,
        ),
        # aurea.PhotoViewer ProgId.
        ("R_pv", -1, pv, None, "Image AUREA", component),
        ("R_pv_icon", -1, pv + r"\DefaultIcon", None, "[BINDIR]aurea.ico", component),
        ("R_pv_open", -1, pv + r"\shell\open", "MUIVerb", "Visionneuse de photos Windows", component),
        ("R_pv_cmd", -1, pv + r"\shell\open\command", None, photoviewer_cmd, component),
        # DropTarget: Explorer hands the viewer its shell context, which is
        # what enables prev/next navigation (the command line never does).
        (
            "R_pv_drop",
            -1,
            pv + r"\shell\open\DropTarget",
            "Clsid",
            PHOTOVIEWER_DROPTARGET,
            component,
        ),
        (
            "R_pv_vw",
            -1,
            pv + r"\shell\aureaviewer",
            "MUIVerb",
            "Ouvrir dans AUREA Viewer",
            component,
        ),
        ("R_pv_vwi", -1, pv + r"\shell\aureaviewer", "Icon", "[BINDIR]aurea.ico", component),
        (
            "R_pv_vwc",
            -1,
            pv + r"\shell\aureaviewer\command",
            None,
            '"[BINDIR]aurea-viewer.exe" "%1"',
            component,
        ),
        ("R_pv_dec", -1, pv + r"\shell\decode", "MUIVerb", "Convertir en PNG", component),
        ("R_pv_deci", -1, pv + r"\shell\decode", "Icon", "[BINDIR]aurea.ico", component),
        (
            "R_pv_decc",
            -1,
            pv + r"\shell\decode\command",
            None,
            '"[PRESETDIR]aurea-decode.cmd" "%1"',
            component,
        ),
        # Legacy ProgId kept as a secondary "Ouvrir avec" entry.
        ("R_prog", -1, r"Software\Classes\AUREA.Image", None, "Image AUREA (AUR2)", component),
        (
            "R_icon",
            -1,
            r"Software\Classes\AUREA.Image\DefaultIcon",
            None,
            "[BINDIR]aurea.ico",
            component,
        ),
        (
            "R_open",
            -1,
            r"Software\Classes\AUREA.Image\shell\open\command",
            None,
            '"[BINDIR]aurea-viewer.exe" "%1"',
            component,
        ),
        # Photo Gallery / Explorer integration (WIC integration doc): the
        # viewer's browse list requires the photo property handler on the
        # extension; KindMap makes System.Kind = picture for search and
        # grouping; OpenWithList/ShellImagePreview declare viewer support.
        (
            "R_ph",
            -1,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\PropertySystem\PropertyHandlers\.aur",
            None,
            PHOTO_PROPERTY_HANDLER,
            component,
        ),
        (
            "R_ph_ca",
            -1,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\PhotoPropertyHandler\ContainerAssociations",
            WIC_CONTAINER,
            TIFF_CONTAINER,
            component,
        ),
        (
            "R_kind",
            -1,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\KindMap",
            ".aur",
            "picture",
            component,
        ),
        ("R_owl1", -1, r"Software\Classes\.aur\OpenWithList\PhotoViewer.dll", None, None, component),
        (
            "R_owl2",
            -1,
            r"Software\Classes\SystemFileAssociations\.aur\OpenWithList\PhotoViewer.dll",
            None,
            None,
            component,
        ),
        (
            "R_sip1",
            -1,
            r"Software\Classes\.aur\ShellEx\ContextMenuHandlers\ShellImagePreview",
            None,
            PHOTOVIEWER_DROPTARGET,
            component,
        ),
        (
            "R_sip2",
            -1,
            r"Software\Classes\SystemFileAssociations\.aur\ShellEx\ContextMenuHandlers\ShellImagePreview",
            None,
            PHOTOVIEWER_DROPTARGET,
            component,
        ),
        # Cascading "Convertir en AUR" on every image type.
        (
            "R_cv",
            -1,
            r"Software\Classes\SystemFileAssociations\image\shell\AUREA",
            "MUIVerb",
            "Convertir en AUR",
            component,
        ),
        (
            "R_cv_ic",
            -1,
            r"Software\Classes\SystemFileAssociations\image\shell\AUREA",
            "Icon",
            "[BINDIR]aurea.ico",
            component,
        ),
        (
            "R_cv_sub",
            -1,
            r"Software\Classes\SystemFileAssociations\image\shell\AUREA",
            "ExtendedSubCommandsKey",
            "AUREA.Convert",
            component,
        ),
    ]
    for key, cmd, label in CONVERT_MENU:
        base = rf"Software\Classes\AUREA.Convert\shell\{key}"
        rows += [
            (f"R_{key}", -1, base, "MUIVerb", label, component),
            (f"R_{key}i", -1, base, "Icon", "[BINDIR]aurea.ico", component),
            (f"R_{key}c", -1, base + r"\command", None, f'"[PRESETDIR]{cmd}" "%1"', component),
        ]
    return rows


def build_msi() -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    msi_path = OUTDIR / f"AUREA-Codec-{VERSION}-x64.msi"
    if msi_path.exists():
        msi_path.unlink()

    product_code = msilib.gen_uuid()
    db = msilib.init_database(
        str(msi_path), schema, PRODUCT_NAME, product_code, VERSION, MANUFACTURER
    )
    add_tables(db, sequence)

    # x64 package.
    si = db.GetSummaryInformation(20)
    si.SetProperty(7, "x64;1033")  # PID_TEMPLATE
    si.Persist()

    add_data(
        db,
        "Property",
        [
            ("ALLUSERS", "1"),
            ("UpgradeCode", UPGRADE_CODE),
            ("ARPPRODUCTICON", "aurea.ico"),
            ("ARPCOMMENTS", "Codec d'image AUREA (format AUR2 v33)"),
            ("ARPCONTACT", MANUFACTURER),
        ],
    )
    add_data(db, "Icon", [("aurea.ico", Binary(str(STAGING / "aurea.ico")))])
    add_data(db, "LaunchCondition", [("VersionNT64", "AUREA Codec necessite Windows 64 bits.")])
    # Replace older versions on upgrade, block downgrades.
    add_data(
        db,
        "Upgrade",
        [
            (UPGRADE_CODE, None, VERSION, None, 1281, None, "OLDVERSIONFOUND"),
            (UPGRADE_CODE, VERSION, None, None, 1026, None, "NEWERVERSIONFOUND"),
        ],
    )
    # RemoveExistingProducts is already scheduled at 6700 by msilib.sequence.
    add_data(db, "LaunchCondition", [("NOT NEWERVERSIONFOUND", "Une version plus recente d'AUREA Codec est deja installee.")])

    cab = CAB("aurea")
    win64 = 256  # msidbComponentAttributes64bit

    feature = Feature(db, "Complete", "AUREA Codec", "Codec, viewer, presets", 1, level=1)
    feature.set_current()

    root = Directory(db, cab, None, str(STAGING), "TARGETDIR", "SourceDir")
    pfiles = Directory(db, cab, root, ".", "ProgramFiles64Folder", "PFiles")
    appdir = Directory(db, cab, pfiles, ".", "AUREADIR", "AUREA|AUREA")

    bindir = Directory(db, cab, appdir, "bin", "BINDIR", "bin")
    bindir.start_component("C_bin", feature, flags=win64, keyfile="aurea.exe")
    bindir.add_file("aurea.exe")
    bindir.add_file("aurea-viewer.exe")
    bindir.add_file("aurea_wic.dll")
    bindir.add_file("aurea.ico")

    presetdir = Directory(db, cab, appdir, "presets", "PRESETDIR", "presets")
    presetdir.start_component("C_presets", feature, flags=win64, keyfile="aurea-web.cmd")
    for name in list(PRESETS) + ["PRESETS.txt"]:
        presetdir.add_file(name)

    docdir = Directory(db, cab, appdir, "docs", "DOCDIR", "docs")
    docdir.start_component("C_docs", feature, flags=win64, keyfile="README.txt")
    docdir.add_file("README.txt")
    docdir.add_file("LICENSE.txt")

    cab.commit(db)

    # System PATH gains [BINDIR]; removed on uninstall.
    add_data(db, "Environment", [("E_path", "=-*Path", "[~];[BINDIR]", "C_bin")])

    # Registry-only component: WIC decoder + associations + context menu.
    assoc_component = "C_assoc"
    add_data(
        db,
        "Component",
        [(assoc_component, msilib.gen_uuid(), "BINDIR", 4 | win64, None, "R_ext")],
    )
    add_data(db, "FeatureComponents", [("Complete", assoc_component)])
    add_data(db, "Registry", assoc_registry_rows(assoc_component))
    add_data(db, "Registry", wic_registry_rows(assoc_component))

    # Start menu: AUREA folder with the viewer and the presets folder.
    add_data(
        db,
        "Directory",
        [
            ("ProgramMenuFolder", "TARGETDIR", "PMenu"),
            ("AUREAMENU", "ProgramMenuFolder", "AUREA|AUREA"),
        ],
    )
    add_data(
        db,
        "Shortcut",
        [
            (
                "S_viewer",
                "AUREAMENU",
                "AUREAVW|AUREA Viewer",
                assoc_component,
                "[BINDIR]aurea-viewer.exe",
                None,
                "Visionneuse d'images AUREA (.aur)",
                None,
                "aurea.ico",
                0,
                None,
                "BINDIR",
            ),
            (
                "S_presets",
                "AUREAMENU",
                "AUREAPR|Presets AUREA",
                assoc_component,
                "[PRESETDIR]",
                None,
                "Dossier des presets d'encodage",
                None,
                None,
                None,
                None,
                "PRESETDIR",
            ),
        ],
    )
    add_data(db, "RemoveFile", [("RM_menu", assoc_component, None, "AUREAMENU", 2)])

    add_ui(db)

    db.Commit()
    return msi_path


def main() -> int:
    print("staging...", flush=True)
    build_staging()
    print("msi...", flush=True)
    path = build_msi()
    size = path.stat().st_size
    print(f"OK: {path} ({size / 1024 / 1024:.2f} Mo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
