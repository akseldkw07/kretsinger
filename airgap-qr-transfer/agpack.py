#!/usr/bin/env python3
"""
agpack.py -- pack a source tree into photographable QR code sheets.

Runs on the OFFLINE machine. Read-only with respect to your source tree:
it never creates, modifies, moves or deletes anything under --root. All
output goes into a single --out directory.

Pipeline:
    files -> filtered -> deterministic tar -> lzma -> base32 -> QR frames
          -> HTML sheets (grid of QRs per screen, photograph one at a time)

Base32 is used deliberately: its alphabet is a subset of the QR
"alphanumeric" mode charset, so the payload survives every QR decoder
without the binary/ECI mangling that byte-mode QR sometimes suffers.
Cost is ~10% capacity versus raw byte mode, which is a fair trade for
not having to debug silent corruption over a camera link.

Requires: Python 3.8+, and one of: segno / qrcode / the qrencode CLI.

Typical use:
    python3 agpack.py --root ~/myrepo --ext .py,.md,.json --estimate
    python3 agpack.py --root ~/myrepo --ext .py,.md,.json --out ~/qr-out
    python3 agpack.py --calibrate --out ~/qr-cal
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zlib

MAGIC = b"AG"
FORMAT_VERSION = 1
HEADER_STRUCT = ">2sBIHHI"  # magic, ver, transfer_id, seq, total, crc32
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)

# Directories that are never worth sending over a camera link.
DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".tox",
    ".next",
    ".nuxt",
    "dist",
    "build",
    ".gradle",
    ".idea",
    ".vscode-test",
    "target",
    ".terraform",
    ".DS_Store",
    "site-packages",
    ".cache",
    "coverage",
    ".nyc_output",
]

# Skipped unless --include-secrets. Every skip is printed, never silent.
DEFAULT_SECRET_GLOBS = [
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "*.keystore",
    "*.jks",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.*",
    "*secret*",
    "*_token",
    "*.token",
    ".htpasswd",
]

# Quality level -> pixels per QR module ON THE SENDING SCREEN.
#
# This is the number that actually decides whether a transfer works. A QR
# symbol the display renders at 2 px/module contains no recoverable detail
# no matter how good the camera is, so layout is derived from the sending
# machine's resolution rather than from a fixed grid.
QUALITY = {
    "safe": 5.5,
    "balanced": 4.5,
    "dense": 3.5,
}

QUIET_ZONE = 3  # modules of white margin around each symbol
CHROME_FRACTION = 0.09  # vertical space taken by sheet header/footer/gaps


# --------------------------------------------------------------------------
# base32 helpers (unpadded, uppercase -- stays inside QR alphanumeric mode)
# --------------------------------------------------------------------------


def b32_encode(raw: bytes) -> str:
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def b32_decode(text: str) -> bytes:
    text = text.strip().upper()
    pad = (-len(text)) % 8
    return base64.b32decode(text + "=" * pad)


# --------------------------------------------------------------------------
# QR backends -- each returns a matrix of 0/1 rows, without a quiet zone
# --------------------------------------------------------------------------


class QRBackend:
    name = "none"

    def matrix(self, text: str, ecc: str):
        raise NotImplementedError

    def describe(self, text: str, ecc: str) -> str:
        return ""


class SegnoBackend(QRBackend):
    name = "segno"

    def __init__(self):
        import segno  # noqa: F401

        self.segno = segno

    def _make(self, text, ecc):
        # mode="alphanumeric" is asserted rather than inferred so that an
        # unexpected character fails loudly here instead of silently
        # downgrading the symbol to byte mode.
        return self.segno.make(text, error=ecc.lower(), mode="alphanumeric", boost_error=False)

    def matrix(self, text, ecc):
        qr = self._make(text, ecc)
        return [list(row) for row in qr.matrix]

    def describe(self, text, ecc):
        qr = self._make(text, ecc)
        return f"version {qr.version}, ecc {qr.error.upper()}"


class QRCodeLibBackend(QRBackend):
    name = "qrcode"

    def __init__(self):
        import qrcode
        import qrcode.constants as C

        self.qrcode = qrcode
        self._levels = {
            "L": C.ERROR_CORRECT_L,
            "M": C.ERROR_CORRECT_M,
            "Q": C.ERROR_CORRECT_Q,
            "H": C.ERROR_CORRECT_H,
        }

    def _make(self, text, ecc):
        qr = self.qrcode.QRCode(
            version=None,
            error_correction=self._levels[ecc.upper()],
            box_size=1,
            border=0,
        )
        qr.add_data(text)
        qr.make(fit=True)
        return qr

    def matrix(self, text, ecc):
        qr = self._make(text, ecc)
        return [[1 if cell else 0 for cell in row] for row in qr.get_matrix()]

    def describe(self, text, ecc):
        return f"version {self._make(text, ecc).version}, ecc {ecc.upper()}"


class QREncodeCLIBackend(QRBackend):
    """Fallback: shell out to the qrencode binary and read its ASCII art."""

    name = "qrencode(cli)"

    def __init__(self):
        if not shutil.which("qrencode"):
            raise RuntimeError("qrencode not on PATH")

    def matrix(self, text, ecc):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(text)
            path = fh.name
        try:
            out = subprocess.run(
                ["qrencode", "-l", ecc.upper(), "-t", "ASCII", "-m", "0", "-o", "-", "-r", path],
                check=True,
                capture_output=True,
            ).stdout.decode("utf-8", "replace")
        finally:
            os.unlink(path)
        rows = []
        for line in out.splitlines():
            if not line.strip():
                continue
            # qrencode ASCII prints two characters per module.
            row = [0 if line[i] == "#" else 1 for i in range(0, len(line) - 1, 2)]
            if row:
                rows.append(row)
        if not rows:
            raise RuntimeError("qrencode produced no output")
        return rows

    def describe(self, text, ecc):
        n = len(self.matrix(text, ecc))
        return f"{n}x{n} modules, ecc {ecc.upper()}"


def pick_backend(preferred: str | None = None) -> QRBackend:
    candidates = [SegnoBackend, QRCodeLibBackend, QREncodeCLIBackend]
    if preferred:
        by_name = {"segno": SegnoBackend, "qrcode": QRCodeLibBackend, "qrencode": QREncodeCLIBackend}
        if preferred not in by_name:
            sys.exit(f"unknown backend {preferred!r}")
        candidates = [by_name[preferred]]
    errors = []
    for cls in candidates:
        try:
            return cls()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"  {cls.name}: {exc}")
    sys.exit(
        "No QR backend available. Tried:\n"
        + "\n".join(errors)
        + "\n\nInstall one of:  pip install segno   |   pip install qrcode"
        "\n                 apt install qrencode | brew install qrencode"
    )


def matrix_to_svg(matrix, border: int = QUIET_ZONE, crisp: bool = True) -> str:
    """Render a module matrix as a compact, crisply-scaling inline SVG."""
    n = len(matrix)
    size = n + 2 * border
    runs = []
    for y, row in enumerate(matrix):
        x = 0
        while x < n:
            if row[x]:
                start = x
                while x < n and row[x]:
                    x += 1
                runs.append(f"M{start + border} {y + border}h{x - start}v1h-{x - start}z")
            else:
                x += 1
    sr = ' shape-rendering="crispEdges"' if crisp else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}"'
        f'{sr} preserveAspectRatio="xMidYMid meet">'
        f'<rect width="{size}" height="{size}" fill="#fff"/>'
        f'<path fill="#000" d="{"".join(runs)}"/></svg>'
    )


# --------------------------------------------------------------------------
# layout: fit the symbol to the screen, not the other way round
# --------------------------------------------------------------------------


def _probe_modules(backend, ecc, payload_bytes, cache):
    """Module count (incl. quiet zone) of a frame carrying N payload bytes."""
    if payload_bytes in cache:
        return cache[payload_bytes]
    head = struct.pack(HEADER_STRUCT, MAGIC, FORMAT_VERSION, 0, 1, 1, 0)
    text = b32_encode(head + b"\x00" * payload_bytes)
    try:
        n = len(backend.matrix(text, ecc)) + 2 * QUIET_ZONE
    except Exception:  # noqa: BLE001  (payload exceeds version 40)
        n = 10**6
    cache[payload_bytes] = n
    return n


def fit_chunk_bytes(backend, ecc, max_modules, cache, lo=40, hi=2400):
    """Largest payload whose symbol still fits inside max_modules."""
    if _probe_modules(backend, ecc, lo, cache) > max_modules:
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _probe_modules(backend, ecc, mid, cache) <= max_modules:
            lo = mid
        else:
            hi = mid - 1
    return lo


def choose_layout(backend, ecc, screen_w, screen_h, px_per_module, forced_grid=None, max_cols=8, max_rows=6):
    """Pick the grid that moves the most bytes per photographed sheet."""
    cache = {}
    usable_h = screen_h * (1 - CHROME_FRACTION)
    usable_w = screen_w * 0.985
    grids = [forced_grid] if forced_grid else [(c, r) for c in range(1, max_cols + 1) for r in range(1, max_rows + 1)]
    best = None
    for cols, rows in grids:
        gap = 0.01 * min(usable_w, usable_h)
        cell = min((usable_w - gap * (cols - 1)) / cols, (usable_h - gap * (rows - 1)) / rows)
        max_modules = int(cell / px_per_module)
        if max_modules < 21 + 2 * QUIET_ZONE:
            continue
        chunk = fit_chunk_bytes(backend, ecc, max_modules, cache)
        if chunk <= 0:
            continue
        modules = _probe_modules(backend, ecc, chunk, cache)
        # Snap to an integer number of screen pixels per module so the browser
        # renders every module the same width instead of dithering edges.
        scale = max(1, int(cell // modules))
        per_sheet = chunk * cols * rows
        cand = {
            "cols": cols,
            "rows": rows,
            "chunk_bytes": chunk,
            "modules": modules,
            "px_per_module": scale,
            "qr_px": scale * modules,
            "bytes_per_sheet": per_sheet,
            "cell_px": int(cell),
        }
        if best is None or per_sheet > best["bytes_per_sheet"]:
            best = cand
    return best


# --------------------------------------------------------------------------
# file collection
# --------------------------------------------------------------------------


def looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(8192)
    except OSError:
        return True


def matches_any(name: str, globs) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatch(low, g.lower()) for g in globs)


def collect_files(root, exts, exclude_dirs, exclude_globs, secret_globs, max_bytes, follow_symlinks=False):
    root = os.path.abspath(root)
    picked, skipped_secret, skipped_large, skipped_binary = [], [], [], []
    exts = {e.lower() for e in exts} if exts else None

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)
            if os.path.islink(full) and not follow_symlinks:
                continue
            if not os.path.isfile(full):
                continue
            if matches_any(rel, exclude_globs) or matches_any(fname, exclude_globs):
                continue
            # The secret check runs BEFORE the extension filter on purpose: a
            # credential file must be reported and excluded no matter how --ext
            # is set, so a broad filter can never quietly sweep one onto a
            # screen you are about to photograph.
            if matches_any(fname, secret_globs):
                skipped_secret.append(rel)
                continue
            if exts is not None:
                ext = os.path.splitext(fname)[1].lower()
                # A dotfile with no extension (".bashrc") is matched by its
                # whole name, so `--ext .bashrc` behaves the way people expect.
                if ext not in exts and fname.lower() not in exts:
                    continue
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size > max_bytes:
                skipped_large.append((rel, size))
                continue
            if looks_binary(full):
                skipped_binary.append(rel)
                continue
            picked.append((rel, full, size))

    return picked, skipped_secret, skipped_large, skipped_binary


def build_archive(files, root, note=""):
    """Deterministic tar.xz: sorted, zeroed timestamps and ownership."""
    manifest = {
        "format": "agpack/1",
        "source_root": os.path.abspath(root),
        "note": note,
        "files": [],
    }
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel, full, size in files:
            with open(full, "rb") as fh:
                data = fh.read()
            info = tarfile.TarInfo(name=rel.replace(os.sep, "/"))
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            mode = os.stat(full).st_mode & 0o777
            info.mode = 0o755 if mode & 0o111 else 0o644
            tar.addfile(info, io.BytesIO(data))
            manifest["files"].append(
                {
                    "path": info.name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "mode": oct(info.mode),
                }
            )
        blob = json.dumps(manifest, indent=1, sort_keys=True).encode()
        info = tarfile.TarInfo(name=".agpack-manifest.json")
        info.size = len(blob)
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(blob))

    tar_bytes = raw.getvalue()
    packed = lzma.compress(tar_bytes, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    return tar_bytes, packed, manifest


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def make_frames(packed: bytes, chunk_bytes: int, transfer_id: int, header_json: str):
    """Frame 0 is the header/manifest; frames 1..N carry the payload."""
    chunks = [packed[i : i + chunk_bytes] for i in range(0, len(packed), chunk_bytes)]
    total = len(chunks)
    frames = []

    def frame(seq: int, payload: bytes) -> str:
        head = struct.pack(
            HEADER_STRUCT, MAGIC, FORMAT_VERSION, transfer_id, seq, total, zlib.crc32(payload) & 0xFFFFFFFF
        )
        return b32_encode(head + payload)

    frames.append((0, frame(0, header_json.encode())))
    for i, chunk in enumerate(chunks, start=1):
        frames.append((i, frame(i, chunk)))
    return frames, total


# --------------------------------------------------------------------------
# HTML sheets
# --------------------------------------------------------------------------

SHEET_CSS = """
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background:#555; font-family: ui-monospace,
  SFMono-Regular, Menlo, Consolas, monospace; }
.sheet { position:relative; width:100vw; height:100vh; background:#fff;
  page-break-after:always; break-after:page; display:flex; flex-direction:column;
  padding: 1.2vmin 1.2vmin 0.4vmin; }
.hdr { display:flex; justify-content:space-between; align-items:baseline;
  font-size: 2.0vmin; font-weight:700; letter-spacing:.02em; padding-bottom:.6vmin; }
.hdr .tid { font-weight:400; color:#444; font-size:1.6vmin; }
.grid { flex:1; display:grid; justify-content:center; align-content:center;
  min-height:0; }
.cell { display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:0; min-width:0; }
.cell svg { display:block; max-width:100%; max-height:100%; }
.cap { font-size:1.5vmin; font-weight:700; padding-top:.2vmin; color:#000;
  line-height:1; }
.foot { font-size:1.4vmin; color:#333; padding-top:.4vmin; display:flex;
  justify-content:space-between; }
.blank { visibility:hidden; }
#nav { position:fixed; left:50%; bottom:2vmin; transform:translateX(-50%);
  background:rgba(0,0,0,.82); color:#fff; padding:.6em 1.1em; border-radius:999px;
  font-size:14px; z-index:99; }
body.shoot #nav { opacity:0; }
body.shoot .sheet { display:none; }
body.shoot .sheet.active { display:flex; }
@media print { #nav { display:none; } body { background:#fff; }
  .sheet { width:auto; height:100vh; } }
"""

SHEET_JS = """
const sheets = [...document.querySelectorAll('.sheet')];
let idx = 0, shoot = false;
function render(){
  document.body.classList.toggle('shoot', shoot);
  sheets.forEach((s,i)=>s.classList.toggle('active', i===idx));
  if (shoot) sheets[idx].scrollIntoView();
  nav.textContent = shoot
    ? `SHOOT MODE  |  sheet ${idx+1} / ${sheets.length}  |  \\u2190 \\u2192 to move  |  S to exit`
    : `scroll to review  |  press S for shoot mode (one sheet per screen)`;
}
addEventListener('keydown', e => {
  if (e.key === 's' || e.key === 'S') { shoot = !shoot; idx = shoot ? idx : idx; }
  else if (!shoot) return;
  else if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') idx = Math.min(idx+1, sheets.length-1);
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') idx = Math.max(idx-1, 0);
  else if (e.key === 'Home') idx = 0;
  else if (e.key === 'End') idx = sheets.length-1;
  else return;
  e.preventDefault(); render();
});
render();
"""


def render_sheets(frames, layout, transfer_id, backend, ecc, title, crisp=True):
    cols, rows = layout["cols"], layout["rows"]
    qr_px = layout["qr_px"]
    per_sheet = cols * rows
    svg_cache = {}
    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title><style>{SHEET_CSS}</style></head><body>",
    ]
    total_sheets = (len(frames) + per_sheet - 1) // per_sheet
    tid_hex = f"{transfer_id:08x}"

    for s in range(total_sheets):
        batch = frames[s * per_sheet : (s + 1) * per_sheet]
        lo, hi = batch[0][0], batch[-1][0]
        out.append("<section class='sheet'>")
        out.append(
            f"<div class='hdr'><span>SHEET {s + 1} / {total_sheets}</span>"
            f"<span>frames {lo}–{hi}</span>"
            f"<span class='tid'>id {tid_hex}</span></div>"
        )
        out.append(
            f"<div class='grid' style='grid-template-columns:repeat({cols},{qr_px}px);"
            f"grid-template-rows:repeat({rows},auto);gap:6px'>"
        )
        for seq, text in batch:
            if text not in svg_cache:
                svg_cache[text] = matrix_to_svg(backend.matrix(text, ecc), crisp=crisp)
            svg = svg_cache[text].replace("<svg ", f"<svg width='{qr_px}' height='{qr_px}' ", 1)
            out.append(f"<div class='cell'>{svg}" f"<div class='cap'>{seq}</div></div>")
        for _ in range(per_sheet - len(batch)):
            out.append("<div class='cell blank'></div>")
        out.append("</div>")
        out.append(
            f"<div class='foot'><span>agpack · transfer {tid_hex}</span>"
            f"<span>photograph this whole sheet, square-on, in focus</span></div>"
        )
        out.append("</section>")

    out.append(f"<div id='nav'></div><script>{SHEET_JS}</script></body></html>")
    return "\n".join(out), total_sheets


# --------------------------------------------------------------------------
# human-readable reporting
# --------------------------------------------------------------------------


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n} B"


def plan_counts(packed_len, layout):
    frames = (packed_len + layout["chunk_bytes"] - 1) // layout["chunk_bytes"] + 1
    per_sheet = layout["cols"] * layout["rows"]
    return frames, (frames + per_sheet - 1) // per_sheet


def report_layout(packed_len, layout, label, selected=False):
    frames, sheets = plan_counts(packed_len, layout)
    star = " *" if selected else ""
    print(
        f"    {label + star:<13} {layout['cols']}x{layout['rows']} grid  "
        f"{layout['chunk_bytes']:>4} B/QR  {layout['modules']:>3} mod  "
        f"{layout['px_per_module']:>2} px/mod  "
        f"{human(layout['bytes_per_sheet']):>9}/sheet  ->  "
        f"{frames:>5} frames  {sheets:>4} photos"
    )
    return frames, sheets


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def cmd_calibrate(args, backend, screen):
    """One sheet per quality level, so the camera can vote on the density."""
    os.makedirs(args.out, exist_ok=True)
    sample = os.urandom(2400)
    for name, ppm in QUALITY.items():
        layout = choose_layout(backend, args.ecc, screen[0], screen[1], ppm)
        if not layout:
            print(f"  {name:<12} screen too small for any usable symbol")
            continue
        payload = sample[: layout["chunk_bytes"]]
        head = struct.pack(HEADER_STRUCT, MAGIC, FORMAT_VERSION, 0, 1, 1, zlib.crc32(payload) & 0xFFFFFFFF)
        text = b32_encode(head + payload)
        n = layout["cols"] * layout["rows"]
        frames = [(i, text) for i in range(1, n + 1)]
        html, _ = render_sheets(
            frames, layout, 0, backend, args.ecc, f"agpack calibration - {name}", crisp=not args.antialias
        )
        path = os.path.join(args.out, f"calibrate-{name}.html")
        with open(path, "w") as fh:
            fh.write(html)
        print(
            f"  {name:<12} {layout['cols']}x{layout['rows']} grid  "
            f"{n:>2} symbols  {layout['px_per_module']} px/mod  "
            f"{human(layout['bytes_per_sheet'])}/sheet  -> {path}"
        )
    print(
        "\nOpen each file, go fullscreen, press S for shoot mode, and "
        "photograph it exactly\nas you plan to shoot the real thing. Then on "
        "the receiving machine:\n\n"
        "    python3 agunpack.py calibrate <folder-of-photos>\n\n"
        "It reports how many symbols decoded per photo. Use the densest level "
        "that\nreads 100%, then step down one for margin."
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Pack a source tree into photographable QR sheets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default=".", help="source tree (read-only)")
    p.add_argument(
        "--ext",
        default="",
        help="comma-separated extensions to include, e.g. .py,.md,.json " "(default: all text files)",
    )
    p.add_argument("--out", default="agpack-out", help="output directory")
    p.add_argument(
        "--screen",
        default="1920x1080",
        help="resolution of the display you will photograph, e.g. " "2560x1440. THIS IS THE MOST IMPORTANT SETTING.",
    )
    p.add_argument(
        "--quality",
        choices=sorted(QUALITY),
        default="balanced",
        help="screen pixels per QR module: safe=5.5, balanced=4.5, " "dense=3.5 (default balanced)",
    )
    p.add_argument("--px-per-module", type=float, default=None, help="override the pixels-per-module target directly")
    p.add_argument("--grid", default=None, help="force a grid such as 4x2 instead of auto-choosing")
    p.add_argument("--antialias", action="store_true", help="render smooth rather than pixel-snapped module edges")
    p.add_argument("--ecc", default="M", choices=list("LMQH"), help="QR error correction (default M)")
    p.add_argument("--backend", default=None, choices=["segno", "qrcode", "qrencode"])
    p.add_argument("--max-file-size", type=int, default=1_048_576, help="skip files larger than this (default 1 MiB)")
    p.add_argument("--exclude", default="", help="extra comma-separated glob patterns to skip")
    p.add_argument("--exclude-dir", default="", help="extra comma-separated directory names to skip")
    p.add_argument("--include-secrets", action="store_true", help="do NOT skip .env / key / credential files")
    p.add_argument("--follow-symlinks", action="store_true")
    p.add_argument("--note", default="", help="free-text note stored in the manifest")
    p.add_argument("--estimate", action="store_true", help="report sizes and frame counts, write nothing")
    p.add_argument("--calibrate", action="store_true", help="emit test sheets to find the safe density for your camera")
    p.add_argument("--force", action="store_true", help="allow writing into a non-empty --out")
    args = p.parse_args(argv)

    backend = pick_backend(args.backend)
    sm = re.fullmatch(r"(\d+)\s*[xX*]\s*(\d+)", args.screen.strip())
    if not sm:
        sys.exit("--screen must look like 1920x1080")
    screen = (int(sm.group(1)), int(sm.group(2)))
    print(f"agpack: QR backend = {backend.name}, target display = " f"{screen[0]}x{screen[1]}\n")

    if args.calibrate:
        return cmd_calibrate(args, backend, screen)

    forced = None
    if args.grid:
        gm = re.fullmatch(r"(\d+)\s*[xX*]\s*(\d+)", args.grid.strip())
        if not gm:
            sys.exit("--grid must look like 4x2")
        forced = (int(gm.group(1)), int(gm.group(2)))

    ppm = args.px_per_module or QUALITY[args.quality]
    layout = choose_layout(backend, args.ecc, screen[0], screen[1], ppm, forced)
    if not layout:
        sys.exit(
            f"Cannot fit a usable QR symbol on a {screen[0]}x{screen[1]} "
            f"display at {ppm} px/module. Lower --quality or use a bigger "
            f"screen."
        )

    exts = [e.strip() if e.strip().startswith(".") else "." + e.strip() for e in args.ext.split(",") if e.strip()]
    exclude_globs = [g.strip() for g in args.exclude.split(",") if g.strip()]
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | {d.strip() for d in args.exclude_dir.split(",") if d.strip()}
    secret_globs = [] if args.include_secrets else DEFAULT_SECRET_GLOBS

    files, sec, large, binary = collect_files(
        args.root, exts, exclude_dirs, exclude_globs, secret_globs, args.max_file_size, args.follow_symlinks
    )

    if not files:
        sys.exit("No files matched. Check --root and --ext.")

    raw_total = sum(f[2] for f in files)
    print(f"  matched {len(files)} files, {human(raw_total)} on disk")
    if sec:
        print(f"  SKIPPED {len(sec)} possible-secret file(s) " f"(--include-secrets to override):")
        for r in sec[:12]:
            print(f"      {r}")
        if len(sec) > 12:
            print(f"      ... and {len(sec) - 12} more")
    if large:
        print(f"  skipped {len(large)} file(s) over {human(args.max_file_size)}:")
        for r, s in large[:6]:
            print(f"      {r}  ({human(s)})")
        if len(large) > 6:
            print(f"      ... and {len(large) - 6} more")
    if binary:
        print(f"  skipped {len(binary)} binary file(s)")

    tar_bytes, packed, manifest = build_archive(files, args.root, args.note)
    ratio = raw_total / len(packed) if packed else 0
    print(f"\n  tar        {human(len(tar_bytes))}")
    print(f"  compressed {human(len(packed))}   ({ratio:.1f}x)")
    b32_len = (len(packed) + 4) // 5 * 8
    print(f"  base32     {human(b32_len)} of QR-safe text\n")

    print(f"  capture cost on a {screen[0]}x{screen[1]} display:")
    for name in ("safe", "balanced", "dense"):
        alt = (
            layout
            if (name == args.quality and not args.px_per_module and not forced)
            else choose_layout(backend, args.ecc, screen[0], screen[1], QUALITY[name], forced)
        )
        if alt:
            report_layout(len(packed), alt, name, selected=(alt is layout))
    if args.px_per_module or forced:
        print()
        report_layout(len(packed), layout, "custom", selected=True)
    print("\n  * = selected.  'px/mod' is how many screen pixels each QR " "module gets;")
    print("  below about 3 the display physically cannot show the data.")

    if args.estimate:
        print("\n(--estimate: nothing written)")
        return

    if os.path.exists(args.out) and os.listdir(args.out) and not args.force:
        sys.exit(f"\n{args.out} exists and is not empty. Use --force or pick another --out.")
    os.makedirs(args.out, exist_ok=True)

    transfer_id = int.from_bytes(hashlib.sha256(packed).digest()[:4], "big")
    header = {
        "format": "agpack/1",
        "transfer_id": f"{transfer_id:08x}",
        "sha256": hashlib.sha256(packed).hexdigest(),
        "compressed_bytes": len(packed),
        "tar_bytes": len(tar_bytes),
        "file_count": len(files),
        "codec": "tar+xz+base32",
        "chunk_bytes": layout["chunk_bytes"],
        "grid": f"{layout['cols']}x{layout['rows']}",
        "note": args.note,
    }
    header_json = json.dumps(header, sort_keys=True, separators=(",", ":"))

    frames, total = make_frames(packed, layout["chunk_bytes"], transfer_id, header_json)
    sample_text = max((t for _, t in frames), key=len)
    print(
        f"\n  QR symbol: {backend.describe(sample_text, args.ecc)}, "
        f"drawn at {layout['qr_px']}px ({layout['px_per_module']} px/module)"
    )

    html, sheet_count = render_sheets(
        frames, layout, transfer_id, backend, args.ecc, f"agpack {transfer_id:08x}", crisp=not args.antialias
    )

    html_path = os.path.join(args.out, "sheets.html")
    with open(html_path, "w") as fh:
        fh.write(html)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(
            {
                **header,
                "frames": total + 1,
                "sheets": sheet_count,
                "ecc": args.ecc,
                "layout": layout,
                "screen": f"{screen[0]}x{screen[1]}",
                "contents": manifest["files"],
            },
            fh,
            indent=2,
        )
    # Kept so that if a real transfer channel ever appears, you can copy this
    # one file instead of photographing anything.
    with open(os.path.join(args.out, "payload.tar.xz"), "wb") as fh:
        fh.write(packed)

    print(f"\n  wrote {html_path}")
    print(f"        {os.path.join(args.out, 'manifest.json')}")
    print(f"        {os.path.join(args.out, 'payload.tar.xz')}  " f"(shortcut if you ever get a real channel)")
    print(f"\n  {total + 1} frames across {sheet_count} sheet(s) to photograph.")
    print("\nNext: open sheets.html in a browser, press F for browser fullscreen,")
    print("then S for shoot mode. Arrow keys advance one sheet at a time.")
    print("Take ONE well-lit, square-on, in-focus photo per sheet.")
    print("Source tree was not modified.")


if __name__ == "__main__":
    main()
