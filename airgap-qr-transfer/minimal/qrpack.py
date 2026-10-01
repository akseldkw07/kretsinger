"""
qrpack - turn a source tree into QR sheets you photograph off the screen.

    import qrpack
    sheets = qrpack.pack("~/kretsinger")               # DEFAULT filetypes
    sheets = qrpack.pack("~/kretsinger", (".py",))     # or pick your own
    display(HTML(sheets[1]))

Selection is inclusion-driven: a file is sent only if its name ends with one
of DEFAULT. Notebooks are sent as input cells only - no outputs, execution
counts or metadata - which on a real tree is ~24 photographs instead of ~848.

Nothing is written to disk. Rebuild with agunpack.py. Needs segno.
"""

import base64, hashlib, io, json, lzma, os, struct, tarfile, zlib
from collections.abc import Sequence

import segno

StrPath = str | os.PathLike[str]

DEFAULT = (".py", ".ipynb", ".zsh", ".md", ".bash", ".json", ".sh", ".toml", ".code-workspace")

# DEFAULT decides what is sent; these two only stop it reaching somewhere silly.
# Both earn their line: .mypy_cache alone is 145 .json files (a .venv would be
# thousands of .py), and vault/source_tokens.zsh is a credential that .zsh would
# otherwise put on a screen you are about to photograph.
SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".mypy_cache", ".pytest_cache"}
SECRET = ("secret", "token", "password", "credential", ".env", "vault/", ".pem", ".key", "id_rsa", ".netrc")


def read(path: StrPath, rel: str) -> bytes:
    """Bytes as sent. Notebooks lose outputs, execution counts and metadata -
    usually 90%+ of the file, and the part you regenerate by re-running.
    outputs/execution_count survive as empty only because nbformat needs them."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if not rel.endswith(".ipynb"):
        return raw
    try:
        nb = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw
    cells: list[dict] = []
    for c in nb.get("cells", []):
        if (k := c.get("cell_type")) not in ("code", "markdown", "raw"):
            continue
        s = c.get("source", "")
        cells.append({"cell_type": k, "metadata": {}, "source": s.splitlines(True) if isinstance(s, str) else s})
        if k == "code":
            cells[-1].update(execution_count=None, outputs=[])
    # minor 4 deliberately: 4.5 requires a per-cell "id" and we emit none, so
    # declaring 5 would produce notebooks that are invalid by their own schema.
    doc = {"cells": cells, "metadata": {}, "nbformat": nb.get("nbformat", 4), "nbformat_minor": 4}
    return (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode()


def collect(root: StrPath, include: Sequence[str] = DEFAULT, max_size: int = 1 << 20) -> list[str]:
    """Relative paths pack() would send, cheapest checks first."""
    root = os.path.abspath(os.path.expanduser(root))
    exts, out = tuple(include), []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP)
        for n in sorted(fn):
            full = os.path.join(dp, n)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if not n.endswith(exts) or os.path.islink(full) or not os.path.isfile(full):
                continue
            if any(s in rel.lower() for s in SECRET):
                continue
            blob = read(full, rel)  # notebooks are measured after stripping
            if len(blob) > max_size or b"\0" in blob[:8192]:
                continue
            out.append(rel)
    return out


def pack(
    root: StrPath,
    include: Sequence[str] = DEFAULT,
    chunk: int = 1200,
    cols: int = 6,
    rows: int = 3,
    scale: int = 4,
    max_size: int = 1 << 20,
    quiet: bool = False,
) -> dict[int, str]:
    """Return {sheet_number: html_fragment} - one entry per photograph.

    Each fragment carries inline styles and a white background, so it renders
    correctly in a dark-theme notebook.

    Defaults suit a 3840x2160 sending display: a 3500x1834 sheet carrying
    21.1 KB, leaving ~340x326 px of headroom. Smaller screens need less on a
    sheet - 1440p: chunk=1350, cols=4, rows=2. 1080p: chunk=750, cols=4, rows=2.

    scale is *CSS* pixels per QR module, the setting that decides whether a
    photo can be decoded at all; below ~3 the display never renders the detail.
    Keep it a whole number: at 4.5 every module edge lands on a half pixel and
    the browser antialiases it, which is the softness that costs you a decode.
    On a HiDPI desktop (macOS 2x, where a 4K panel reports a 1920x1080 CSS
    viewport) halve it - scale=2 there is still 4 device pixels per module.

    Raising chunk grows the symbol, so it pays only until the symbol stops
    fitting the grid; area efficiency flattens past ~1200 bytes and the
    alphanumeric/ECC-M ceiling is ~2100.
    """
    root = os.path.abspath(os.path.expanduser(root))
    files = collect(root, include, max_size)
    if not files:
        raise ValueError(f"no files matched under {root}")

    entries: list[dict[str, str | int]] = []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in [*files, None]:  # None is the manifest, so it lands complete
            if rel is None:
                name = ".agpack-manifest.json"
                blob = json.dumps({"format": "agpack/1", "files": entries}, sort_keys=True).encode()
                mode = 0o644
            else:
                full = os.path.join(root, rel)
                name, blob = rel, read(full, rel)
                mode = 0o755 if os.stat(full).st_mode & 0o111 else 0o644
                entries.append({"path": rel, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid, info.mode = len(blob), 0, 0, 0, mode
            tar.addfile(info, io.BytesIO(blob))

    payload = lzma.compress(buf.getvalue(), preset=9)
    sha = hashlib.sha256(payload).hexdigest()
    tid = int(sha[:8], 16)
    parts = [payload[i : i + chunk] for i in range(0, len(payload), chunk)]

    def frame(seq: int, blob: bytes) -> str:
        head = struct.pack(">2sBIHHI", b"AG", 1, tid, seq, len(parts), zlib.crc32(blob) & 0xFFFFFFFF)
        return base64.b32encode(head + blob).decode().rstrip("=")

    header = json.dumps(
        {
            "format": "agpack/1",
            "transfer_id": "%08x" % tid,
            "sha256": sha,
            "compressed_bytes": len(payload),
            "codec": "tar+xz+base32",
            "chunk_bytes": chunk,
            "grid": "%dx%d" % (cols, rows),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    frames = [(0, frame(0, header))] + [(i, frame(i, c)) for i, c in enumerate(parts, 1)]

    per = cols * rows
    total = -(-len(frames) // per)
    sheets = {}
    for s in range(0, len(frames), per):
        cells = "".join(
            "<div style='text-align:center'>%s<div style='font:700 12px monospace;color:#000'>%d</div></div>"
            % (segno.make(t, error="m", mode="alphanumeric").svg_inline(scale=scale, border=3), seq)
            for seq, t in frames[s : s + per]
        )
        sheets[s // per + 1] = (
            "<div style='background:#fff;padding:14px;display:inline-block'>"
            "<div style='font:700 14px monospace;color:#000;padding-bottom:8px'>"
            "SHEET %d / %d &nbsp;&middot;&nbsp; id %08x &nbsp;&middot;&nbsp; frames %d-%d</div>"
            "<div style='display:grid;grid-template-columns:repeat(%d,auto);gap:8px'>%s</div></div>"
            % (s // per + 1, total, tid, frames[s][0], frames[min(s + per, len(frames)) - 1][0], cols, cells)
        )

    if not quiet:
        raw = sum(int(e["bytes"]) for e in entries)  # as sent, not as on disk
        print(
            f"{len(files)} files, {raw/1024:,.0f} KB -> {len(payload)/1024:,.0f} KB "
            f"compressed -> {len(frames)} frames -> {total} photographs"
        )
        print(f"transfer id {tid:08x}   sha256 {sha[:32]}...")
        print("nothing written to disk")
    return sheets
