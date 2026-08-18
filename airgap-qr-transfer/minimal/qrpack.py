"""
qrpack - turn a source tree into QR sheets you photograph off the screen.

Nothing is written to disk. pack() returns the sheets as HTML fragments you
render inline in a notebook:

    import qrpack
    sheets = qrpack.pack("~/kretsinger", include=[".py", ".md"])
    display(HTML(sheets[1]))

Rebuild on the receiving machine with agunpack.py.
Needs segno (pip install segno). Read-only with respect to `root`.
"""

import base64, fnmatch, hashlib, io, json, lzma, os, struct, tarfile, zlib
import segno

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

# Never packed unless secrets=True, and checked before `include` so a broad
# include list cannot sweep one onto a screen you are about to photograph.
# vault/* and *token* are here because kretsinger/vault/ holds API tokens whose
# names ("hugging-face-token.txt") the generic patterns miss.
SECRETS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    ".netrc",
    "credentials*",
    "*secret*",
    "*.token",
    "*token*",
    "vault/*",
    "*-api.txt",
    "*password*",
)


def _globs(pats):
    """'.py' -> '*.py'; anything else passes through as a glob."""
    if pats is None:
        return None
    if isinstance(pats, str):
        pats = [pats]
    return [("*" + p if p.startswith(".") and not any(c in p for c in "*?/") else p) for p in pats if p.strip()]


def _hit(rel, name, globs):
    return any(fnmatch.fnmatch(name.lower(), g.lower()) or fnmatch.fnmatch(rel.lower(), g.lower()) for g in globs)


def collect(root, include=None, exclude=(), max_size=1 << 20, secrets=False):
    """Relative paths that pack() would send. include/exclude take extensions
    ('.py') and globs ('*_test.py', 'docs/*'). include=None means all text."""
    root = os.path.abspath(os.path.expanduser(root))
    inc, exc = _globs(include), _globs(exclude) or []
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if not os.path.isfile(full) or os.path.islink(full):
                continue
            if not secrets and _hit(rel, name, SECRETS):
                continue
            if inc is not None and not _hit(rel, name, inc):
                continue
            if exc and _hit(rel, name, exc):
                continue
            if os.path.getsize(full) > max_size:
                continue
            with open(full, "rb") as fh:
                if b"\0" in fh.read(8192):  # skip binaries
                    continue
            out.append(rel)
    return out


def pack(
    root, include=None, exclude=(), chunk=600, cols=4, rows=2, scale=4, max_size=1 << 20, secrets=False, quiet=False
):
    """Return {sheet_number: html_fragment} - one entry per photograph.

    Writes nothing anywhere. Each fragment carries its own inline styles and a
    white background, so it renders correctly in a dark-theme notebook.

    scale is screen pixels per QR module, the setting that decides whether a
    photo can be decoded at all; below ~3 the display cannot show the detail.
    Defaults suit 1080p. On 1440p try chunk=1100; on 4K add cols=6, rows=3.
    """
    root = os.path.abspath(os.path.expanduser(root))
    files = collect(root, include, exclude, max_size, secrets)
    if not files:
        raise ValueError(f"no files matched under {root}")

    manifest = {"format": "agpack/1", "files": []}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in files:
            full = os.path.join(root, rel)
            with open(full, "rb") as fh:
                blob = fh.read()
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.uid, info.gid = len(blob), 0, 0, 0
            info.mode = 0o755 if os.stat(full).st_mode & 0o111 else 0o644
            tar.addfile(info, io.BytesIO(blob))
            manifest["files"].append({"path": rel, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
        blob = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo(".agpack-manifest.json")
        info.size, info.mtime, info.uid, info.gid, info.mode = len(blob), 0, 0, 0, 0o644
        tar.addfile(info, io.BytesIO(blob))

    payload = lzma.compress(buf.getvalue(), preset=9)
    sha = hashlib.sha256(payload).hexdigest()
    tid = int(sha[:8], 16)
    chunks = [payload[i : i + chunk] for i in range(0, len(payload), chunk)]

    def frame(seq, blob):
        head = struct.pack(">2sBIHHI", b"AG", 1, tid, seq, len(chunks), zlib.crc32(blob) & 0xFFFFFFFF)
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
    frames = [(0, frame(0, header))] + [(i, frame(i, c)) for i, c in enumerate(chunks, 1)]

    per = cols * rows
    total = -(-len(frames) // per)
    sheets = {}
    for s in range(0, len(frames), per):
        n = s // per + 1
        cells = []
        for seq, text in frames[s : s + per]:
            qr = segno.make(text, error="m", mode="alphanumeric")
            cells.append(
                "<div style='text-align:center'>%s"
                "<div style='font:700 12px monospace;color:#000'>%d</div></div>"
                % (qr.svg_inline(scale=scale, border=3), seq)
            )
        sheets[n] = (
            "<div style='background:#fff;padding:14px;display:inline-block'>"
            "<div style='font:700 14px monospace;color:#000;padding-bottom:8px'>"
            "SHEET %d / %d &nbsp;&middot;&nbsp; id %08x &nbsp;&middot;&nbsp; "
            "frames %d-%d</div>"
            "<div style='display:grid;grid-template-columns:repeat(%d,auto);"
            "gap:8px'>%s</div></div>"
            % (n, total, tid, frames[s][0], frames[min(s + per, len(frames)) - 1][0], cols, "".join(cells))
        )

    if not quiet:
        raw = sum(os.path.getsize(os.path.join(root, f)) for f in files)
        print(
            f"{len(files)} files, {raw/1024:,.0f} KB -> {len(payload)/1024:,.0f} KB "
            f"compressed -> {len(frames)} frames -> {total} photographs"
        )
        print(f"transfer id {tid:08x}   sha256 {sha[:32]}...")
        print("nothing written to disk")
    return sheets
