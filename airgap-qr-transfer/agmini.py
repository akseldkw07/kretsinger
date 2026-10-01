#!/usr/bin/env python3
"""
agmini - pack a source tree into photographable QR sheets, as a function.

    import agmini

    agmini.collect("~/repo", include=[".py", ".md"])      # check the file list
    info = agmini.pack("~/repo", "sheets.html",
                       include=[".py", ".md"], exclude=["*_test.py"])

Open sheets.html fullscreen, photograph one screen at a time, then rebuild on
the other machine with agunpack.py. Read-only with respect to `root`.

Needs segno (pip install segno).
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
SECRETS = (".env", ".env.*", "*.pem", "*.key", "id_rsa*", ".netrc", "credentials*", "*secret*", "*.token")


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
    """Return the relative paths that pack() would send.

    include/exclude take extensions ('.py') and globs ('*_test.py', 'docs/*').
    include=None means every text file. Credential files are skipped unless
    secrets=True, and skipping happens before `include` is applied.
    """
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
    root,
    out="sheets.html",
    include=None,
    exclude=(),
    chunk=600,
    cols=4,
    rows=2,
    scale=4,
    max_size=1 << 20,
    secrets=False,
):
    """Write QR sheets for `root` to `out`. Returns a summary dict.

    chunk    bytes of payload per QR symbol
    cols,rows  symbols per sheet
    scale    screen pixels per QR module - the setting that decides whether a
             photo can be decoded at all. Below ~3 the display cannot show the
             detail. Defaults suit a 1080p screen; on a 1440p+ display try
             chunk=1100, or cols=6, rows=3 on 4K.
    """
    root = os.path.abspath(os.path.expanduser(root))
    files = collect(root, include, exclude, max_size, secrets)
    if not files:
        raise ValueError(f"no files matched under {root}")

    manifest = {"format": "agpack/1", "files": []}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in files:
            with open(os.path.join(root, rel), "rb") as fh:
                blob = fh.read()
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.uid, info.gid = len(blob), 0, 0, 0
            info.mode = 0o755 if os.stat(os.path.join(root, rel)).st_mode & 0o111 else 0o644
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
    sheets = -(-len(frames) // per)
    w = [
        "<html><head><meta charset=utf-8><style>"
        "body{margin:0;background:#fff;font:700 13px monospace}"
        ".s{width:100vw;height:100vh;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;page-break-after:always}"
        ".g{display:grid;grid-template-columns:repeat(%d,auto);gap:6px}"
        ".c{text-align:center}svg{display:block}</style></head><body>" % cols
    ]
    for s in range(0, len(frames), per):
        w.append("<div class=s><div>SHEET %d/%d &nbsp; id %08x</div><div class=g>" % (s // per + 1, sheets, tid))
        for seq, text in frames[s : s + per]:
            qr = segno.make(text, error="m", mode="alphanumeric")
            w.append("<div class=c>%s<div>%d</div></div>" % (qr.svg_inline(scale=scale, border=3), seq))
        w.append("</div></div>")
    out = os.path.abspath(os.path.expanduser(out))
    with open(out, "w") as fh:
        fh.write("".join(w) + "</body></html>")

    return {
        "path": out,
        "files": len(files),
        "raw_bytes": sum(os.path.getsize(os.path.join(root, f)) for f in files),
        "compressed_bytes": len(payload),
        "frames": len(frames),
        "sheets": sheets,
        "sha256": sha,
        "transfer_id": "%08x" % tid,
    }


if __name__ == "__main__":
    import sys

    inc = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    print(json.dumps(pack(sys.argv[1], sys.argv[2], include=inc), indent=2))
