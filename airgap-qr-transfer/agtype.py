#!/usr/bin/env python3
# agtype.py - the hand-typeable sender.
#
# This is the minimal version of agmini, kept deliberately short because the
# offline machine cannot receive files: whatever runs there has to arrive
# through the keyboard. ~65 lines, about ten minutes of typing.
#
# Use agmini.py instead if you can get a file onto that machine, or if you want
# the include/exclude API, layout optimiser and notebook integration.
# Output is decoded by the same agunpack.py either way.
#
# Usage: python3 agtype.py ROOT OUT.html .py,.md,.json [CHUNK] [COLS] [ROWS]
import base64, io, json, hashlib, lzma, os, struct, sys, tarfile, zlib
import segno

root, out = sys.argv[1], sys.argv[2]
exts = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
CH = int(sys.argv[4]) if len(sys.argv) > 4 else 600
CO = int(sys.argv[5]) if len(sys.argv) > 5 else 4
RO = int(sys.argv[6]) if len(sys.argv) > 6 else 2
SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as t:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP]
        for n in sorted(fn):
            if n.startswith(".env") or n.endswith((".pem", ".key")):
                continue
            if exts and not n.endswith(exts):
                continue
            f = os.path.join(dp, n)
            if not os.path.isfile(f) or os.path.islink(f):
                continue
            d = open(f, "rb").read()
            if b"\0" in d[:8192] or len(d) > 1 << 20:
                continue
            i = tarfile.TarInfo(os.path.relpath(f, root).replace(os.sep, "/"))
            i.size, i.mtime, i.mode = len(d), 0, 0o644
            t.addfile(i, io.BytesIO(d))

p = lzma.compress(buf.getvalue(), preset=9)
tid = int.from_bytes(hashlib.sha256(p).digest()[:4], "big")
hdr = json.dumps(
    {
        "format": "agpack/1",
        "transfer_id": "%08x" % tid,
        "sha256": hashlib.sha256(p).hexdigest(),
        "compressed_bytes": len(p),
        "codec": "tar+xz+base32",
        "chunk_bytes": CH,
        "grid": "%dx%d" % (CO, RO),
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()
ck = [p[i : i + CH] for i in range(0, len(p), CH)]
N = len(ck)


def fr(s, d):
    h = struct.pack(">2sBIHHI", b"AG", 1, tid, s, N, zlib.crc32(d) & 0xFFFFFFFF)
    return base64.b32encode(h + d).decode().rstrip("=")


F = [(0, fr(0, hdr))] + [(i, fr(i, c)) for i, c in enumerate(ck, 1)]
w = [
    "<html><head><meta charset=utf-8><style>"
    "body{margin:0;background:#fff;font:700 13px monospace}"
    ".s{width:100vw;height:100vh;display:flex;flex-direction:column;"
    "align-items:center;justify-content:center;page-break-after:always}"
    ".g{display:grid;grid-template-columns:repeat(%d,auto);gap:6px}"
    ".c{text-align:center}svg{display:block}</style></head><body>" % CO
]
per = CO * RO
for s in range(0, len(F), per):
    w.append(
        "<div class=s><div>SHEET %d/%d &nbsp; id %08x</div><div class=g>"
        % (s // per + 1, (len(F) + per - 1) // per, tid)
    )
    for seq, txt in F[s : s + per]:
        q = segno.make(txt, error="m", mode="alphanumeric")
        w.append("<div class=c>%s<div>%d</div></div>" % (q.svg_inline(scale=4, border=3), seq))
    w.append("</div></div>")
open(out, "w").write("".join(w) + "</body></html>")
print("%d files -> %d bytes -> %d frames, %d sheets -> %s" % (len(ck), len(p), len(F), (len(F) + per - 1) // per, out))
print("sha256", hashlib.sha256(p).hexdigest())
