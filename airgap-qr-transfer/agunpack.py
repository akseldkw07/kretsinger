#!/usr/bin/env python3
"""
agunpack.py -- rebuild a source tree from photographs of agpack QR sheets.

Runs on the ONLINE machine. Non-destructive: it writes only into its own
work directory and into a --out directory that must be new or empty.

    python3 agunpack.py decode ~/Photos/transfer --out ~/recovered
    python3 agunpack.py decode ~/clip.mov --out ~/recovered      # video too
    python3 agunpack.py status                                   # what's missing
    python3 agunpack.py calibrate ~/Photos/calib                 # density test

Decoded frames are cached in the work directory, so you can shoot a few
sheets, run decode, see exactly which frames are still missing, re-shoot
only those, and run decode again. Nothing is lost between runs.

Requires: pip install pyzbar pillow  (plus the zbar shared library:
apt install libzbar0  /  brew install zbar), pillow-heif for iPhone HEIC,
and ffmpeg only if you feed it a video.
"""

from __future__ import annotations

import argparse
import base64
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
HEADER_STRUCT = ">2sBIHHI"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp", ".dng"}
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}


def b32_decode(text: str) -> bytes:
    text = text.strip().upper().replace(" ", "")
    pad = (-len(text)) % 8
    return base64.b32decode(text + "=" * pad)


# --------------------------------------------------------------------------
# imaging
# --------------------------------------------------------------------------


def load_pil():
    try:
        from PIL import Image, ImageOps
    except ImportError:
        sys.exit("Pillow is required:  pip install pillow")
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    return Image, ImageOps


def load_zbar():
    try:
        from pyzbar.pyzbar import decode as zdecode, ZBarSymbol
    except ImportError:
        sys.exit(
            "pyzbar is required:  pip install pyzbar\n"
            "  plus the zbar library: apt install libzbar0 / brew install zbar"
        )
    return zdecode, ZBarSymbol


def iter_inputs(paths, tmpdir, fps):
    """Yield image paths, expanding directories and extracting video frames."""
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if os.path.splitext(name)[1].lower() in IMAGE_EXT:
                        yield os.path.join(dirpath, name)
                    elif os.path.splitext(name)[1].lower() in VIDEO_EXT:
                        yield from explode_video(os.path.join(dirpath, name), tmpdir, fps)
        elif os.path.splitext(path)[1].lower() in VIDEO_EXT:
            yield from explode_video(path, tmpdir, fps)
        elif os.path.isfile(path):
            yield path


def explode_video(path, tmpdir, fps):
    if not shutil.which("ffmpeg"):
        print(f"  ! ffmpeg not found, skipping video {path}")
        return
    dest = os.path.join(tmpdir, "frames_" + hashlib.md5(path.encode()).hexdigest()[:8])
    os.makedirs(dest, exist_ok=True)
    print(f"  extracting frames from {os.path.basename(path)} at {fps} fps ...")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            path,
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            os.path.join(dest, "f%05d.jpg"),
        ],
        check=False,
    )
    for name in sorted(os.listdir(dest)):
        yield os.path.join(dest, name)


def escalation_views(img, ImageOps):
    """Progressively more aggressive views, tried only when the cheap pass
    came up short. Each is a separate zbar run, so order matters: cheapest
    and most likely to help first."""
    w, h = img.size
    # Underexposure and screen glare respond to a contrast stretch.
    yield ImageOps.autocontrast(img, cutoff=1)
    # Mild downscale often *helps* zbar on noisy phone sensors.
    yield img.resize((int(w * 0.6), int(h * 0.6)), 1)
    # Overlapping tiles: a symbol filling more of the frame is easier to find.
    for n in (2, 3):
        tw, th = w // n, h // n
        ov_x, ov_y = tw // 5, th // 5
        for r in range(n):
            for c in range(n):
                yield img.crop(
                    (
                        max(0, c * tw - ov_x),
                        max(0, r * th - ov_y),
                        min(w, (c + 1) * tw + ov_x),
                        min(h, (r + 1) * th + ov_y),
                    )
                )


def decode_photo(path, escalate=False):
    """Return {seq: (payload, total, tid)} found in one photo.

    Imports live inside the function so this works as a multiprocessing
    worker without relying on inherited module state.
    """
    from PIL import Image, ImageOps

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    from pyzbar.pyzbar import decode as zdecode, ZBarSymbol

    found = {}

    def harvest(view):
        for sym in zdecode(view, symbols=[ZBarSymbol.QRCODE]):
            parsed = parse_frame(sym.data)
            if parsed:
                seq, payload, total, tid = parsed
                found[seq] = (payload, total, tid)

    try:
        with Image.open(path) as im:
            im.load()
            img = ImageOps.exif_transpose(im).convert("L")
        harvest(img)
        if escalate:
            for view in escalation_views(img, ImageOps):
                harvest(view)
    except Exception as exc:  # noqa: BLE001
        return {"__error__": str(exc)}
    return found


def parse_frame(data: bytes):
    try:
        text = data.decode("ascii", "strict")
    except UnicodeDecodeError:
        return None
    try:
        raw = b32_decode(text)
    except Exception:  # noqa: BLE001
        return None
    if len(raw) < HEADER_SIZE:
        return None
    magic, ver, tid, seq, total, crc = struct.unpack(HEADER_STRUCT, raw[:HEADER_SIZE])
    if magic != MAGIC or ver != FORMAT_VERSION:
        return None
    payload = raw[HEADER_SIZE:]
    if zlib.crc32(payload) & 0xFFFFFFFF != crc:
        return None  # corrupt read, drop it silently and wait for a better one
    return seq, payload, total, tid


# --------------------------------------------------------------------------
# work directory
# --------------------------------------------------------------------------


class Store:
    def __init__(self, workdir):
        self.dir = workdir
        self.chunks = os.path.join(workdir, "chunks")
        os.makedirs(self.chunks, exist_ok=True)
        self.meta_path = os.path.join(workdir, "state.json")
        self.meta = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as fh:
                self.meta = json.load(fh)

    def have(self):
        out = set()
        for name in os.listdir(self.chunks):
            m = re.fullmatch(r"(\d+)\.bin", name)
            if m:
                out.add(int(m.group(1)))
        return out

    def put(self, seq, payload):
        path = os.path.join(self.chunks, f"{seq}.bin")
        if os.path.exists(path):
            return False
        with open(path, "wb") as fh:
            fh.write(payload)
        return True

    def get(self, seq):
        with open(os.path.join(self.chunks, f"{seq}.bin"), "rb") as fh:
            return fh.read()

    def save(self):
        with open(self.meta_path, "w") as fh:
            json.dump(self.meta, fh, indent=2)


def describe_missing(missing, grid, total):
    """Translate missing frame numbers into 'go re-shoot sheet N'."""
    if not missing:
        return []
    per_sheet = None
    if grid:
        m = re.fullmatch(r"(\d+)x(\d+)", grid)
        if m:
            per_sheet = int(m.group(1)) * int(m.group(2))
    lines = []
    ranges, start, prev = [], None, None
    for seq in sorted(missing):
        if start is None:
            start = prev = seq
        elif seq == prev + 1:
            prev = seq
        else:
            ranges.append((start, prev))
            start = prev = seq
    if start is not None:
        ranges.append((start, prev))
    for lo, hi in ranges:
        label = f"frame {lo}" if lo == hi else f"frames {lo}-{hi}"
        if per_sheet:
            s_lo = lo // per_sheet + 1
            s_hi = hi // per_sheet + 1
            sheet = f"sheet {s_lo}" if s_lo == s_hi else f"sheets {s_lo}-{s_hi}"
            lines.append(f"    {label:<22} -> re-shoot {sheet}")
        else:
            lines.append(f"    {label}")
    return lines


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def safe_extract(packed: bytes, out_dir: str):
    tar_bytes = lzma.decompress(packed)
    written, manifest = [], None
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = tar.getmembers()
        for m in members:
            if m.name.startswith("/") or os.path.isabs(m.name):
                raise ValueError(f"refusing absolute path in archive: {m.name}")
            norm = os.path.normpath(m.name)
            if norm.startswith("..") or os.sep + ".." + os.sep in os.sep + norm + os.sep:
                raise ValueError(f"refusing path traversal in archive: {m.name}")
            if m.issym() or m.islnk() or m.isdev():
                raise ValueError(f"refusing link/device member: {m.name}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"refusing unusual member type: {m.name}")
        os.makedirs(out_dir, exist_ok=True)
        for m in members:
            # filter="data" is the 3.14 default and must be explicit before then,
            # or extraction warns now and changes behaviour later. It keeps 0o755
            # and 0o644 intact - the executable bits this transfer preserves - and
            # strips setuid/setgid, which nothing here should ever carry anyway.
            tar.extract(m, out_dir, filter="data")
            if m.isfile():
                written.append(m.name)
        mpath = os.path.join(out_dir, ".agpack-manifest.json")
        if os.path.exists(mpath):
            with open(mpath) as fh:
                manifest = json.load(fh)
    return written, manifest


def verify_against_manifest(out_dir, manifest):
    bad, ok = [], 0
    for entry in manifest.get("files", []):
        path = os.path.join(out_dir, entry["path"])
        if not os.path.exists(path):
            bad.append((entry["path"], "missing"))
            continue
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != entry["sha256"]:
            bad.append((entry["path"], "sha256 mismatch"))
        else:
            ok += 1
    return ok, bad


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def run_pass(paths, escalate, jobs, label):
    """Decode a batch of photos, in parallel when it is worth it."""
    results = {}
    if jobs > 1 and len(paths) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(decode_photo, p, escalate): p for p in paths}
            for i, fut in enumerate(as_completed(futures), 1):
                path = futures[fut]
                results[path] = fut.result()
                print(f"\r  {label}: {i}/{len(paths)} photos", end="", flush=True)
    else:
        for i, path in enumerate(paths, 1):
            results[path] = decode_photo(path, escalate)
            print(f"\r  {label}: {i}/{len(paths)} photos", end="", flush=True)
    print()
    return results


def cmd_decode(args):
    load_pil()
    load_zbar()
    store = Store(args.work)

    header = store.meta.get("header")
    total = store.meta.get("total")
    tid = store.meta.get("transfer_id")

    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = list(iter_inputs(args.inputs, tmpdir, args.fps))
        if not inputs:
            sys.exit("No images or videos found in the given path(s).")
        jobs = args.jobs or min(8, (os.cpu_count() or 2))
        print(f"scanning {len(inputs)} image(s) with {jobs} worker(s)\n")

        # Phase 1: one cheap full-frame decode per photo. On a well-shot
        # sheet this already returns every symbol.
        results = run_pass(inputs, False, jobs, "pass 1")

        counts = {p: len([k for k in r if k != "__error__"]) for p, r in results.items()}
        for p, r in results.items():
            if "__error__" in r:
                print(f"  ! could not read {os.path.basename(p)}: {r['__error__']}")

        # Expected symbols per sheet: authoritative from the header if we have
        # it, otherwise the best any single photo managed.
        expected = 0
        for r in results.values():
            if 0 in r:
                try:
                    header = json.loads(r[0][0].decode())
                except Exception:  # noqa: BLE001
                    pass
        if header and header.get("grid"):
            m = re.fullmatch(r"(\d+)x(\d+)", header["grid"])
            if m:
                expected = int(m.group(1)) * int(m.group(2))
        if not expected:
            expected = max(counts.values(), default=0)

        # A transfer smaller than one full sheet can never yield a full grid,
        # and a sheet that is already complete needs no second look.
        seen = set(store.have())
        for r in results.values():
            seen.update(k for k in r if k != "__error__")
        frame_total = total or max(
            (v[1] for r in results.values() for k, v in r.items() if k != "__error__"), default=0
        )
        if frame_total:
            expected = min(expected, frame_total + 1)
        complete = frame_total and seen >= set(range(frame_total + 1))

        # Phase 2: only the photos that under-delivered get the slow treatment.
        weak = [] if complete else [p for p, n in counts.items() if n < expected]
        if complete:
            print(f"\n  all {frame_total + 1} frames recovered on the first pass")
        if weak and not args.fast:
            print(f"\n  {len(weak)} photo(s) yielded fewer than {expected} " f"symbols; retrying those harder")
            for path, res in run_pass(weak, True, jobs, "pass 2").items():
                if "__error__" not in res:
                    results[path].update(res)
        elif not weak and not complete:
            print(f"\n  every photo yielded the full {expected} symbols")

    new_count = 0
    for path, res in sorted(results.items()):
        for seq, val in sorted(res.items()):
            if seq == "__error__":
                continue
            payload, ftotal, ftid = val
            if tid is None:
                tid = ftid
            elif ftid != tid:
                continue
            total = total or ftotal
            if seq == 0 and header is None:
                try:
                    header = json.loads(payload.decode())
                except Exception:  # noqa: BLE001
                    pass
            if store.put(seq, payload):
                new_count += 1

    store.meta.update({"header": header, "total": total, "transfer_id": tid})
    store.save()

    have = store.have()
    print(f"\n{new_count} new frame(s) this run; {len(have)} cached in {args.work}")
    return finish(args, store, header, total, have)


def finish(args, store, header, total, have):
    if total is None:
        print("\nNo agpack frames recognised yet. Nothing to assemble.")
        return 1

    want = set(range(0, total + 1))
    missing = want - have
    print(f"have {len(have)} / {total + 1} frames " f"(frame 0 = header, 1-{total} = payload)")

    if missing:
        print(f"\nMISSING {len(missing)} frame(s):")
        for line in describe_missing(missing, (header or {}).get("grid"), total):
            print(line)
        print(
            "\nRe-photograph those sheets and run decode again with the new "
            "photos.\nAlready-decoded frames are kept, so you only need the "
            "missing ones."
        )
        return 2

    if header is None:
        print(
            "\nAll payload frames present but the header frame (0) is missing;"
            "\ncannot verify integrity. Re-shoot the sheet containing frame 0."
        )
        return 2

    packed = b"".join(store.get(i) for i in range(1, total + 1))
    digest = hashlib.sha256(packed).hexdigest()
    print(f"\nreassembled {len(packed):,} bytes")
    print(f"  expected sha256 {header['sha256']}")
    print(f"  actual   sha256 {digest}")
    if digest != header["sha256"]:
        print(
            "\nCHECKSUM MISMATCH. Frames decoded cleanly but the whole does "
            "not match.\nThis usually means two different transfers got mixed "
            "together. Clear the\nwork directory and re-decode from a single "
            "capture session."
        )
        return 1
    print("  MATCH")

    if not args.out:
        print("\n(no --out given, stopping before extraction)")
        return 0

    if os.path.exists(args.out) and os.listdir(args.out) and not args.force:
        print(f"\n{args.out} exists and is not empty. Use --force or pick " f"another --out.\nNothing was written.")
        return 1

    written, manifest = safe_extract(packed, args.out)
    print(f"\nextracted {len(written)} file(s) to {args.out}")
    if manifest:
        ok, bad = verify_against_manifest(args.out, manifest)
        print(f"  per-file sha256: {ok} ok, {len(bad)} bad")
        for path, why in bad[:20]:
            print(f"    ! {path}: {why}")
    print("\nDone. Nothing outside --out and --work was touched.")
    return 0


def cmd_status(args):
    store = Store(args.work)
    header = store.meta.get("header")
    total = store.meta.get("total")
    return finish(args, store, header, total, store.have())


def cmd_calibrate(args):
    load_pil()
    load_zbar()
    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = list(iter_inputs(args.inputs, tmpdir, args.fps))
        if not inputs:
            sys.exit("No images found.")
        print(f"checking {len(inputs)} calibration photo(s)\n")
        print(f"{'photo':<32} {'decoded':>8}   verdict")
        print("-" * 68)
        for path in inputs:
            found = decode_photo(path, escalate=not args.fast)
            n = len([k for k in found if k != "__error__"])
            name = os.path.basename(path)
            if n == 0:
                verdict = "NOTHING read - too small, blurry, or glared"
            else:
                verdict = f"{n} symbol(s) read - compare against the symbol " f"count agpack printed"
            print(f"{name[:31]:<32} {n:>8}   {verdict}")
    print(
        "\nA level passes only if a photo decodes EVERY symbol on its sheet.\n"
        "Use the densest passing level, then step down one for margin.\n"
        "Shoot square-on, fill the frame, tap to focus, kill reflections."
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--work", default="agunpack-work", help="cache directory for decoded frames (resumable)")
    p.add_argument("--fps", type=float, default=4.0, help="frame extraction rate if given a video (default 4)")
    p.add_argument("--fast", action="store_true", help="single-pass decode only; faster, finds fewer codes")
    p.add_argument("--jobs", type=int, default=0, help="parallel decode workers (default: CPU count, max 8)")
    p.add_argument("--force", action="store_true", help="allow writing into a non-empty --out")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decode", help="scan photos/video and assemble")
    d.add_argument("inputs", nargs="+", help="photo files, folders, or a video")
    d.add_argument("--out", default=None, help="destination directory (new/empty)")
    d.set_defaults(func=cmd_decode)

    s = sub.add_parser("status", help="report what is cached and what is missing")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_status)

    c = sub.add_parser("calibrate", help="check photos of agpack calibration sheets")
    c.add_argument("inputs", nargs="+")
    c.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
