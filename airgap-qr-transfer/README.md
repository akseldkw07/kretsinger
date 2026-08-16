# Moving code off an air-gapped machine via QR sheets

Screen → phone camera → QR decode → byte-identical repository.

| file | runs on | what it does |
|---|---|---|
| `agmini.py` | the **offline** machine | 150 lines. `collect()` and `pack()` — import it and pass in the paths and filetypes. |
| `agpack.py` | the **offline** machine | Command-line front end, plus `--estimate` and `--calibrate` for working out the best layout for a given screen. |
| `agtype.py` | the **offline** machine | 77 lines, no arguments, nothing importable — the version you type by hand when the machine can't receive files at all. |
| `agunpack.py` | your **Mac** | Reads the photos, verifies, rebuilds the tree. |
| `library_usage.ipynb` | anywhere | Walkthrough of the `agmini` API. |
| `demo.ipynb` | anywhere | The whole pipeline end to end, with simulated photos. |

All three senders emit the same frame format, so `agunpack.py` decodes any of them.

---

## Why QR and not OCR of text

OCR was the original plan, and it's the one thing here that reliably fails. Code is
the worst possible OCR input: leading-whitespace depth is semantically load-bearing in
Python, `l`/`1`/`I` and `O`/`0` are visually identical in most fonts, and quotes get
"smartened". A single silently-wrong character produces a bug you hunt for a week,
with no signal that anything went wrong.

QR carries **bytes**, with Reed–Solomon error correction inside each symbol, a CRC32
per frame, and a SHA-256 over the whole payload. Either it reconstructs your repo
exactly or it tells you precisely which sheet to re-photograph. There is no
quiet-corruption failure mode.

The payload is base32-encoded rather than raw binary. Base32's alphabet is a subset of
QR's "alphanumeric" mode charset, which sidesteps the ECI/charset mangling that some
QR decoders inflict on byte-mode data. It costs about 10% capacity and buys certainty.

---

## The number that decides everything: pixels per QR module

Not camera megapixels. **The resolution of the screen you photograph.**

A QR symbol drawn at 2 pixels per module contains no recoverable detail — the display
never rendered the data, so no camera can recover it. My first attempt packed 16
symbols onto a 1080p screen, which worked out to 1.8 px/module, and exactly 1 of 100
frames decoded.

`agpack.py` therefore derives its entire layout from `--screen`. Give it the real
resolution of the sending display and it picks the grid and symbol size that move the
most bytes per photo while staying above your chosen pixels-per-module floor.

| quality | px/module | use when |
|---|---|---|
| `safe` | 5.5 | older/smaller display, or your calibration shots were marginal |
| `balanced` | 4.5 | default |
| `dense` | 3.5 | good display, steady hands, verified by calibration |

Throughput scales hard with the sending screen. If the offline machine can drive an
external monitor, plug in the biggest one you can find — 1080p → 1440p roughly halves
the number of photos, and 4K roughly halves it again.

---

## Do this first: check what you're actually sending

Measured on your `kretsinger/` folder, as the size reference you gave:

| selection | files | raw | compressed | photos @1080p | photos @1440p |
|---|---|---|---|---|---|
| code only | 244 | 493 KB | 101 KB | 23 | 13 |
| code + notebooks | 287 | 4.4 MB | 1.5 MB | 328 | 182 |

At 2–3× that, code-only lands around **50–70 photos at 1080p, or 30–40 at 1440p**. That
is an afternoon, not a project.

**Notebook outputs are the entire problem.** Your 45 notebooks are 7.3 MB, of which
6.9 MB is stored cell output — rendered plots, base64 PNGs, dumped dataframes:

| | raw | compressed | photos @1080p |
|---|---|---|---|
| notebooks as-is | 7.3 MB | 3.8 MB | 821 |
| notebooks, outputs cleared | 380 KB | 98 KB | 22 |

**38× less to photograph**, and you lose nothing but rendered output you'd regenerate
by re-running anyway. On the offline machine, before packing:

```bash
jupyter nbconvert --clear-output --inplace **/*.ipynb
```

No jupyter there? This does the same with only the standard library:

```python
import json, glob
for f in glob.glob("**/*.ipynb", recursive=True):
    nb = json.load(open(f))
    for c in nb.get("cells", []):
        c["outputs"] = []; c["execution_count"] = None
    json.dump(nb, open(f, "w"))          # writes in place - copy the tree first
```

Also worth excluding outright: `.parquet` (the one in `kretsinger/` is 253 MB — a
million-plus photos, so treat data as non-transferable and regenerate it), plus
`node_modules`, `.venv`, and `__pycache__`, all of which `agpack` skips by default.

---

## Using `agmini` from a notebook

Two functions. `agtype.py`'s behaviour, with the paths and filetypes as arguments.

```python
import agmini

agmini.collect("~/repo", include=[".py", ".md"])     # what would be sent
info = agmini.pack("~/repo", "sheets.html",
                   include=[".py", ".md"], exclude=["*_test.py"])
# {'files': 38, 'compressed_bytes': 108284, 'frames': 182, 'sheets': 23,
#  'sha256': '4efb8d4c...', 'transfer_id': '4efb8d4c'}
```

`include` and `exclude` take extensions (`".py"`) and globs (`"*_test.py"`,
`"docs/generated/*"`) in the same list; `include=None` means every text file. `*`
crosses `/`, so `docs/*` matches nested paths.

Other keywords, all optional: `chunk` (payload bytes per QR, default 600), `cols`,
`rows` (symbols per sheet, default 4×2), `scale` (screen pixels per QR module,
default 4), `max_size` (skip files above 1 MiB), `secrets=True` to stop excluding
`.env` and key files.

The defaults suit a 1080p display. On 1440p try `chunk=1100`; on 4K try `cols=6,
rows=3` as well. `agpack.py --estimate` will work out the best combination for a
given resolution if you want it computed rather than guessed.

`library_usage.ipynb` has a worked example.

## The chicken-and-egg problem

The offline machine can't receive files — that's the whole premise — so **whatever you
run there has to arrive through the keyboard.**

`agtype.py` exists for that: ~65 dense lines, maybe ten minutes of typing, and it does
the real job. Type it, run it, send the repo. `agmini.py` is the better tool but it's
~700 lines with docstrings, which is a different kind of afternoon.

```bash
python3 -c "import segno; print('ok')"        # agtype needs segno
python3 agtype.py . /tmp/test.html .md        # tiny dry run first
```

If `segno` isn't there but `qrencode` is, `agpack.py` has a `qrencode` backend. Check
`python3 -c "import qrcode"` too — `agpack` and `agmini` both support that as well.

---

## Procedure

### 1. Calibrate (10 minutes, saves hours)

On the offline machine:

```bash
python3 agpack.py --calibrate --screen 1920x1080 --out ~/qr-cal
```

Open each `calibrate-*.html`, go fullscreen, press `S` for shoot mode, photograph each
one exactly as you plan to shoot the real thing. Move those photos to your Mac:

```bash
python3 agunpack.py calibrate ~/photos/calib
```

It reports how many symbols each photo yielded. A level passes only if a photo decodes
**every** symbol on its sheet. Use the densest passing level, then step down one for
margin.

### 2. Estimate

```bash
python3 agpack.py --root ~/myrepo --ext .py,.md,.toml,.sh \
                  --screen 1920x1080 --estimate
```

Writes nothing. Shows compressed size and photo count at every quality level. Tune
`--ext` until the count is something you're willing to shoot.

### 3. Pack

```bash
python3 agpack.py --root ~/myrepo --ext .py,.md,.toml,.sh \
                  --screen 1920x1080 --quality balanced --out ~/qr-out
```

### 4. Shoot

Open `~/qr-out/sheets.html`, browser-fullscreen it, press `S`, advance with `→`.

- One photo per sheet, square-on, filling the frame
- Tap to focus before each shot; check the first one is sharp before doing 40 more
- Kill reflections — a ceiling light in the middle of the sheet takes out four symbols
- Sheet number is printed top-left, so you can tell later which photo is which

Order doesn't matter. Duplicates don't matter. Every frame carries its own index.

### 5. Rebuild

```bash
python3 agunpack.py decode ~/photos/transfer --out ~/recovered
```

If frames are missing it tells you which **sheets** to re-shoot:

```
MISSING 121 frame(s):
    frames 40-71     -> re-shoot sheets 6-9
    frames 96-184    -> re-shoot sheets 13-24
```

Re-photograph only those, run `decode` again pointed at the new photos. Decoded frames
are cached in the work directory, so nothing is repeated and nothing is lost. Then:

```
  expected sha256 037ceb07...
  actual   sha256 037ceb07...
  MATCH
extracted 39 file(s) to ~/recovered
  per-file sha256: 38 ok, 0 bad
```

Then `cd ~/recovered && git init`. The archive carries a `.agpack-manifest.json` with
a SHA-256 for every file, which `agunpack` verifies individually after extraction.

---

## Non-destructive guarantees

- `agpack` opens your source tree **read-only**. It never creates, modifies, moves or
  deletes anything under `--root`; it doesn't need write permission there. All output
  goes to `--out`, which must be new or empty unless you pass `--force`.
- `agunpack` writes only to `--out` and `--work`, and refuses a non-empty `--out`
  without `--force`.
- Tar members are validated before extraction: absolute paths, `../` traversal,
  symlinks, hardlinks and device nodes are rejected rather than written.
- **Secrets are skipped by default** — `.env`, `*.pem`, `*.key`, `id_rsa*`,
  `credentials*` and similar. That check runs *before* the `--ext` filter, so a broad
  extension list can't sweep a credential onto a screen you're about to photograph.
  Every skip is printed by name, never silent. Photographs leak easily;
  `--include-secrets` overrides, deliberately.
- `--max-file-size` (default 1 MiB) and a binary-content check keep large blobs out.
- `agpack` also writes `payload.tar.xz` next to the sheets. If a real transfer channel
  ever appears, copy that one file instead of photographing anything.

---

## Verification performed

End-to-end on a 529 KB, 38-file tree (real Python sources), with sheets rendered
headlessly at 1920×1080 and then degraded to imitate a phone photo: perspective tilt,
±1.4° rotation, resampling onto a simulated 12 MP sensor, lens blur, a specular glare
blob, sensor noise, and JPEG quality 72.

| run | sheets | frames | result |
|---|---|---|---|
| `balanced` (4 px/module) | 24 | 185/185 | SHA-256 match, `diff -r` identical |
| `dense` (3 px/module) | 14 | 109/109 | SHA-256 match, `diff -r` identical |
| `agmini.py` | 23 | 179/179 | SHA-256 match, `diff -r` identical |
| partial capture (8 of 24) | 8 | 64/185 | correctly named sheets 6–9, 13–24 as missing; resumed to a clean match |

Executable bits survived. Decode ran ~0.4 s per photo.

Caveat worth stating plainly: a simulation is not a phone. It models optics and
compression, not autofocus hunting, rolling shutter, or your particular screen's
subpixel layout and refresh interaction. That's exactly why step 1 is calibration with
your actual hardware.

---

## Useful flags

**agpack**

```
--root PATH            source tree (read-only)
--ext .py,.md,.json    extensions to include; omit for all text files
--screen 2560x1440     resolution of the display you'll photograph
--quality safe|balanced|dense
--px-per-module 4.2    override the density target directly
--grid 4x2             force a grid instead of auto-choosing
--ecc L|M|Q|H          error correction; M is the default and the right trade
--exclude '*.lock,*.min.js'
--exclude-dir 'fixtures,vendor'
--max-file-size 262144
--include-secrets      opt back in to .env / key files
--estimate             report and write nothing
--calibrate            emit density test sheets
```

**agunpack**

```
decode PATH...   photos, folders, or a video file
status           what's cached, what's missing
calibrate PATH   score calibration photos
--out DIR        destination (must be new or empty)
--work DIR       frame cache; keep it between runs to resume
--jobs 8         parallel decode workers
--fast           single-pass decode, skips the expensive retry
--fps 4          frame extraction rate if you hand it a video
```

Video works too (`decode clip.mov`) if you'd rather film than shoot stills, but expect
motion blur and compression to cost you frames; stills are meaningfully more reliable
per unit of effort.

## Requirements

Offline machine: Python 3.8+, and one of `segno` / `qrcode` / the `qrencode` CLI.

Mac: `pip install pyzbar pillow` plus the zbar library (`brew install zbar`).
Add `pillow-heif` if you shoot HEIC rather than JPEG, and `ffmpeg` only for video.
