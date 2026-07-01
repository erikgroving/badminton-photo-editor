"""
Scans the training data directory for paired Raws/Edited folder sets and
produces mapping.json: a per-event/category list of {raw, edited, label}.

Matching strategy (in priority order):
  1. Filename: 0V2A####.CR3 <-> 0V2A####.jpg  (exact, e.g. 202602 NW CRC)
  2. Timestamp: CR3 CMT2-box EXIF DateTimeOriginal+SubsecTimeOriginal matched
     against edited JPG EXIF  (handles renamed exports like "EventName-N.jpg")

Run modes:
  python -m data.mapping                  → build only if mapping.json doesn't exist
  python -m data.mapping --rebuild        → always rebuild
  python -m data.mapping --verify         → print summary statistics
  python -m data.mapping --sanity-check N → save N random pair comparison images to
                                            sanity_check/ so you can visually verify
                                            the raw→edited pairing is correct
"""
import argparse
import json
import random
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, EXCLUDED_CATEGORIES, MAPPING_FILE, SPLIT_SEED, TEST_RATIO, TRAIN_RATIO, TRAINING_DATA_DIR, VAL_RATIO

_NUM_RE = re.compile(r"0V2A(\d{4,})", re.IGNORECASE)

# Canon CR3 ISOBMFF UUID that contains CMT1/CMT2 EXIF boxes
_CANON_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")


# ── CR3 timestamp reader ───────────────────────────────────────────────────────

def _iter_isobmff_boxes(data: bytes, offset: int = 0):
    while offset + 8 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        box_type = data[offset + 4 : offset + 8]
        if size == 0:
            body_start, next_off = offset + 8, len(data)
        elif size == 1:
            if offset + 16 > len(data):
                break
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            body_start, next_off = offset + 16, offset + size
        else:
            body_start, next_off = offset + 8, offset + size
        yield box_type, data[body_start : min(next_off, len(data))]
        if next_off <= offset or size < 8:
            break
        offset = next_off


def _parse_tiff_tags(data: bytes, target_tags: set) -> dict:
    """Return a dict of {tag: value_str} for the requested EXIF tags."""
    if len(data) < 8:
        return {}
    endian = "<" if data[0:2] == b"II" else ">"
    ifd0_off = struct.unpack_from(endian + "I", data, 4)[0]

    def read_ifd(off: int, visited: set | None = None) -> dict:
        visited = visited or set()
        if off in visited or off + 2 > len(data):
            return {}
        visited.add(off)
        count = struct.unpack_from(endian + "H", data, off)[0]
        result: dict = {}
        for i in range(min(count, 200)):
            e = off + 2 + i * 12
            if e + 12 > len(data):
                break
            tag, typ, cnt = struct.unpack_from(endian + "HHI", data, e)
            vo = e + 8
            if tag in target_tags and typ == 2:  # ASCII
                if cnt <= 4:
                    raw = data[vo : vo + cnt]
                else:
                    str_off = struct.unpack_from(endian + "I", data, vo)[0]
                    raw = data[str_off : str_off + cnt]
                result[tag] = raw.decode("ascii", errors="replace").rstrip("\x00")
            # Follow ExifIFD sub-directory pointer
            if tag == 34665 and typ in (4, 9):
                sub_off = struct.unpack_from(endian + "I", data, vo)[0]
                result.update(read_ifd(sub_off, visited))
        return result

    return read_ifd(ifd0_off)


_TS_TAGS = {36867, 37521}  # DateTimeOriginal, SubsecTimeOriginal


def _read_cr3_ts(path: str, max_bytes: int = 600_000) -> str | None:
    """
    Return 'YYYY:MM:DD HH:MM:SS.subsec' timestamp from a CR3 file's CMT2 EXIF,
    or None if unreadable.  Reads only the first max_bytes of the file (the
    metadata boxes always appear before the large mdat sensor-data block).
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
        for bt, body in _iter_isobmff_boxes(data):
            if bt != b"moov":
                continue
            for bt2, body2 in _iter_isobmff_boxes(body):
                if bt2 != b"uuid" or not body2.startswith(_CANON_UUID):
                    continue
                for bt3, body3 in _iter_isobmff_boxes(body2[16:]):
                    if bt3 != b"CMT2":
                        continue
                    tags = _parse_tiff_tags(body3, _TS_TAGS)
                    dto = tags.get(36867)
                    sub = tags.get(37521, "00")
                    if dto:
                        return f"{dto}.{sub}"
    except Exception:
        pass
    return None


def _read_jpg_ts(path: str) -> str | None:
    """Return 'YYYY:MM:DD HH:MM:SS.subsec' timestamp from a JPG EXIF, or None."""
    try:
        from PIL import Image
        exif = Image.open(path)._getexif() or {}
        dto = exif.get(36867)
        sub = exif.get(37521, "00")
        if dto:
            return f"{dto}.{sub}"
    except Exception:
        pass
    return None


BURST_SECONDS = 1.0  # photos within this many seconds of each other = same burst


def _ts_str_to_float(ts: str) -> float | None:
    """Convert 'YYYY:MM:DD HH:MM:SS.sub' (CR3 EXIF format) to a float timestamp."""
    try:
        from datetime import datetime
        parts = ts.split(".")
        dt = datetime.strptime(parts[0], "%Y:%m:%d %H:%M:%S")
        sub = float("0." + parts[1]) if len(parts) > 1 else 0.0
        return dt.timestamp() + sub
    except Exception:
        return None


def _group_into_bursts_filenumber(raw_files: list, gap: int = 20, size: int = 60) -> list:
    """Legacy burst grouping by file-number proximity + size cap."""
    numbered = sorted(
        [(int(n), f) for f in raw_files if (n := _extract_number(f.name))],
        key=lambda x: x[0]
    )
    gap_groups: list = []
    current: list = []
    last = None
    for num, f in numbered:
        if last is None or num - last <= gap:
            current.append(f)
        else:
            gap_groups.append(current)
            current = [f]
        last = num
    if current:
        gap_groups.append(current)
    groups: list = []
    for g in gap_groups:
        for i in range(0, len(g), size):
            groups.append(g[i : i + size])
    unnumbered = [[f] for f in raw_files if not _extract_number(f.name)]
    return groups + unnumbered


def _group_into_bursts(raw_files: list, raw_timestamps: dict) -> list:
    """
    Group raw files into burst sequences using EXIF timestamps.
    A new burst starts whenever the gap to the previous shot exceeds BURST_SECONDS.
    Files without a readable timestamp fall back to file-number ordering and are
    treated as if each is 1/30 s apart (EOS R3 max frame rate).
    """
    timed, untimed = [], []
    for f in raw_files:
        ts = raw_timestamps.get(str(f))
        if ts is not None:
            timed.append((ts, f))
        else:
            n = _extract_number(f.name)
            untimed.append((int(n) / 30.0 if n else 0.0, f))

    all_sorted = sorted(timed + untimed, key=lambda x: x[0])

    groups: list = []
    current: list = []
    last_ts: float | None = None
    for ts, f in all_sorted:
        if last_ts is None or ts - last_ts <= BURST_SECONDS:
            current.append(f)
        else:
            groups.append(current)
            current = [f]
        last_ts = ts
    if current:
        groups.append(current)

    return groups


def _assign_event_splits(event_names: list[str]) -> dict[str, str]:
    """
    Legacy: assign each event entirely to one partition (no cross-event leakage).
    Kept for reference; build_mapping() uses burst-group splitting instead.
    """
    import math
    rng = random.Random(SPLIT_SEED)
    shuffled = list(event_names)
    rng.shuffle(shuffled)
    n = len(shuffled)

    n_test  = max(1, math.ceil(n * TEST_RATIO))
    n_val   = max(1, math.ceil(n * VAL_RATIO))
    n_train = max(1, n - n_test - n_val)

    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    result = {}
    for i, event in enumerate(shuffled):
        if i < n_train:
            result[event] = "train"
        elif i < n_train + n_val:
            result[event] = "val"
        else:
            result[event] = "test"
    return result


def _extract_number(filename: str) -> str | None:
    m = _NUM_RE.search(filename)
    return m.group(1) if m else None


def _event_prefix(folder_name: str) -> str:
    return folder_name[:6]


_TS_WORKERS = 16  # parallel threads for EXIF timestamp extraction


def _build_ts_edited_map(edited_path: Path) -> dict[str, Path]:
    """
    Build a {timestamp_str: edited_path} map for all JPGs in an edited folder.
    """
    files = [f for f in edited_path.iterdir()
             if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    def _worker(f: Path):
        return f, _read_jpg_ts(str(f))

    ts_map: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=_TS_WORKERS) as ex:
        for f, ts in ex.map(_worker, files):
            if ts:
                ts_map[ts] = f
    return ts_map


def _build_ts_raw_map(raws_path: Path) -> dict[str, Path]:
    """
    Build a {timestamp_str: raw_path} map for all RAW files in a raws folder.
    Uses parallel threads for fast EXIF extraction.
    """
    files = [f for f in raws_path.iterdir()
             if f.suffix.lower() in {".cr3", ".cr2", ".nef", ".arw", ".rw2", ".dng"}]

    def _worker(f: Path):
        return f, _read_cr3_ts(str(f))

    ts_map: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=_TS_WORKERS) as ex:
        for f, ts in ex.map(_worker, files):
            if ts:
                ts_map[ts] = f
    return ts_map


def build_mapping(split_mode: str = "burst1s") -> dict:
    """
    Returns nested dict:
      {
        event_prefix: {
          category_label: [
            {"raw": path|null, "edited": path|null, "label": 0|1|null, "split": str}
          ]
        }
      }
    label=1  → Jay kept this photo (raw has a matching edited)
    label=0  → Jay culled this photo (raw has no matching edited)
    label=None → edited-only entry (judge model training, no culling label)

    Matching order per folder pair:
      1. Filename: 0V2A####.CR3 <-> 0V2A####.jpg
      2. Timestamp: CR3 CMT2 EXIF DateTimeOriginal+SubsecTimeOriginal
         (used when edited files were renamed on export, e.g. "EventName-N.jpg")
    """
    all_folders = [f for f in TRAINING_DATA_DIR.iterdir() if f.is_dir()]
    raws_folders   = {f.name: f for f in all_folders if f.name.endswith("Raws")}
    edited_folders = {f.name: f for f in all_folders if f.name.endswith("Edited")}

    # ── Pass 1: scan folders, build edited maps, collect burst groups ──────────
    category_data: dict = {}  # (event_prefix, category) -> scan results

    for raws_name, raws_path in sorted(raws_folders.items()):
        base = raws_name[: -len("Raws")].rstrip()
        if base.strip() in EXCLUDED_CATEGORIES:
            print(f"  [skip] {base.strip()} (in EXCLUDED_CATEGORIES)")
            continue
        edited_name = base + " Edited"
        edited_path = edited_folders.get(edited_name)
        event_prefix = _event_prefix(raws_name)
        category     = base.strip()

        raw_files: list[Path] = [
            f for f in raws_path.iterdir()
            if f.suffix.lower() in {".cr3", ".cr2", ".nef", ".arw", ".rw2", ".dng"}
        ]

        edited_map: dict[str, Path] = {}
        if edited_path:
            for f in edited_path.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    n = _extract_number(f.name)
                    if n:
                        edited_map[n] = f

        # Always read raw timestamps — used for burst grouping regardless of
        # whether filename matching succeeded.
        ts_raw = _build_ts_raw_map(raws_path)  # {ts_str: raw_path}
        raw_timestamps: dict[str, float] = {}
        for ts_str, raw_f in ts_raw.items():
            ts_f = _ts_str_to_float(ts_str)
            if ts_f is not None:
                raw_timestamps[str(raw_f)] = ts_f

        ts_raw_to_edited: dict[str, Path] = {}
        used_ts_fallback = False
        if edited_path and len(edited_map) == 0 and len(raw_files) > 0:
            print(f"  [{category}] No filename matches — building timestamp maps…")
            ts_edited = _build_ts_edited_map(edited_path)
            print(f"    Edited JPGs with parseable timestamp: {len(ts_edited)}")
            if ts_edited:
                print(f"    Raws with parseable timestamp:       {len(ts_raw)}")
                for ts_str, raw_f in ts_raw.items():
                    edited_f = ts_edited.get(ts_str)
                    if edited_f:
                        ts_raw_to_edited[str(raw_f)] = edited_f
                print(f"    Timestamp-matched pairs:             {len(ts_raw_to_edited)}")
                used_ts_fallback = True

        category_data[(event_prefix, category)] = {
            "raw_files":        raw_files,
            "raw_timestamps":   raw_timestamps,
            "edited_map":       edited_map,
            "ts_raw_to_edited": ts_raw_to_edited,
            "used_ts_fallback": used_ts_fallback,
        }

    # ── Assign splits ─────────────────────────────────────────────────────────
    photo_split: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}

    if split_mode == "event":
        # Whole events → one partition (strongest generalisation, fewest groups)
        all_event_prefixes = sorted({ep for ep, _ in category_data})
        event_map = _assign_event_splits(all_event_prefixes)
        print(f"\nEvent-level split ({len(all_event_prefixes)} events):")
        for ev, sp in sorted(event_map.items()):
            print(f"  {ev} -> {sp}")
        for (event_prefix, _), data in category_data.items():
            sp = event_map.get(event_prefix, "train")
            for f in data["raw_files"]:
                photo_split[str(f)] = sp
                split_counts[sp] += 1

    elif split_mode == "burst60":
        # File-number proximity + 60-photo size cap (legacy intermediate approach)
        all_groups: list = []
        for (ep, cat), data in category_data.items():
            for group in _group_into_bursts_filenumber(data["raw_files"]):
                all_groups.append((ep, cat, group))
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(all_groups)
        n = len(all_groups)
        n_train, n_val = round(n * TRAIN_RATIO), round(n * VAL_RATIO)
        for i, (_, _, files) in enumerate(all_groups):
            sp = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            for f in files:
                photo_split[str(f)] = sp
                split_counts[sp] += 1
        print(f"\nBurst-60 split ({n} groups):")
        total_photos = sum(split_counts.values())
        for sp, cnt in split_counts.items():
            print(f"  {sp:5s}: {cnt:6,} photos ({cnt/total_photos:.1%})")

    elif split_mode == "random":
        # Naive baseline: shuffle individual photos with no burst awareness
        all_files: list[str] = []
        for data in category_data.values():
            all_files.extend(str(f) for f in data["raw_files"])
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(all_files)
        n = len(all_files)
        n_train, n_val = round(n * TRAIN_RATIO), round(n * VAL_RATIO)
        for i, fpath in enumerate(all_files):
            sp = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            photo_split[fpath] = sp
            split_counts[sp] += 1
        total_photos = sum(split_counts.values())
        print(f"\nRandom split ({total_photos:,} photos, no burst grouping):")
        for sp, cnt in split_counts.items():
            print(f"  {sp:5s}: {cnt:6,} photos ({cnt/total_photos:.1%})")

    else:  # burst1s (default)
        # EXIF timestamp proximity: 1-second window defines burst boundary
        all_groups = []
        for (ep, cat), data in category_data.items():
            for group in _group_into_bursts(data["raw_files"], data["raw_timestamps"]):
                all_groups.append((ep, cat, group))
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(all_groups)
        n = len(all_groups)
        n_train, n_val = round(n * TRAIN_RATIO), round(n * VAL_RATIO)
        for i, (_, _, files) in enumerate(all_groups):
            sp = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            for f in files:
                photo_split[str(f)] = sp
                split_counts[sp] += 1
        print(f"\nBurst-1s split ({n} groups, BURST_SECONDS={BURST_SECONDS}):")
        total_photos = sum(split_counts.values())
        for sp, cnt in split_counts.items():
            print(f"  {sp:5s}: {cnt:6,} photos ({cnt/total_photos:.1%})")

    # ── Pass 2: build mapping entries ─────────────────────────────────────────
    mapping: dict = {}

    for (event_prefix, category), data in category_data.items():
        entries = []
        for f in data["raw_files"]:
            n = _extract_number(f.name)
            if data["used_ts_fallback"]:
                edited_file = data["ts_raw_to_edited"].get(str(f))
            else:
                edited_file = data["edited_map"].get(n) if n else None

            entries.append({
                "raw":    str(f),
                "edited": str(edited_file) if edited_file else None,
                "label":  1 if edited_file else 0,
                "split":  photo_split.get(str(f), "train"),
            })

        mapping.setdefault(event_prefix, {})[category] = entries

    # Edited-only folders (no matching Raws) — e.g. Boba Cup — judge model only
    for edited_name, edited_path in edited_folders.items():
        base = edited_name[: -len("Edited")].rstrip()
        if (base + " Raws") in raws_folders:
            continue
        if base.strip() in EXCLUDED_CATEGORIES:
            continue
        event_prefix = _event_prefix(edited_name)
        category     = base.strip()
        entries = []
        for f in sorted(edited_path.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            entries.append({
                "raw":    None,
                "edited": str(f),
                "label":  None,
                "split":  "judge_only",
            })
        mapping.setdefault(event_prefix, {})[category] = entries

    return mapping


def save_mapping(mapping: dict) -> None:
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, "w") as fh:
        json.dump(mapping, fh, indent=2)
    print(f"Saved: {MAPPING_FILE}")


def load_mapping() -> dict:
    with open(MAPPING_FILE) as fh:
        return json.load(fh)


def flat_entries(mapping: dict, split: str | None = None, label=None):
    """Yield individual entry dicts, optionally filtered by split and/or label."""
    for event in mapping.values():
        for entries in event.values():
            for e in entries:
                if split is not None and e["split"] != split:
                    continue
                if label is not None and e["label"] != label:
                    continue
                yield e


# ── Verification ──────────────────────────────────────────────────────────────

def _verify(mapping: dict) -> None:
    print(f"\n{'Event':8s}  {'Category':45s}  {'Total':>6}  {'Kept':>5}  {'Rate':>6}  {'Tr/Va/Te'}")
    print("-" * 95)
    for event_prefix, cats in sorted(mapping.items()):
        for cat, entries in sorted(cats.items()):
            labeled = [e for e in entries if e["label"] is not None]
            kept    = sum(1 for e in labeled if e["label"] == 1)
            total   = len(labeled)
            rate    = f"{kept/total:.1%}" if total else "—"
            tr = sum(1 for e in labeled if e["split"] == "train")
            va = sum(1 for e in labeled if e["split"] == "val")
            te = sum(1 for e in labeled if e["split"] == "test")
            print(f"{event_prefix:8s}  {cat:45s}  {total:6d}  {kept:5d}  {rate:>6}  {tr}/{va}/{te}")

    all_labeled  = [e for e in flat_entries(mapping) if e["label"] is not None]
    total        = len(all_labeled)
    kept         = sum(1 for e in all_labeled if e["label"] == 1)
    judge_only   = sum(1 for e in flat_entries(mapping) if e["split"] == "judge_only")
    train_total  = sum(1 for e in all_labeled if e["split"] == "train")
    val_total    = sum(1 for e in all_labeled if e["split"] == "val")
    test_total   = sum(1 for e in all_labeled if e["split"] == "test")
    print(f"\nTotal labeled raws : {total:,}")
    print(f"Kept  (label=1)    : {kept:,}  ({kept/total:.1%})")
    print(f"Culled(label=0)    : {total-kept:,}  ({(total-kept)/total:.1%})")
    print(f"Judge-only edits   : {judge_only:,}")
    print(f"Train / Val / Test : {train_total:,} / {val_total:,} / {test_total:,}")


# ── Sanity-check image grid ───────────────────────────────────────────────────

def _sanity_check(mapping: dict, n: int = 10) -> None:
    """
    Save a side-by-side comparison of N random raw→edited pairs so you can
    visually confirm the filename-based mapping is correct.
    Each row: raw thumbnail (left) | edited JPEG (right)
    Output: sanity_check/pair_XXXX.jpg  (one file per pair for easy browsing)
    """
    from PIL import Image, ImageDraw, ImageFont

    pairs = [e for e in flat_entries(mapping) if e["label"] == 1 and e["edited"] and e["raw"]]
    if not pairs:
        print("No matched pairs found in mapping.")
        return

    sample = random.sample(pairs, min(n, len(pairs)))
    out_dir = BASE_DIR / "sanity_check"
    out_dir.mkdir(exist_ok=True)

    # Import here so the script works even without rawpy during testing
    from data.raw_reader import extract_thumbnail

    saved = []
    for i, entry in enumerate(sample):
        try:
            raw_img    = extract_thumbnail(entry["raw"], size=(512, 512))
            edited_img = Image.open(entry["edited"]).convert("RGB")
            edited_img.thumbnail((512, 512))
            # Pad edited to exactly 512×512 (letterbox)
            padded = Image.new("RGB", (512, 512), (30, 30, 30))
            x_off  = (512 - edited_img.width)  // 2
            y_off  = (512 - edited_img.height) // 2
            padded.paste(edited_img, (x_off, y_off))
            edited_img = padded

            # Header bar
            header_h = 30
            canvas   = Image.new("RGB", (1024 + 4, 512 + header_h), (20, 20, 20))
            draw     = ImageDraw.Draw(canvas)

            raw_name    = Path(entry["raw"]).name
            edited_name = Path(entry["edited"]).name
            draw.text((6,  4), f"RAW: {raw_name}",    fill=(200, 200, 200))
            draw.text((518, 4), f"EDITED: {edited_name}", fill=(200, 200, 200))

            canvas.paste(raw_img,    (0,   header_h))
            canvas.paste(edited_img, (512 + 4, header_h))

            # Divider line
            draw.line([(512 + 2, header_h), (512 + 2, 512 + header_h)], fill=(100, 100, 100), width=3)

            out_path = out_dir / f"pair_{i+1:02d}_{raw_name.replace('.CR3','').replace('.cr3','')}.jpg"
            canvas.save(out_path, format="JPEG", quality=90)
            saved.append(out_path)
            print(f"  pair {i+1:2d}: {raw_name}  <->  {edited_name}")
        except Exception as exc:
            print(f"  pair {i+1}: FAILED ({exc})")

    if saved:
        print(f"\n{len(saved)} sanity-check images saved to: {out_dir}")
        print("Open them to confirm the raw-to-edited pairing looks correct.")
        print("If they look good, the mapping.json is valid -- it will be reused automatically on future runs.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild",      action="store_true", help="Rebuild even if mapping.json exists")
    parser.add_argument("--verify",       action="store_true", help="Print per-event summary stats")
    parser.add_argument("--sanity-check", type=int, metavar="N", dest="sanity_n",
                        help="Save N sample pair images to sanity_check/")
    parser.add_argument("--output",       type=Path, default=None,
                        help="Save to this path instead of data/mapping.json (implies --rebuild)")
    parser.add_argument("--split-mode",   type=str,  default="burst1s",
                        choices=["burst1s", "burst60", "event", "random"],
                        help="Split strategy: burst1s (default), burst60, or event")
    args = parser.parse_args()

    out_file = args.output or MAPPING_FILE
    if out_file.exists() and not args.rebuild and not args.output:
        print(f"Mapping already exists: {out_file}")
        print("(Use --rebuild to regenerate it)")
        with open(out_file) as fh:
            mapping = json.load(fh)
    else:
        print(f"Building mapping (split_mode={args.split_mode})…")
        mapping = build_mapping(split_mode=args.split_mode)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as fh:
            json.dump(mapping, fh, indent=2)
        print(f"Saved: {out_file}")

    if args.verify:
        _verify(mapping)

    if args.sanity_n:
        _sanity_check(mapping, n=args.sanity_n)
