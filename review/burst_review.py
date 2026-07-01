"""
Burst-by-burst review UI — second culling pass after AI selection.

Workflow:
  1. AI has already passed top X% of photos into culls/passed/
  2. This server groups those photos into bursts (1-second EXIF gap)
  3. Jay reviews each burst: click to deselect, N to confirm and advance
  4. Unselected photos move from culls/passed/ -> culls/culled/
  5. rankings.json is updated with final decisions

Usage:
    python -m review.burst_review --output /path/to/session_output
    # Open http://127.0.0.1:8767
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

# ── Thumbnail extraction ───────────────────────────────────────────────────────

_yolo_orient_model = None


def _load_orient_detector():
    global _yolo_orient_model
    if _yolo_orient_model is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference.player_coverage import build_detector
        _yolo_orient_model = build_detector()
        print("[orient] YOLO detector loaded OK")
    return _yolo_orient_model


def _upright_score(img, detector) -> float:
    """
    Score how upright the main subject is in this image.
    Uses only the LARGEST detected person (the main athlete) to ignore
    background audience members who can appear upright in the wrong orientation.
    Metric: (bbox_h / img_h) / (bbox_w / img_w) on the largest detection.
    """
    iw, ih = img.size
    if ih == 0 or iw == 0:
        return 0.0
    results = detector(img, classes=[0], verbose=False, imgsz=320)
    largest_area = 0
    largest_score = 0.0
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            if area > largest_area and bw > 0:
                largest_area = area
                largest_score = (bh / ih) / (bw / iw)
    return largest_score


def _auto_orient(img):
    """
    Rotate thumbnail so the athlete appears upright using YOLO person detection.
    Tries 0°, 90° CW, 90° CCW — picks whichever gives the highest
    (person_height / image_height) score.
    """
    try:
        det = _load_orient_detector()
        score0 = _upright_score(img, det)
        best_img, best_score = img, score0

        # 90° CW = PIL rotate(270), 90° CCW = PIL rotate(90)
        for angle in (270, 90):
            candidate = img.rotate(angle, expand=True)
            score = _upright_score(candidate, det)
            print(f"[orient] angle={angle}° score={score:.3f} (current best={best_score:.3f})")
            if score > best_score:
                best_score = score
                best_img = candidate

        return best_img
    except Exception as exc:
        import traceback
        print(f"[orient] YOLO orientation failed: {exc}")
        traceback.print_exc()
        return img


def _extract_thumb(raw_path: Path, size: int = 900) -> Optional[bytes]:
    """Extract thumbnail from CR3 and orient using rawpy flip metadata."""
    try:
        import rawpy
        from PIL import Image
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.raw_reader import _apply_flip
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
            flip  = int(raw.sizes.flip)
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        else:
            img = Image.fromarray(thumb.data).convert("RGB")
        img = _apply_flip(img, flip)
        img.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
    except Exception as exc:
        print(f"[thumb] failed for {raw_path}: {exc}")
        return None


# ── EXIF timestamp reading (copied from data/mapping.py) ──────────────────────

_CANON_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
_TS_TAGS    = {36867, 37521}


def _iter_boxes(data: bytes, offset: int = 0):
    while offset + 8 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        bt   = data[offset + 4: offset + 8]
        if size == 0:
            body, next_off = offset + 8, len(data)
        elif size == 1:
            if offset + 16 > len(data):
                break
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            body, next_off = offset + 16, offset + size
        else:
            body, next_off = offset + 8, offset + size
        yield bt, data[body: min(next_off, len(data))]
        if next_off <= offset or size < 8:
            break
        offset = next_off


def _parse_tiff(data: bytes, tags: set) -> dict:
    if len(data) < 8:
        return {}
    end = "<" if data[:2] == b"II" else ">"
    off = struct.unpack_from(end + "I", data, 4)[0]
    seen: set = set()
    result: dict = {}

    def _read(o):
        if o in seen or o + 2 > len(data):
            return
        seen.add(o)
        n = struct.unpack_from(end + "H", data, o)[0]
        for i in range(min(n, 200)):
            e = o + 2 + i * 12
            if e + 12 > len(data):
                break
            tag, typ, cnt = struct.unpack_from(end + "HHI", data, e)
            vo = e + 8
            if tag in tags and typ == 2:
                raw = data[vo: vo + cnt] if cnt <= 4 else data[struct.unpack_from(end + "I", data, vo)[0]: struct.unpack_from(end + "I", data, vo)[0] + cnt]
                result[tag] = raw.decode("ascii", errors="replace").rstrip("\x00")
            if tag == 34665 and typ in (4, 9):
                _read(struct.unpack_from(end + "I", data, vo)[0])

    _read(off)
    return result


def _cr3_ts(path: Path) -> Optional[float]:
    try:
        with open(path, "rb") as fh:
            data = fh.read(600_000)
        for bt, body in _iter_boxes(data):
            if bt != b"moov":
                continue
            for bt2, b2 in _iter_boxes(body):
                if bt2 != b"uuid" or not b2.startswith(_CANON_UUID):
                    continue
                for bt3, b3 in _iter_boxes(b2[16:]):
                    if bt3 != b"CMT2":
                        continue
                    tags = _parse_tiff(b3, _TS_TAGS)
                    dto  = tags.get(36867)
                    sub  = tags.get(37521, "00")
                    if dto:
                        from datetime import datetime
                        parts = dto.split(".")
                        dt = datetime.strptime(parts[0], "%Y:%m:%d %H:%M:%S")
                        return dt.timestamp() + float("0." + (sub if len(parts) < 2 else parts[1]))
    except Exception:
        pass
    return None


_NUM_RE = re.compile(r"(\d{4,})", re.IGNORECASE)


def _filenum(path: Path) -> float:
    m = _NUM_RE.search(path.stem)
    return float(m.group(1)) / 30.0 if m else 0.0


# ── Burst grouping ────────────────────────────────────────────────────────────

BURST_GAP = 1.0   # seconds


def group_into_bursts(paths: list[Path]) -> list[list[Path]]:
    """Group paths into burst sequences using EXIF timestamps, fallback file numbers."""
    print(f"Reading timestamps for {len(paths)} passed photos...")
    ts_map: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(_cr3_ts, p): p for p in paths}
        for fut, p in futures.items():
            ts = fut.result()
            ts_map[str(p)] = ts if ts is not None else _filenum(p)

    ordered = sorted(paths, key=lambda p: ts_map[str(p)])

    bursts: list[list[Path]] = []
    current: list[Path] = []
    last_ts: Optional[float] = None
    for p in ordered:
        ts = ts_map[str(p)]
        if last_ts is None or ts - last_ts <= BURST_GAP:
            current.append(p)
        else:
            bursts.append(current)
            current = [p]
        last_ts = ts
    if current:
        bursts.append(current)

    print(f"Grouped into {len(bursts)} bursts")
    return bursts


# ── Server state ──────────────────────────────────────────────────────────────

class ReviewState:
    def __init__(self, output_dir: Path):
        self.output_dir   = output_dir
        self.passed_dir   = output_dir / "culls" / "passed"
        self.culled_dir   = output_dir / "culls" / "culled"
        self.rankings_path = output_dir / "culls" / "rankings.json"
        self.state_path   = output_dir / "burst_review_state.json"
        self.bursts: list[list[str]] = []   # list of bursts; each is list of filenames
        self.current_idx: int = 0           # which burst we're on
        self._lock = threading.Lock()
        self._thumb_ready: set[str] = set()

        self._load()

    def _load(self):
        # Find passed CR3 files — deduplicate by lowercase name (Windows glob is case-insensitive
        # so *.CR3 and *.cr3 both match the same files, producing duplicates)
        _seen: set[str] = set()
        paths: list[Path] = []
        for p in sorted(self.passed_dir.glob("*.CR3")) + sorted(self.passed_dir.glob("*.cr3")):
            if p.name.lower() not in _seen:
                _seen.add(p.name.lower())
                paths.append(p)
        if not paths:
            print("No passed photos found.")
            return

        # Restore saved state if it exists
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            self.bursts      = saved.get("bursts", [])
            self.current_idx = saved.get("current_idx", 0)
            # Filter to only still-existing files
            self.bursts = [
                [f for f in burst if (self.passed_dir / f).exists()]
                for burst in self.bursts
            ]
            self.bursts = [b for b in self.bursts if b]
            print(f"Restored state: burst {self.current_idx + 1}/{len(self.bursts)}")
            # Queue thumbnails for current + next few bursts
            threading.Thread(target=self._warm_thumbnails, daemon=True).start()
            return

        # First run: group into bursts
        bursts = group_into_bursts(paths)
        self.bursts = [[p.name for p in burst] for burst in bursts]
        self.current_idx = 0
        self._save_state()
        threading.Thread(target=self._warm_thumbnails, daemon=True).start()

    def _save_state(self):
        self.state_path.write_text(json.dumps({
            "bursts":      self.bursts,
            "current_idx": self.current_idx,
        }, indent=2))

    def _warm_thumbnails(self):
        """Pre-generate thumbnails starting from current burst."""
        idx = self.current_idx
        while idx < len(self.bursts):
            for fname in self.bursts[idx]:
                self.serve_thumbnail(fname)
            idx += 1
            if idx > self.current_idx + 5:
                break  # only warm ahead 5 bursts, rest on demand

    def _resolve_raw(self, fname: str) -> Optional[Path]:
        for d in (self.passed_dir, self.culled_dir):
            p = d / fname
            if p.exists():
                try:
                    return p.resolve()
                except Exception:
                    return p
        return None

    def serve_thumbnail(self, fname: str) -> Optional[bytes]:
        stem = Path(fname).stem
        thumb_dir = self.output_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        cache_path = thumb_dir / f"{stem}_burst.jpg"
        if cache_path.exists():
            return cache_path.read_bytes()
        raw_path = self._resolve_raw(fname)
        if raw_path is None:
            return None
        try:
            import sys as _sys, pathlib as _pathlib
            _sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))
            from data.raw_reader import extract_thumbnail
            img = extract_thumbnail(raw_path, size=(720, 720), oriented=True)
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            data = buf.getvalue()
            cache_path.write_bytes(data)
            return data
        except Exception as exc:
            print(f"[thumb] failed for {fname}: {exc}", flush=True)
            return None

    def current_burst(self) -> list[str]:
        if 0 <= self.current_idx < len(self.bursts):
            return self.bursts[self.current_idx]
        return []

    def scores(self) -> dict[str, float]:
        """Return filename -> score map from rankings.json."""
        try:
            data = json.loads(self.rankings_path.read_text())
            return {p["filename"]: p["score"] for p in data.get("photos", [])}
        except Exception:
            return {}

    def confirm_burst(self, keep: list[str]) -> dict:
        """
        Confirm current burst: move non-kept photos to culled, advance to next burst.
        Returns {"advanced_to": idx, "moved": n}
        """
        with self._lock:
            burst = self.current_burst()
            keep_set = set(keep)
            moved = 0
            for fname in burst:
                if fname not in keep_set:
                    src = self.passed_dir / fname
                    dst = self.culled_dir / fname
                    if src.exists():
                        shutil.move(str(src), str(dst))
                        moved += 1

            # Update rankings.json decisions
            self._update_rankings(keep_set, set(burst) - keep_set)

            self.current_idx += 1
            if self.current_idx < len(self.bursts):
                self._save_state()
                # Warm thumbnails ahead
                threading.Thread(target=self._warm_thumbnails, daemon=True).start()
            else:
                self._save_state()

            return {"advanced_to": self.current_idx, "moved": moved, "done": self.current_idx >= len(self.bursts)}

    def go_prev(self) -> dict:
        with self._lock:
            if self.current_idx > 0:
                self.current_idx -= 1
                self._save_state()
            return {"current_idx": self.current_idx}

    def _update_rankings(self, newly_passed: set[str], newly_culled: set[str]):
        try:
            data = json.loads(self.rankings_path.read_text())
            for p in data["photos"]:
                if p["filename"] in newly_culled:
                    p["decision"] = "culled"
                    p["burst_culled"] = True
                elif p["filename"] in newly_passed:
                    p["decision"] = "passed"
            self.rankings_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            print(f"Warning: could not update rankings.json: {exc}")

    def culled_list(self, sort: str = "score") -> list:
        _seen: set[str] = set()
        paths: list[Path] = []
        for p in sorted(self.culled_dir.glob("*.CR3")) + sorted(self.culled_dir.glob("*.cr3")):
            if p.name.lower() not in _seen:
                _seen.add(p.name.lower())
                paths.append(p)
        try:
            data = json.loads(self.rankings_path.read_text())
            by_name = {p["filename"]: p for p in data.get("photos", [])}
        except Exception:
            by_name = {}
        photos = []
        for p in paths:
            entry = by_name.get(p.name, {})
            photos.append({"filename": p.name, "score": entry.get("score"), "rank": entry.get("rank")})
        if sort == "score":
            photos.sort(key=lambda x: x["score"] or 0, reverse=True)
        elif sort == "score_asc":
            photos.sort(key=lambda x: x["score"] or 0)
        else:
            photos.sort(key=lambda x: x["filename"])
        return photos

    def restore_photo(self, fname: str) -> dict:
        with self._lock:
            src = self.culled_dir / fname
            dst = self.passed_dir / fname
            if not src.exists():
                return {"ok": False, "error": "not found"}
            shutil.move(str(src), str(dst))
            self._update_rankings({fname}, set())
            return {"ok": True}

    def serve_culled_thumbnail(self, fname: str) -> Optional[bytes]:
        p = self.culled_dir / fname
        if not p.exists():
            return None
        try:
            from data.raw_reader import extract_thumbnail
            img = extract_thumbnail(p, size=(600, 600), oriented=True)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception as exc:
            print(f"[culled_thumb] failed for {fname}: {exc}")
            return None

    @property
    def total_bursts(self):
        return len(self.bursts)

    @property
    def done(self):
        return self.current_idx >= len(self.bursts)


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Burst Review — Jay Ma Photography</title>
<link rel="icon" type="image/png" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAEzUlEQVR4nL1WbUhbVxi+556bxN4Vg9UF1Np1ndrV2DV+1YJ0MD9mcZuD/JjDlJVo8V8HM4N2o5vUf9M6WP2ko9YJHbof/mh/qGCDZa5TUUexNhOdOu3UYDTGJCY3995zxs3Ra4xGN9A+gXDv+Xif977v877nAIwxdZigD9U69QoIGIqiMMYIIQAAGcIY0zQd+IoQoultrpARec0eADzPM4xEEwRBEAAAGONdZ4PW7G4aAAihND02NlZZWRkerhYEASoYp8NRVlaWk5MjLzWZTBQALMuKoqhgGKfLRVHU9zU1+7ov0bS0/PTg5wc8z9fV1q2vr7MsW1X13fDwiMFgWF9fV6mUZrNZo9FUV1dDBkpu+R0v/9JkXVy8kHmB47jA6GE/IAMFQUjRpRiNRnD0aPiVK5/p9frs7PfIotHR5+3t7QghjLGP98XGxpR/Ub7TtXvN92y2ZQghRW2FiPaDoijLC8uzZ6P9/U+p1NR0QisIAkJIFEW8G9B24NDwejmM8fj4+KVLBZJeeN7ncDhkVdA0jRASAiCKIsmYHAHyLIqiNLcJjuNEUbzx1dfHIqN8Ps6+usoL/IZMGYahaVoWg/yZMhBCZFzWpSiK/uBsgbj3aVGRNilJoVDKPoWUYKB12s/38p+XC/ML7GusNkkLIQwqDgCkZ53unE53LnD7PgTIb2V09PnNb24uzM+r1WqPx4sxvnq11Gg07lqAkpQViq2hs2ffcblcJI1B6SIJ7+npOXnyzdq6ervdjjHmeaH3yZPzmZnXPr8mr9m5q39gIDfvfSnsoQiIWhYWFhISE/v6+uRBWXJp6enNzc3keQ+CkM1OFEUAQF19XX5+flZWls/nk/XD8zyEsLGhoampief5oCgFIeQcEcnw0LBer0cISV1lU0IKhQJjnJGRcYRlJyYmAABEZrtinySvOZ1hYWEke5iSfoRm45+i19bW9rawOwGJLAB0akoK7a++nXHgeT4iQh0dHS3z/ScCkkkIIZFabW3t9PS02Wz+e3bW7XZjhJVKhUajOXXqVHx8fEdHR+BGcpCEJJBNQwg9Hg93d3dXV/cLi4WBMDom5vWoKKVSSdOQ57mRkT8WF61O51pkVKTBYPi4sFCO285kMEGlPzU11djY2NvbGxkZlZ2dXVpawjCKoaGh2bnZ5eUVhJBKpYqLi7v47sXjscctf1rs9hUAQFVV1cNHDysqKvJy8ziOU6lUWwRo03Gr1XrrVuVvT/vycvNaWlq0Wi1Zcf9+y+PHPekZGYkJCdLHeb0zMzNtbW1vnDhRX19PUdTS0tIPd+5cvmwoN5luXL9uKDZsO+N0KakY44aGhjNJSeUmk9VqleuFtFIcAj6fz+v1IoRu3645/faZ1dXVycnJxNOJra2tGOPf+/tJoTFut7ug4AOO479pb09OTiYnbaBssP8Tg4RA0zRRgcfjuXv3x6KiT9R+pKefLykpjTgWEf9W/EaI/pqcNBQXV1R8S0xDCINOeeA/u3fqj8QBQpiWltrX9+uHhR+5XK7BgaEjLOtyuhBCDodD2t7V1ZWfny93fOr/AG82j8HBwZWVFbVavWSzLdtsRqNxbm6us7OzrKxMulXs7LoHCOZArCN/lyV3JHKVkqsq5LXpoEAfqvVXQfAvtL6ESpuk6AEAAAAASUVORK5CYII=">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: -apple-system, sans-serif;
         height: 100vh; overflow: hidden; display: flex; flex-direction: column; }

  #header {
    display: flex; align-items: center; gap: 16px;
    padding: 10px 20px; background: #1a1a1a; border-bottom: 1px solid #333;
    flex-shrink: 0; height: 56px;
  }
  #burst-info { font-size: 13px; color: #aaa; }
  #burst-num  { font-size: 17px; font-weight: 600; color: #fff; }
  #progress-bar-wrap { flex: 1; height: 6px; background: #333; border-radius: 3px; }
  #progress-bar { height: 100%; background: #4a90d9; border-radius: 3px; transition: width 0.3s; }
  #btn-prev { padding: 7px 16px; background: #333; color: #ccc; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; }
  #btn-prev:hover { background: #444; }
  #btn-next { padding: 8px 24px; background: #4a90d9; color: #fff; border: none;
              border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; }
  #btn-next:hover { background: #3a7fc9; }
  #btn-next.done { background: #5cb85c; }

  #subheader {
    display: flex; align-items: center; gap: 12px;
    padding: 5px 20px; background: #161616; border-bottom: 1px solid #2a2a2a;
    flex-shrink: 0; font-size: 12px; color: #777; height: 36px;
  }
  #sel-count { color: #ddd; font-weight: 600; }
  .sbtn { padding: 3px 10px; background: #2a2a2a; border: 1px solid #444; color: #ccc;
          border-radius: 4px; cursor: pointer; font-size: 11px; }
  .sbtn:hover { background: #3a3a3a; }
  .kbd { background: #2a2a2a; border: 1px solid #444; border-radius: 3px;
         padding: 1px 5px; font-family: monospace; font-size: 11px; color: #aaa; }

  #grid {
    flex: 1; min-height: 0;
    display: grid;
    gap: 8px; padding: 8px;
    overflow: hidden;
  }

  .thumb-wrap {
    position: relative; cursor: pointer; border-radius: 4px;
    overflow: hidden; min-height: 0; min-width: 0;
    transition: opacity 0.12s, box-shadow 0.12s;
  }
  .thumb-wrap img { display: block; width: 100%; height: 100%; object-fit: contain;
                    background: #1a1a1a; }
  .thumb-wrap.selected  { box-shadow: 0 0 0 3px #4a90d9; }
  .thumb-wrap.deselected { opacity: 0.38; box-shadow: none; }
  .thumb-wrap.deselected::after {
    content: ""; position: absolute; inset: 0; background: rgba(0,0,0,0.55);
  }

  .score-badge {
    position: absolute; bottom: 5px; right: 5px;
    background: rgba(0,0,0,0.72); color: #fff;
    font-size: 11px; padding: 2px 6px; border-radius: 3px; pointer-events: none;
  }
  .rank-badge {
    position: absolute; top: 5px; left: 5px;
    background: rgba(74,144,217,0.88); color: #fff;
    font-size: 11px; padding: 2px 6px; border-radius: 3px; pointer-events: none;
  }

  #done-screen {
    display: none; flex: 1; align-items: center; justify-content: center;
    flex-direction: column; gap: 18px; text-align: center; padding: 40px;
  }
  #done-screen h1 { font-size: 32px; color: #5cb85c; }
  #done-screen p  { color: #888; font-size: 15px; max-width: 600px; line-height: 1.6; }
  #done-screen .file-tree { background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
    padding: 16px 24px; text-align: left; font-family: monospace; font-size: 13px; color: #aaa;
    max-width: 560px; line-height: 1.8; }
  #done-screen .file-tree .hl { color: #4a90d9; }
  #btn-close { padding: 12px 32px; background: #5cb85c; color: #fff; border: none;
               border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 700; margin-top: 8px; }
  #btn-close:hover { background: #4aa84c; }
</style>
</head>
<body>

<div id="header">
  <div>
    <div id="burst-num">Burst 1</div>
    <div id="burst-info">Loading...</div>
  </div>
  <div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
  <button id="btn-prev" onclick="goPrev()">&#8592; Prev</button>
  <button id="btn-next" onclick="confirmAndNext()">Confirm &amp; Next &nbsp; N</button>
</div>

<div id="subheader">
  <span id="sel-count">0 selected</span>
  <button class="sbtn" onclick="selectAll()">Select All &nbsp; A</button>
  <button class="sbtn" onclick="selectNone()">None &nbsp; X</button>
  <a href="/culled" target="_blank" style="margin-left:auto; color:#666; font-size:11px; text-decoration:none; white-space:nowrap;">Culled Photos &#8599;</a>
  <span>
    <span class="kbd">click</span> toggle &nbsp;
    <span class="kbd">1–9</span> toggle nth &nbsp;
    <span class="kbd">N</span> next &nbsp;
    <span class="kbd">P</span> prev &nbsp;
    <span class="kbd">A</span> all &nbsp;
    <span class="kbd">X</span> none
  </span>
</div>

<div id="grid"></div>
<div id="done-screen">
  <h1>Review Complete</h1>
  <p id="done-summary"></p>
  <div class="file-tree" id="done-tree"></div>
  <p id="done-next" style="color:#ccc"></p>
  <button id="btn-close" onclick="window.close()">Close Window</button>
</div>

<script>
let state = { burst_idx: 0, total: 0, photos: [], selected: new Set() };
let _lastColCount = -1;

async function loadState() {
  const r = await fetch('/api/state');
  const d = await r.json();
  if (d.done) { showDone(d); return; }
  state.burst_idx = d.burst_idx;
  state.total     = d.total_bursts;
  state.photos    = d.photos;
  state.selected  = new Set(d.photos.map(p => p.filename));  // all selected by default
  renderGrid();
  updateHeader();
}

function calcBestCols(n) {
  if (n <= 1) return 1;
  const gap = 8;
  const gw  = window.innerWidth  - gap * 2;
  const gh  = window.innerHeight - 56 - 36 - gap * 2;
  const photoAR = 3 / 4;
  let bestCols = 1, bestScore = -1;
  for (let c = 1; c <= n; c++) {
    const r  = Math.ceil(n / c);
    const cw = (gw - gap * (c - 1)) / c;
    const ch = (gh - gap * (r - 1)) / r;
    const pW = Math.min(cw, ch * photoAR);
    const pH = pW / photoAR;
    const squareness = Math.min(c / r, r / c);
    const score = squareness * pW * pH;
    if (score > bestScore) { bestScore = score; bestCols = c; }
  }
  return bestCols;
}

function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  const n    = state.photos.length;
  const cols = calcBestCols(n);
  _lastColCount = cols;
  const rows = Math.ceil(n / cols);
  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  grid.style.gridTemplateRows    = `repeat(${rows}, 1fr)`;

  state.photos.forEach(photo => {
    const isSelected = state.selected.has(photo.filename);
    const wrap = document.createElement('div');
    wrap.className     = 'thumb-wrap ' + (isSelected ? 'selected' : 'deselected');
    wrap.dataset.fname = photo.filename;

    const img = document.createElement('img');
    img.src = '/thumb/' + encodeURIComponent(photo.filename);
    img.alt = photo.filename;
    wrap.appendChild(img);

    if (photo.score != null) {
      const badge = document.createElement('div');
      badge.className   = 'score-badge';
      badge.textContent = Math.round(photo.score * 100) + '%';
      wrap.appendChild(badge);
    }
    if (photo.rank != null) {
      const badge = document.createElement('div');
      badge.className   = 'rank-badge';
      badge.textContent = '#' + photo.rank;
      wrap.appendChild(badge);
    }

    wrap.addEventListener('click', () => togglePhoto(photo.filename, wrap));
    grid.appendChild(wrap);
  });

  updateSelCount();
}

function togglePhoto(fname, wrap) {
  if (state.selected.has(fname)) {
    state.selected.delete(fname);
    wrap.className = 'thumb-wrap deselected';
  } else {
    state.selected.add(fname);
    wrap.className = 'thumb-wrap selected';
  }
  updateSelCount();
}

function selectAll() {
  state.photos.forEach(p => state.selected.add(p.filename));
  document.querySelectorAll('.thumb-wrap').forEach(w => w.className = 'thumb-wrap selected');
  updateSelCount();
}

function selectNone() {
  state.selected.clear();
  document.querySelectorAll('.thumb-wrap').forEach(w => w.className = 'thumb-wrap deselected');
  updateSelCount();
}

function updateSelCount() {
  const n = state.selected.size;
  const t = state.photos.length;
  document.getElementById('sel-count').textContent =
    n + ' of ' + t + ' selected' + (n < t ? '  (' + (t - n) + ' will be culled)' : '');
}

function updateHeader() {
  const pct = state.total > 0 ? (state.burst_idx / state.total * 100) : 0;
  document.getElementById('burst-num').textContent  = 'Burst ' + (state.burst_idx + 1);
  document.getElementById('burst-info').textContent =
    'of ' + state.total + ' — ' + state.photos.length + ' photo' + (state.photos.length === 1 ? '' : 's');
  document.getElementById('progress-bar').style.width = pct + '%';
}

async function confirmAndNext() {
  const keep = Array.from(state.selected);
  const r = await fetch('/api/confirm', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ keep }),
  });
  const d = await r.json();
  if (d.done) { showDone(d); return; }
  await loadState();
  window.scrollTo(0, 0);
}

async function goPrev() {
  await fetch('/api/prev', { method: 'POST' });
  // If on done screen, restore normal view before loading
  document.getElementById('done-screen').style.display = 'none';
  document.getElementById('grid').style.display       = '';
  document.getElementById('subheader').style.display  = '';
  document.getElementById('btn-next').style.display   = '';
  document.getElementById('btn-next').className       = '';
  document.getElementById('btn-next').textContent     = 'Confirm & Next   N';
  document.getElementById('btn-next').onclick         = confirmAndNext;
  await loadState();
  window.scrollTo(0, 0);
}

function showDone(d) {
  document.getElementById('grid').style.display      = 'none';
  document.getElementById('subheader').style.display = 'none';
  document.getElementById('done-screen').style.display = 'flex';
  document.getElementById('btn-next').style.display  = 'none';
  document.getElementById('progress-bar').style.width = '100%';

  const passed = d.n_passed || 0;
  const culled = d.n_culled || 0;
  const outDir = d.output_dir || '';

  document.getElementById('done-summary').textContent =
    'All ' + (d.total_bursts || state.total) + ' bursts reviewed. ' +
    passed + ' photos kept, ' + culled + ' culled.';

  document.getElementById('done-tree').innerHTML =
    '<span style="color:#666">' + outDir + '/</span>\n' +
    '├── <span class="hl">culls/passed/</span>   ← <b>' + passed + '</b> kept RAW files\n' +
    '├── culls/culled/    ← ' + culled + ' removed\n' +
    '├── crops/           ← cropped JPEGs (run next step)\n' +
    '└── colors_and_lightning/ ← color-corrected (run next step)';

  document.getElementById('done-next').innerHTML =
    'Back in the Jay Pipeline app, click <b>Run Crop &amp; Color</b> to generate the final JPEGs.';
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'n' || e.key === 'N') { e.preventDefault(); confirmAndNext(); }
  if (e.key === 'p' || e.key === 'P') { e.preventDefault(); goPrev(); }
  if (e.key === 'a' || e.key === 'A') { e.preventDefault(); selectAll(); }
  if (e.key === 'x' || e.key === 'X') { e.preventDefault(); selectNone(); }
  const num = parseInt(e.key, 10);
  if (!isNaN(num) && num >= 1 && num <= 9) {
    const wraps = document.querySelectorAll('.thumb-wrap');
    const wrap = wraps[num - 1];
    if (wrap) {
      const fname = wrap.dataset.fname;
      if (fname) togglePhoto(fname, wrap);
    }
  }
});

window.addEventListener('resize', () => {
  if (!state || !state.photos || !state.photos.length) return;
  const newCols = calcBestCols(state.photos.length);
  if (newCols !== _lastColCount) renderGrid();
});
loadState();
</script>
</body>
</html>
"""


# ── Culled photos HTML page ───────────────────────────────────────────────────

_CULLED_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Culled Photos — Jay Ma Photography</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: -apple-system, sans-serif; }
  #header {
    position: sticky; top: 0; z-index: 10;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 12px 20px; background: #1a1a1a; border-bottom: 1px solid #333;
  }
  #header h1 { font-size: 17px; font-weight: 700; color: #fff; white-space: nowrap; }
  #stats { font-size: 13px; color: #aaa; }
  .sort-group { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; }
  .sort-btn {
    padding: 5px 12px; background: #2a2a2a; border: 1px solid #444;
    color: #ccc; border-radius: 4px; cursor: pointer; font-size: 12px;
  }
  .sort-btn:hover { background: #3a3a3a; }
  .sort-btn.active { background: #4a90d9; border-color: #4a90d9; color: #fff; }
  #grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px; padding: 16px;
  }
  .card {
    background: #1a1a1a; border-radius: 6px; overflow: hidden;
    border: 1px solid #2a2a2a; display: flex; flex-direction: column;
  }
  .thumb-wrap { position: relative; aspect-ratio: 2/3; background: #222; overflow: hidden; }
  .thumb-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; background: #222; }
  .score-badge {
    position: absolute; bottom: 5px; right: 5px;
    background: rgba(0,0,0,0.72); color: #fff;
    font-size: 11px; padding: 2px 6px; border-radius: 3px;
  }
  .card-footer {
    padding: 8px 10px; display: flex; align-items: center; gap: 6px;
    border-top: 1px solid #2a2a2a;
  }
  .fname { font-size: 11px; color: #888; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .restore-btn {
    padding: 4px 10px; background: #5cb85c; color: #fff; border: none;
    border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 600; white-space: nowrap;
  }
  .restore-btn:hover { background: #4aa84c; }
  .restore-btn:disabled { background: #333; color: #666; cursor: default; }
  #empty { padding: 60px 20px; text-align: center; color: #555; font-size: 15px; }
</style>
</head>
<body>
<div id="header">
  <h1>Culled Photos</h1>
  <span id="stats"></span>
  <div class="sort-group">
    <span style="font-size:12px;color:#777;align-self:center">Sort:</span>
    <button class="sort-btn active" data-sort="score" onclick="setSort('score',this)">Score ↓</button>
    <button class="sort-btn" data-sort="score_asc" onclick="setSort('score_asc',this)">Score ↑</button>
    <button class="sort-btn" data-sort="name" onclick="setSort('name',this)">Name</button>
  </div>
</div>
<div id="grid"></div>
<div id="empty" style="display:none">No culled photos found.</div>
<script>
let currentSort = 'score';
let photos = [];

async function loadPhotos() {
  const r = await fetch('/api/culled_list?sort=' + currentSort);
  photos = await r.json();
  renderGrid();
}

function renderGrid() {
  const grid = document.getElementById('grid');
  const empty = document.getElementById('empty');
  grid.innerHTML = '';
  if (photos.length === 0) { empty.style.display = ''; return; }
  empty.style.display = 'none';
  document.getElementById('stats').textContent = photos.length + ' culled photo' + (photos.length === 1 ? '' : 's');
  photos.forEach(p => {
    const card = document.createElement('div');
    card.className = 'card';
    card.id = 'card-' + p.filename;

    const wrap = document.createElement('div');
    wrap.className = 'thumb-wrap';

    const img = document.createElement('img');
    img.src = '/culled_thumb/' + encodeURIComponent(p.filename);
    img.alt = p.filename;
    img.loading = 'lazy';
    wrap.appendChild(img);

    if (p.score != null) {
      const badge = document.createElement('div');
      badge.className = 'score-badge';
      badge.textContent = Math.round(p.score * 100) + '%';
      wrap.appendChild(badge);
    }
    card.appendChild(wrap);

    const footer = document.createElement('div');
    footer.className = 'card-footer';

    const fname = document.createElement('div');
    fname.className = 'fname';
    fname.title = p.filename;
    fname.textContent = p.filename;
    footer.appendChild(fname);

    const btn = document.createElement('button');
    btn.className = 'restore-btn';
    btn.textContent = 'Add back';
    btn.onclick = () => restorePhoto(p.filename, btn);
    footer.appendChild(btn);

    card.appendChild(footer);
    grid.appendChild(card);
  });
}

async function restorePhoto(fname, btn) {
  btn.disabled = true;
  btn.textContent = '…';
  const r = await fetch('/api/restore', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: fname}),
  });
  const d = await r.json();
  if (d.ok) {
    const card = document.getElementById('card-' + fname);
    if (card) { card.style.opacity = '0.3'; card.style.pointerEvents = 'none'; }
    btn.textContent = 'Restored';
    photos = photos.filter(p => p.filename !== fname);
    document.getElementById('stats').textContent = photos.length + ' culled photo' + (photos.length === 1 ? '' : 's');
    if (photos.length === 0) {
      document.getElementById('grid').innerHTML = '';
      document.getElementById('empty').style.display = '';
    }
  } else {
    btn.disabled = false;
    btn.textContent = 'Add back';
    alert('Failed: ' + (d.error || 'unknown error'));
  }
}

function setSort(sort, el) {
  currentSort = sort;
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  photos.sort((a, b) => {
    if (sort === 'score') return (b.score || 0) - (a.score || 0);
    if (sort === 'score_asc') return (a.score || 0) - (b.score || 0);
    return a.filename.localeCompare(b.filename);
  });
  renderGrid();
}

loadPhotos();
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

_state: Optional[ReviewState] = None


class BurstReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # suppress access log spam

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if _state is None:
            body = (
                b'<!DOCTYPE html><html><head><meta charset="UTF-8">'
                b'<meta http-equiv="refresh" content="2">'
                b'<style>body{background:#111;color:#eee;font-family:-apple-system,sans-serif;'
                b'display:flex;align-items:center;justify-content:center;height:100vh;margin:0}'
                b'p{font-size:18px;color:#aaa}</style></head>'
                b'<body><p>Loading photos...</p></body></html>'
            )
            self._send(200, body, "text/html; charset=utf-8")
            return

        if path == "/" or path == "/index.html":
            self._send(200, _HTML.encode(), "text/html; charset=utf-8")
            return

        if path == "/culled":
            self._send(200, _CULLED_HTML.encode(), "text/html; charset=utf-8")
            return

        if path == "/api/state":
            self._api_state()
            return

        if path == "/api/culled_list":
            qs   = parse_qs(urlparse(self.path).query)
            sort = qs.get("sort", ["score"])[0]
            data = _state.culled_list(sort)
            self._send(200, json.dumps(data).encode())
            return

        if path.startswith("/thumb/"):
            fname = Path(path[7:]).name
            self._serve_thumb(fname)
            return

        if path.startswith("/culled_thumb/"):
            fname = Path(path[14:]).name
            data  = _state.serve_culled_thumbnail(fname)
            if data:
                self._send(200, data, "image/jpeg")
            else:
                self._send(200, _grey_placeholder(), "image/jpeg")
            return

        self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if _state is None:
            self._send(503, b'{"error":"loading"}')
            return

        path = urlparse(self.path).path

        if path == "/api/confirm":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            result = _state.confirm_burst(body.get("keep", []))
            # After advancing, return new state
            if not result["done"]:
                result.update(self._build_state_dict())
            self._send(200, json.dumps(result).encode())
            return

        if path == "/api/prev":
            _state.go_prev()
            self._send(200, json.dumps(self._build_state_dict()).encode())
            return

        if path == "/api/restore":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            result = _state.restore_photo(body.get("filename", ""))
            self._send(200, json.dumps(result).encode())
            return

        self._send(404, b'{"error":"not found"}')

    def _api_state(self):
        self._send(200, json.dumps(self._build_state_dict()).encode())

    def _build_state_dict(self) -> dict:
        if _state.done:
            try:
                data     = json.loads(_state.rankings_path.read_text())
                n_passed = sum(1 for p in data.get("photos", []) if p.get("decision") == "passed")
                n_culled = sum(1 for p in data.get("photos", []) if p.get("decision") != "passed")
            except Exception:
                n_passed = n_culled = 0
            return {
                "done":         True,
                "total_bursts": _state.total_bursts,
                "n_passed":     n_passed,
                "n_culled":     n_culled,
                "output_dir":   str(_state.output_dir),
            }
        burst   = _state.current_burst()
        try:
            data    = json.loads(_state.rankings_path.read_text())
            by_name = {p["filename"]: p for p in data.get("photos", [])}
        except Exception:
            by_name = {}
        photos = []
        for fname in burst:
            entry = by_name.get(fname, {})
            photos.append({
                "filename": fname,
                "score":    entry.get("score"),
                "rank":     entry.get("rank"),
            })
        return {
            "done":         False,
            "burst_idx":    _state.current_idx,
            "total_bursts": _state.total_bursts,
            "photos":       photos,
        }

    def _serve_thumb(self, fname: str):
        data = _state.serve_thumbnail(fname)
        if data:
            self._send(200, data, "image/jpeg")
        else:
            # Return a tiny grey placeholder
            self._send(200, _grey_placeholder(), "image/jpeg")


def _grey_placeholder() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (4, 3), (40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _state
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Pipeline output folder (contains culls/)")
    parser.add_argument("--port",   type=int, default=8767)
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not (output_dir / "culls" / "passed").exists():
        print(f"ERROR: {output_dir / 'culls' / 'passed'} not found. Run the pipeline first.")
        return

    _state = ReviewState(output_dir)

    if _state.total_bursts == 0:
        print("No passed photos to review.")
        return

    import webbrowser
    url = f"http://127.0.0.1:{args.port}"
    print(f"Burst review server: {url}")
    print(f"Bursts to review: {_state.total_bursts}  (current: {_state.current_idx + 1})")
    print("Press Ctrl+C to stop.")

    server = HTTPServer(("127.0.0.1", args.port), BurstReviewHandler)
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
