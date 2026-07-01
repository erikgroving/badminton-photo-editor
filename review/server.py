"""
review/server.py — Jay Ma Photo Review Server

Reads a pipeline output directory and serves a browser-based review UI:
  - Cull Review:     passed vs culled grid, sortable by score or filename
  - Stage Selection: 3-panel per photo (original | cropped | colored), click to select for finals

Output directory layout expected:
  <output>/
    culls/
      passed/              ← symlinks or copies of .CR3 files
      culled/              ← symlinks or copies of .CR3 files
      rankings.json        ← written by pipeline
    thumbnails/            ← JPEG previews cached here by this server
    crops/                 ← cropped JPEGs from pipeline
    colors_and_lightning/  ← color-corrected JPEGs from pipeline
    finals/
      original/            ← Jay selected: needs full rework (large thumbnail)
      cropped/             ← Jay selected: crop OK, color needs work
      colored/             ← Jay selected: looks great, use this
      selections.json      ← saved automatically on each click

Usage:
    python review/server.py /path/to/PhotoSession_output/
    python review/server.py /path/to/PhotoSession_output/ --port 8765 --no-browser
"""
import argparse
import io
import json
import os
import shutil
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

try:
    import rawpy
    from PIL import Image
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

# ─── Global state ──────────────────────────────────────────────────────────────
_output_dir: Path = None
_data_cache: Optional[dict] = None
_cache_lock = threading.Lock()


# ─── Data loading ──────────────────────────────────────────────────────────────

def _parse_rankings(path: Path) -> tuple[list[dict], dict]:
    """Parse rankings.json or legacy pipeline_results.json. Returns (photos, meta)."""
    data = json.loads(path.read_text())

    if isinstance(data, dict) and "photos" in data:
        # Canonical format
        photos = []
        for i, p in enumerate(data["photos"], 1):
            photos.append({
                "filename": p.get("filename", p.get("file", "")),
                "stem":     Path(p.get("filename", p.get("file", ""))).stem,
                "score":    float(p.get("score", 0.0)),
                "rank":     p.get("rank", i),
                "decision": p.get("decision", "passed"),
            })
        return photos, data

    if isinstance(data, dict) and "results" in data:
        # Legacy pipeline_results.json
        photos = []
        for i, r in enumerate(data["results"], 1):
            fname = r.get("file", "")
            photos.append({
                "filename": fname,
                "stem":     Path(fname).stem,
                "score":    float(r.get("cull_conf", r.get("confidence", 0.0))),
                "rank":     i,
                "decision": "culled" if r.get("culled") else "passed",
            })
        return photos, data

    return [], {}


def _load_data() -> dict:
    # No in-memory caching — re-read each time so fresh crops/colors show immediately
    # after a processing run without needing a server restart.
    for candidate in (
        _output_dir / "culls" / "rankings.json",
        _output_dir / "pipeline_results.json",
    ):
        if candidate.exists():
            photos, meta = _parse_rankings(candidate)
            break
    else:
        return {"error": "No rankings.json found. Run the pipeline first.", "photos": []}

    # Enrich with which pipeline outputs actually exist
    crops_dir  = _output_dir / "crops"
    colors_dir = _output_dir / "colors_and_lightning"
    for p in photos:
        p["has_crop"]  = (crops_dir  / f"{p['stem']}.jpg").exists()
        p["has_color"] = (colors_dir / f"{p['stem']}.jpg").exists()

    # Load saved selections
    sel_path = _output_dir / "finals" / "selections.json"
    selections = {}
    if sel_path.exists():
        try:
            selections = json.loads(sel_path.read_text())
        except Exception:
            pass
    for p in photos:
        p["selection"] = selections.get(p["stem"])

    return {
        "photos":   photos,
        "metadata": {
            "recall_target":  meta.get("recall_target"),
            "threshold_used": meta.get("threshold_used"),
            "model_version":  meta.get("model_version"),
        },
    }


def _invalidate_cache():
    global _data_cache
    with _cache_lock:
        _data_cache = None


# ─── Thumbnail extraction ───────────────────────────────────────────────────────

def _find_cr3(filename: str) -> Optional[Path]:
    """Locate a CR3 file in the output directory tree."""
    for sub in ("culls/passed", "culls/culled", "."):
        p = _output_dir / sub / filename
        if p.exists():
            return p.resolve() if p.is_symlink() else p
    return None


def _get_thumbnail_bytes(filename: str, size: int = 1200) -> bytes:
    """
    Return JPEG bytes for the CR3 developed thumbnail.
    Uses develop_raw (same pipeline as crop output) so color and orientation
    match the Cropped and Color Corrected columns exactly.
    Results are disk-cached keyed by filename + size.
    """
    stem = Path(filename).stem
    thumb_dir = _output_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    cache_path = thumb_dir / f"{stem}_{size}.jpg"

    if cache_path.exists():
        return cache_path.read_bytes()

    cr3 = _find_cr3(filename)
    if cr3 is None:
        raise FileNotFoundError(f"CR3 not found: {filename}")

    try:
        # Develop the RAW — same processing as the crop pipeline, so colors match.
        # develop_raw applies camera orientation automatically via rawpy.postprocess().
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.raw_reader import develop_raw
        img = develop_raw(cr3, size=(size, size), neutral=False)
    except Exception:
        # Fallback: extract embedded JPEG (colour won't match crop, but better than nothing)
        if not HAS_RAWPY:
            raise RuntimeError("rawpy not installed")
        with rawpy.imread(str(cr3)) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(thumb.data))
                else:
                    img = Image.fromarray(thumb.data)
            except Exception:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
                img = Image.fromarray(rgb)
        img.thumbnail((size, size), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    data = buf.getvalue()
    cache_path.write_bytes(data)
    return data


# ─── Selection writing ──────────────────────────────────────────────────────────

def _save_selection(filename: str, version: Optional[str]):
    """Persist a selection (or deselection) to finals/selections.json."""
    stem = Path(filename).stem
    finals_dir = _output_dir / "finals"
    finals_dir.mkdir(exist_ok=True)
    sel_path = finals_dir / "selections.json"

    sels: dict = {}
    if sel_path.exists():
        try:
            sels = json.loads(sel_path.read_text())
        except Exception:
            pass

    if version is None:
        sels.pop(stem, None)
    else:
        sels[stem] = version

    sel_path.write_text(json.dumps(sels, indent=2))
    _invalidate_cache()


def _export_all_selections() -> tuple[int, list[str]]:
    """Copy/save selected versions to finals/<version>/ subdirectories."""
    sel_path = _output_dir / "finals" / "selections.json"
    if not sel_path.exists():
        return 0, []

    sels = json.loads(sel_path.read_text())
    copied = 0
    errors = []

    for stem, version in sels.items():
        try:
            dest_dir = _output_dir / "finals" / version
            dest_dir.mkdir(parents=True, exist_ok=True)
            dst = dest_dir / f"{stem}.jpg"

            if version == "original":
                # Use a larger cached thumbnail as the proxy for "original"
                cr3_name = next(
                    (f.name for f in [
                        _output_dir / "culls" / "passed" / f"{stem}.CR3",
                        _output_dir / "culls" / "passed" / f"{stem}.cr3",
                        _output_dir / "culls" / "culled" / f"{stem}.CR3",
                        _output_dir / "culls" / "culled" / f"{stem}.cr3",
                    ] if f.exists()),
                    f"{stem}.CR3",
                )
                data = _get_thumbnail_bytes(cr3_name, size=1200)
                dst.write_bytes(data)
            elif version == "cropped":
                src = _output_dir / "crops" / f"{stem}.jpg"
                shutil.copy2(src, dst)
            elif version == "colored":
                src = _output_dir / "colors_and_lighting" / f"{stem}.jpg"
                shutil.copy2(src, dst)
            else:
                errors.append(f"Unknown version '{version}' for {stem}")
                continue

            copied += 1
        except Exception as e:
            errors.append(f"{stem}: {e}")

    return copied, errors


# ─── HTTP Handler ───────────────────────────────────────────────────────────────

class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress request logs

    def _send(self, code: int, content_type: str, body: bytes, cache: str = "no-cache"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self._send(code, "application/json", body)

    def _send_error(self, code: int, msg: str):
        self._send_json({"error": msg}, code)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send(200, "text/html; charset=utf-8", HTML.encode())
            return

        if path == "/api/data":
            self._send_json(_load_data())
            return

        # ── Image serving ──────────────────────────────────────────────────────
        if path.startswith("/img/thumb/"):
            filename = urllib.parse.unquote(path[len("/img/thumb/"):])
            try:
                size = int(params.get("size", ["1200"])[0])
                data = _get_thumbnail_bytes(filename, size=size)
                self._send(200, "image/jpeg", data, cache="public, max-age=3600")
            except FileNotFoundError:
                self._send_error(404, f"CR3 not found: {filename}")
            except Exception as e:
                self._send_error(500, str(e))
            return

        if path.startswith("/img/crop/"):
            rel = urllib.parse.unquote(path[len("/img/crop/"):])
            img_path = _output_dir / "crops" / rel
            if img_path.exists():
                self._send(200, "image/jpeg", img_path.read_bytes(), cache="public, max-age=3600")
            else:
                self._send_error(404, f"Not found: {rel}")
            return

        if path.startswith("/img/color/"):
            rel = urllib.parse.unquote(path[len("/img/color/"):])
            img_path = _output_dir / "colors_and_lightning" / rel
            if img_path.exists():
                self._send(200, "image/jpeg", img_path.read_bytes(), cache="public, max-age=3600")
            else:
                self._send_error(404, f"Not found: {rel}")
            return

        self._send_error(404, "Not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        if self.path == "/api/select":
            try:
                payload  = json.loads(body)
                filename = payload.get("filename", "")
                version  = payload.get("version")  # None = deselect
                _save_selection(filename, version)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_error(400, str(e))
            return

        if self.path == "/api/export":
            try:
                n, errors = _export_all_selections()
                msg = f"Exported {n} photo(s) to finals/"
                if errors:
                    msg += f" ({len(errors)} failed)"
                self._send_json({"ok": True, "message": msg, "errors": errors})
            except Exception as e:
                self._send_error(500, str(e))
            return

        self._send_error(404, "Not found")


# ─── HTML ───────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Jay Ma Photo Review</title>
<style>
  :root {
    --bg:#111; --surface:#1c1c1c; --border:#2a2a2a; --text:#e0e0e0; --muted:#666;
    --green:#2ecc71; --yellow:#f59e0b;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px}

  /* ── Header ── */
  #sticky-head { position: sticky; top: 0; z-index: 100; background: #0a0a0a; }
  #hdr{
    height:48px;display:flex;align-items:center;gap:10px;padding:0 16px;
    background:#0a0a0a;border-bottom:1px solid var(--border);
  }
  #hdr h1{font-size:14px;font-weight:600;color:#fff;white-space:nowrap;margin-right:6px}
  #hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
  select.sort{background:var(--surface);color:var(--text);border:1px solid var(--border);
    border-radius:4px;padding:4px 8px;font-size:12px;cursor:pointer}
  #stat{color:var(--muted);font-size:12px;white-space:nowrap}

  /* ── Stage View ── */
  .sbar{
    background:#0a0a0a;
    border-bottom:1px solid var(--border);padding:8px 16px;
    display:flex;align-items:center;gap:12px;
  }
  .export-btn{
    background:var(--green);color:#000;font-weight:700;
    border:none;border-radius:5px;padding:6px 16px;cursor:pointer;font-size:12px;
  }
  .export-btn:hover{opacity:.85}
  #sel-count{font-size:12px;color:var(--muted)}
  .col-labels{
    display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;
    padding:6px 16px;border-bottom:1px solid var(--border);
    background:#111;
  }
  .col-labels span{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:#999;text-align:center}
  #stage-rows{padding:8px 16px 60px}
  .prow{
    display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;
    margin-bottom:10px;border-radius:6px;overflow:hidden;
    border:1px solid var(--border);align-items:stretch;
  }
  .prow.focused { outline: 2px solid #fff; outline-offset: 2px; }
  .prow-label{
    grid-column:1/-1;padding:5px 10px;background:var(--surface);
    font-size:11px;color:var(--muted);display:flex;align-items:center;gap:8px;
    border-bottom:1px solid var(--border);
  }
  .rnk{background:#2a2a2a;color:#aaa;border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700}
  .sc{color:var(--yellow);font-weight:600}
  .spanel{
    background:var(--surface);cursor:pointer;position:relative;
    transition:background .12s;border-right:1px solid var(--border);
  }
  .spanel:last-child{border-right:none}
  .spanel:hover{background:#222}
  .spanel.sel{background:#0b1a10;outline:3px solid var(--green);outline-offset:-3px}
  .img-wrap { aspect-ratio: 3/2; overflow: hidden; background: linear-gradient(90deg, #1a1a1a 25%, #242424 50%, #1a1a1a 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; display:flex; align-items:center; justify-content:center; }
  .img-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .nophoto { width:100%; aspect-ratio: 3/2; display:flex; align-items:center; justify-content:center; color:#333; font-size:11px; }
  .plabel{padding:4px 8px;display:flex;justify-content:space-between;align-items:center;border-top: 3px solid transparent;}
  .sel .plabel { border-top-color: var(--green); }
  .pname{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
  .sel .pname{color:var(--green)}
  .pcheck{display:none;font-size:11px;font-weight:700;background:var(--green);color:#000;padding:3px 9px;border-radius:3px}
  .sel .pcheck{display:block}
  .prow.has-selection .spanel:not(.sel) { opacity: 0.6; }
  .unreviewed-badge { color: #444; font-size: 10px; margin-left: 6px; }

  /* ── Toast ── */
  #toast{
    position:fixed;bottom:18px;right:18px;background:#222;color:#fff;
    padding:9px 16px;border-radius:5px;font-size:12px;opacity:0;pointer-events:none;
    transition:opacity .25s;z-index:999;border:1px solid var(--border);
  }
  #toast.show{opacity:1}

  @keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
  }
</style>
</head>
<body>

<div id="sticky-head">
<div id="hdr">
  <h1>Jay Ma · Stage Selection</h1>
  <div id="hdr-right">
    <select class="sort" id="srt" onchange="applySort()">
      <option value="name-asc">Name A → Z</option>
      <option value="name-desc">Name Z → A</option>
      <option value="score-desc">Score: High → Low</option>
      <option value="score-asc">Score: Low → High</option>
    </select>
    <span id="stat">Loading…</span>
  </div>
</div>

<div class="sbar">
  <button class="export-btn" onclick="doExport()">Export Selections → finals/</button>
  <span id="sel-count">0 selected</span>
  <span style="font-size:10px; color:#555; margin-left:12px;">J/K move · 1/2/3 pick version</span>
</div>
<div class="col-labels">
  <span>Original</span><span>Cropped</span><span>Color Corrected</span>
</div>
</div>

<div id="stage-rows"></div>

<div id="toast"></div>

<div id="export-modal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:200; align-items:center; justify-content:center;">
  <div style="background:#1c1c1c; border:1px solid #333; border-radius:8px; padding:28px; max-width:420px; width:90%;">
    <h2 style="font-size:15px; margin:0 0 12px; color:#fff;">Confirm Export</h2>
    <p id="modal-body" style="font-size:13px; color:#aaa; line-height:1.7; margin:0 0 20px;"></p>
    <div style="display:flex; gap:10px; justify-content:flex-end;">
      <button onclick="closeModal()" style="padding:7px 18px; background:#333; color:#ccc; border:none; border-radius:5px; cursor:pointer; font-size:13px;">Cancel</button>
      <button onclick="confirmExport()" style="padding:7px 18px; background:var(--green); color:#000; font-weight:700; border:none; border-radius:5px; cursor:pointer; font-size:13px;">Export</button>
    </div>
  </div>
</div>

<script>
let photos = [];
let selections = {};
let focusedIdx = -1;

async function init() {
  const d = await fetch('/api/data').then(r => r.json());
  if (d.error) { document.getElementById('stat').textContent = d.error; return; }
  photos = d.photos || [];

  const passed = photos.filter(p => p.decision === 'passed').length;
  document.getElementById('stat').textContent = `${passed} photos`;

  renderStage();
  updateCount();
}

function sorted(arr) {
  const s = document.getElementById('srt').value;
  return [...arr].sort((a, b) =>
    s === 'score-desc' ? b.score - a.score :
    s === 'score-asc'  ? a.score - b.score :
    s === 'name-asc'   ? a.filename.localeCompare(b.filename) :
                         b.filename.localeCompare(a.filename)
  );
}
function applySort() { renderStage(); }

function renderStage() {
  const passed = sorted(photos.filter(p => p.decision === 'passed'));
  document.getElementById('stage-rows').innerHTML = passed.map(stageRow).join('');
  const rows = document.querySelectorAll('.prow');
  focusRow(Math.min(Math.max(focusedIdx, 0), rows.length - 1));
}

function stageRow(p) {
  const so = selections[p.stem] === 'original' ? 'sel' : '';
  const sc = selections[p.stem] === 'cropped'  ? 'sel' : '';
  const sk = selections[p.stem] === 'colored'  ? 'sel' : '';
  const fn = encodeURIComponent(p.filename);
  const pct = (p.score * 100).toFixed(1) + '%';
  const hasSelClass = selections[p.stem] ? ' has-selection' : '';
  const unrevBadge = selections[p.stem] ? '' : '<span class="unreviewed-badge">· unreviewed ·</span>';

  const cropImg  = p.has_crop  ? `<div class="img-wrap"><img src="/img/crop/${p.stem}.jpg" loading="lazy" onload="this.parentElement.style.animation='none'"></div>` : `<div class="nophoto">no output</div>`;
  const colorImg = p.has_color ? `<div class="img-wrap"><img src="/img/color/${p.stem}.jpg" loading="lazy" onload="this.parentElement.style.animation='none'"></div>` : `<div class="nophoto">no output</div>`;

  return `<div class="prow${hasSelClass}" id="row-${p.stem}">
  <div class="prow-label">
    <span class="rnk">#${p.rank}</span>
    <span>${p.filename}</span>
    <span class="sc">${pct}</span>
    ${unrevBadge}
  </div>
  <div class="spanel ${so}" onclick="pick('${p.stem}','${p.filename}','original')">
    <div class="img-wrap"><img src="/img/thumb/${fn}" loading="lazy" onload="this.parentElement.style.animation='none'" onerror="this.style.background='#1f1f1f'"></div>
    <div class="plabel"><span class="pname">Original</span><span class="pcheck">✓ SELECTED</span></div>
  </div>
  <div class="spanel ${sc}" onclick="pick('${p.stem}','${p.filename}','cropped')">
    ${cropImg}
    <div class="plabel"><span class="pname">Cropped</span><span class="pcheck">✓ SELECTED</span></div>
  </div>
  <div class="spanel ${sk}" onclick="pick('${p.stem}','${p.filename}','colored')">
    ${colorImg}
    <div class="plabel"><span class="pname">Color Corrected</span><span class="pcheck">✓ SELECTED</span></div>
  </div>
</div>`;
}

function pick(stem, filename, version) {
  const prev = selections[stem];
  if (prev === version) {
    delete selections[stem];
    version = null;
  } else {
    selections[stem] = version;
  }

  const row = document.getElementById('row-' + stem);
  if (row) {
    row.querySelectorAll('.spanel').forEach(p => p.classList.remove('sel'));
    if (version) {
      const idx = ['original','cropped','colored'].indexOf(version);
      const panels = row.querySelectorAll('.spanel');
      if (idx >= 0 && panels[idx]) panels[idx].classList.add('sel');
    }
    if (version) row.classList.add('has-selection');
    else row.classList.remove('has-selection');
  }

  const unrev = document.querySelector(`#row-${stem} .unreviewed-badge`);
  if (version && unrev) unrev.remove();

  updateCount();

  fetch('/api/select', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({filename, version}),
  });
}

function updateCount() {
  const n = Object.keys(selections).length;
  document.getElementById('sel-count').textContent =
    n === 1 ? '1 photo selected' : `${n} photos selected`;
}

function focusRow(idx) {
  const rows = Array.from(document.querySelectorAll('.prow'));
  if (rows[focusedIdx]) rows[focusedIdx].classList.remove('focused');
  focusedIdx = idx;
  if (rows[focusedIdx]) {
    rows[focusedIdx].classList.add('focused');
    rows[focusedIdx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

function pickFocused(version) {
  if (focusedIdx < 0) return;
  const rows = Array.from(document.querySelectorAll('.prow'));
  const row = rows[focusedIdx];
  if (!row) return;
  const stem = row.id.replace('row-', '');
  const photo = photos.find(p => p.stem === stem);
  if (photo) {
    if (version && selections[stem] === version) {
      pick(stem, photo.filename, version);
    } else if (version) {
      pick(stem, photo.filename, version);
    } else {
      if (selections[stem]) pick(stem, photo.filename, selections[stem]);
    }
  }
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const rows = document.querySelectorAll('.prow');
  if (e.key === 'ArrowDown' || e.key === 'j' || e.key === 'J') {
    e.preventDefault();
    focusRow(Math.min(focusedIdx + 1, rows.length - 1));
  } else if (e.key === 'ArrowUp' || e.key === 'k' || e.key === 'K') {
    e.preventDefault();
    focusRow(Math.max(focusedIdx - 1, 0));
  } else if (e.key === '1' || e.key === 'o' || e.key === 'O') {
    pickFocused('original');
  } else if (e.key === '2' || e.key === 'c' || e.key === 'C') {
    pickFocused('cropped');
  } else if (e.key === '3' || e.key === 'v' || e.key === 'V') {
    pickFocused('colored');
  } else if (e.key === '0' || e.key === 'Escape') {
    pickFocused(null);
  }
});

function doExport() {
  if (!Object.keys(selections).length) { toast('Nothing selected yet'); return; }
  const passed = photos.filter(p => p.decision === 'passed');
  const nSel = Object.keys(selections).length;
  const nUnreviewed = passed.length - nSel;
  const counts = { original: 0, cropped: 0, colored: 0 };
  for (const v of Object.values(selections)) if (counts[v] !== undefined) counts[v]++;
  document.getElementById('modal-body').textContent =
    `${counts.original} originals · ${counts.cropped} cropped · ${counts.colored} color corrected` +
    (nUnreviewed > 0 ? ` · ${nUnreviewed} photo(s) not yet reviewed` : ' · all reviewed');
  document.getElementById('export-modal').style.display = 'flex';
}
function closeModal() { document.getElementById('export-modal').style.display = 'none'; }
async function confirmExport() {
  closeModal();
  toast('Exporting…');
  const r = await fetch('/api/export', {method:'POST'}).then(r => r.json()).catch(() => ({}));
  toast(r.message || 'Export complete');
}

function toast(msg, ms = 3500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), ms);
}

init();
</script>
</body>
</html>"""


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    global _output_dir

    parser = argparse.ArgumentParser(description="Jay Ma Photo Review Server")
    parser.add_argument("output_dir",  help="Pipeline output directory (contains culls/, crops/, colors_and_lightning/)")
    parser.add_argument("--port",      type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    _output_dir = Path(args.output_dir).resolve()
    if not _output_dir.is_dir():
        print(f"ERROR: {_output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if not HAS_RAWPY:
        print("WARNING: rawpy/Pillow not installed — thumbnail extraction disabled.", file=sys.stderr)
        print("         Run:  pip install rawpy Pillow", file=sys.stderr)

    # Reset selections so every server session starts with a clean slate
    sel_path = _output_dir / "finals" / "selections.json"
    if sel_path.exists():
        sel_path.write_text("{}")

    def _prewarm():
        from concurrent.futures import ThreadPoolExecutor
        try:
            data = _load_data()
            passed = [p for p in data.get("photos", []) if p.get("decision") == "passed"]
            print(f"[prewarm] warming {len(passed)} thumbnails…", flush=True)
            with ThreadPoolExecutor(max_workers=4) as ex:
                for p in passed:
                    ex.submit(_get_thumbnail_bytes, p["filename"])
            print("[prewarm] done", flush=True)
        except Exception as exc:
            print(f"[prewarm] error: {exc}", flush=True)
    threading.Thread(target=_prewarm, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    url    = f"http://127.0.0.1:{args.port}"
    print(f"Review server: {url}")
    print(f"Output dir:    {_output_dir}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
