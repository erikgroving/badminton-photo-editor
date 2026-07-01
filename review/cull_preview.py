"""
review/cull_preview.py — Culling result browser.

Runs DINOv2-large culling inference on a folder of CR3s and serves a
scrollable HTML page: thumbnail on the left, score bar + value on the right.
Scores are cached to <folder>_cull_scores.json so re-opening is instant.

Usage:
    python review/cull_preview.py "C:/path/to/Raws"
    python review/cull_preview.py "C:/path/to/Raws" --show culled
    python review/cull_preview.py "C:/path/to/Raws" --show passed
    python review/cull_preview.py "C:/path/to/Raws" --port 8766 --no-browser
"""
import argparse
import io
import json
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(raw_dir: Path, ckpt_path: Path, cache_path: Path) -> dict:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from config import THUMB_SIZE
    from data.raw_reader import extract_thumbnail
    from models.culling.model import build_model

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )

    saved            = torch.load(ckpt_path, map_location=device)
    backbone         = saved["backbone"]
    dynamic_img_size = saved.get("dynamic_img_size", False)
    model    = build_model(backbone=backbone, pretrained=False,
                           dynamic_img_size=dynamic_img_size).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    threshold = float(saved.get("threshold", 0.5))

    # Use the size the model was trained at; fall back to ViT-default 518
    is_vit     = "vit" in backbone or "dinov2" in backbone
    input_size = saved.get("input_size") or (518 if is_vit else THUMB_SIZE[0])
    thumb_size = (input_size, input_size)

    TF = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Deduplicate (Windows glob returns both .CR3 and .cr3 for the same file)
    seen: dict[str, Path] = {}
    for p in raw_dir.iterdir():
        if p.suffix.lower() == ".cr3":
            seen[p.name.lower()] = p
    raws = sorted(seen.values(), key=lambda p: p.name)

    print(f"Scoring {len(raws)} photos with {backbone} @ {input_size}px…", flush=True)
    photos = []
    for i, raw in enumerate(raws, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(raws)}", flush=True)
        img = extract_thumbnail(raw, size=thumb_size)
        inp = TF(img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = float(torch.sigmoid(model(inp)).item())
        photos.append({"filename": raw.name, "score": round(score, 6)})

    photos.sort(key=lambda x: x["score"])
    for rank, p in enumerate(photos, 1):
        p["rank"]     = rank
        p["decision"] = "passed" if p["score"] >= threshold else "culled"

    result = {
        "backbone":  backbone,
        "threshold": threshold,
        "photos":    photos,
        "raw_dir":   str(raw_dir),
    }
    cache_path.write_text(json.dumps(result, indent=2))
    print(f"Saved scores -> {cache_path}", flush=True)
    return result


# ─── Thumbnail extraction ───────────────────────────────────────────────────────

def _thumb_bytes(raw_path: Path, size: int = 420) -> bytes:
    import rawpy
    from PIL import Image

    thumb_dir  = raw_path.parent / ".cull_thumbs"
    thumb_dir.mkdir(exist_ok=True)
    cache_file = thumb_dir / f"{raw_path.stem}_{size}.jpg"

    if cache_file.exists():
        return cache_file.read_bytes()

    with rawpy.imread(str(raw_path)) as raw:
        try:
            t = raw.extract_thumb()
            img = (Image.open(io.BytesIO(t.data)) if t.format == rawpy.ThumbFormat.JPEG
                   else Image.fromarray(t.data))
        except Exception:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            img = Image.fromarray(rgb)

    img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    data = buf.getvalue()
    cache_file.write_bytes(data)
    return data


# ─── HTTP server ────────────────────────────────────────────────────────────────

_data:    dict = {}
_raw_dir: Path = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML.encode())
            return

        if path == "/api/data":
            self._send(200, "application/json", json.dumps(_data).encode())
            return

        if path.startswith("/thumb/"):
            filename = urllib.parse.unquote(path[len("/thumb/"):])
            raw_path = _raw_dir / filename
            if not raw_path.exists():
                self._send(404, "text/plain", b"not found")
                return
            try:
                self._send(200, "image/jpeg", _thumb_bytes(raw_path, size=900))
            except Exception as e:
                self._send(500, "text/plain", str(e).encode())
            return

        self._send(404, "text/plain", b"not found")


# ─── HTML ───────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cull Preview</title>
<style>
  :root{
    --bg:#0e0e0e;--surface:#161616;--border:#222;--text:#e0e0e0;--muted:#555;
    --green:#2ecc71;--red:#e74c3c;--blue:#3b82f6;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px}

  /* ── Header ── */
  #hdr{
    position:sticky;top:0;z-index:100;
    height:52px;display:flex;align-items:center;gap:12px;padding:0 20px;
    background:#080808;border-bottom:1px solid var(--border);
  }
  #hdr h1{font-size:14px;font-weight:600;color:#fff;white-space:nowrap}
  .pill{
    padding:5px 14px;border-radius:20px;border:1px solid var(--border);
    background:var(--surface);color:var(--muted);cursor:pointer;font-size:12px;transition:all .12s;
  }
  .pill.on{background:var(--blue);color:#fff;border-color:var(--blue)}
  .pill:hover:not(.on){color:var(--text);border-color:#444}
  #hdr-right{margin-left:auto;display:flex;align-items:center;gap:12px}
  #counts{font-size:12px;color:var(--muted);white-space:nowrap}
  select.srt{background:var(--surface);color:var(--text);border:1px solid var(--border);
    border-radius:4px;padding:5px 10px;font-size:12px;cursor:pointer}
  #thr-line{font-size:11px;color:var(--muted);white-space:nowrap}

  /* ── Cards ── */
  #list{padding:14px 20px 80px}

  .row{
    display:flex;align-items:stretch;
    border:1px solid var(--border);border-radius:8px;
    margin-bottom:10px;overflow:hidden;
    background:var(--surface);transition:background .1s;
  }
  .row:hover{background:#1b1b1b}
  .row.passed{border-left:4px solid var(--green)}
  .row.culled{border-left:4px solid var(--red)}

  /* Large preview */
  .thumb-wrap{
    flex-shrink:0;width:640px;
    background:#0a0a0a;overflow:hidden;
    display:flex;align-items:center;justify-content:center;
    min-height:427px;
  }
  .thumb-wrap img{
    width:640px;height:427px;
    object-fit:cover;display:block;
  }
  .thumb-wrap .no-img{color:#2a2a2a;font-size:13px}

  /* Right info panel */
  .info{
    flex:1;padding:24px 28px;
    display:flex;flex-direction:column;justify-content:center;gap:14px;
    min-width:0;
  }

  .fname{
    font-size:17px;font-weight:600;color:#fff;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  }
  .meta{font-size:12px;color:var(--muted)}

  /* Score display */
  .score-block{display:flex;align-items:baseline;gap:10px}
  .score-num{
    font-size:52px;font-weight:800;line-height:1;
    font-variant-numeric:tabular-nums;letter-spacing:-1px;
  }
  .passed .score-num{color:var(--green)}
  .culled .score-num{color:var(--red)}
  .score-label{font-size:13px;color:var(--muted)}

  /* Bar */
  .bar-bg{
    height:8px;background:#1e1e1e;border-radius:4px;overflow:hidden;
    margin-top:4px;
  }
  .bar-fill{height:100%;border-radius:4px}
  .passed .bar-fill{background:var(--green)}
  .culled .bar-fill{background:var(--red)}

  /* Badge */
  .badge{
    display:inline-block;font-size:10px;font-weight:800;
    letter-spacing:.8px;text-transform:uppercase;
    padding:3px 10px;border-radius:4px;width:fit-content;
  }
  .passed .badge{background:#0a1f0f;color:var(--green);border:1px solid #1a4a1f}
  .culled .badge{background:#1f0a0a;color:var(--red);border:1px solid #4a1a1a}

  /* Empty */
  #empty{display:none;text-align:center;padding:80px 0;color:var(--muted);font-size:14px}
</style>
</head>
<body>

<div id="hdr">
  <h1>Cull Preview</h1>
  <button class="pill on" id="f-all"    onclick="setFilter('all')">All</button>
  <button class="pill"    id="f-culled" onclick="setFilter('culled')">Culled</button>
  <button class="pill"    id="f-passed" onclick="setFilter('passed')">Passed</button>
  <div id="hdr-right">
    <select class="srt" id="srt" onchange="applySort()">
      <option value="score-asc">Score: Low → High</option>
      <option value="score-desc">Score: High → Low</option>
      <option value="name-asc">Name A → Z</option>
      <option value="name-desc">Name Z → A</option>
    </select>
    <span id="counts"></span>
    <span id="thr-line"></span>
  </div>
</div>

<div id="list"></div>
<div id="empty">No photos match this filter.</div>

<script>
let photos = [];
let filter = 'all';
let total  = 0;

async function init() {
  const d = await fetch('/api/data').then(r => r.json());
  photos = d.photos || [];
  total  = photos.length;
  document.getElementById('thr-line').textContent = `threshold ${(d.threshold ?? 0).toFixed(4)}`;
  render();
}

function sorted(arr) {
  const s = document.getElementById('srt').value;
  return [...arr].sort((a, b) =>
    s === 'score-asc'  ? a.score - b.score :
    s === 'score-desc' ? b.score - a.score :
    s === 'name-asc'   ? a.filename.localeCompare(b.filename) :
                         b.filename.localeCompare(a.filename)
  );
}

function setFilter(f) {
  filter = f;
  ['all','culled','passed'].forEach(k =>
    document.getElementById('f-'+k).classList.toggle('on', k === f)
  );
  render();
}

function applySort() { render(); }

function render() {
  const visible = sorted(
    filter === 'all' ? photos : photos.filter(p => p.decision === filter)
  );
  const passed = photos.filter(p => p.decision === 'passed').length;
  document.getElementById('counts').textContent =
    `${total} total · ${passed} passed · ${total - passed} culled · showing ${visible.length}`;

  const list  = document.getElementById('list');
  const empty = document.getElementById('empty');
  if (!visible.length) { list.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';

  list.innerHTML = visible.map(p => {
    const fn  = encodeURIComponent(p.filename);
    const pct = (p.score * 100).toFixed(1);
    return `
<div class="row ${p.decision}">
  <div class="thumb-wrap">
    <img src="/thumb/${fn}" loading="lazy"
         onerror="this.parentElement.innerHTML='<span class=no-img>no preview</span>'">
  </div>
  <div class="info">
    <div>
      <div class="fname">${p.filename}</div>
      <div class="meta">rank #${p.rank} of ${total}</div>
    </div>
    <div>
      <div class="score-block">
        <span class="score-num">${p.score.toFixed(4)}</span>
        <span class="score-label">confidence</span>
      </div>
      <div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div>
    </div>
    <span class="badge">${p.decision}</span>
  </div>
</div>`;
  }).join('');
}

init();
</script>
</body>
</html>"""


# ─── Entry point ────────────────────────────────────────────────────────────────

def main():
    global _data, _raw_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir",     help="Folder containing CR3 files")
    parser.add_argument("--show",      choices=["all", "culled", "passed"], default="all")
    parser.add_argument("--port",      type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip inference, load cached scores only")
    parser.add_argument("--force-inference", action="store_true",
                        help="Re-score even if a cache exists")
    parser.add_argument("--ckpt", default=None,
                        help="Path to culling checkpoint (default: best in checkpoints/)")
    args = parser.parse_args()

    _raw_dir = Path(args.raw_dir).resolve()
    if not _raw_dir.is_dir():
        print(f"ERROR: {_raw_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Cache lives next to the input folder
    cache_path = _raw_dir.parent / f"{_raw_dir.name}_cull_scores.json"

    if args.no_inference and not cache_path.exists():
        print(f"ERROR: no cached scores at {cache_path}", file=sys.stderr)
        sys.exit(1)

    if cache_path.exists() and not args.force_inference:
        print(f"Loading cached scores from {cache_path}")
        _data = json.loads(cache_path.read_text())
    else:
        # Find best culling checkpoint if not specified
        if args.ckpt:
            ckpt_path = Path(args.ckpt)
        else:
            import torch
            ckpts_dir = PROJECT_ROOT / "checkpoints"
            # Exclude non-standard-split variants (_event, _random, _burst*, _w*, _attr, _lstm)
            _skip = ("_event", "_random", "_burst", "_attr", "_lstm")
            def _is_standard(p: Path) -> bool:
                stem = p.stem.replace("culling_", "")
                return not any(stem.endswith(s) or s[1:] in stem.split("_w")[1:2] or s in stem
                               for s in _skip)
            candidates = [f for f in ckpts_dir.glob("culling_*.pt") if _is_standard(f)]
            if not candidates:
                candidates = list(ckpts_dir.glob("culling_*.pt"))
            if not candidates:
                print("ERROR: no culling checkpoints found", file=sys.stderr)
                sys.exit(1)
            ckpt_path = max(
                candidates,
                key=lambda f: torch.load(f, map_location="cpu").get("metrics", {}).get("f2", 0),
            )
            print(f"Auto-selected checkpoint: {ckpt_path.name}")
        _data = _run_inference(_raw_dir, ckpt_path, cache_path)

    n = len(_data.get("photos", []))
    passed = sum(1 for p in _data["photos"] if p["decision"] == "passed")
    print(f"\n{n} photos  |  {passed} passed  |  {n-passed} culled")
    print(f"Threshold: {_data['threshold']:.4f}")

    url = f"http://127.0.0.1:{args.port}"
    print(f"Review:    {url}\n")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
