"""
Read Lightroom develop settings from XMP embedded in exported JPEGs.

Lightroom embeds the full develop history in each exported JPEG as an inline
XMP packet. We read it directly — no Nelder-Mead fitting, no approximation.

Extracts:
  temperature, tint, exposure, contrast, highlights, shadows,
  whites, blacks, saturation, vibrance, clarity, texture, dehaze,
  hue_yellow (the one HSL knob Jay occasionally uses)

Also reads:
  tone_curve  — the [0,0; 75,92; 174,198; 255,255] lift curve Jay uses on NW CRC
  camera_profile — always "Adobe Standard" in this dataset

Usage (standalone):
    python -m data.xmp_reader                     # print stats
    python -m data.xmp_reader --out data/color_gt.json  # write GT file
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROP_GT_FILE, COLOR_GT_FILE

# ── Parameter catalogue ───────────────────────────────────────────────────────
# (xmp_key, default_value, (lo, hi) for normalisation to [-1, 1])
_PARAMS: list[tuple[str, float, tuple[float, float]]] = [
    ("Temperature",   5500.0, (2000.0,  8000.0)),
    ("Tint",             0.0, (-150.0,   150.0)),
    ("Exposure2012",     0.0,   (-5.0,     5.0)),
    ("Contrast2012",     0.0, (-100.0,  100.0)),
    ("Highlights2012",   0.0, (-100.0,  100.0)),
    ("Shadows2012",      0.0, (-100.0,  100.0)),
    ("Whites2012",       0.0, (-100.0,  100.0)),
    ("Blacks2012",       0.0, (-100.0,  100.0)),
    ("Vibrance",         0.0, (-100.0,  100.0)),
    ("Saturation",       0.0, (-100.0,  100.0)),
    ("Clarity2012",      0.0, (-100.0,  100.0)),
    ("Texture",          0.0, (-100.0,  100.0)),
    ("Dehaze",           0.0, (-100.0,  100.0)),
    ("HueAdjustmentYellow", 0.0, (-100.0, 100.0)),
]

PARAM_NAMES: list[str] = [p[0] for p in _PARAMS]
PARAM_DEFAULTS: dict[str, float] = {p[0]: p[1] for p in _PARAMS}
PARAM_RANGES:   dict[str, tuple[float, float]] = {p[0]: p[2] for p in _PARAMS}


def _xmp_bytes(jpg_path: str | Path) -> bytes | None:
    """Return the raw XMP bytes from a JPEG, or None if not found."""
    with open(jpg_path, "rb") as fh:
        data = fh.read()
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        return None
    end = data.find(b"</x:xmpmeta>", start)
    if end < 0:
        return None
    return data[start: end + len(b"</x:xmpmeta>")]


def read_xmp_params(jpg_path: str | Path) -> dict | None:
    """
    Parse Lightroom develop settings from a JPEG's embedded XMP.

    Returns a dict with all PARAM_NAMES as float values, plus:
      "tone_curve"    : list of [input, output] pairs (or None if linear)
      "white_balance" : "As Shot" | "Custom" | str
      "has_xmp"       : True

    Returns None if the file has no embedded XMP.
    """
    raw = _xmp_bytes(jpg_path)
    if raw is None:
        return None

    xmp = raw.decode("utf-8", errors="replace")

    # Extract all attribute-style crs: tags
    attrs = dict(re.findall(r'crs:(\w+)="([^"]*)"', xmp))

    params: dict[str, float | str | list | bool] = {"has_xmp": True}
    params["white_balance"] = attrs.get("WhiteBalance", "As Shot")

    for key, default, _ in _PARAMS:
        raw_val = attrs.get(key)
        if raw_val is None:
            params[key] = default
        else:
            try:
                params[key] = float(raw_val.lstrip("+"))
            except ValueError:
                params[key] = default

    # Tone curve (element form, not attribute)
    curve_name = attrs.get("ToneCurveName2012", "Linear")
    if curve_name != "Linear":
        m = re.search(r'<crs:ToneCurvePV2012>(.*?)</crs:ToneCurvePV2012>', xmp, re.DOTALL)
        if m:
            pts = re.findall(r'<rdf:li>([^<]+)</rdf:li>', m.group(1))
            params["tone_curve"] = [[int(x) for x in p.split(",")] for p in pts]
        else:
            params["tone_curve"] = None
    else:
        params["tone_curve"] = None   # linear = no custom curve

    return params


def params_to_vec(params: dict) -> list[float]:
    """Normalise param dict → [-1, 1] float vector (length = len(PARAM_NAMES))."""
    vec = []
    for key, _, (lo, hi) in _PARAMS:
        v = float(params.get(key, PARAM_DEFAULTS[key]))
        vec.append(max(-1.0, min(1.0, (v - lo) / (hi - lo) * 2 - 1)))
    return vec


def vec_to_params(vec: list[float]) -> dict[str, float]:
    """Denormalise [-1, 1] vector → param dict with real Lightroom units."""
    out = {}
    for i, (key, _, (lo, hi)) in enumerate(_PARAMS):
        out[key] = (vec[i] + 1) / 2 * (hi - lo) + lo
    return out


# ── Build color GT ────────────────────────────────────────────────────────────

def build_color_gt(crop_gt_path: Path = CROP_GT_FILE,
                   out_path: Path = COLOR_GT_FILE) -> list[dict]:
    """
    Read XMP params from every edited JPEG in crop_gt.json.
    Output list:  [{"raw": ..., "edited": ..., "params": {...}, "split": ...}, ...]
    """
    with open(crop_gt_path) as fh:
        crop_records = json.load(fh)

    results, missing = [], 0
    for r in crop_records:
        edited = r.get("edited")
        if not edited or not Path(edited).exists():
            missing += 1
            continue
        p = read_xmp_params(edited)
        if p is None:
            missing += 1
            continue
        results.append({
            "raw":        r["raw"],
            "edited":     edited,
            "split":      r["split"],
            "params":     {k: p[k] for k in PARAM_NAMES},
            "tone_curve": p.get("tone_curve"),
        })

    print(f"XMP GT: {len(results):,} records extracted  |  {missing:,} missing XMP")
    if results:
        temps = [r["params"]["Temperature"] for r in results]
        tints = [r["params"]["Tint"] for r in results]
        expos = [r["params"]["Exposure2012"] for r in results]
        print(f"  Temperature: mean={sum(temps)/len(temps):.0f}K  "
              f"range=[{min(temps):.0f}, {max(temps):.0f}]")
        print(f"  Tint:        mean={sum(tints)/len(tints):.1f}  "
              f"range=[{min(tints):.1f}, {max(tints):.1f}]")
        print(f"  Exposure:    mean={sum(expos)/len(expos):.3f}  "
              f"range=[{min(expos):.3f}, {max(expos):.3f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(COLOR_GT_FILE))
    args = parser.parse_args()
    build_color_gt(out_path=Path(args.out))
