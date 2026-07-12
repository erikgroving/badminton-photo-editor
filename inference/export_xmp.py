"""
Export predicted color correction parameters as Adobe XMP sidecar files.

Jay's workflow:
  1. Run the pipeline → XMP files are written next to each RAW.
  2. Open Lightroom Classic → Camera Raw automatically picks up the XMP.
  3. All sliders are pre-populated; Jay can fine-tune and export.

XMP keys used (Adobe Camera Raw / Lightroom namespace `crs:`):
    Exposure2012, Contrast2012, Highlights2012, Shadows2012,
    Whites2012, Blacks2012, Temperature, Tint, Saturation

Usage:
    from inference.export_xmp import write_xmp, write_xmp_for_results

    # Write one file
    write_xmp("photo.CR3", {"exposure": 0.5, "contrast": 20, ...})

    # Write from pipeline results list
    write_xmp_for_results([{"raw": "path.CR3", "params": {...}}, ...])
"""
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLOR_PARAM_NAMES, COLOR_PARAM_RANGES

# ── Mapping: our param names → XMP / LR slider names + scale conversion ──────
# Lightroom Classic / Camera Raw slider ranges:
#   Exposure:   -5.0 .. +5.0 (we use -2..+2)
#   Contrast:   -100 .. +100
#   Highlights: -100 .. +100
#   Shadows:    -100 .. +100
#   Whites:     -100 .. +100
#   Blacks:     -100 .. +100
#   Temperature: 2000 .. 50000 K  (delta from a reference)
#   Tint:       -150 .. +150
#   Saturation: -100 .. +100

# Params dicts use XMP names directly (config.COLOR_PARAM_NAMES / vec_to_params
# output) and values are already on the Lightroom scale — keys map 1:1.
# NOTE: the previous map used legacy lowercase keys ("exposure", "temp_shift"),
# which no params dict has contained since the XMP-GT migration — every lookup
# silently defaulted and sidecars came out all-zero.
_XMP_MAP: dict[str, tuple[str, callable]] = {
    "Temperature":         ("Temperature",         lambda v: f"{int(round(v))}"),
    "Tint":                ("Tint",                lambda v: f"{int(round(v))}"),
    "Exposure2012":        ("Exposure2012",        lambda v: f"{v:+.2f}"),
    "Contrast2012":        ("Contrast2012",        lambda v: f"{int(round(v))}"),
    "Highlights2012":      ("Highlights2012",      lambda v: f"{int(round(v))}"),
    "Shadows2012":         ("Shadows2012",         lambda v: f"{int(round(v))}"),
    "Whites2012":          ("Whites2012",          lambda v: f"{int(round(v))}"),
    "Blacks2012":          ("Blacks2012",          lambda v: f"{int(round(v))}"),
    "Vibrance":            ("Vibrance",            lambda v: f"{int(round(v))}"),
    "Saturation":          ("Saturation",          lambda v: f"{int(round(v))}"),
    "Clarity2012":         ("Clarity2012",         lambda v: f"{int(round(v))}"),
    "Texture":             ("Texture",             lambda v: f"{int(round(v))}"),
    "Dehaze":              ("Dehaze",              lambda v: f"{int(round(v))}"),
    "HueAdjustmentYellow": ("HueAdjustmentYellow", lambda v: f"{int(round(v))}"),
}

_XMP_TEMPLATE = """\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
        xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
      <crs:Version>15.0</crs:Version>
      <crs:ProcessVersion>11.0</crs:ProcessVersion>
      <crs:WhiteBalance>Custom</crs:WhiteBalance>
      {sliders}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""


def params_to_xmp_sliders(params: dict) -> str:
    """Convert a params dict to XMP slider XML lines."""
    lines = []
    for param_name, (xmp_key, scale_fn) in _XMP_MAP.items():
        val = params.get(param_name, 0.0)
        lines.append(f"<crs:{xmp_key}>{scale_fn(val)}</crs:{xmp_key}>")
    return "\n      ".join(lines)


def write_xmp(raw_path: Union[str, Path], params: dict,
              out_path: Union[str, Path, None] = None) -> Path:
    """
    Write an XMP sidecar for the given RAW file.

    raw_path: path to the CR3/RAW file
    params:   dict with color correction params (exposure, contrast, ...)
    out_path: optional override for the .xmp output path
              (default: same directory + same stem + .xmp)

    Returns the path to the written XMP file.
    """
    raw_path = Path(raw_path)
    if out_path is None:
        out_path = raw_path.with_suffix(".xmp")
    out_path = Path(out_path)

    sliders = params_to_xmp_sliders(params)
    xmp     = _XMP_TEMPLATE.format(sliders=sliders)
    out_path.write_text(xmp, encoding="utf-8")
    return out_path


def write_xmp_for_results(results: list[dict],
                           out_dir: Union[str, Path, None] = None) -> list[Path]:
    """
    Write XMP files for a list of pipeline results.

    results: list of dicts with keys "raw" and "params"
    out_dir: optional directory for XMP files (default: same dir as each RAW)

    Returns list of written XMP paths.
    """
    written = []
    for r in results:
        raw   = Path(r["raw"])
        out   = Path(out_dir) / (raw.stem + ".xmp") if out_dir else None
        path  = write_xmp(raw, r["params"], out_path=out)
        written.append(path)
    return written


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Write XMP sidecars from a color_gt.json or pipeline output JSON.")
    parser.add_argument("results_json", help="JSON file with list of {raw, params} dicts")
    parser.add_argument("--out-dir",    default=None,
                        help="Directory for XMP files (default: alongside each RAW)")
    args = parser.parse_args()

    with open(args.results_json) as fh:
        results = json.load(fh)

    written = write_xmp_for_results(results, out_dir=args.out_dir)
    print(f"Wrote {len(written)} XMP sidecar files.")
    for p in written[:5]:
        print(f"  {p}")
    if len(written) > 5:
        print(f"  ... and {len(written) - 5} more")
