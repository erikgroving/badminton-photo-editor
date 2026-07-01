"""Check XMP contents in edited JPGs — full elements, tone curve, across events."""
import json, re
from pathlib import Path

with open("data/crop_gt.json") as f:
    records = json.load(f)

# Sample one from each event folder
seen_events = {}
for r in records:
    event = Path(r["edited"]).parent.name
    if event not in seen_events:
        seen_events[event] = r

print(f"Events: {list(seen_events.keys())}\n")

for event, r in seen_events.items():
    print(f"=== {event} ===")
    with open(r["edited"], "rb") as fh:
        raw = fh.read()

    xmp_start = raw.find(b"<x:xmpmeta")
    if xmp_start < 0:
        print("  NO XMP EMBEDDED\n")
        continue

    xmp_end = raw.find(b"</x:xmpmeta>", xmp_start) + len(b"</x:xmpmeta>")
    xmp = raw[xmp_start:xmp_end].decode("utf-8", errors="replace")

    # Key sliders (attribute form)
    key_tags = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
                "Whites2012", "Blacks2012", "Vibrance", "Saturation", "Clarity2012",
                "Texture", "Dehaze", "Temperature", "Tint", "ToneCurveName2012"]
    attrs = dict(re.findall(r'crs:(\w+)="([^"]*)"', xmp))
    for t in key_tags:
        print(f"  {t} = {attrs.get(t, '(missing)')}")

    # Tone curve points (element form, not attribute)
    curve_match = re.search(r'<crs:ToneCurvePV2012>(.*?)</crs:ToneCurvePV2012>', xmp, re.DOTALL)
    if curve_match:
        points = re.findall(r'<rdf:li>([^<]+)</rdf:li>', curve_match.group(1))
        print(f"  ToneCurvePV2012 ({len(points)} pts): {points[:4]}...")
    else:
        print("  ToneCurvePV2012: (no element found)")

    # Any per-channel curves?
    for chan in ["Red", "Green", "Blue"]:
        m = re.search(rf'<crs:ToneCurvePV2012{chan}>(.*?)</crs:ToneCurvePV2012{chan}>', xmp, re.DOTALL)
        if m:
            pts = re.findall(r'<rdf:li>([^<]+)</rdf:li>', m.group(1))
            print(f"  ToneCurvePV2012{chan} ({len(pts)} pts): {pts[:3]}...")

    # HSL — are any non-zero?
    hsl = {k: v for k, v in attrs.items()
           if any(x in k for x in ["HueAdj", "SatAdj", "LumAdj", "Luminance"]) and v != "0"}
    if hsl:
        print(f"  Non-zero HSL: {hsl}")
    else:
        print("  HSL: all zero")

    print()
