import json, sys
sys.path.insert(0, ".")
from data.color_analyzer import _process_record

with open("data/crop_gt.json") as f:
    records = json.load(f)

for i in [0, 1, 50, 100]:
    r = records[i]
    result = _process_record(r)
    if result:
        p = result["params"]
        print(f"[{i}] OK  mse={result['mse']:.5f}  exp={p['exposure']:.2f}  "
              f"contrast={p['contrast']:.1f}  sat={p['saturation']:.1f}")
    else:
        # Re-run with exceptions exposed
        try:
            from PIL import Image
            import numpy as np
            from data.raw_reader import develop_raw
            from config import COLOR_SIZE
            x1, y1, x2, y2 = r["box"]
            angle_deg = r.get("angle_deg", 0.0)
            raw_pil = develop_raw(r["raw"], size=COLOR_SIZE)
            print(f"[{i}] raw_pil ok: {raw_pil.size}  angle={angle_deg}")
        except Exception as e:
            print(f"[{i}] FAILED: {e}")
