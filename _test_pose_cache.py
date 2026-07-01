"""Quick test of YOLO pose pipeline and cache script debug."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.raw_reader import extract_thumbnail

def main():
    from ultralytics import YOLO
    m = YOLO("yolo11n-pose.pt")
    print("YOLO loaded OK")

    recs = json.load(open("data/crop_gt.json"))
    tested = 0
    for r in recs[:20]:
        p = Path(r["raw"])
        if not p.exists():
            continue
        img = extract_thumbnail(r["raw"], size=(512, 512))
        res = m(img, classes=[0], verbose=False, device="cpu")
        n_persons = len(res[0].boxes)
        has_kps = res[0].keypoints is not None
        print(f"  OK: {n_persons} persons, kps={has_kps} in {p.name}")
        tested += 1
        if tested >= 3:
            break
    print(f"Tested {tested} images successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
