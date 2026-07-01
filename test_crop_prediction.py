"""
Quick test: run the crop model on a given raw file and save a thumbnail
with the predicted crop box overlaid.

Usage:
    python test_crop_prediction.py <raw_path> [--ckpt-tag _pb_test]
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR
from data.raw_reader import extract_thumbnail
from inference.pipeline import predict_crop
from models.cropping.model import build_crop_model


def _load_model(ckpt_tag: str = "_pb_test", backbone: str = "efficientnet_b0") -> tuple:
    head = "angle_"
    ckpt_path = CHECKPOINTS_DIR / f"cropping_{head}{backbone.replace('/', '_')}{ckpt_tag}.pt"
    if not ckpt_path.exists():
        # Try without angle head
        ckpt_path = CHECKPOINTS_DIR / f"cropping_{backbone.replace('/', '_')}{ckpt_tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path.name}")
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = build_crop_model(
        backbone=ck["backbone"], pretrained=False,
        use_angle_head=ck.get("use_angle_head", False),
        use_player_bbox=ck.get("use_player_bbox", False),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    model._input_size = ck.get("input_size", 224)
    m = ck.get("metrics", {})
    print(f"  epoch={ck.get('epoch')}  val mean_iou={m.get('mean_iou', '?'):.4f}"
          f"  use_player_bbox={ck.get('use_player_bbox', False)}")
    return model, ck.get("use_player_bbox", False)


def _get_player_bbox(raw_path: str):
    from config import CHECKPOINTS_DIR, COLOR_SIZE
    from ultralytics import YOLO
    bundled = CHECKPOINTS_DIR / "yolo11n.pt"
    detector = YOLO(str(bundled) if bundled.exists() else "yolo11n.pt")
    img = extract_thumbnail(raw_path, size=COLOR_SIZE)
    w, h = img.size
    min_area = w * h * 0.01
    results = detector(img, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = map(float, box)
            if (x2 - x1) * (y2 - y1) >= min_area:
                boxes.append((x1, y1, x2, y2))
    if not boxes:
        print("  No player detected — model will get zero bbox")
        return None
    px1, py1, px2, py2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    return [px1/w, py1/h, px2/w, py2/h]


def visualize(raw_path: str, ckpt_tag: str = "_pb_test",
              backbone: str = "efficientnet_b0", out_dir: str = "logs"):
    raw_path = str(raw_path)
    model, use_pb = _load_model(ckpt_tag, backbone)
    device = torch.device("cpu")

    player_bbox = _get_player_bbox(raw_path) if use_pb else None
    if player_bbox:
        print(f"  Primary player bbox: [{player_bbox[0]:.3f}, {player_bbox[1]:.3f}, "
              f"{player_bbox[2]:.3f}, {player_bbox[3]:.3f}]")

    from config import DEVELOP_SIZE
    from data.raw_reader import extract_thumbnail_ar
    crop_box, angle = predict_crop(model, raw_path, device, player_bbox=player_bbox)
    # crop_box is [x, y, w, h] in DEVELOP_SIZE pixel space
    W0, H0 = DEVELOP_SIZE
    px0, py0, cw, ch = crop_box
    cn_x1 = px0 / W0
    cn_y1 = py0 / H0
    cn_x2 = (px0 + cw) / W0
    cn_y2 = (py0 + ch) / H0
    print(f"  Predicted crop (norm): [{cn_x1:.3f}, {cn_y1:.3f}, {cn_x2:.3f}, {cn_y2:.3f}]  angle={angle:.1f}°")
    if player_bbox:
        feet_norm = player_bbox[3]
        clipped = feet_norm > cn_y2
        print(f"  Crop bottom={cn_y2:.3f}  player feet={feet_norm:.3f}  "
              f"{'FEET CLIPPED by {:.3f}'.format(feet_norm - cn_y2) if clipped else 'feet included OK'}")

    # Draw on thumbnail
    thumb = extract_thumbnail_ar(raw_path, max_size=512)
    tw, th = thumb.size
    draw = ImageDraw.Draw(thumb)
    draw.rectangle([cn_x1*tw, cn_y1*th, cn_x2*tw, cn_y2*th], outline="red", width=3)
    if player_bbox:
        pb_x1, pb_y1, pb_x2, pb_y2 = player_bbox
        draw.rectangle([pb_x1*tw, pb_y1*th, pb_x2*tw, pb_y2*th], outline="blue", width=2)

    stem = Path(raw_path).stem
    out_path = Path(out_dir) / f"{stem}_crop_preview_{ckpt_tag.strip('_')}.jpg"
    thumb.save(out_path, quality=85)
    print(f"  Saved: {out_path}")
    return crop_box, angle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", help="Path to raw CR3 file")
    parser.add_argument("--ckpt-tag", default="_pb_test")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--out-dir", default="logs")
    args = parser.parse_args()
    visualize(args.raw_path, args.ckpt_tag, args.backbone, args.out_dir)
