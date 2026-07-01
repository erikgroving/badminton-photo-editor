import json, random, sys, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, DEVELOP_SIZE
from data.raw_reader import extract_thumbnail_ar
from inference.pipeline import predict_crop
from models.cropping.model import build_crop_model, box_iou_numpy

ckpt_path = CHECKPOINTS_DIR / "cropping_angle_vit_base_patch14_reg4_dinov2_pb.pt"
ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
model = build_crop_model(backbone=ck["backbone"], pretrained=False,
                         use_angle_head=ck["use_angle_head"],
                         use_player_bbox=ck.get("use_player_bbox", False))
model.load_state_dict(ck["model_state"])
model.eval()
model._input_size = ck.get("input_size", 518)
print(f"Loaded ep{ck['epoch']}  val_iou={ck['metrics']['mean_iou']:.4f}")

pb_cache = json.load(open("data/primary_player_bboxes.json"))

recs = [r for r in json.load(open("data/crop_gt.json")) if r["split"] == "test"]
random.seed(99)
samples = random.sample(recs, 15)

W0, H0 = DEVELOP_SIZE
device = torch.device("cpu")
Path("logs/crop_samples").mkdir(exist_ok=True)

print()
for i, r in enumerate(samples):
    raw = r["raw"]
    gt  = r["box"]
    pb  = pb_cache.get(raw)

    pred_box, angle = predict_crop(model, raw, device, player_bbox=pb)
    px, py, pw, ph = pred_box
    p_x1, p_y1 = px/W0, py/H0
    p_x2, p_y2 = (px+pw)/W0, (py+ph)/H0

    iou = box_iou_numpy(np.array([[p_x1, p_y1, p_x2, p_y2]]), np.array([gt]))[0]
    if pb:
        feet_ok = "OK" if p_y2 >= pb[3] else f"CLIP {pb[3]-p_y2:.3f}"
    else:
        feet_ok = "no player"

    print(f"[{i+1}] {Path(raw).name}")
    print(f"     GT:   [{gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f},{gt[3]:.3f}]")
    print(f"     Pred: [{p_x1:.3f},{p_y1:.3f},{p_x2:.3f},{p_y2:.3f}]  angle={angle:.1f}  IoU={iou:.3f}  feet={feet_ok}")
    if pb:
        print(f"     Player: [{pb[0]:.3f},{pb[1]:.3f},{pb[2]:.3f},{pb[3]:.3f}]")

    thumb = extract_thumbnail_ar(raw, max_size=512)
    tw, th = thumb.size
    draw = ImageDraw.Draw(thumb)
    draw.rectangle([gt[0]*tw, gt[1]*th, gt[2]*tw, gt[3]*th], outline="green", width=3)
    draw.rectangle([p_x1*tw, p_y1*th, p_x2*tw, p_y2*th], outline="red", width=2)
    if pb:
        draw.rectangle([pb[0]*tw, pb[1]*th, pb[2]*tw, pb[3]*th], outline="blue", width=2)
    out = f"logs/crop_samples/sample_{i+1}_{Path(raw).stem}.jpg"
    thumb.save(out, quality=85)
    print(f"     -> {out}")
    print()
