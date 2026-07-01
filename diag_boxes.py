import sys, torch, json
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from config import DEVELOP_SIZE, CROP_GT_FILE
from data.raw_reader import develop_raw, get_raw_flip, extract_thumbnail_ar
from inference.pipeline import _load_crop, predict_crop, _developed_image_box, _detect_players, _load_yolo

stem = sys.argv[1] if len(sys.argv) > 1 else "1362"
passed = Path(r'C:\Users\erikg\Desktop\JayPipeline\Demo\culls\passed')
cr3 = next(p for p in passed.iterdir() if stem in p.name)
flip = get_raw_flip(cr3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = _load_crop(device)
raw_img = develop_raw(cr3, size=DEVELOP_SIZE, neutral=False)
W, H = raw_img.size

yolo = _load_yolo()

# Match NEW run.py: detect on SN thumbnail → convert to DEV → predict_crop converts back to SN
thumb_sn = extract_thumbnail_ar(cr3, max_size=512)
union_sn, primary_sn = _detect_players(yolo, thumb_sn)
union_dev_in   = _developed_image_box(union_sn,   flip)
primary_dev_in = _developed_image_box(primary_sn, flip)
print(f'YOLO SN union:       {[round(v,3) for v in union_sn]}')
print(f'YOLO SN → DEV union: {[round(v,3) for v in union_dev_in]}')

crop_box, model_angle = predict_crop(model, cr3, device,
    img_size=raw_img.size, player_bbox=union_dev_in, primary_bbox=primary_dev_in)

x, y, w, h = crop_box

DISP_W = 700
scale = DISP_W / W
DISP_H = int(H * scale)
vis = raw_img.resize((DISP_W, DISP_H), Image.LANCZOS)
draw = ImageDraw.Draw(vis)

# Red = predicted crop
px1, py1 = int(x*scale), int(y*scale)
px2, py2 = int((x+w)*scale), int((y+h)*scale)
draw.rectangle([px1, py1, px2, py2], outline=(255, 0, 0), width=4)
draw.text((px1+6, py1+6), f'PRED y2={y+h} ({100*(y+h)/H:.0f}%H)', fill=(255,0,0))

# Blue = YOLO union in DEV space (already in DEV since we detected on portrait)
union_dev_draw = union_dev_in if union_dev_in else [0,0,0,0]
bx1=int(union_dev_draw[0]*W*scale); by1=int(union_dev_draw[1]*H*scale)
bx2=int(union_dev_draw[2]*W*scale); by2=int(union_dev_draw[3]*H*scale)
draw.rectangle([bx1, by1, bx2, by2], outline=(0, 120, 255), width=2)
draw.text((bx1+6, by2-22), 'YOLO union', fill=(0,120,255))

# Green = GT (if exists)
with open(CROP_GT_FILE) as f:
    gt_recs = json.load(f)
gt = next((r for r in gt_recs if stem in r.get('raw', '')), None)
if gt:
    gt_dev = _developed_image_box(gt['box'], flip)
    gx1=int(gt_dev[0]*W*scale); gy1=int(gt_dev[1]*H*scale)
    gx2=int(gt_dev[2]*W*scale); gy2=int(gt_dev[3]*H*scale)
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 220, 0), width=3)
    draw.text((gx1+6, gy1+6), 'GT', fill=(0,220,0))
    print(f'GT (dev): {[round(v,3) for v in gt_dev]}')

print(f'flip={flip}  develop={W}x{H}')
print(f'PRED: x={x} y={y} w={w} h={h}  y_bottom={y+h}  ({100*(y+h)/H:.1f}% of H)')
print(f'YOLO union DEV (portrait): {[round(v,3) for v in union_dev_draw]}')
print(f'model_angle: {model_angle:.1f}  (zeroed for portrait)')

out = f'diag_{stem}_boxes.jpg'
vis.save(out, quality=92)
print(f'Saved {out}')
