"""
Crop-quality GAN fine-tuning.

Phase 1: Pretrain crop judge (discriminator) to distinguish real GT crops
         from random/bad crops of the same raw thumbnails.

Phase 2: GAN fine-tune the crop model.
         G = crop model (dinov2 + angle head)
         D = crop judge
         G loss = (1-lam) * supervised(SmoothL1+IoU) + lam * adversarial
         D loss = BCE(real GT crops=1, G(raw).stop_grad()=0)

The crop judge operates on box crops extracted from raw thumbnails (no rotation —
composition is judged on the content of the selected region, orientation aside).

Usage:
    python train_crop_gan.py                       # both phases
    python train_crop_gan.py --skip-pretrain       # GAN only (use existing judge ckpt)
    python train_crop_gan.py --judge-only          # pretrain judge only
    python train_crop_gan.py --judge-backbone efficientnet_b3
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CROP_GT_FILE
from data.raw_reader import extract_thumbnail_ar
from models.cropping.model import ANGLE_SCALE, CropLoss, build_crop_model

# ── Constants ─────────────────────────────────────────────────────────────────
JUDGE_CKPT        = CHECKPOINTS_DIR / "crop_judge.pt"
GAN_CKPT          = CHECKPOINTS_DIR / "cropping_angle_dinov2_gan.pt"
BASE_CKPT         = CHECKPOINTS_DIR / "cropping_angle_vit_base_patch14_reg4_dinov2.pt"

JUDGE_EPOCHS      = 10
GAN_EPOCHS        = 15
JUDGE_BATCH       = 32
GAN_BATCH         = 8          # dinov2 is big
JUDGE_LR          = 1e-4
GAN_LR            = 1e-5
ADV_LAMBDA        = 0.05       # adversarial weight vs supervised
CROP_SIZE         = 224        # judge input patch size
EXTRACT_SIZE      = 512        # raw thumbnail size (must match crop training)


# ── Crop judge model ──────────────────────────────────────────────────────────

def build_judge(backbone: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    return timm.create_model(backbone, pretrained=pretrained, num_classes=1)


# ── Differentiable crop extraction ────────────────────────────────────────────

def extract_crop_differentiable(img: torch.Tensor, box: torch.Tensor,
                                out_size: int = CROP_SIZE) -> torch.Tensor:
    """
    Differentiably extract box crops from images using affine_grid + grid_sample.

    img: (B, C, H, W) float tensor
    box: (B, 4) [x1, y1, x2, y2] in [0, 1], must be on same device as img
    returns: (B, C, out_size, out_size)

    Rotation is intentionally omitted — the judge scores composition (framing),
    not orientation. The supervised angle loss handles rotation quality separately.
    """
    B, C, H, W = img.shape
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    # Convert box center to affine [-1, 1] space: p_norm = p * 2 - 1
    cx  = (x1 + x2) - 1.0      # == (x1+x2)/2 * 2 - 1
    cy  = (y1 + y2) - 1.0
    sx  = x2 - x1               # width  in [0, 1]
    sy  = y2 - y1               # height in [0, 1]
    z   = torch.zeros_like(cx)

    theta = torch.stack([sx, z, cx, z, sy, cy], dim=1).view(B, 2, 3)
    grid  = F.affine_grid(theta, (B, C, out_size, out_size), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)


# ── PIL-based crop (for judge pretraining dataset) ────────────────────────────

def pil_crop(img_pil: Image.Image, box: list[float]) -> Image.Image:
    """Extract box crop from a PIL image. box = [x1, y1, x2, y2] in [0, 1]."""
    w, h  = img_pil.size
    x1, y1, x2, y2 = box
    left  = int(x1 * w)
    upper = int(y1 * h)
    right = int(x2 * w)
    lower = int(y2 * h)
    right  = max(right,  left + 1)
    lower  = max(lower,  upper + 1)
    return img_pil.crop((left, upper, right, lower))


def random_box() -> list[float]:
    """Generate a random crop box in [0, 1] that is plausibly valid."""
    rng = random
    x1 = rng.uniform(0.0, 0.4)
    y1 = rng.uniform(0.0, 0.4)
    x2 = rng.uniform(x1 + 0.2, min(x1 + 0.8, 1.0))
    y2 = rng.uniform(y1 + 0.2, min(y1 + 0.8, 1.0))
    return [x1, y1, x2, y2]


# ── Judge pretraining dataset ─────────────────────────────────────────────────

class CropJudgeDataset(Dataset):
    """
    Positive: GT crop from raw thumbnail (label=1)
    Negative: random crop from same raw thumbnail (label=0)
    Records are from crop_gt.json (same split-aware list as crop training).
    """
    def __init__(self, records: list[dict], tf, augment: bool = False):
        self.records = records
        self.tf      = tf
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records) * 2   # one real + one fake per record

    def __getitem__(self, idx: int):
        r      = self.records[idx % len(self.records)]
        is_real = idx < len(self.records)   # first half = real, second half = fake

        img = extract_thumbnail_ar(r["raw"], max_size=EXTRACT_SIZE)

        if is_real:
            box  = r["box"]
            crop = pil_crop(img, box)
            label = 1.0
        else:
            # Random box — very likely to be a "bad" crop relative to Jay's choice
            crop  = pil_crop(img, random_box())
            label = 0.0

        if self.augment and random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

        crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
        return self.tf(crop), torch.tensor(label, dtype=torch.float32)


# ── GAN dataset (raw thumbnails + GT targets for supervised loss) ─────────────

class GANDataset(Dataset):
    def __init__(self, records: list[dict], tf):
        self.records = records
        self.tf      = tf

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r         = self.records[idx]
        img       = extract_thumbnail_ar(r["raw"], max_size=EXTRACT_SIZE)
        img_t     = self.tf(img)
        box_t     = torch.tensor(r["box"],        dtype=torch.float32)
        angle_t   = torch.tensor(r.get("angle_deg", 0.0) / ANGLE_SCALE,
                                  dtype=torch.float32)
        return img_t, box_t, angle_t


# ── Judge pretraining ─────────────────────────────────────────────────────────

def pretrain_judge(records_train: list[dict], records_val: list[dict],
                   backbone: str, epochs: int, batch_size: int, lr: float) -> nn.Module:
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Crop judge pretraining ({backbone})")
    print("="*60)

    import timm
    data_cfg   = timm.data.resolve_data_config({}, model=backbone)
    input_size = data_cfg.get("input_size", (3, CROP_SIZE, CROP_SIZE))[1]
    norm_mean  = list(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    norm_std   = list(data_cfg.get("std",  (0.229, 0.224, 0.225)))

    tf_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    nw           = min(4, batch_size)
    train_loader = DataLoader(CropJudgeDataset(records_train, tf_train, augment=True),
                              batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(CropJudgeDataset(records_val,   tf_val),
                              batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge     = build_judge(backbone, pretrained=True).to(device)
    opt       = torch.optim.AdamW(judge.parameters(), lr=lr, weight_decay=1e-4)
    sched     = CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    JUDGE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        judge.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[judge] ep{epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(judge(imgs).squeeze(1), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        judge.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = (torch.sigmoid(judge(imgs.to(device)).squeeze(1).cpu()) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total   += len(labels)
        acc = correct / total
        print(f"  [judge] ep{epoch:02d}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": epoch, "backbone": backbone,
                        "model_state": judge.state_dict(),
                        "metrics": {"accuracy": acc}}, JUDGE_CKPT)
            print(f"    [OK] Saved best judge (val_acc={best_acc:.4f})")

    ck = torch.load(JUDGE_CKPT, map_location="cpu", weights_only=False)
    judge.load_state_dict(ck["model_state"])
    judge.to(device)
    print(f"\nJudge pretrained — best val accuracy: {best_acc:.4f}")
    return judge


# ── GAN fine-tuning ───────────────────────────────────────────────────────────

def gan_finetune(records_train: list[dict], records_val: list[dict],
                 judge: nn.Module, epochs: int, batch_size: int, lr: float,
                 adv_lambda: float) -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Crop GAN fine-tuning  (adv_lambda={adv_lambda})")
    print("="*60)

    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"Base crop checkpoint not found: {BASE_CKPT}\n"
                                "Run sweep_crop_with_rotation.py --sizes large first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base crop model
    gen = build_crop_model(backbone="vit_base_patch14_reg4_dinov2",
                           pretrained=True, use_angle_head=True)
    gen.set_grad_checkpointing(enable=True)
    ck  = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    gen.load_state_dict(ck["model_state"])
    gen = gen.to(device)
    print(f"  Loaded base crop model (val IoU={ck['metrics']['mean_iou']:.4f}  "
          f"angle_mae={ck['metrics'].get('angle_mae_deg', float('nan')):.2f}°)")

    judge = judge.to(device)

    # Transforms matching crop training (same as models/cropping/train.py)
    import timm as _timm
    data_cfg  = _timm.data.resolve_model_data_config(gen.backbone)
    inp_size  = data_cfg.get("input_size", (3, 224, 224))[1]
    norm_mean = list(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    norm_std  = list(data_cfg.get("std",  (0.229, 0.224, 0.225)))
    tf = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # The judge expects unnormalized [0, 1] crops — we need a separate normalizer
    import timm as _timm2
    jck       = torch.load(JUDGE_CKPT, map_location="cpu", weights_only=False)
    j_cfg     = _timm2.data.resolve_data_config({}, model=jck["backbone"])
    j_inp     = j_cfg.get("input_size", (3, CROP_SIZE, CROP_SIZE))[1]
    j_mean    = torch.tensor(list(j_cfg.get("mean", (0.485, 0.456, 0.406))),
                              device=device).view(1, 3, 1, 1)
    j_std     = torch.tensor(list(j_cfg.get("std",  (0.229, 0.224, 0.225))),
                              device=device).view(1, 3, 1, 1)

    nw           = min(4, batch_size)
    train_loader = DataLoader(GANDataset(records_train, tf),
                              batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(GANDataset(records_val, tf),
                              batch_size=batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True)

    criterion  = CropLoss(alpha=0.5, angle_weight=0.25)
    gen_opt    = torch.optim.AdamW(gen.parameters(),   lr=lr,       weight_decay=1e-4)
    judge_opt  = torch.optim.AdamW(judge.parameters(), lr=lr * 0.5, weight_decay=1e-4)
    gen_sched  = CosineAnnealingLR(gen_opt,   T_max=epochs)
    judge_sched= CosineAnnealingLR(judge_opt, T_max=epochs)

    GAN_CKPT.parent.mkdir(parents=True, exist_ok=True)
    best_iou = ck["metrics"]["mean_iou"]

    def _judge_crop(imgs_norm: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Extract crops from generator-normalized images and score them."""
        # Undo generator normalization to get [0, 1] range
        gm = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
        gs = torch.tensor(norm_std,  device=device).view(1, 3, 1, 1)
        imgs_01 = (imgs_norm * gs + gm).clamp(0, 1)
        # Extract box crop
        patch = extract_crop_differentiable(imgs_01, box, out_size=j_inp)
        # Apply judge normalization
        patch_j = (patch - j_mean) / j_std
        return judge(patch_j).squeeze(1)

    for epoch in range(1, epochs + 1):
        gen.train()
        judge.train()
        sum_sup = sum_adv = sum_d = 0.0
        n_batches = len(train_loader)

        for imgs, boxes, angle_norms in tqdm(train_loader,
                                              desc=f"[crop-gan] ep{epoch}/{epochs}", leave=False):
            imgs, boxes, angle_norms = imgs.to(device), boxes.to(device), angle_norms.to(device)

            # Forward: get predicted box + angle
            box_pred, angle_pred = gen(imgs)

            # ── D step ─────────────────────────────────────────────────────────
            judge_opt.zero_grad()
            real_logits = _judge_crop(imgs, boxes)
            fake_logits = _judge_crop(imgs, box_pred.detach())
            d_loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                    + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))) * 0.5
            d_loss.backward()
            judge_opt.step()
            sum_d += d_loss.item()

            # ── G step ─────────────────────────────────────────────────────────
            gen_opt.zero_grad()
            sup_loss = criterion(box_pred, boxes, angle_pred, angle_norms)
            adv_logits = _judge_crop(imgs, box_pred)
            adv_loss   = F.binary_cross_entropy_with_logits(adv_logits,
                                                             torch.ones_like(adv_logits))
            total = (1.0 - adv_lambda) * sup_loss + adv_lambda * adv_loss
            total.backward()
            gen_opt.step()
            sum_sup += sup_loss.item()
            sum_adv += adv_loss.item()

        gen_sched.step()
        judge_sched.step()

        # ── Validation ─────────────────────────────────────────────────────────
        gen.eval()
        import numpy as np
        from models.cropping.model import box_iou_numpy
        all_pred, all_gt, all_ap, all_at = [], [], [], []
        with torch.no_grad():
            for imgs, boxes, angle_norms in val_loader:
                bp, ap = gen(imgs.to(device))
                all_pred.append(bp.cpu().numpy())
                all_gt.append(boxes.numpy())
                all_ap.append(ap.cpu().numpy())
                all_at.append(angle_norms.numpy())
        pred_arr  = np.concatenate(all_pred)
        gt_arr    = np.concatenate(all_gt)
        ious      = box_iou_numpy(pred_arr, gt_arr)
        mean_iou  = float(ious.mean())
        angle_mae = float(np.mean(np.abs(
            np.concatenate(all_ap) * ANGLE_SCALE - np.concatenate(all_at) * ANGLE_SCALE)))
        print(f"  ep{epoch:02d}  sup={sum_sup/n_batches:.4f}  adv={sum_adv/n_batches:.4f}"
              f"  d={sum_d/n_batches:.4f}  [val] iou={mean_iou:.4f}  angle_mae={angle_mae:.2f}°")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({
                "epoch":          epoch,
                "backbone":       "vit_base_patch14_reg4_dinov2",
                "use_angle_head": True,
                "model_state":    gen.state_dict(),
                "metrics":        {"mean_iou": mean_iou, "angle_mae_deg": angle_mae},
                "input_size":     inp_size,
                "norm_mean":      tuple(norm_mean),
                "norm_std":       tuple(norm_std),
                "angle_scale":    ANGLE_SCALE,
            }, GAN_CKPT)
            print(f"    [OK] Saved GAN checkpoint (val_iou={best_iou:.4f})")

    print(f"\nGAN fine-tuning complete — best val IoU: {best_iou:.4f}")
    if best_iou > ck["metrics"]["mean_iou"]:
        print(f"  Improvement over base: +{best_iou - ck['metrics']['mean_iou']:.4f}")
    else:
        print(f"  No improvement — base checkpoint remains best (use {BASE_CKPT.name})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if not CROP_GT_FILE.exists():
        raise FileNotFoundError(f"Run data/crop_detector.py first — {CROP_GT_FILE} not found.")

    with open(CROP_GT_FILE) as fh:
        all_records = json.load(fh)

    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs   = [r for r in all_records if r["split"] == "val"]
    print(f"Dataset: {len(train_recs):,} train / {len(val_recs):,} val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.judge_only or not args.skip_pretrain:
        judge = pretrain_judge(train_recs, val_recs,
                               backbone=args.judge_backbone,
                               epochs=args.judge_epochs,
                               batch_size=JUDGE_BATCH,
                               lr=JUDGE_LR)
    else:
        if not JUDGE_CKPT.exists():
            raise FileNotFoundError(f"--skip-pretrain but no judge ckpt at {JUDGE_CKPT}")
        print(f"Loading existing judge from {JUDGE_CKPT}")
        ck    = torch.load(JUDGE_CKPT, map_location="cpu", weights_only=False)
        judge = build_judge(ck["backbone"], pretrained=False)
        judge.load_state_dict(ck["model_state"])
        print(f"  val_acc={ck['metrics']['accuracy']:.4f}")

    if args.judge_only:
        return

    gan_finetune(train_recs, val_recs, judge,
                 epochs=args.gan_epochs,
                 batch_size=GAN_BATCH,
                 lr=GAN_LR,
                 adv_lambda=args.adv_lambda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-backbone", default="efficientnet_b3")
    parser.add_argument("--judge-epochs",   type=int,   default=JUDGE_EPOCHS)
    parser.add_argument("--gan-epochs",     type=int,   default=GAN_EPOCHS)
    parser.add_argument("--adv-lambda",     type=float, default=ADV_LAMBDA)
    parser.add_argument("--skip-pretrain",  action="store_true")
    parser.add_argument("--judge-only",     action="store_true")
    args = parser.parse_args()
    main(args)
