"""
Train color correction models with proper GAN-style alternating training.

Training loop (alternating D / G):
  D step: judge trained on (real_edited=1, G(raw).detach()=0)
  G step: generator trained on pixel + perceptual + adversarial (fool D) losses

This is more powerful than a fixed judge: the discriminator keeps getting
better at spotting fakes, which forces the generator to improve continuously.

Usage:
    python -m models.color_correction.train --model param   # param model only
    python -m models.color_correction.train --model unet    # U-Net only
    python -m models.color_correction.train --model both    # train both, compare
    python -m models.color_correction.train --judge-backbone convnext_small
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CHECKPOINTS_DIR, COLOR_GT_FILE, COLOR_PARAM_BATCH_SIZE, COLOR_PARAM_CKPT,
    COLOR_PARAM_EPOCHS, COLOR_PARAM_LR, COLOR_SIZE, COLOR_UNET_BATCH_SIZE,
    COLOR_UNET_CKPT, COLOR_UNET_EPOCHS, COLOR_UNET_LAMBDA_JUDGE,
    COLOR_UNET_LAMBDA_PERCEPT, COLOR_UNET_LAMBDA_PIXEL, COLOR_UNET_LR,
    JUDGE_GAN_D_STEPS, JUDGE_MODEL_NAME,
)
from data.xmp_reader import params_to_vec
from data.mapping import load_mapping
from data.raw_reader import develop_raw, mask_watermark
from data.split_lookup import build_split_lookup
from models.color_correction.param_model import build_param_model
from models.color_correction.unet import ColorUNet, _VGGPerceptualLoss
from models.judge.model import build_model as build_judge


def _load_pretrained_judge(backbone: str, device) -> nn.Module | None:
    from config import CHECKPOINTS_DIR
    safe = backbone.replace("/", "_")
    ckpt_path = CHECKPOINTS_DIR / f"judge_{safe}.pt"
    if not ckpt_path.exists():
        print(f"WARNING: No pre-trained judge at {ckpt_path}. Starting judge from scratch.")
        return build_judge(backbone=backbone, pretrained=True).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    judge = build_judge(backbone=backbone, pretrained=False).to(device)
    judge.load_state_dict(ckpt["model_state"])
    print(f"Loaded pre-trained judge ({backbone}) accuracy={ckpt['metrics']['accuracy']:.4f}")
    return judge


# ── Dataset ────────────────────────────────────────────────────────────────────

def _load_player_bboxes() -> dict:
    """Load pre-computed player bbox cache, if available."""
    cache_path = Path(__file__).parent.parent.parent / "data" / "player_bboxes.json"
    if cache_path.exists():
        import json
        return json.load(open(cache_path))
    return {}

_PLAYER_BBOXES: dict = {}  # loaded once on first ColorDataset instantiation


def _crop_to_player(img: Image.Image, bbox_norm: list, pad: float = 0.15) -> Image.Image:
    """Crop image to player union bbox (normalised coords) with padding, then resize."""
    W, H = img.size
    x1n, y1n, x2n, y2n = bbox_norm
    px = (x2n - x1n) * pad
    py = (y2n - y1n) * pad
    x1 = max(0.0, x1n - px)
    y1 = max(0.0, y1n - py)
    x2 = min(1.0, x2n + px)
    y2 = min(1.0, y2n + py)
    crop = img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
    return crop.resize(COLOR_SIZE, Image.LANCZOS)


class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], augment: bool = False, input_size: tuple = COLOR_SIZE):
        global _PLAYER_BBOXES
        self.records = records
        self.augment = augment
        self.input_size = input_size
        if not _PLAYER_BBOXES:
            _PLAYER_BBOXES = _load_player_bboxes()
            if _PLAYER_BBOXES:
                found = sum(1 for v in _PLAYER_BBOXES.values() if v)
                print(f"  [ColorDataset] Player bbox cache: {found}/{len(_PLAYER_BBOXES)} with detections")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        raw_pil    = develop_raw(r["raw"],    size=self.input_size, neutral=False)
        edited_pil = Image.open(r["edited"]).convert("RGB").resize(self.input_size, Image.LANCZOS)

        # Mask watermark in both images so neither model can exploit it
        raw_pil    = mask_watermark(raw_pil)
        edited_pil = mask_watermark(edited_pil)

        # Crop to player region if available — forces model to focus on player colours
        bbox = _PLAYER_BBOXES.get(r["raw"])
        if bbox:
            raw_pil    = _crop_to_player(raw_pil,    bbox)
            edited_pil = _crop_to_player(edited_pil, bbox)
        else:
            raw_pil    = raw_pil.resize(self.input_size,    Image.LANCZOS)
            edited_pil = edited_pil.resize(self.input_size, Image.LANCZOS)

        if self.augment and torch.rand(1).item() > 0.5:
            raw_pil    = raw_pil.transpose(Image.FLIP_LEFT_RIGHT)
            edited_pil = edited_pil.transpose(Image.FLIP_LEFT_RIGHT)

        tf = transforms.ToTensor()
        raw_t    = tf(raw_pil)
        edited_t = tf(edited_pil)
        param_vec = torch.tensor(params_to_vec(r["params"]), dtype=torch.float32)
        return raw_t, edited_t, param_vec


# ── Discriminator update (shared) ─────────────────────────────────────────────

def _update_discriminator(judge, judge_opt, real_imgs, fake_imgs, device):
    """One D step: maximise log D(real) + log(1 - D(fake))."""
    judge.train()
    judge_opt.zero_grad()
    real_logits = judge(real_imgs.to(device)).squeeze(1)
    fake_logits = judge(fake_imgs.detach().to(device)).squeeze(1)
    d_loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))) / 2
    d_loss.backward()
    judge_opt.step()
    return d_loss.item()


# ── Judge score evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def _judge_score(judge, imgs, device) -> float:
    judge.eval()
    return float(torch.sigmoid(judge(imgs.to(device)).squeeze(1)).mean().item())


# ── Param model training ───────────────────────────────────────────────────────

def train_param_model(records_train, records_val, records_test, device, judge, epochs, batch_size, lr, param_backbone=None) -> dict:
    gen       = build_param_model(pretrained=True, backbone_name=param_backbone).to(device)
    # ViT patch size 14 requires H/W divisible by 14; use 518 (14×37) instead of 512
    bb_name = param_backbone or ""
    is_vit = "vit" in bb_name or "swin" in bb_name
    input_size = (518, 518) if is_vit else COLOR_SIZE

    train_loader = DataLoader(ColorDataset(records_train, augment=True,    input_size=input_size),
                              batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ColorDataset(records_val,   input_size=input_size),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ColorDataset(records_test,  input_size=input_size),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    gen_opt   = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=1e-4)
    judge_opt = torch.optim.AdamW(judge.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gen_opt, T_max=epochs)

    COLOR_PARAM_CKPT.parent.mkdir(parents=True, exist_ok=True)
    best_l1 = float("inf")
    best_metrics: dict = {}

    for epoch in range(1, epochs + 1):
        gen.train()
        sum_l1 = 0.0
        for raw_t, edited_t, param_vec in tqdm(train_loader, desc=f"[param] Ep {epoch}/{epochs}", leave=False):
            raw_t = raw_t.to(device)
            edited_t = edited_t.to(device)
            param_vec = param_vec.to(device)

            pred_params = gen(raw_t)

            # Param model predicts LR slider values — no pixel output, so no GAN step.
            # Pure L1 regression on the normalised parameter vector.
            gen_opt.zero_grad()
            l1_loss = F.l1_loss(pred_params, param_vec)
            l1_loss.backward()
            gen_opt.step()
            sum_l1 += l1_loss.item()

        scheduler.step()
        n = len(train_loader)
        print(f"  [param] l1={sum_l1/n:.4f}", end="")

        # Eval on val set
        gen.eval()
        v_l1 = 0.0
        judge_scores = []
        with torch.no_grad():
            for raw_t, edited_t, param_vec in val_loader:
                v_l1 += F.l1_loss(gen(raw_t.to(device)), param_vec.to(device)).item()
                judge_scores.append(_judge_score(judge, raw_t, device))
        v_l1  /= len(val_loader)
        j_score = sum(judge_scores) / len(judge_scores)
        print(f"  [val] l1={v_l1:.4f}  judge={j_score:.3f}")

        if v_l1 < best_l1:
            best_l1      = v_l1
            # "val_l1" + top-level "backbone"/"player_crop" are REQUIRED by
            # inference/pipeline._load_color_param — checkpoints without them
            # crash the loader ('?' formatted as float) or rebuild the wrong
            # backbone from None.
            best_metrics = {"model": "param", "val_l1": v_l1, "param_l1": v_l1,
                            "judge_score": j_score, "epoch": epoch}
            torch.save({"epoch": epoch, "model_state": gen.state_dict(),
                        "backbone": param_backbone,
                        "player_crop": True,
                        "judge_state": judge.state_dict(),
                        "metrics": best_metrics}, COLOR_PARAM_CKPT)
            print(f"    [OK] Saved best param model (val_l1={best_l1:.4f})")

    # Final test evaluation
    t_l1 = 0.0
    t_judge = []
    with torch.no_grad():
        for raw_t, edited_t, param_vec in test_loader:
            t_l1 += F.l1_loss(gen(raw_t.to(device)), param_vec.to(device)).item()
            t_judge.append(_judge_score(judge, raw_t, device))
    t_l1 /= len(test_loader)
    print(f"  [test] l1={t_l1:.4f}  judge={sum(t_judge)/len(t_judge):.3f}")
    best_metrics["test_metrics"] = {"param_l1": t_l1, "judge_score": sum(t_judge) / len(t_judge)}
    return best_metrics


# ── U-Net training ─────────────────────────────────────────────────────────────

def train_unet(records_train, records_val, records_test, device, judge, epochs, batch_size, lr) -> dict:
    train_loader = DataLoader(ColorDataset(records_train, augment=True),
                              batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ColorDataset(records_val),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ColorDataset(records_test),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    gen         = ColorUNet(pretrained=True).to(device)
    perceptual  = _VGGPerceptualLoss().to(device)
    gen_opt     = torch.optim.AdamW(gen.parameters(),   lr=lr,       weight_decay=1e-4)
    judge_opt   = torch.optim.AdamW(judge.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(gen_opt, T_max=epochs)

    COLOR_UNET_CKPT.parent.mkdir(parents=True, exist_ok=True)
    best_l1 = float("inf")
    best_metrics: dict = {}

    for epoch in range(1, epochs + 1):
        gen.train()
        sum_pixel = sum_percept = sum_adv = sum_d = 0.0
        for raw_t, edited_t, _ in tqdm(train_loader, desc=f"[unet]  Ep {epoch}/{epochs}", leave=False):
            raw_t    = raw_t.to(device)
            edited_t = edited_t.to(device)

            fake = gen(raw_t)

            # D step (update judge on real edited vs fake generated)
            for _ in range(JUDGE_GAN_D_STEPS):
                d_loss = _update_discriminator(judge, judge_opt, edited_t, fake, device)
            sum_d += d_loss

            # G step
            gen_opt.zero_grad()
            pixel_loss   = COLOR_UNET_LAMBDA_PIXEL   * F.l1_loss(fake, edited_t)
            percept_loss = COLOR_UNET_LAMBDA_PERCEPT  * perceptual(fake, edited_t)
            adv_loss     = COLOR_UNET_LAMBDA_JUDGE    * F.binary_cross_entropy_with_logits(
                judge(fake).squeeze(1), torch.ones(fake.size(0), device=device))
            loss = pixel_loss + percept_loss + adv_loss
            loss.backward()
            gen_opt.step()
            sum_pixel   += pixel_loss.item()
            sum_percept += percept_loss.item()
            sum_adv     += adv_loss.item()

        scheduler.step()
        n = len(train_loader)
        print(f"  [unet] px={sum_pixel/n:.4f} pc={sum_percept/n:.4f} adv={sum_adv/n:.4f} d={sum_d/n:.4f}", end="")

        # Eval on val set
        gen.eval()
        v_l1 = 0.0
        judge_scores = []
        with torch.no_grad():
            for raw_t, edited_t, _ in val_loader:
                fake = gen(raw_t.to(device))
                v_l1 += F.l1_loss(fake, edited_t.to(device)).item()
                judge_scores.append(_judge_score(judge, fake, device))
        v_l1  /= len(val_loader)
        j_score = sum(judge_scores) / len(judge_scores)
        print(f"  [val] l1={v_l1:.4f}  judge={j_score:.3f}")

        if v_l1 < best_l1:
            best_l1      = v_l1
            best_metrics = {"model": "unet", "pixel_l1": v_l1, "judge_score": j_score, "epoch": epoch}
            torch.save({"epoch": epoch, "model_state": gen.state_dict(),
                        "judge_state": judge.state_dict(), "metrics": best_metrics}, COLOR_UNET_CKPT)
            print(f"    [OK] Saved best U-Net (val_l1={best_l1:.4f})")

    # Final test evaluation
    t_l1 = 0.0
    t_judge = []
    with torch.no_grad():
        for raw_t, edited_t, _ in test_loader:
            fake = gen(raw_t.to(device))
            t_l1 += F.l1_loss(fake, edited_t.to(device)).item()
            t_judge.append(_judge_score(judge, fake, device))
    t_l1 /= len(test_loader)
    print(f"  [test] l1={t_l1:.4f}  judge={sum(t_judge)/len(t_judge):.3f}")
    best_metrics["test_metrics"] = {"pixel_l1": t_l1, "judge_score": sum(t_judge) / len(t_judge)}
    return best_metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def train(model_choice: str, judge_backbone: str, **kwargs) -> None:
    if not COLOR_GT_FILE.exists():
        raise FileNotFoundError(f"Run data/color_analyzer.py first — {COLOR_GT_FILE} not found.")

    with open(COLOR_GT_FILE) as fh:
        records = json.load(fh)

    split_lookup  = build_split_lookup()
    train_records = [r for r in records if split_lookup.get(r["raw"]) == "train"]
    val_records   = [r for r in records if split_lookup.get(r["raw"]) == "val"]
    test_records  = [r for r in records if split_lookup.get(r["raw"]) == "test"]
    print(f"Color GT  train: {len(train_records):,}  val: {len(val_records):,}  test: {len(test_records):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Each training run gets its own unfrozen judge copy (so they train independently)
    results = {}
    if model_choice in ("param", "both"):
        print("\n--- Param model + judge ---")
        judge = _load_pretrained_judge(judge_backbone, device)
        results["param"] = train_param_model(
            train_records, val_records, test_records, device, judge,
            kwargs.get("param_epochs",    COLOR_PARAM_EPOCHS),
            kwargs.get("param_batch",     COLOR_PARAM_BATCH_SIZE),
            kwargs.get("param_lr",        COLOR_PARAM_LR),
            param_backbone=kwargs.get("param_backbone", None),
        )

    if model_choice in ("unet", "both"):
        print("\n--- U-Net + judge ---")
        judge = _load_pretrained_judge(judge_backbone, device)
        results["unet"] = train_unet(
            train_records, val_records, test_records, device, judge,
            kwargs.get("unet_epochs", COLOR_UNET_EPOCHS),
            kwargs.get("unet_batch",  COLOR_UNET_BATCH_SIZE),
            kwargs.get("unet_lr",     COLOR_UNET_LR),
        )

    if len(results) == 2:
        print("\n--- Final comparison (judge score = % looking like Jay's edits) ---")
        print(f"{'Model':<10} {'Judge Score':>12} {'Pixel L1':>10}")
        print("-" * 36)
        for name, m in results.items():
            l1 = m.get("param_l1") or m.get("pixel_l1", 0)
            print(f"{name:<10} {m.get('judge_score',0):>12.3f} {l1:>10.4f}")
        winner = max(results, key=lambda k: results[k].get("judge_score", 0))
        print(f"\nWinner: {winner}  (will be used as default in UI)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          choices=["param", "unet", "both"], default="both")
    parser.add_argument("--judge-backbone", type=str,   default=JUDGE_MODEL_NAME)
    parser.add_argument("--param-backbone", type=str,   default=None)
    parser.add_argument("--param-epochs",   type=int,   default=COLOR_PARAM_EPOCHS)
    parser.add_argument("--unet-epochs",    type=int,   default=COLOR_UNET_EPOCHS)
    parser.add_argument("--param-batch",    type=int,   default=COLOR_PARAM_BATCH_SIZE)
    parser.add_argument("--unet-batch",     type=int,   default=COLOR_UNET_BATCH_SIZE)
    parser.add_argument("--param-lr",       type=float, default=COLOR_PARAM_LR)
    parser.add_argument("--unet-lr",        type=float, default=COLOR_UNET_LR)
    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if k not in ("model", "judge_backbone")}
    train(args.model, args.judge_backbone, **kwargs)
