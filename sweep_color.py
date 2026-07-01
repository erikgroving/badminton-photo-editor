"""
Sweep color correction param model across 6 backbones (2 small / 2 medium / 2 large).
Each backbone gets its own checkpoint. Reports val L1 and test L1 per slider group.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, COLOR_GT_FILE, COLOR_SIZE, COLOR_PARAM_LR, COLOR_PARAM_EPOCHS
from data.xmp_reader import params_to_vec, PARAM_NAMES
from data.split_lookup import build_split_lookup
from models.color_correction.train import ColorDataset
from models.color_correction.param_model import build_param_model

BACKBONES = [
    # (label,            backbone_name,                      batch, epochs, lr,    input_size)
    ("b0 [small]",      "efficientnet_b0",                  32,    20,    1e-4,   512),
    ("mv3 [small]",     "mobilenetv3_large_100",             32,    20,    1e-4,   512),
    ("b3 [medium]",     "efficientnet_b3",                   16,    20,    1e-4,   512),
    ("convnext [med]",  "convnext_small",                    16,    20,    1e-4,   512),
    ("b4 [large]",      "efficientnet_b4",                    8,    20,    5e-5,   512),
    ("dinov2 [large]",  "vit_base_patch14_reg4_dinov2",       4,    20,    5e-5,   518),
]

# Parameter groups for per-group L1 breakdown
PARAM_GROUPS = {
    "white_bal": ["Temperature", "Tint"],
    "exposure":  ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012"],
    "color":     ["Vibrance", "Saturation", "HueAdjustmentYellow"],
    "texture":   ["Clarity2012", "Texture", "Dehaze"],
}


def load_records():
    with open(COLOR_GT_FILE) as fh:
        records = json.load(fh)
    lookup = build_split_lookup()
    train = [r for r in records if lookup.get(r["raw"]) == "train"]
    val   = [r for r in records if lookup.get(r["raw"]) == "val"]
    test  = [r for r in records if lookup.get(r["raw"]) == "test"]
    print(f"Color GT  train={len(train):,}  val={len(val):,}  test={len(test):,}\n")
    return train, val, test


def ckpt_path(backbone_name: str) -> Path:
    safe = backbone_name.replace("/", "_")
    return CHECKPOINTS_DIR / f"color_param_{safe}.pt"


def eval_l1(model, loader, device, maybe_resize=None):
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for raw_t, _, param_vec in loader:
            inp = raw_t.to(device)
            if maybe_resize is not None:
                inp = maybe_resize(inp)
            pred = model(inp).cpu()
            all_pred.append(pred)
            all_gt.append(param_vec)
    pred_cat = torch.cat(all_pred)
    gt_cat   = torch.cat(all_gt)
    total_l1 = float(F.l1_loss(pred_cat, gt_cat).item())

    group_l1 = {}
    for grp, keys in PARAM_GROUPS.items():
        idxs = [PARAM_NAMES.index(k) for k in keys]
        group_l1[grp] = float(F.l1_loss(pred_cat[:, idxs], gt_cat[:, idxs]).item())
    return total_l1, group_l1


def train_one(label, backbone_name, batch_size, epochs, train_recs, val_recs, test_recs, device, lr=COLOR_PARAM_LR, input_size=512):
    print(f"\n{'='*60}")
    print(f"  {label}  ({backbone_name})  batch={batch_size}  lr={lr:.0e}  input={input_size}  epochs={epochs}")
    print(f"{'='*60}")

    train_loader = DataLoader(ColorDataset(train_recs, augment=True),
                              batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ColorDataset(val_recs),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ColorDataset(test_recs),
                              batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Resize raw input to model's expected size if it differs from COLOR_SIZE
    import torch.nn.functional as TF
    def maybe_resize(t):
        if input_size != COLOR_SIZE[0]:
            return TF.interpolate(t, size=(input_size, input_size), mode="bilinear", align_corners=False)
        return t

    model     = build_param_model(pretrained=True, backbone_name=backbone_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_path = ckpt_path(backbone_name)
    best_val = float("inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for raw_t, _, param_vec in tqdm(train_loader, desc=f"  ep{epoch}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss = F.l1_loss(model(maybe_resize(raw_t.to(device))), param_vec.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_l1, val_groups = eval_l1(model, val_loader, device, maybe_resize)
        grp_str = "  ".join(f"{k}={v:.4f}" for k, v in val_groups.items())
        print(f"  ep{epoch:02d}  train={total_loss/len(train_loader):.4f}  val={val_l1:.4f}  [{grp_str}]")

        if val_l1 < best_val:
            best_val = val_l1
            best_metrics = {"backbone": backbone_name, "epoch": epoch,
                            "val_l1": val_l1, "val_groups": val_groups}
            torch.save({"backbone": backbone_name, "epoch": epoch,
                        "model_state": model.state_dict(), "metrics": best_metrics}, out_path)
            print(f"    >> Saved best (val_l1={best_val:.4f})")

    # Reload best checkpoint (last saved epoch may not be best) and run test eval
    saved = torch.load(out_path, map_location=device)
    model.load_state_dict(saved["model_state"])
    test_l1, test_groups = eval_l1(model, test_loader, device, maybe_resize)
    best_metrics["test_l1"]     = test_l1
    best_metrics["test_groups"] = test_groups
    # Persist test metrics back into the checkpoint file
    saved["metrics"] = best_metrics
    torch.save(saved, out_path)
    print(f"  TEST  l1={test_l1:.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in test_groups.items()))
    return best_metrics


def main():
    if not COLOR_GT_FILE.exists():
        print(f"ERROR: {COLOR_GT_FILE} not found. Run: python -m data.xmp_reader --out data/color_gt.json")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_recs, val_recs, test_recs = load_records()

    results = []
    for label, backbone, batch, epochs, lr, input_size in BACKBONES:
        try:
            m = train_one(label, backbone, batch, epochs, train_recs, val_recs, test_recs, device, lr, input_size)
            results.append((label, m))
        except Exception as e:
            print(f"  FAILED {label}: {e}")
            results.append((label, None))
        # Free GPU memory between runs
        torch.cuda.empty_cache()

    print("\n\n" + "="*70)
    print("  SWEEP RESULTS")
    print("="*70)
    hdr = f"  {'Model':<20}  {'Val L1':>8}  {'Test L1':>8}  {'wb':>7}  {'exp':>7}  {'color':>7}  {'texture':>7}  {'epoch':>6}"
    print(hdr)
    print("  " + "-" * 72)
    for label, m in results:
        if m is None:
            print(f"  {label:<20}  FAILED")
            continue
        tg = m.get("test_groups", {})
        print(f"  {label:<20}  {m['val_l1']:>8.4f}  {m['test_l1']:>8.4f}"
              f"  {tg.get('white_bal',0):>7.4f}  {tg.get('exposure',0):>7.4f}"
              f"  {tg.get('color',0):>7.4f}  {tg.get('texture',0):>7.4f}"
              f"  {m['epoch']:>6}")

    best = min((r for _, r in results if r), key=lambda x: x["test_l1"])
    print(f"\n  Winner: {best['backbone']}  (test_l1={best['test_l1']:.4f})")
    print(f"  Checkpoint: checkpoints/color_param_{best['backbone'].replace('/', '_')}.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, default=None,
                    help="Run only the backbone whose label contains this substring (e.g. 'dinov2')")
    args = ap.parse_args()
    if args.label:
        BACKBONES[:] = [b for b in BACKBONES if args.label.lower() in b[0].lower()]
        if not BACKBONES:
            print(f"No backbone matched --label={args.label!r}")
            sys.exit(1)
    main()
