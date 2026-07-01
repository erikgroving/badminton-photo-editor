"""
For each model: what selection rate does unconstrained need to hit ~47% recall?
Compares burst-max-1 (fixed ~29% selection, ~47% recall) vs unconstrained at same recall.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINTS_DIR, CULL_BATCH_SIZE, THUMB_SIZE
from data.mapping import flat_entries, load_mapping
from models.culling.model import build_model
from models.culling.train import CullingDataset

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

CHECKPOINTS = [
    ("efficientnet_b0", "efficientnet_b0"),
    ("efficientnet_b3", "efficientnet_b3"),
    ("siglip_512",      "vit_base_patch16_siglip_512"),
    ("dinov2_base",     "vit_base_patch14_reg4_dinov2"),
    ("dinov2_large",    "vit_large_patch14_dinov2"),
]

# Max recall achieved by burst-max-1 per model (from previous run)
BURST_MAX_RECALL = {
    "efficientnet_b0": 0.47,
    "efficientnet_b3": 0.46,
    "siglip_512":      0.47,
    "dinov2_base":     0.45,
    "dinov2_large":    0.50,
}

def load_val_entries():
    mapping = load_mapping()
    return [e for e in flat_entries(mapping, split="val")
            if e["label"] is not None and e["raw"]]

def get_logits(backbone, ckpt_path, val_entries, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    input_size = ck.get("input_size", THUMB_SIZE[0])
    norm_mean  = ck.get("norm_mean",  _IMAGENET_MEAN)
    norm_std   = ck.get("norm_std",   _IMAGENET_STD)
    resize = [transforms.Resize((input_size, input_size))] if input_size != THUMB_SIZE[0] else []
    tf = transforms.Compose(resize + [
        transforms.ToTensor(),
        transforms.Normalize(list(norm_mean), list(norm_std)),
    ])
    ds = CullingDataset(val_entries, tf)
    loader = DataLoader(ds, batch_size=CULL_BATCH_SIZE * 2,
                        shuffle=False, num_workers=4, pin_memory=True)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_logits.append(model(imgs.to(device)).squeeze(1).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_entries = load_val_entries()
    n_total = len(val_entries)

    # burst-max-1 always selects exactly 1 per burst = 1,711 / 5,949
    n_bursts = 1711
    burst_sel = n_bursts / n_total

    print(f"Val: {n_total:,} photos  |  {n_bursts:,} bursts  |  burst-max-1 selection={burst_sel:.1%}\n")
    print(f"  {'Model':<16}  {'burst max recall':>17}  {'unconstrained sel @ same recall':>32}  {'verdict'}")
    print("  " + "-" * 80)

    for label, backbone in CHECKPOINTS:
        ckpt_path = CHECKPOINTS_DIR / f"culling_{backbone.replace('/', '_')}.pt"
        if not ckpt_path.exists():
            continue

        logits, labels = get_logits(backbone, str(ckpt_path), val_entries, device)
        probs = torch.sigmoid(logits)
        target_recall = BURST_MAX_RECALL[label]

        # Find unconstrained selection at target_recall
        best_t, best_sel, best_actual = None, None, None
        for t in torch.linspace(0.001, 0.999, 500):
            t = float(t)
            preds = (probs >= t).float()
            tp = int(((preds == 1) & (labels == 1)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall >= target_recall:
                sel = float(preds.sum()) / len(preds)
                if best_t is None or t > best_t:
                    best_t, best_sel, best_actual = t, sel, recall

        if best_sel is None:
            unc_str = "N/A"
            verdict = "both bad"
        else:
            unc_str = f"{best_sel:.1%} (actual {best_actual:.0%})"
            if best_sel < burst_sel:
                verdict = f"unconstrained wins by {burst_sel - best_sel:.1%}"
            else:
                verdict = f"burst wins by {best_sel - burst_sel:.1%}"

        print(f"  {label:<16}  {target_recall:.0%} max recall  {unc_str:>32}  {verdict}")

    print(f"\n  burst-max-1 always selects {burst_sel:.1%} of photos (1 per burst, no threshold tuning)")

if __name__ == "__main__":
    main()
