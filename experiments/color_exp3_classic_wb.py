"""
Color Exp 3: Classic WB algorithm outputs from thumbnail → 6 dim.
Gray-world, max-RGB, and p95-based R/G and B/G estimates.
Hypothesis: These classic CV WB estimators are cheap, rotation-invariant
color summaries that complement deep features.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.color_base import features_classic_wb, train_conditioned

if __name__ == "__main__":
    train_conditioned(
        exp_name   = "color_exp3_classic_wb",
        feature_fn = features_classic_wb,
        cond_dim   = 6,
    )
