"""
Color Exp 5: 16-bin normalised histograms for R, G, B → 48 dim.
Hypothesis: The full per-channel distribution (not just mean) captures
exposure bias, clipping, and colour cast in a compact, shift-invariant way.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.color_base import features_histogram, train_conditioned

if __name__ == "__main__":
    train_conditioned(
        exp_name   = "color_exp5_histogram",
        feature_fn = features_histogram,
        cond_dim   = 48,
    )
