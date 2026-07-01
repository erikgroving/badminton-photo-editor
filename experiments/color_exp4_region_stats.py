"""
Color Exp 4: 3x3 spatial grid of (brightness, saturation) per cell → 18 dim.
Hypothesis: Where shadows and highlights fall in the frame, and where
saturation is concentrated, informs tone curve and vibrance corrections.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.color_base import features_region_stats, train_conditioned

if __name__ == "__main__":
    train_conditioned(
        exp_name   = "color_exp4_region_stats",
        feature_fn = features_region_stats,
        cond_dim   = 18,
    )
