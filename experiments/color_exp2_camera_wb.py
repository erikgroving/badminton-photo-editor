"""
Color Exp 2: Camera white balance coefficients from rawpy → 4 dim.
Hypothesis: The camera's measured WB (R/G, B/G ratios vs daylight reference)
is the strongest single prior for predicting Temperature correction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.color_base import features_camera_wb, train_conditioned

if __name__ == "__main__":
    train_conditioned(
        exp_name   = "color_exp2_camera_wb",
        feature_fn = features_camera_wb,
        cond_dim   = 4,
    )
