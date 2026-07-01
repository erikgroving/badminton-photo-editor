"""
Color Exp 6: Combined — all signals from c1-c5 → 79 dim.
EXIF (3) + camera WB (4) + classic WB (6) + region stats (18) + histogram (48).
Hypothesis: Additive benefit; the model can learn to weight each signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.color_base import features_combined, train_conditioned

if __name__ == "__main__":
    train_conditioned(
        exp_name   = "color_exp6_combined",
        feature_fn = features_combined,
        cond_dim   = 79,
    )
