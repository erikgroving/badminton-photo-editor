"""
Color Exp 1: EXIF conditioning (ISO, shutter speed, f-number) → 3 dim.
Hypothesis: ISO tells the model about noise/brightness bias; shutter+aperture
reveal the photographer's exposure intent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.color_base import features_exif, train_conditioned

if __name__ == "__main__":
    train_conditioned(
        exp_name   = "color_exp1_exif",
        feature_fn = features_exif,
        cond_dim   = 3,
    )
