import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TRAINING_DATA_DIR = Path(r"C:\Users\erikg\Downloads\Jay Ma Photography Training")
CACHE_DIR = BASE_DIR / "cache"

def _weights_dir() -> Path:
    """Return the model weights directory, handling PyInstaller frozen apps."""
    if not getattr(sys, 'frozen', False):
        return BASE_DIR / "checkpoints"
    exe = Path(sys.executable)
    if sys.platform == 'darwin' and '.app' in str(exe):
        # Jay Pipeline.app/Contents/MacOS/exe → Contents/Resources/weights
        return exe.parent.parent / 'Resources' / 'weights'
    # Windows onedir: weights/ sits next to the .exe
    return exe.parent / 'weights'

CHECKPOINTS_DIR = _weights_dir()
MAPPING_FILE = BASE_DIR / "data" / "mapping.json"
CROP_GT_FILE      = BASE_DIR / "data" / "crop_gt.json"
CROP_GT_YOLO_FILE = BASE_DIR / "data" / "crop_gt_yolo.json"
COLOR_GT_FILE = BASE_DIR / "data" / "color_gt.json"

# ── Excluded categories (base folder name without " Raws" / " Edited" suffix) ──
# Any folder pair whose base name is in this set is skipped during mapping.
# e.g. "202605 WA Open Audience" — only 24/114 photos matched (different camera)
EXCLUDED_CATEGORIES: set[str] = {"202605 WA Open Audience"}

# ── Per-folder train / val / test split ───────────────────────────────────────
# Each Raws/Edited folder pair is split randomly (seeded) into these fractions.
# This ensures every event and category contributes to all three splits.
SPLIT_SEED       = 42
TRAIN_RATIO      = 0.70   # 70 % train
VAL_RATIO        = 0.15   # 15 % val   (used for threshold tuning / early stopping)
TEST_RATIO       = 0.15   # 15 % test  (final held-out evaluation, never seen during training)

# ── Watermark masking ─────────────────────────────────────────────────────────
# Jay's watermark appears in edited images but NOT in raws, which would let the
# judge model cheat by detecting the watermark instead of learning photo style.
# Set this to (left, top, right, bottom) as fractions of image dimensions.
# The same region is zeroed out in BOTH raws and edited images before any model
# sees them — so neither can exploit watermark presence as a signal.
#
# After running step 1 (mapping --sanity-check 10), open sanity_check/*.jpg,
# note where the watermark is, and set this accordingly.  Examples:
#   Bottom-right 25% wide × 8% tall  →  (0.75, 0.92, 1.0, 1.0)
#   Bottom-center                    →  (0.35, 0.92, 0.65, 1.0)
#   Bottom-left                      →  (0.0,  0.92, 0.25, 1.0)
#
# Leave as None until you have confirmed the watermark position.
WATERMARK_REGION: tuple[float, float, float, float] | None = (0.0, 0.92, 0.60, 1.0)

# ── Raw reader ─────────────────────────────────────────────────────────────────
THUMB_SIZE = (512, 512)
DEVELOP_SIZE = (1536, 1024)   # 3:2 matches Canon sensor AR; no center-crop in _resize_cover
COLOR_SIZE = (512, 512)

# ── Culling model ──────────────────────────────────────────────────────────────
CULL_MODEL_NAME = "efficientnet_b0"
CULL_BACKBONE_CANDIDATES = [
    "efficientnet_b0",
    "efficientnet_b3",
    "resnet50",
    "convnext_small",
    "vit_small_patch16_224",
    "mobilenetv3_large_100",
]
CULL_BATCH_SIZE = 8
CULL_LR = 1e-4
CULL_EPOCHS = 20
CULL_CKPT = CHECKPOINTS_DIR / "culling.pt"
# False-negative weight: cost of wrongly rejecting a "keep" photo vs wrongly keeping a "reject".
# Set > 1 to bias strongly toward recall (never miss a good photo).
# 15 means missing a kept photo is 15× worse than a false alarm.
CULL_FN_WEIGHT = 15.0
# F-beta score beta for evaluation — beta>1 weights recall over precision.
CULL_FBETA = 2.0

# ── Crop model ─────────────────────────────────────────────────────────────────
# Default backbone (used when not sweeping)
CROP_MODEL_NAME = "resnet50"
# All backbones to try in the multi-backbone sweep
CROP_BACKBONE_CANDIDATES = [
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "efficientnet_b4",
    "convnext_small",
    "vit_small_patch16_224",
]
CROP_BATCH_SIZE = 16
CROP_LR = 1e-4
CROP_EPOCHS = 30
CROP_CKPT = CHECKPOINTS_DIR / "cropping.pt"
CROP_MIN_INLIER_RATIO = 0.3   # below this, homography match is discarded

# ── Judge model ────────────────────────────────────────────────────────────────
JUDGE_MODEL_NAME = "efficientnet_b0"
JUDGE_BACKBONE_CANDIDATES = [
    "efficientnet_b0",
    "efficientnet_b3",
    "resnet50",
    "convnext_small",
    "mobilenetv3_large_100",
]
# GAN alternating training: how many generator steps per discriminator step
JUDGE_GAN_D_STEPS = 1   # discriminator updates per generator update
JUDGE_GAN_G_STEPS = 1
JUDGE_BATCH_SIZE = 32
JUDGE_LR = 1e-4
JUDGE_EPOCHS = 15
JUDGE_CKPT = CHECKPOINTS_DIR / "judge.pt"

# ── Color correction models ────────────────────────────────────────────────────
COLOR_PARAM_MODEL_NAME = "efficientnet_b4"
COLOR_PARAM_BATCH_SIZE = 16
COLOR_PARAM_LR = 1e-4
COLOR_PARAM_EPOCHS = 30
COLOR_PARAM_CKPT = CHECKPOINTS_DIR / "color_param.pt"

COLOR_UNET_BATCH_SIZE = 8
COLOR_UNET_LR = 1e-4
COLOR_UNET_EPOCHS = 40
COLOR_UNET_LAMBDA_PIXEL = 1.0
COLOR_UNET_LAMBDA_PERCEPT = 0.1
COLOR_UNET_LAMBDA_JUDGE = 0.05
COLOR_UNET_CKPT = CHECKPOINTS_DIR / "color_unet.pt"

# ── Color correction parameter names (read from XMP embedded in edited JPEGs) ──
# Ground truth comes directly from Lightroom via data/xmp_reader.py — no fitting.
# Jay's main edits: Temperature + Tint (white balance), then minor Exposure/Contrast.
from data.xmp_reader import PARAM_NAMES as COLOR_PARAM_NAMES, PARAM_RANGES as COLOR_PARAM_RANGES
