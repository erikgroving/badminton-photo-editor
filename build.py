"""
Build script for Badminton AI Photo Editor.

Run on each platform to produce its binary:
  Windows  ->  dist/Badminton AI Photo Editor/Badminton AI Photo Editor.exe
               (+ dist/BadmintonAIPhotoEditor-Windows.zip)
  Mac      ->  dist/Badminton AI Photo Editor.app
               (+ dist/BadmintonAIPhotoEditor-Mac.dmg)

Usage:
    python build.py            # full build
    python build.py --no-zip   # skip zip/dmg step (faster, for iteration)

Weights are copied to the right location inside/alongside the bundle so the
app is self-contained:
  Windows: dist/Badminton AI Photo Editor/weights/
  Mac:     dist/Badminton AI Photo Editor.app/Contents/Resources/weights/
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT      = Path(__file__).parent
DIST      = ROOT / 'dist'
CKPTS_DIR = ROOT / 'checkpoints'

APP_NAME  = 'Badminton AI Photo Editor'
SPEC_FILE = 'BadmintonAIPhotoEditor.spec'

# Exactly the checkpoints the app loads at runtime — nothing else ships.
#   culling:  _load_cull picks best asym_cost among culling_*.pt
#   crop:     _load_crop prefers exp9d_full (test IoU 0.8777, angle MAE 0.74 deg)
#   color:    _load_color_param picks player-crop-trained best val_l1 -> b3
#   yolo11n:  player detection for crop conditioning
#   yolo11n-pose: pose keypoints for exp9d crop conditioning
SHIP_WEIGHTS = [
    'culling_vit_large_patch14_dinov2.pt',
    'cropping_angle_vit_large_patch14_reg4_dinov2_exp9d_full.pt',
    'color_affine_efficientnet_b0.pt',   # production color model (dE2000 5.53)
    'color_param_efficientnet_b3.pt',    # legacy slider fallback + not-yet-migrated paths
    'yolo11n.pt',
    'yolo11n-pose.pt',
]

# Keys stripped from checkpoints on copy — training-resume state the app
# never reads (the exp9 checkpoints carry ~2.4 GB of optimizer state).
_TRAIN_STATE_KEYS = ('optimizer_state', 'scheduler_state')


def collect_weights() -> list[Path]:
    """Return the checkpoint files to include in the distribution."""
    weights, missing = [], []
    for name in SHIP_WEIGHTS:
        p = CKPTS_DIR / name
        (weights if p.exists() else missing).append(p)
    if missing:
        print('ERROR: required weights missing:')
        for p in missing:
            print(f'  {p}')
        sys.exit(1)
    total_mb = sum(p.stat().st_size for p in weights) / 1024**2
    print(f'Weights to bundle: {len(weights)} files  ({total_mb:.0f} MB pre-strip)')
    for w in weights:
        print(f'  {w.name}')
    return weights


def _copy_weight(src: Path, dst: Path):
    """Copy a checkpoint, stripping training-resume state if present."""
    import torch
    try:
        d = torch.load(src, map_location='cpu', weights_only=False)
    except Exception:
        shutil.copy2(src, dst)          # non-dict formats (e.g. YOLO) copy as-is
        return
    if isinstance(d, dict) and any(k in d for k in _TRAIN_STATE_KEYS):
        for k in _TRAIN_STATE_KEYS:
            d.pop(k, None)
        torch.save(d, dst)
        print(f'  {src.name}: stripped train state '
              f'({src.stat().st_size / 1024**2:.0f} -> {dst.stat().st_size / 1024**2:.0f} MB)')
    else:
        shutil.copy2(src, dst)


def copy_weights(weights: list[Path]):
    if sys.platform == 'darwin':
        dst = DIST / f'{APP_NAME}.app' / 'Contents' / 'Resources' / 'weights'
    else:
        dst = DIST / APP_NAME / 'weights'
    dst.mkdir(parents=True, exist_ok=True)
    for w in weights:
        _copy_weight(w, dst / w.name)
    print(f'Weights copied -> {dst}')


def build_pyinstaller():
    print('\n-- Running PyInstaller ----------------------------------')
    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', SPEC_FILE],
        cwd=ROOT,
    )
    # PyInstaller exits 1 for non-fatal "Hidden import not found" warnings on
    # deprecated torch.distributed._shard submodules; the bundle still builds.
    # Only abort if the expected output artifact is missing.
    bundle = DIST / APP_NAME / f'{APP_NAME}.exe'
    if sys.platform == 'darwin':
        bundle = DIST / f'{APP_NAME}.app'
    if not bundle.exists():
        print(f'ERROR: PyInstaller exited {result.returncode} and bundle not found at {bundle}')
        sys.exit(1)


def zip_windows():
    zip_path = DIST / 'BadmintonAIPhotoEditor-Windows.zip'
    src      = DIST / APP_NAME
    print(f'\n-- Creating {zip_path.name} ------------------------------')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in src.rglob('*'):
            if f.is_file():
                zf.write(f, Path(APP_NAME) / f.relative_to(src))
    size_mb = zip_path.stat().st_size / 1024**2
    print(f'Created {zip_path}  ({size_mb:.0f} MB)')


def dmg_mac():
    app     = DIST / f'{APP_NAME}.app'
    dmg_dir = DIST / '_dmg_stage'
    dmg_out = DIST / 'BadmintonAIPhotoEditor-Mac.dmg'

    print(f'\n-- Creating {dmg_out.name} ------------------------------')
    if dmg_dir.exists():
        shutil.rmtree(dmg_dir)
    dmg_dir.mkdir()

    # Symlink to /Applications so the DMG has a drag-to-install target
    shutil.copytree(app, dmg_dir / app.name, symlinks=True)
    (dmg_dir / 'Applications').symlink_to('/Applications')

    if dmg_out.exists():
        dmg_out.unlink()

    subprocess.run([
        'hdiutil', 'create',
        '-volname', APP_NAME,
        '-srcfolder', str(dmg_dir),
        '-ov', '-format', 'UDZO',
        str(dmg_out),
    ], check=True)

    shutil.rmtree(dmg_dir)
    size_mb = dmg_out.stat().st_size / 1024**2
    print(f'Created {dmg_out}  ({size_mb:.0f} MB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-zip', action='store_true',
                        help='Skip zip/dmg packaging step')
    args = parser.parse_args()

    weights = collect_weights()

    build_pyinstaller()
    copy_weights(weights)

    if sys.platform == 'win32':
        if not args.no_zip:
            zip_windows()
        print('\nDone.  Distribute:  dist/BadmintonAIPhotoEditor-Windows.zip')

    elif sys.platform == 'darwin':
        if not args.no_zip:
            dmg_mac()
        print('\nDone.  Distribute:  dist/BadmintonAIPhotoEditor-Mac.dmg')

    else:
        print(f'Platform {sys.platform} not yet supported for packaging.')
        sys.exit(1)


if __name__ == '__main__':
    main()
