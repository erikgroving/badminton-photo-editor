"""
Build script for Jay Pipeline.

Run on each platform to produce its binary:
  Windows  ->  dist/Jay Pipeline/Jay Pipeline.exe  (+ dist/JayPipeline-Windows.zip)
  Mac      ->  dist/Jay Pipeline.app               (+ dist/JayPipeline-Mac.dmg)

Usage:
    python build.py            # full build
    python build.py --no-zip   # skip zip/dmg step (faster, for iteration)

Weights are copied to the right location inside/alongside the bundle so the
app is self-contained:
  Windows: dist/Jay Pipeline/weights/
  Mac:     dist/Jay Pipeline.app/Contents/Resources/weights/
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

# Which checkpoint files to ship (skip training-only variants)
_SKIP_PATTERNS = ('_event', '_random', '_burst', '_attr', '_lstm', '_lr1e5',
                  'crop_judge', 'color_unet', 'judge_')

def _should_ship(p: Path) -> bool:
    name = p.name
    return not any(pat in name for pat in _SKIP_PATTERNS)


def collect_weights() -> list[Path]:
    """Return the checkpoint files to include in the distribution."""
    weights = [p for p in CKPTS_DIR.glob('*.pt') if _should_ship(p)]
    if not weights:
        print('WARNING: no checkpoints found — the app will launch but cannot run the pipeline.')
    else:
        total_mb = sum(p.stat().st_size for p in weights) / 1024**2
        print(f'Weights to bundle: {len(weights)} files  ({total_mb:.0f} MB)')
        for w in sorted(weights):
            print(f'  {w.name}')
    return weights


def copy_weights_windows(weights: list[Path]):
    dst = DIST / 'Jay Pipeline' / 'weights'
    dst.mkdir(parents=True, exist_ok=True)
    for w in weights:
        shutil.copy2(w, dst / w.name)
    print(f'Weights copied -> {dst}')


def copy_weights_mac(weights: list[Path]):
    dst = DIST / 'Jay Pipeline.app' / 'Contents' / 'Resources' / 'weights'
    dst.mkdir(parents=True, exist_ok=True)
    for w in weights:
        shutil.copy2(w, dst / w.name)
    print(f'Weights copied -> {dst}')


def build_pyinstaller():
    print('\n-- Running PyInstaller ----------------------------------')
    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', 'JayPipeline.spec'],
        cwd=ROOT,
    )
    # PyInstaller exits 1 for non-fatal "Hidden import not found" warnings on
    # deprecated torch.distributed._shard submodules; the bundle still builds.
    # Only abort if the expected output artifact is missing.
    bundle = DIST / 'Jay Pipeline' / 'Jay Pipeline.exe'
    if sys.platform == 'darwin':
        bundle = DIST / 'Jay Pipeline.app'
    if not bundle.exists():
        print(f'ERROR: PyInstaller exited {result.returncode} and bundle not found at {bundle}')
        sys.exit(1)


def zip_windows():
    zip_path = DIST / 'JayPipeline-Windows.zip'
    src      = DIST / 'Jay Pipeline'
    print(f'\n-- Creating {zip_path.name} ------------------------------')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in src.rglob('*'):
            if f.is_file():
                zf.write(f, Path('Jay Pipeline') / f.relative_to(src))
    size_mb = zip_path.stat().st_size / 1024**2
    print(f'Created {zip_path}  ({size_mb:.0f} MB)')


def dmg_mac():
    app     = DIST / 'Jay Pipeline.app'
    dmg_dir = DIST / '_dmg_stage'
    dmg_out = DIST / 'JayPipeline-Mac.dmg'

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
        '-volname', 'Jay Pipeline',
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

    if sys.platform == 'win32':
        copy_weights_windows(weights)
        if not args.no_zip:
            zip_windows()
        print('\nDone.  Distribute:  dist/JayPipeline-Windows.zip')

    elif sys.platform == 'darwin':
        copy_weights_mac(weights)
        if not args.no_zip:
            dmg_mac()
        print('\nDone.  Distribute:  dist/JayPipeline-Mac.dmg')

    else:
        print(f'Platform {sys.platform} not yet supported for packaging.')
        sys.exit(1)


if __name__ == '__main__':
    main()
