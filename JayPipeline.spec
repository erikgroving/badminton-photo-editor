# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Jay Pipeline.
Build on each platform to get the correct binary:
  Windows  →  dist/Jay Pipeline/Jay Pipeline.exe
  Mac      →  dist/Jay Pipeline.app  (then wrapped into .dmg by build.py)

Run via build.py, not directly.
"""
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# Collect all data files, binaries, and hidden imports for heavyweight packages
torch_datas,        torch_bins,        torch_hidden        = collect_all('torch')
torchvision_datas,  torchvision_bins,  torchvision_hidden  = collect_all('torchvision')
timm_datas,         timm_bins,         timm_hidden         = collect_all('timm')
rawpy_datas,        rawpy_bins,        rawpy_hidden        = collect_all('rawpy')
pil_datas,          pil_bins,          pil_hidden          = collect_all('PIL')
ultralytics_datas,  ultralytics_bins,  ultralytics_hidden  = collect_all('ultralytics')
sklearn_datas,      sklearn_bins,      sklearn_hidden      = collect_all('sklearn')
scipy_datas,        scipy_bins,        scipy_hidden        = collect_all('scipy')
# NumPy 2.x moved its core to numpy._core; PyInstaller hooks don't auto-collect it
numpy_datas,        numpy_bins,        numpy_hidden        = collect_all('numpy')
numpy_core_datas,   numpy_core_bins,   numpy_core_hidden   = collect_all('numpy._core')
# DO NOT call collect_all('cv2') here.
# opencv-python installs cv2 as a namespace package with the actual extension module
# at cv2/cv2.cpXXX-win_amd64.pyd (one level deep).  collect_all() walks the package
# with importlib and tries to import the extension before the bootloader has finished
# setting up sys.path, which triggers the "recursion is detected during loading of
# cv2 binary extensions" error at runtime.  The ultralytics hook already drags in
# the cv2 binaries it needs; we just need to declare 'cv2' as a hidden import so
# PyInstaller knows the name exists.  The .pyd itself is discovered automatically
# via the binary-dependency analysis of ultralytics/rawpy.

all_datas    = (torch_datas + torchvision_datas + timm_datas + rawpy_datas
                + pil_datas + ultralytics_datas + sklearn_datas + scipy_datas
                + numpy_datas + numpy_core_datas
                + [('assets', 'assets')])
all_binaries = (torch_bins + torchvision_bins + timm_bins + rawpy_bins
                + pil_bins + ultralytics_bins + sklearn_bins + scipy_bins
                + numpy_bins + numpy_core_bins)
all_hidden   = (torch_hidden + torchvision_hidden + timm_hidden + rawpy_hidden
                + pil_hidden + ultralytics_hidden + sklearn_hidden + scipy_hidden
                + numpy_hidden + numpy_core_hidden)

# Our own modules that PyInstaller won't auto-discover (dynamic imports in inference)
project_hidden = [
    'config',
    'data.raw_reader',
    'data.xmp_reader',
    'data.split_lookup',
    'data.mapping',
    'inference.run',
    'inference.pipeline',
    'inference.player_coverage',
    'inference.apply_params',
    'inference.export_xmp',
    'review.server',
    'review.burst_review',
    'review.__init__',
    'models.culling.model',
    'models.cropping.model',
    'models.color_correction.param_model',
    'models.color_correction.unet',
    'models.judge.model',
    'tqdm',
    'safetensors',
    'safetensors.torch',
    'matplotlib',
    'matplotlib.font_manager',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    # cv2 is pulled in by ultralytics and rawpy; listing it here tells PyInstaller
    # the import name is valid without collect_all() triggering the recursion bug.
    'cv2',
]

a = Analysis(
    ['app/main.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden + project_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_cv2.py'],
    excludes=[
        'notebook', 'IPython',
        'pandas',
        'tkinter', '_tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Jay Pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,       # no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Jay Pipeline',
)

# Mac: wrap the COLLECT output into a .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Jay Pipeline.app',
        icon='assets/icon.png',
        bundle_identifier='com.jayma.pipeline',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '1.0.0',
        },
    )
