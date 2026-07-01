"""
PyInstaller runtime hook for OpenCV (cv2).

cv2's extension module needs to find its companion DLLs at startup.
In a frozen PyInstaller bundle the DLLs land in _MEIPASS (and possibly
_MEIPASS/cv2/), but Windows DLL search doesn't look there automatically.
Adding these directories early (before any import of cv2 or ultralytics)
prevents the "recursion is detected during loading of cv2 binary extensions"
error that otherwise surfaces when ultralytics first imports cv2.
"""
import os
import sys

if getattr(sys, "frozen", False):
    base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

    dirs_to_add = [base]
    cv2_sub = os.path.join(base, "cv2")
    if os.path.isdir(cv2_sub):
        dirs_to_add.append(cv2_sub)

    # Python 3.8+: os.add_dll_directory is the correct way on Windows
    if hasattr(os, "add_dll_directory"):
        for d in dirs_to_add:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

    # Also extend PATH for older loaders / sub-processes
    extra = os.pathsep.join(dirs_to_add)
    os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
