"""
ComfyUI Workflow Preview Handler - Installer
Run this script as Administrator to install or uninstall the preview handler.

Usage:
    python install.py           # install
    python install.py uninstall # uninstall
"""

import ctypes
import subprocess
import sys
from pathlib import Path


def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def relaunch_as_admin(args: list[str]):
    script = str(Path(__file__).resolve())
    params = " ".join(f'"{a}"' for a in [script] + args)
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, params, None, 1
    )
    if ret <= 32:
        print("Failed to elevate. Please run this script as Administrator manually.")
        sys.exit(1)


def install():
    handler = Path(__file__).parent / "preview_handler.py"
    if not handler.exists():
        print(f"ERROR: preview_handler.py not found at {handler}")
        sys.exit(1)

    print("Installing ComfyUI Workflow Preview Handler...")
    result = subprocess.run(
        [sys.executable, str(handler), "--register"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print("Installation failed.")
        sys.exit(result.returncode)


def uninstall():
    handler = Path(__file__).parent / "preview_handler.py"
    if not handler.exists():
        print(f"ERROR: preview_handler.py not found at {handler}")
        sys.exit(1)

    print("Uninstalling ComfyUI Workflow Preview Handler...")
    result = subprocess.run(
        [sys.executable, str(handler), "--unregister"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def main():
    action = "uninstall" if "uninstall" in sys.argv else "install"

    if not is_admin():
        print(f"Elevating to Administrator for {action}...")
        relaunch_as_admin([action] if action == "uninstall" else [])
        return

    if action == "uninstall":
        uninstall()
    else:
        install()

    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
