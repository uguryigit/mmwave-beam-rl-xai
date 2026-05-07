import importlib
import os
import sys
import subprocess
import venv
from pathlib import Path
import platform

# List of required packages
REQUIRED_PACKAGES = ["pygame", "numpy", "gymnasium","pandas","matplotlib","stable_baselines3","boto3","streamlit","requests","tensorboard"]
ENV_FLAG = "VENV_BOOTSTRAPPED"

def in_venv() -> bool:
    # Check if we are already inside a virtual environment
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)

def venv_python(venv_dir: Path) -> str:
    # Return the correct python path inside the virtual environment
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")

def ensure_packages():
    missing = []
    # Check which packages are missing
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return  # All required packages are already installed

    # If we are already in a venv, install missing packages there
    if in_venv():
        try:
            print(f"[INFO] Installing missing packages into active venv: {missing}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            return
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Installation into active venv failed: {e}")

    # Otherwise, create a new .venv in the project directory
    proj_dir = Path(__file__).resolve().parent
    vdir = proj_dir / ".venv"

    if not vdir.exists():
        print("[INFO] .venv not found, creating one...")
        venv.create(str(vdir), with_pip=True)

    py = venv_python(vdir)

    # Install missing packages into the new venv
    try:
        print(f"[INFO] Installing {missing} into {vdir.name}...")
        subprocess.check_call([py, "-m", "pip", "install", *missing])
    except subprocess.CalledProcessError as e:
        msg = (
            f"[ERROR] Installation failed for {missing}: {e}\n"
            "Try manually:\n"
            f"  {py} -m pip install {' '.join(missing)}\n"
            "Or create and activate a venv manually:\n"
            "  python3 -m venv .venv && source .venv/bin/activate && pip install pygame numpy gymnasium\n"
        )
        print(msg)
        raise

    # Restart the script using the new venv Python (to ensure imports succeed)
    if os.environ.get(ENV_FLAG) != "1":
        print("[INFO] Restarting script with virtual environment...")
        env = os.environ.copy()
        env[ENV_FLAG] = "1"
        os.execvpe(py, [py, *sys.argv], env)


ensure_packages()