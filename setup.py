import argparse
import os
import subprocess
import sys
import venv
import shutil

MIN_PYTHON_VERSION = (3, 11, 0)
RECOMMENDED_PYTHON_VERSION_STR = ">= 3.11.9"

def check_python_version():
    print("Checking Python version...")
    current_version = sys.version_info
    if current_version < MIN_PYTHON_VERSION:
        print(f"Error: Python version {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}.{MIN_PYTHON_VERSION[2]} or higher is required.")
        print(f"Your version is {current_version.major}.{current_version.minor}.{current_version.micro}.")
        print(f"The recommended version for this project is {RECOMMENDED_PYTHON_VERSION_STR}.")
        sys.exit(1)
    print(f"Python version {current_version.major}.{current_version.minor}.{current_version.micro} is compatible.")

def run_command(command, cwd=None, env=None):
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    try:
        process = subprocess.run(
            command if isinstance(command, list) else command.split(),
            check=True, cwd=cwd, env=env, capture_output=True, text=True,
        )
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: '{command[0] if isinstance(command, list) else command.split()[0]}' not found in PATH.")
        sys.exit(1)


def create_venv(project_root):
    venv_dir = os.path.join(project_root, "venv")
    if not os.path.exists(venv_dir):
        print("Creating virtual environment at 'venv'...")
        venv.create(venv_dir, with_pip=True)
        print("Virtual environment created.")
    else:
        print("Virtual environment 'venv' already exists.")
    return venv_dir

def get_python_executable(venv_dir):
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir, "bin", "python")

def get_pip_executable(venv_dir):
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_dir, "bin", "pip")

def patch_file(path, replacements):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def setup_melo_tts(project_root, pip_executable):
    print("\nSetting up MeloTTS...")
    if not shutil.which("git"):
        print("Error: git not found in PATH.")
        sys.exit(1)

    melo_tts_dir = os.path.join(project_root, "MeloTTS")
    if not os.path.isdir(melo_tts_dir):
        run_command(["git", "clone", "https://github.com/Felixdiamond/MeloTTS.git", melo_tts_dir], cwd=project_root)
    else:
        run_command(["git", "-C", melo_tts_dir, "pull"])

    run_command([pip_executable, "install", "--no-build-isolation", "-e", "."], cwd=melo_tts_dir)
    python_executable = get_python_executable(os.path.join(project_root, "venv"))
    run_command([python_executable, "-m", "unidic", "download"])
    print("MeloTTS setup complete.")


def setup_whisperx(project_root, pip_executable):
    print("\nSetting up whisperX...")
    whisperx_dir = os.path.join(project_root, ".deps", "whisperx")
    os.makedirs(os.path.dirname(whisperx_dir), exist_ok=True)
    if not os.path.isdir(whisperx_dir):
        run_command(["git", "clone", "https://github.com/m-bain/whisperX.git", whisperx_dir])
    else:
        run_command(["git", "-C", whisperx_dir, "pull"])
    patch_file(os.path.join(whisperx_dir, "pyproject.toml"), [
        ("torch~=2.8.0", "torch>=2.8.0"),
        ("torchaudio~=2.8.0", "torchaudio>=2.8.0"),
    ])
    run_command([pip_executable, "install", "-e", whisperx_dir])
    print("whisperX setup complete.")


def setup_qwen3_tts(project_root, pip_executable):
    print("\nSetting up Qwen3-TTS...")
    qwen3_dir = os.path.join(project_root, ".deps", "qwen3tts")
    os.makedirs(os.path.dirname(qwen3_dir), exist_ok=True)
    if not os.path.isdir(qwen3_dir):
        run_command(["git", "clone", "https://github.com/QwenLM/Qwen3-TTS.git", qwen3_dir])
    else:
        run_command(["git", "-C", qwen3_dir, "pull"])
    patch_file(os.path.join(qwen3_dir, "pyproject.toml"), [
        ("transformers==4.57.3", "transformers>=4.47.1"),
    ])
    run_command([pip_executable, "install", "-e", qwen3_dir])
    print("Qwen3-TTS setup complete.")


def setup_backend(project_root, venv_dir, tts):
    print("\nSetting up backend dependencies...")
    pip_executable = get_pip_executable(venv_dir)
    requirements_file = os.path.join(project_root, "requirements.txt")
    run_command([pip_executable, "install", "-r", requirements_file])
    setup_whisperx(project_root, pip_executable)
    if "melo" in tts:
        setup_melo_tts(project_root, pip_executable)
    if "qwen3" in tts:
        setup_qwen3_tts(project_root, pip_executable)


def setup_frontend(project_root):
    print("\nSetting up frontend...")
    frontend_dir = os.path.join(project_root, "video-translator")
    if not os.path.isdir(frontend_dir):
        print(f"Error: Frontend directory '{frontend_dir}' not found.")
        # Decide if this is a fatal error or if frontend setup is optional
        return

    if not shutil.which("npm"):
        print("Warning: npm command not found. Cannot install frontend dependencies.")
        print("Please install Node.js and npm: https://nodejs.org/")
        return # Or sys.exit(1) if frontend is mandatory

    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        run_command(["npm", "install"], cwd=frontend_dir)
    else:
        print("node_modules already exists, skipping npm install.")
    print("Frontend setup complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap videoTranslator. whisperX is always installed. MeloTTS is installed by default; use --qwen3 to add or replace it."
    )
    parser.add_argument("--melo", action="store_true", default=True, help="Install Felixdiamond/MeloTTS (default TTS engine, on by default).")
    parser.add_argument("--no-melo", dest="melo", action="store_false", help="Skip MeloTTS install.")
    parser.add_argument("--qwen3", action="store_true", help="Also install Qwen3-TTS (enables voice cloning).")
    return parser.parse_args()


def main():
    args = parse_args()
    tts = set()
    if args.melo:
        tts.add("melo")
    if args.qwen3:
        tts.add("qwen3")

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print(f"Project root: {project_root}")
    print(f"TTS engines to install: {', '.join(sorted(tts)) if tts else 'none (gtts fallback only)'}")

    check_python_version()
    venv_dir = create_venv(project_root)
    setup_backend(project_root, venv_dir, tts)
    setup_frontend(project_root)

    print("\nSetup complete.")
    print("\nNext Steps:")
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print(f"   On Windows: .\\venv\\Scripts\\activate")
    else:
        print(f"   On macOS/Linux: source ./venv/bin/activate")
    
    print("\n2. To run the application (backend and frontend):")
    print(f"   python run.py")
    
    print("\n   (This will start the FastAPI backend on http://localhost:8000")
    print(f"    and the Next.js frontend on http://localhost:3000)")

    print("\n   Alternatively, to run backend and frontend separately:")
    print(f"   - Backend: python server.py (after activating venv)")
    print(f"   - Frontend: cd video-translator && npm run dev")

if __name__ == "__main__":
    main()