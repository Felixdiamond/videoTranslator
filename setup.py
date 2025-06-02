import os
import subprocess
import sys
import venv
import shutil # For checking if git is available

MIN_PYTHON_VERSION = (3, 11, 0) # Minimum required Python version (3.11.0)
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
    print(f"Running command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        # Use a list of args for better cross-platform compatibility and security than shell=True
        # For commands like 'source', shell=True might still be needed or handled differently.
        if isinstance(command, str) and ("&&" in command or ">" in command or "<" in command or "|" in command or "source" in command):
             # For complex shell commands, keep shell=True but be mindful of security.
            process = subprocess.run(command, check=True, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
        else:
            process = subprocess.run(command if isinstance(command, list) else command.split(), check=True, cwd=cwd, env=env, capture_output=True, text=True)
        
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(f"Stderr from command: {process.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Command '{command[0] if isinstance(command, list) else command.split()[0]}' not found. Is it in your PATH?")
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

def setup_melo_tts(project_root, pip_executable):
    print("\nSetting up MeloTTS...")
    melo_tts_dir = os.path.join(project_root, "MeloTTS") # Expecting MeloTTS to be cloned here or already present

    if not shutil.which("git"):
        print("Error: git command not found. Please install git and ensure it's in your PATH to clone MeloTTS.")
        # Optionally, provide instructions or skip this step if MeloTTS dir exists.
        if not os.path.isdir(melo_tts_dir):
             sys.exit(1)
        else:
            print(f"git not found, but MeloTTS directory '{melo_tts_dir}' exists. Assuming manual setup or pre-cloned.")


    if not os.path.isdir(melo_tts_dir):
        print("Cloning MeloTTS repository...")
        run_command(["git", "clone", "https://github.com/myshell-ai/MeloTTS.git", melo_tts_dir], cwd=project_root)
    else:
        print(f"MeloTTS directory '{melo_tts_dir}' already exists. Skipping clone.")
        # Optionally, add logic to pull latest changes:
        # print("Pulling latest changes for MeloTTS...")
        # run_command(["git", "pull"], cwd=melo_tts_dir)


    print("Installing MeloTTS package...")
    run_command([pip_executable, "install", "-e", "."], cwd=melo_tts_dir)

    print("Downloading UniDic for Japanese support in MeloTTS...")
    # This command uses the python from the venv
    python_executable = get_python_executable(os.path.join(project_root, "venv"))
    run_command([python_executable, "-m", "unidic", "download"], cwd=melo_tts_dir)
    print("MeloTTS setup complete.")


def setup_backend(project_root, venv_dir):
    print("\nSetting up backend dependencies...")
    pip_executable = get_pip_executable(venv_dir)
    requirements_file = os.path.join(project_root, "requirements.txt")
    run_command([pip_executable, "install", "-r", requirements_file])
    
    # After installing base requirements, setup MeloTTS
    setup_melo_tts(project_root, pip_executable)


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
        print("Installing frontend dependencies (npm install)...")
        run_command(["npm", "install"], cwd=frontend_dir)
    else:
        print("Frontend dependencies (node_modules) already exist. Skipping npm install.")
    print("Frontend setup complete.")


def main():
    # Ensure we're in the project root directory where setup.py is located
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root) # Change current working directory to project root

    print(f"Project root directory: {project_root}")

    check_python_version()

    # Create/ensure virtual environment
    venv_dir = create_venv(project_root)

    # Setup backend (including MeloTTS)
    setup_backend(project_root, venv_dir)

    # Setup frontend
    setup_frontend(project_root)

    print("\n🎉🎉🎉 Setup complete! 🎉🎉🎉")
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