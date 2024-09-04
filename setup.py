import os
import subprocess
import sys
import venv

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def create_venv():
    print("Creating virtual environment...")
    venv.create("venv", with_pip=True)

def get_venv_activate_command():
    if sys.platform == "win32":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def setup_backend():
    print("Setting up backend...")
    activate_command = get_venv_activate_command()
    run_command(f"{activate_command} && pip install -r requirements.txt")

def setup_frontend():
    print("Setting up frontend...")
    os.chdir("video-translator")
    if not os.path.exists("node_modules"):
        run_command("npm install")
    os.chdir("..")

def main():
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Create virtual environment
    create_venv()

    # Setup backend
    setup_backend()

    # Setup frontend
    setup_frontend()

    print("\nSetup complete!")
    print("\nTo activate the virtual environment:")
    print(get_venv_activate_command())
    print("\nTo run the backend server:")
    print("python server.py")
    print("\nTo run the frontend development server:")
    print("cd video-translator")
    print("npm run dev")

if __name__ == "__main__":
    main()