import os
import subprocess
import sys

def run_command(command, cwd=None):
    try:
        return subprocess.Popen(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def get_venv_python_executable(project_root):
    venv_dir = os.path.join(project_root, "venv")
    if sys.platform == "win32":
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(python_exe):
        print(f"Error: Python executable not found in virtual environment: {python_exe}")
        print("Please ensure the virtual environment 'venv' exists and was created correctly (e.g., by running setup.py).")
        sys.exit(1)
    return python_exe

def main():
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Get Python executable from venv
    python_from_venv = get_venv_python_executable(project_root)

    # Start backend server
    print(f"Starting backend server using: {python_from_venv} server.py")
    # Running "python_from_venv server.py" directly ensures it uses the venv's Python and packages
    backend_process = run_command(f'"{python_from_venv}" server.py')


    # Start frontend server
    print("Starting frontend server (npm run dev)...")
    frontend_process = run_command("npm run dev", cwd="video-translator")

    print("\nBoth servers are now running!")
    print("Backend server is running on http://localhost:8000")
    print("Frontend server is running on http://localhost:3000")
    print("\nPress CTRL+C to stop both servers.")

    try:
        # Wait for user to stop the servers
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Servers stopped.")

if __name__ == "__main__":
    main()