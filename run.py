import os
import subprocess
import sys

def run_command(command, cwd=None):
    try:
        return subprocess.Popen(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def get_venv_activate_command():
    if sys.platform == "win32":
        return "call venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def main():
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Start backend server
    print("Starting backend server...")
    activate_command = get_venv_activate_command()
    backend_process = run_command(f"{activate_command} && python server.py")

    # Start frontend server
    print("Starting frontend server...")
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