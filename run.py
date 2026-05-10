"""
Opens http://localhost:5050 in your browser automatically.
"""
import subprocess, sys, time, webbrowser, os

os.chdir(os.path.dirname(__file__))

proc = subprocess.Popen(
    [sys.executable, "backend/app.py"],
    env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
)

time.sleep(3)   # wait for server to start
webbrowser.open("http://localhost:5050")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    print("\nStopped.")
