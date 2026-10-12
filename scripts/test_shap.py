import sys
import subprocess

try:
    import shap
    print("shap is already installed! Version:", shap.__version__)
except ImportError:
    print("shap not installed. Testing pip install shap --no-deps...")
    res = subprocess.run([sys.executable, "-m", "pip", "install", "shap", "--no-deps"], capture_output=True, text=True)
    print("Return code:", res.returncode)
    print("Stdout:", res.stdout[:500])
    try:
        import shap
        print("shap installed successfully! Version:", shap.__version__)
    except Exception as e:
        print("Import failed:", e)
