import os
import shutil

ROOT_DIR = r"d:\Financial Distress"

dirs_to_create = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/config",
    "src/data",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/fusion",
    "src/explainability",
    "src/risk_engine",
    "src/utils",
    "dashboard/components",
    "dashboard/pages",
    "models/ews",
    "models/bankruptcy",
    "models/anomaly",
    "models/fusion",
    "reports/data_audit",
    "reports/eda",
    "reports/model_evaluation",
    "reports/explainability",
    "tests",
    "config"
]

for d in dirs_to_create:
    full_path = os.path.join(ROOT_DIR, d)
    os.makedirs(full_path, exist_ok=True)
    init_file = os.path.join(full_path, "__init__.py")
    if d.startswith("src"):
        with open(init_file, "a") as f:
            pass

# Copy raw data from Data/ to data/raw/
src_data_dir = os.path.join(ROOT_DIR, "Data")
dst_data_dir = os.path.join(ROOT_DIR, "data", "raw")

for fname in os.listdir(src_data_dir):
    s_path = os.path.join(src_data_dir, fname)
    d_path = os.path.join(dst_data_dir, fname)
    if os.path.isfile(s_path) and not os.path.exists(d_path):
        print(f"Copying {fname} to data/raw/...")
        shutil.copy2(s_path, d_path)

print("Project directories created and raw data synced successfully!")
