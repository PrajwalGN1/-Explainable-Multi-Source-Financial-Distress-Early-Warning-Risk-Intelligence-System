import os
import pandas as pd
import numpy as np

data_dir = r"d:\Financial Distress\Data"

print("="*60)
print("INSPECTING DATASETS IN", data_dir)
print("="*60)

for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    if not fname.endswith('.csv'):
        continue
    print(f"\n--- FILE: {fname} ---")
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    df = pd.read_csv(fpath, nrows=5)
    print(f"Columns count: {len(df.columns)}")
    print(f"Columns preview: {list(df.columns[:15])}")
    if len(df.columns) > 15:
        print(f"Columns preview (last 5): {list(df.columns[-5:])}")
        
    # Check total rows
    full_df = pd.read_csv(fpath)
    print(f"Full shape: {full_df.shape}")
    print(f"Data types count: {full_df.dtypes.value_counts().to_dict()}")
    print(f"Missing values total: {full_df.isna().sum().sum()}")
    print(f"Duplicates total: {full_df.duplicated().sum()}")
    print("Head (first 2 rows):")
    print(full_df.iloc[:2, :min(10, len(full_df.columns))])
