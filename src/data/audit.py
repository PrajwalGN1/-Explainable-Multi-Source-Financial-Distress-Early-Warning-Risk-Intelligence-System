import os
import json
import pandas as pd
import numpy as np
from src.data.loaders import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("data_audit")

def run_data_audit(raw_dir: str = "data/raw", output_dir: str = "reports/data_audit"):
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(raw_dir)
    df_a = loader.load_dataset_a()
    df_b = loader.load_dataset_b()
    df_c = loader.load_dataset_c()
    
    datasets = {
        "Dataset A (Financial Distress - Longitudinal EWS)": (df_a, "Company", "Time", "Financial Distress"),
        "Dataset B (Taiwan Economic Journal - Bankruptcy)": (df_b, None, None, "Bankrupt?"),
        "Dataset C (Polish Companies Bankruptcy - Unlabelled External)": (df_c, "ID", None, None)
    }
    
    summary_rows = []
    missing_rows = []
    duplicate_rows = []
    feat_stats_list = []
    profile_json = {}
    
    for name, (df, id_col, time_col, target_col) in datasets.items():
        logger.info(f"Auditing {name}...")
        n_rows, n_cols = df.shape
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)
        
        # Check duplicates
        n_dup_rows = int(df.duplicated().sum())
        n_dup_ids = int(df[id_col].duplicated().sum()) if id_col and id_col in df.columns else 0
        
        # Missing values
        missing_counts = df.isna().sum()
        total_missing = int(missing_counts.sum())
        missing_pct = float((total_missing / (n_rows * n_cols)) * 100)
        
        for col, cnt in missing_counts.items():
            if cnt > 0:
                missing_rows.append({
                    "dataset": name,
                    "column": col,
                    "missing_count": int(cnt),
                    "missing_percentage": float((cnt / n_rows) * 100)
                })
                
        duplicate_rows.append({
            "dataset": name,
            "duplicate_rows": n_dup_rows,
            "duplicate_id_count": n_dup_ids,
            "id_column": id_col if id_col else "None"
        })
        
        # Constant / near-constant columns
        constant_cols = [c for c in num_cols if df[c].std() == 0]
        near_constant_cols = [c for c in num_cols if df[c].std() > 0 and df[c].std() < 1e-4]
        
        # Target stats
        target_info = {}
        if target_col and target_col in df.columns:
            target_series = df[target_col]
            target_info = {
                "target_column": target_col,
                "is_numeric": bool(pd.api.types.is_numeric_dtype(target_series)),
                "unique_values": int(target_series.nunique()),
                "sample_values": [float(x) for x in target_series.dropna().unique()[:5]]
            }
            if name.startswith("Dataset A"):
                # Literature threshold is <= -0.5
                n_distressed = int((target_series <= -0.5).sum())
                target_info["distress_threshold"] = -0.5
                target_info["distressed_count"] = n_distressed
                target_info["distressed_rate_pct"] = round((n_distressed / n_rows) * 100, 2)
            elif name.startswith("Dataset B"):
                target_info["class_counts"] = target_series.value_counts().to_dict()
                target_info["bankruptcy_rate_pct"] = round(float(target_series.mean() * 100), 2)
                
        summary_rows.append({
            "dataset_name": name,
            "num_rows": n_rows,
            "num_columns": n_cols,
            "num_numeric_features": len(num_cols),
            "num_categorical_features": len(cat_cols),
            "total_missing_values": total_missing,
            "missing_percentage": round(missing_pct, 4),
            "duplicate_rows": n_dup_rows,
            "id_column": id_col,
            "time_column": time_col,
            "target_column": target_col,
            "constant_columns_count": len(constant_cols),
            "near_constant_columns_count": len(near_constant_cols)
        })
        
        # Feature statistics
        for col in num_cols:
            s = df[col].dropna()
            q25, q50, q75 = s.quantile([0.25, 0.5, 0.75])
            iqr = q75 - q25
            outliers = int(((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum())
            feat_stats_list.append({
                "dataset": name,
                "feature": col,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q25": float(q25),
                "median": float(q50),
                "q75": float(q75),
                "max": float(s.max()),
                "outliers_iqr_count": outliers
            })
            
        profile_json[name] = {
            "rows": n_rows,
            "columns": n_cols,
            "column_names": list(df.columns),
            "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
            "missing_total": total_missing,
            "duplicates": n_dup_rows,
            "target_info": target_info,
            "constant_columns": constant_cols,
            "near_constant_columns": near_constant_cols
        }
        
    # Write audit reports
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)
    pd.DataFrame(missing_rows if missing_rows else [{"info": "No missing values found across all 3 datasets"}]).to_csv(os.path.join(output_dir, "missing_values.csv"), index=False)
    pd.DataFrame(duplicate_rows).to_csv(os.path.join(output_dir, "duplicate_report.csv"), index=False)
    pd.DataFrame(feat_stats_list).to_csv(os.path.join(output_dir, "feature_statistics.csv"), index=False)
    
    with open(os.path.join(output_dir, "dataset_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile_json, f, indent=2)
        
    logger.info("Data audit reports generated successfully in " + output_dir)
    return profile_json

if __name__ == "__main__":
    run_data_audit()
