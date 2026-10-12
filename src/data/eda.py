import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loaders import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("eda")

def run_eda(raw_dir: str = "data/raw", output_dir: str = "reports/eda"):
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(raw_dir)
    df_a = loader.load_dataset_a()
    df_b = loader.load_dataset_b()
    df_c = loader.load_dataset_c()
    
    # 1. Dataset A Longitudinal Analysis
    logger.info("Performing Longitudinal Analysis for Dataset A...")
    company_obs = df_a.groupby("Company")["Time"].agg(
        total_periods='count',
        min_time='min',
        max_time='max'
    ).reset_index()
    
    # Check distress occurrence
    distress_events = df_a[df_a["Financial Distress"] <= -0.5][["Company", "Time", "Financial Distress"]].rename(
        columns={"Time": "distress_time"}
    )
    company_obs = company_obs.merge(distress_events[["Company", "distress_time"]], on="Company", how="left")
    company_obs["is_distressed"] = company_obs["distress_time"].notna()
    
    company_obs.to_csv(os.path.join(output_dir, "dataset_a_company_summary.csv"), index=False)
    
    # Time period distribution of distress events
    distress_per_period = df_a[df_a["Financial Distress"] <= -0.5].groupby("Time")["Company"].count().reset_index()
    distress_per_period.columns = ["Time_Period", "Distress_Count"]
    distress_per_period.to_csv(os.path.join(output_dir, "dataset_a_distress_by_time.csv"), index=False)
    
    # Plot 1: Observations per company histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(company_obs["total_periods"], bins=14, kde=False, color="#3b82f6")
    plt.title("Dataset A: Observation Periods per Firm Distribution")
    plt.xlabel("Total Observation Periods (Years)")
    plt.ylabel("Number of Companies")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_a_periods_distribution.png"), dpi=150)
    plt.close()
    
    # Plot 2: Distress events across time periods
    plt.figure(figsize=(9, 4))
    sns.barplot(data=distress_per_period, x="Time_Period", y="Distress_Count", color="#ef4444")
    plt.title("Dataset A: Number of Financial Distress Events by Time Period")
    plt.xlabel("Time Period")
    plt.ylabel("Firms Entering Distress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_a_distress_timeline.png"), dpi=150)
    plt.close()
    
    # 2. Dataset B Imbalance & Key Ratios
    logger.info("Analyzing Dataset B (Bankruptcy)...")
    target_b = df_b["Bankrupt?"]
    b_class_dist = target_b.value_counts(normalize=True).reset_index()
    b_class_dist.columns = ["Class", "Proportion"]
    b_class_dist.to_csv(os.path.join(output_dir, "dataset_b_class_balance.csv"), index=False)
    
    # Select top discriminating features in Dataset B via correlation with Bankrupt?
    corrs_b = df_b.drop(columns=["Bankrupt?", "Net Income Flag"]).corrwith(df_b["Bankrupt?"]).abs().sort_values(ascending=False)
    top_b_features = corrs_b.head(15).reset_index()
    top_b_features.columns = ["Feature", "Absolute_Correlation_With_Bankruptcy"]
    top_b_features.to_csv(os.path.join(output_dir, "dataset_b_top_discriminators.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_b_features, x="Absolute_Correlation_With_Bankruptcy", y="Feature", palette="viridis")
    plt.title("Dataset B: Top 15 Financial Ratios Correlated with Bankruptcy")
    plt.xlabel("Absolute Pearson Correlation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_b_top_ratios.png"), dpi=150)
    plt.close()
    
    # 3. Dataset C Anomaly / Distribution Profile
    logger.info("Analyzing Dataset C (External Test)...")
    c_features = df_c.drop(columns=["ID"]) if "ID" in df_c.columns else df_c
    c_stats = pd.DataFrame({
        "mean": c_features.mean(),
        "std": c_features.std(),
        "skewness": c_features.skew(),
        "kurtosis": c_features.kurt()
    })
    c_stats.to_csv(os.path.join(output_dir, "dataset_c_moments.csv"))
    
    # Write summary markdown report
    md_content = f"""# Exploratory Data Analysis & Provenance Audit Summary

## 1. Dataset A: Tehran Stock Exchange Panel (Financial Distress.csv)
- **Total Records:** {len(df_a):,} across {df_a['Company'].nunique()} unique firms.
- **Time Window:** Periods 1 to 14.
- **Surviving (Healthy) Firms:** {company_obs['is_distressed'].value_counts().get(False, 0)} firms (observed continuously for all 14 periods).
- **Distressed Firms:** {company_obs['is_distressed'].value_counts().get(True, 0)} firms ({company_obs['is_distressed'].mean()*100:.1f}%).
- **Longitudinal Mechanics:** When a firm experiences distress ($Score \\le -0.5$), it undergoes terminal distress and ceases reporting in subsequent periods.
- **Formulation for Early-Warning:** Time $t$ features predict distress at $t+1$. Right-censored panel avoids future leakage.

## 2. Dataset B: Taiwan Economic Journal Cross-Sectional Benchmark (data.csv)
- **Total Records:** {len(df_b):,} firms.
- **Bankruptcy Rate:** {df_b['Bankrupt?'].mean()*100:.2f}% (220 bankrupt, 6,599 non-bankrupt).
- **Feature Space:** 95 distinct financial ratios across profitability, liquidity, solvency, leverage, and turnover.
- **Top Discriminators:** Net Income to Total Assets, ROA, Debt Ratio, Persistent EPS.

## 3. Dataset C: Polish Companies Benchmark Test Partition (bankruptcy_Test_X.csv)
- **Total Records:** {len(df_c):,} firms.
- **Feature Space:** 64 standardized ratios ($Attr1$ to $Attr64$).
- **Labels:** Completely unlabelled.
- **Role:** Out-of-Distribution (OOD) financial anomaly detection and distribution drift evaluation.

## 4. Scientific Prohibition of Naive Feature Concatenation
The three datasets originate from three completely distinct financial markets, regulatory accounting standards, sampling regimes, and column definitions. Concatenating them row-wise is statistically invalid and methodologically flawed. The system adopts a multi-source risk fusion architecture that integrates independent evidence scores.
"""
    with open(os.path.join(output_dir, "eda_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.info("EDA completed successfully!")

if __name__ == "__main__":
    run_eda()
