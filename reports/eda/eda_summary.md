# Exploratory Data Analysis & Provenance Audit Summary

## 1. Dataset A: Tehran Stock Exchange Panel (Financial Distress.csv)
- **Total Records:** 3,672 across 422 unique firms.
- **Time Window:** Periods 1 to 14.
- **Surviving (Healthy) Firms:** 286 firms (observed continuously for all 14 periods).
- **Distressed Firms:** 136 firms (32.2%).
- **Longitudinal Mechanics:** When a firm experiences distress ($Score \le -0.5$), it undergoes terminal distress and ceases reporting in subsequent periods.
- **Formulation for Early-Warning:** Time $t$ features predict distress at $t+1$. Right-censored panel avoids future leakage.

## 2. Dataset B: Taiwan Economic Journal Cross-Sectional Benchmark (data.csv)
- **Total Records:** 6,819 firms.
- **Bankruptcy Rate:** 3.23% (220 bankrupt, 6,599 non-bankrupt).
- **Feature Space:** 95 distinct financial ratios across profitability, liquidity, solvency, leverage, and turnover.
- **Top Discriminators:** Net Income to Total Assets, ROA, Debt Ratio, Persistent EPS.

## 3. Dataset C: Polish Companies Benchmark Test Partition (bankruptcy_Test_X.csv)
- **Total Records:** 5,000 firms.
- **Feature Space:** 64 standardized ratios ($Attr1$ to $Attr64$).
- **Labels:** Completely unlabelled.
- **Role:** Out-of-Distribution (OOD) financial anomaly detection and distribution drift evaluation.

## 4. Scientific Prohibition of Naive Feature Concatenation
The three datasets originate from three completely distinct financial markets, regulatory accounting standards, sampling regimes, and column definitions. Concatenating them row-wise is statistically invalid and methodologically flawed. The system adopts a multi-source risk fusion architecture that integrates independent evidence scores.
