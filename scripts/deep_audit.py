import os
import pandas as pd
import numpy as np

data_dir = r"d:\Financial Distress\Data"

# 1. Inspect Financial Distress.csv
df_fd = pd.read_csv(os.path.join(data_dir, "Financial Distress.csv"))
print("=== FINANCIAL DISTRESS.CSV ===")
print("Shape:", df_fd.shape)
print("Unique companies:", df_fd['Company'].nunique())
print("Time min, max, unique:", df_fd['Time'].min(), df_fd['Time'].max(), df_fd['Time'].nunique())
obs_per_comp = df_fd.groupby('Company')['Time'].count()
print("Observations per company stats:")
print(obs_per_comp.describe())

target = df_fd['Financial Distress']
print("Financial Distress target stats:")
print(target.describe())
print("Values <= -0.5 count:", (target <= -0.5).sum(), f"({(target <= -0.5).mean()*100:.2f}%)")
print("Values > -0.5 count:", (target > -0.5).sum(), f"({(target > -0.5).mean()*100:.2f}%)")
print("Values < 0 count:", (target < 0).sum())

# Check distress per company over time
distressed_comp = df_fd[df_fd['Financial Distress'] <= -0.5]['Company'].unique()
print("Number of companies experiencing distress (<= -0.5):", len(distressed_comp))

# Check when distress occurs for distressed companies (e.g. at their last time point?)
for c in distressed_comp[:10]:
    c_df = df_fd[df_fd['Company'] == c][['Company', 'Time', 'Financial Distress']]
    print(f"Company {c} timeline:")
    print(c_df.to_string(index=False))

# Check x80 (often categorical industry in this dataset)
print("\nx80 unique values count:", df_fd['x80'].nunique(), "Sample values:", df_fd['x80'].unique()[:10])

# 2. Inspect data.csv
print("\n=== DATA.CSV ===")
df_data = pd.read_csv(os.path.join(data_dir, "data.csv"))
print("Shape:", df_data.shape)
print("Target column 'Bankrupt?' value counts:")
print(df_data['Bankrupt?'].value_counts(dropna=False))
print(f"Bankrupt rate: {df_data['Bankrupt?'].mean()*100:.2f}%")
print("Columns snippet (first 10):", list(df_data.columns[:10]))
print("Columns snippet (middle 10):", list(df_data.columns[40:50]))

# 3. Inspect bankruptcy_Test_X.csv
print("\n=== BANKRUPTCY_TEST_X.CSV ===")
df_test = pd.read_csv(os.path.join(data_dir, "bankruptcy_Test_X.csv"))
print("Shape:", df_test.shape)
print("Columns:", list(df_test.columns))
print("ID min, max:", df_test['ID'].min(), df_test['ID'].max())
print("Attr1 to Attr5 stats:")
print(df_test[['Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5']].describe())
