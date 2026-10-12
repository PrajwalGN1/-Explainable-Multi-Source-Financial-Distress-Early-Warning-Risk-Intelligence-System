import os
import pandas as pd
import numpy as np

df_a = pd.read_csv("data/raw/Financial Distress.csv")
corrs = df_a.drop(columns=['Company', 'Time']).corrwith(df_a['Financial Distress']).sort_values()

print("Top 10 features negatively correlated with Financial Distress:")
print(corrs.head(10))

print("\nTop 10 features positively correlated with Financial Distress:")
print(corrs.tail(10))

# Check distribution of features
print("\nx80 distribution:")
print(df_a['x80'].value_counts())
