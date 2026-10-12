import json
import pandas as pd

with open('data/processed/companies_risk_intelligence.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

companies = db['companies']

thresholds = [
    ("Healthy", 0.0, 4.0),
    ("Low Risk", 4.01, 8.0),
    ("Watch", 8.01, 13.0),
    ("High Risk", 13.01, 17.0),
    ("Critical", 17.01, 100.0)
]

def get_cat(score):
    for label, low, high in thresholds:
        if low <= score <= high:
            return label
    return "Healthy"

cats = []
for cid, cdata in companies.items():
    score = cdata.get('latest_assessment', {}).get('risk_score', 0)
    cats.append((cid, score, get_cat(score)))

df = pd.DataFrame(cats, columns=['company_id', 'risk_score', 'category'])
print("Calibrated Distribution at Latest Reporting Period:")
print(df['category'].value_counts())
print("\nCritical Companies:")
print(df[df['category'] == 'Critical'])
