import json
import time

t0 = time.time()
with open('data/processed/companies_risk_intelligence.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(f"Loaded in {time.time()-t0:.2f}s, {len(d.get('companies', {}))} companies")
print("Metadata:", d.get("metadata", {}))
