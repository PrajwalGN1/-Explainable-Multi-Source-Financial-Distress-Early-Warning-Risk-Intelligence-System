import json
import pandas as pd

with open('data/processed/companies_risk_intelligence.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

companies = db['companies']

latest_risks = []
all_risks = []
for cid, cdata in companies.items():
    latest_assess = cdata.get('latest_assessment', {})
    latest_risks.append({
        'company_id': cid,
        'latest_period': cdata.get('latest_period'),
        'risk_score': latest_assess.get('risk_score', 0),
        'risk_pct': latest_assess.get('final_risk_probability_pct', 0),
        'category': latest_assess.get('risk_category', 'Unknown'),
        'trend': cdata.get('trajectory', {}).get('trend', 'Unknown')
    })
    
    # Check all periods history
    for traj in cdata.get('trajectory', {}).get('trajectory_history', []):
        all_risks.append(traj.get('risk_pct', 0))

df_latest = pd.DataFrame(latest_risks)
print("Latest period risk score distribution:")
print(df_latest['risk_score'].describe())
print("\nCategory counts at latest period:")
print(df_latest['category'].value_counts())
print("\nAll-time risk score max/quantiles:")
s_all = pd.Series(all_risks)
print(s_all.describe())
print(f"Number of company-periods with risk >= 20%: {(s_all >= 20).sum()}")
print(f"Number of company-periods with risk >= 15%: {(s_all >= 15).sum()}")
print(f"Top 10 highest risk companies at any period:")
# Find max risk for each company
max_per_comp = []
for cid, cdata in companies.items():
    hist = cdata.get('trajectory', {}).get('trajectory_history', [])
    if hist:
        max_entry = max(hist, key=lambda x: x['risk_pct'])
        max_per_comp.append((cid, max_entry['period'], max_entry['risk_pct']))
max_per_comp.sort(key=lambda x: x[2], reverse=True)
for item in max_per_comp[:10]:
    print(f"Company {item[0]} at period {item[1]}: {item[2]}%")
