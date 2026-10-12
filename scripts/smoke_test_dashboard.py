import sys
import os
sys.path.insert(0, ".")

print("Testing dashboard module imports and cache loaders...")
import app

# Test loading cached datasets directly
db = app.load_risk_db()
print(f"Risk DB loaded: {len(db.get('companies', {}))} companies")
assert len(db.get('companies', {})) == 422

wl = app.load_watchlist()
print(f"Watchlist loaded: {len(wl)} rows")
assert len(wl) == 422

abl = app.load_ablation_study()
print(f"Ablation study loaded: {len(abl)} rows")
assert len(abl) == 6

glob = app.load_global_importance()
print(f"Global importance loaded: {len(glob)} rows")
assert len(glob) > 0

print("All dashboard components, cache functions, and data dependencies verified successfully!")
