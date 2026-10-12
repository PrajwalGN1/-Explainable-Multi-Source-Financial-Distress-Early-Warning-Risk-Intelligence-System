import sys
import os
sys.path.insert(0, ".")
import json
import yaml
import pandas as pd
from src.config.settings import get_risk_config
from src.fusion.risk_fusion import RiskFusionEngine
from src.risk_engine.recommendations import RecommendationEngine

def refresh():
    cfg = get_risk_config()
    fusion = RiskFusionEngine()
    rec_engine = RecommendationEngine()
    
    with open('data/processed/companies_risk_intelligence.json', 'r', encoding='utf-8') as f:
        db = json.load(f)
        
    companies = db['companies']
    watchlist_records = []
    
    high_count = 0
    critical_count = 0
    total_score = 0.0
    
    for cid, cdata in companies.items():
        # Update latest assessment
        latest = cdata.get('latest_assessment', {})
        score = latest.get('risk_score', 0.0)
        cat_info = fusion.determine_category(score)
        latest['risk_category'] = cat_info['label']
        latest['risk_color'] = cat_info['color']
        latest['risk_description'] = cat_info['description']
        
        # Update recommendations based on new category
        signals = cdata.get('warning_signals', [])
        recs = rec_engine.generate_recommendations(cat_info['label'], signals)
        cdata['recommendations'] = recs['recommended_actions']
        
        # Update periods_data
        for pdata in cdata.get('periods_data', []):
            pscore = pdata.get('assessment', {}).get('risk_score', 0.0)
            pcat = fusion.determine_category(pscore)
            pdata['assessment']['risk_category'] = pcat['label']
            pdata['assessment']['risk_color'] = pcat['color']
            pdata['assessment']['risk_description'] = pcat['description']
            
        if cat_info['label'] == 'Critical':
            critical_count += 1
            priority = 'P1 - Urgent Escalation'
        elif cat_info['label'] == 'High Risk':
            high_count += 1
            priority = 'P2 - High Priority Surveillance'
        elif cat_info['label'] == 'Watch':
            priority = 'P3 - Active Watchlist'
        else:
            priority = 'P4 - Normal Surveillance'
            
        total_score += score
        
        drivers = cdata.get('top_risk_drivers', [])
        top_driver = drivers[0]['feature'] if drivers else 'N/A'
        
        watchlist_records.append({
            'company_id': cid,
            'latest_period': cdata.get('latest_period'),
            'final_risk_probability': latest.get('final_risk_probability'),
            'final_risk_probability_pct': f"{latest.get('final_risk_probability_pct')}%",
            'risk_score': score,
            'risk_category': cat_info['label'],
            'risk_trend': cdata.get('trajectory', {}).get('trend', 'Stable'),
            'financial_health_score': cdata.get('latest_health', {}).get('financial_health_score', 50.0),
            'active_signals_count': len(signals),
            'top_risk_driver': top_driver,
            'priority': priority
        })
        
    df_wl = pd.DataFrame(watchlist_records).sort_values(by=['risk_score', 'active_signals_count'], ascending=[False, False])
    df_wl.to_csv('data/processed/watchlist.csv', index=False)
    
    db['metadata']['high_risk_count'] = high_count
    db['metadata']['critical_count'] = critical_count
    db['metadata']['mean_risk_score'] = round(total_score / max(1, len(companies)), 1)
    
    with open('data/processed/companies_risk_intelligence.json', 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2)
        
    print(f"Successfully refreshed DB! Critical: {critical_count}, High: {high_count}, Watchlist updated.")

if __name__ == '__main__':
    refresh()
