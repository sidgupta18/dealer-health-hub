import json
from pathlib import Path
import numpy as np
import pandas as pd
CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config/scoring.json').read_text())
STATUS_ORDER = ['Healthy', 'Watchlist', 'At Risk', 'Critical']
COLORS = {'Healthy':'#16856b','Watchlist':'#b48613','At Risk':'#d76828','Critical':'#c13949','Unassessed':'#738096'}
def classify(score, config=CONFIG):
    return next(name for name, threshold in sorted(config['classes'].items(),key=lambda item:item[1],reverse=True) if score >= threshold)
def normalize(value, spec):
    return float(np.clip((value-spec['bad'])/(spec['good']-spec['bad'])*100,0,100))
def score_data(df, config=CONFIG, indicator_weights=None, method="Business weights"):
    if indicator_weights is not None:
        if set(indicator_weights)!=set(config["kpis"]) or any(v<0 for v in indicator_weights.values()) or not np.isclose(sum(indicator_weights.values()),1):
            raise ValueError("Indicator weights must cover all scoring KPIs, be nonnegative and sum to one.")
    out=df.copy()
    out['target_attainment_pct']=100*out.sales_units/out.target_units
    scores=[]
    for _, row in out.iterrows():
        missing=[k for k in config['kpis'] if pd.isna(row[k])]
        norms={k:normalize(row[k],s) if pd.notna(row[k]) else np.nan for k,s in config['kpis'].items()}
        pillars={p:np.mean([norms[k] for k,s in config['kpis'].items() if s['pillar']==p]) for p in config['weights']}
        raw=(sum(norms[k]*w for k,w in indicator_weights.items()) if indicator_weights is not None else sum(pillars[p]*w for p,w in config['weights'].items())) if not missing else np.nan
        base=classify(raw,config) if not missing else 'Unassessed'
        status=base; triggers=[]
        if not missing:
            for rule in config['overrides']:
                operators={'>=':lambda a,b:a>=b,'>':lambda a,b:a>b,'<=':lambda a,b:a<=b,'<':lambda a,b:a<b}
                if operators[rule['operator']](row[rule['metric']],rule['threshold']):
                    triggers.append(f"{rule['metric']} = {row[rule['metric']]:.1f} {rule['operator']} {rule['threshold']} → {rule['status']}")
                    if STATUS_ORDER.index(rule['status']) > STATUS_ORDER.index(status): status=rule['status']
        scores.append({'raw_index':raw,'raw_status':base,'status':status,'override_evidence':'; '.join(triggers),'missing_inputs':', '.join(missing),**{p+'_score':v for p,v in pillars.items()}})
    out=pd.concat([out.reset_index(drop=True),pd.DataFrame(scores)],axis=1).sort_values(['dealer_id','month'])
    out['index_method']=method
    out['trajectory']=out.groupby('dealer_id').raw_index.diff()
    out['growth_opportunity']=((100-out.target_attainment_pct).clip(0,100)*0.5+out.customer_satisfaction*0.5).round(1)
    out['review_priority']=out.apply(lambda r:'P1 • Verify data' if r.status=='Unassessed' else 'P1 • Urgent' if r.status=='Critical' else 'P2 • Review' if r.status=='At Risk' or (pd.notna(r.trajectory) and r.trajectory<=-8) else 'P3 • Monitor' if r.status=='Watchlist' else 'P4 • Routine',axis=1)
    return out
