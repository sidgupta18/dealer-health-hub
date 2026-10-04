from pathlib import Path
import numpy as np
import pandas as pd
METRICS=['sales_units','target_units','lead_conversion_pct','operating_margin_pct','overdue_receivables_pct','avg_payment_delay_days','inventory_days','stock_over_90_days_pct','customer_satisfaction','service_retention_pct','complaint_resolution_days']
COLUMNS=['dealer_id','dealer_name','region','city','month']+METRICS
RANGES={k:(0,100) for k in ['lead_conversion_pct','overdue_receivables_pct','stock_over_90_days_pct','customer_satisfaction','service_retention_pct']}
RANGES.update({'sales_units':(0,10000),'target_units':(1,10000),'operating_margin_pct':(-100,100),'avg_payment_delay_days':(0,365),'inventory_days':(0,730),'complaint_resolution_days':(0,365)})
def generate(seed=42):
    rng=np.random.default_rng(seed); rows=[]
    regions=['North','South','East','West','Central']; cities=['Aarav Nagar','Neelapur','Suryabad','Kiranpur','Taravan']
    for i in range(60):
        quality=rng.uniform(.25,.95); pressure=rng.uniform(0,1); delay=rng.uniform(0,25)
        for t,month in enumerate(pd.date_range('2024-10-01',periods=24,freq='MS')):
            q=np.clip(quality+rng.normal(0,.06),0,1)
            if i==0: q=.82; pressure=.1+t*.055
            elif i==1: q=.94; pressure=.05
            elif i==2: q=.3+t*.027; pressure=.8-t*.027
            else: pressure=np.clip(.85*pressure+.15*(1-q)+rng.normal(0,.08),0,1.3)
            delay=np.clip(.55*delay+12*pressure+3*(1-q)+rng.normal(0,5),0,90)
            target=int(rng.integers(70,180)); attain=.55+.6*q+rng.normal(0,.08)
            if i==0: attain=1.2+rng.normal(0,.04)
            if i==1: attain=.65+rng.normal(0,.04)
            vals=[round(target*attain),target,np.clip(5+23*q+rng.normal(0,2),0,100),np.clip(2+9*q-8*pressure+rng.normal(0,1),-20,20),np.clip(2+23*pressure+rng.normal(0,3),0,100),delay,np.clip(115-75*q+20*pressure+rng.normal(0,8),15,180),np.clip(35-32*q+12*pressure+rng.normal(0,4),0,100),np.clip(55+42*q+rng.normal(0,3),0,100),np.clip(35+52*q+rng.normal(0,4),0,100),np.clip(19-17*q+rng.normal(0,1.5),1,30)]
            name=['Aarav Motors','Neel Auto','Surya Mobility','Kiran Wheels'][i] if i<4 else f'{["Tara","Veda","Uday","Nila","Aruna"][i%5]} Motors {i+1:02d}'
            rows.append(dict(zip(COLUMNS,[f'D{i+1:03d}',name,regions[i%5],cities[i%5],month.strftime('%Y-%m-%d')]+[round(float(v),2) for v in vals])))
    df=pd.DataFrame(rows); df.loc[(df.dealer_id=='D004') & (df.month>='2026-07-01'),'overdue_receivables_pct']=np.nan
    return df

def validate(df):
    errors=[]; warnings=[]
    if df.empty: return None,['CSV contains no data rows.'],[]
    missing=sorted(set(COLUMNS)-set(df.columns))
    if missing: return None,[f"Missing columns: {', '.join(missing)}"],[]
    df=df[COLUMNS].copy()
    for k in COLUMNS[:4]:
        if df[k].isna().any() or df[k].astype(str).str.strip().eq('').any(): errors.append(f'{k}: blank identifiers are invalid.')
    dates=pd.to_datetime(df.month,errors='coerce')
    if dates.isna().any(): errors.append('month: use valid dates, preferably YYYY-MM-01.')
    df['month']=dates.dt.to_period('M').dt.to_timestamp()
    if df.duplicated(['dealer_id','month']).any(): errors.append('Duplicate dealer-month keys after monthly normalization.')
    for k,(lo,hi) in RANGES.items():
        numeric=pd.to_numeric(df[k],errors='coerce')
        bad=df[k].notna() & numeric.isna()
        if bad.any(): errors.append(f'{k}: {bad.sum()} nonnumeric values.')
        if ((numeric<lo)|(numeric>hi)|np.isinf(numeric)).any(): errors.append(f'{k}: valid range is {lo}–{hi}.')
        if k in ['sales_units','target_units'] and (numeric.dropna()%1!=0).any(): errors.append(f'{k}: must be whole units.')
        df[k]=numeric
        if numeric.isna().any(): warnings.append(f'{k}: {numeric.isna().sum()} missing inputs; affected health rows are Unassessed.')
    for k in ['dealer_name','region','city']:
        if df.groupby('dealer_id')[k].nunique().gt(1).any(): errors.append(f'{k}: inconsistent dealer metadata across months.')
    return (None if errors else df.sort_values(['dealer_id','month']).reset_index(drop=True)),errors,warnings

DATA_VERSION='synthetic-v2'
def generate_v2(seed=42):
    """Separate revision; v1 generate() and its CSV remain unchanged.

    Adds market seasonality and short dealer-level shocks to latent pressure,
    not to a health classification. Next outcomes remain stochastic.
    """
    df=generate(seed).copy(); rng=np.random.default_rng(seed+2026)
    for dealer,indices in df.groupby('dealer_id',sort=True).groups.items():
        positions=list(indices); delay=float(df.loc[positions[0],'avg_payment_delay_days']); shock=0.0
        for t,i in enumerate(positions):
            month=pd.Timestamp(df.loc[i,'month']); season=np.sin(2*np.pi*(month.month-8)/12)
            innovation=rng.normal(0,2.0)
            # Occasional two-sided operational shocks, with decaying persistence.
            shock=.55*shock+(rng.choice([-1,1])*rng.uniform(6,14) if rng.random()<.07 else 0)
            base_pressure=(float(df.loc[i,'overdue_receivables_pct']) if pd.notna(df.loc[i,'overdue_receivables_pct']) else 14)/25
            delay=np.clip(.55*delay+9.5*base_pressure+2.4*season+shock+innovation+rng.normal(0,3.5),0,110)
            df.loc[i,'avg_payment_delay_days']=round(delay,2)
            df.loc[i,'sales_units']=round(max(0,float(df.loc[i,'sales_units'])*(1+.08*season)+rng.normal(0,3)))
            df.loc[i,'inventory_days']=round(np.clip(float(df.loc[i,'inventory_days'])-5*season+shock*.4,15,200),2)
    return df
