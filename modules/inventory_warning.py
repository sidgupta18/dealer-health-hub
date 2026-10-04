"""Three-month inventory early warning. Explicit training; lazy artifact loading; no torch."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from modules.data import COLUMNS

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=json.loads((ROOT/'config/inventory_warning.json').read_text())
BASELINE='Inventory trend baseline'
LR='Inventory logistic regression'
RF='Inventory random forest'
FEATURES=['stock_over_90_days_pct','inventory_days','sales_units','target_attainment_pct','operating_margin_pct','lead_conversion_pct']
LIMITS='Synthetic-data estimate; probabilities are not established as calibrated. Overlapping horizons and repeated dealers reduce independence. This tests future periods for existing dealers, not new-dealer generalisation. A flag supports investigation, not a claim of distress or causality.'

def fingerprint(df,config=SETTINGS):
    frame=df[COLUMNS].copy().sort_values(['dealer_id','month']).reset_index(drop=True)
    frame['month']=pd.to_datetime(frame.month).dt.strftime('%Y-%m-%d')
    data_hash=hashlib.sha256(frame.to_csv(index=False,float_format='%.10g').encode()).hexdigest()
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    artifact_hash=hashlib.sha256((data_hash+json.dumps(config,sort_keys=True)+source_hash).encode()).hexdigest()[:24]
    return {'data_fingerprint':data_hash,'model_version':config['version'],'implementation_fingerprint':source_hash,'artifact_fingerprint':artifact_hash}

def dataset(df,config=SETTINGS):
    d=df.copy();d['month']=pd.to_datetime(d.month).dt.to_period('M').dt.to_timestamp()
    if d.duplicated(['dealer_id','month']).any(): raise ValueError('Duplicate dealer-month observations.')
    d['target_attainment_pct']=100*d.sales_units/d.target_units
    inputs=[];rows=[];skipped={'current_breach':0,'at_threshold':0,'missing_aged_stock':0,'insufficient_or_gapped_history':0,'incomplete_future_horizon':0}
    names=[f'{k}__lag{lag}' for lag in range(config['input_months']-1,-1,-1) for k in FEATURES]+['aged_stock_trend_pp_per_month','inventory_trend_days_per_month','margin_trend_pp_per_month','attainment_trend_pp_per_month','sales_growth_1m_pct','sales_growth_3m_pct']
    for dealer,g in d.groupby('dealer_id',sort=True):
        g=g.sort_values('month').set_index('month',drop=False)
        for month,row in g.iterrows():
            aged=row.stock_over_90_days_pct
            if pd.isna(aged): skipped['missing_aged_stock']+=1;continue
            if aged>config['event_threshold_pct']: skipped['current_breach']+=1;continue
            if aged==config['event_threshold_pct']: skipped['at_threshold']+=1;continue
            dates=pd.date_range(end=month,periods=config['input_months'],freq='MS')
            if not dates.isin(g.index).all(): skipped['insufficient_or_gapped_history']+=1;continue
            block=g.loc[dates]
            # Baseline needs the same complete aged-stock history as LR/RF. Other feature holes are training-only imputed.
            if block.stock_over_90_days_pct.isna().any(): skipped['insufficient_or_gapped_history']+=1;continue
            x=block[FEATURES].to_numpy(dtype=float)
            def slope(column):
                finite=np.isfinite(column)
                return float(np.polyfit(np.arange(len(column))[finite],column[finite],1)[0]) if finite.sum()>=2 else np.nan
            growth=lambda lag:100*(x[-1,2]/x[-1-lag,2]-1) if np.isfinite(x[-1,2]) and np.isfinite(x[-1-lag,2]) and x[-1-lag,2]>0 else np.nan
            vector=np.concatenate([x.ravel(),[slope(x[:,0]),slope(x[:,1]),slope(x[:,4]),slope(x[:,3]),growth(1),growth(3)]])
            future_dates=pd.date_range(month+pd.offsets.MonthBegin(1),periods=config['horizon_months'],freq='MS')
            complete=future_dates.isin(g.index).all()
            future=g.loc[future_dates,'stock_over_90_days_pct'] if complete else None
            complete=complete and future.notna().all()
            label=float((future>config['event_threshold_pct']).any()) if complete else np.nan
            if not complete: skipped['incomplete_future_horizon']+=1
            inputs.append(vector)
            rows.append({'dealer_id':dealer,'month':month,'prediction_date':month+pd.offsets.MonthEnd(0),
                'label':label,'label_available_at':future_dates[-1]+pd.offsets.MonthEnd(0) if complete else pd.NaT,
                'horizon_start':future_dates[0],'horizon_end':future_dates[-1]+pd.offsets.MonthEnd(0),
                'current_aged_stock':float(aged),'aged_slope':slope(x[:,0]),'aged_history':x[:,0].tolist()})
    return {'x':np.array(inputs,dtype=float).reshape(-1,len(names)), 'rows':pd.DataFrame(rows,columns=['dealer_id','month','prediction_date','label','label_available_at','horizon_start','horizon_end','current_aged_stock','aged_slope','aged_history']), 'feature_names':names,'exclusions':skipped}

def splits(data,config=SETTINGS):
    rows=data['rows']; months=sorted(rows.loc[rows.label.notna(),'month'].unique())
    if len(months)<config['test_months']+config['validation_months']+config['min_train_months']+config['horizon_months']:
        return {'available':False,'reason':'Insufficient complete history for purged validation and an untouched test period.'}
    test_months=months[-config['test_months']:];freeze=pd.Timestamp(test_months[0])
    test=rows.index[rows.month.isin(test_months)&rows.label.notna()].to_numpy()
    # Purge horizons unknown before first test month; not merely rows earlier than the test month.
    final=rows.index[rows.label.notna()&(rows.label_available_at<freeze)].to_numpy()
    before=sorted(rows.loc[rows.label.notna()&(rows.label_available_at<freeze),'month'].unique());folds=[];skipped=[]
    for start in range(max(0,len(before)-config['max_folds']*config['validation_months']),len(before),config['validation_months']):
        val_months=before[start:start+config['validation_months']]
        if len(val_months)<config['validation_months']:continue
        cut=pd.Timestamp(val_months[0]); tr=rows.index[rows.label.notna()&(rows.label_available_at<cut)].to_numpy()
        # Validation labels themselves must be known at the final pre-test freeze.
        va=rows.index[rows.month.isin(val_months)&rows.label.notna()&(rows.label_available_at<freeze)].to_numpy()
        train_labels=rows.loc[tr,'label']; val_labels=rows.loc[va,'label']
        if len(tr)<config['min_train_samples'] or rows.loc[tr,'month'].nunique()<config['min_train_months'] or train_labels.nunique()<2 or val_labels.nunique()<2:
            skipped.append({'validation_months':[str(pd.Timestamp(m).date()) for m in val_months],'reason':'Insufficient training support or single-class/empty split.'});continue
        folds.append({'train':tr,'validation':va,'freeze_at':str(cut.date())})
    if not folds or len(final)<config['min_train_samples'] or rows.loc[final,'label'].nunique()<2 or not len(test):
        return {'available':False,'reason':'Insufficient events or eligible observations after three-month purging. No trained warning fabricated.','skipped_folds':skipped}
    return {'available':True,'folds':folds,'skipped_folds':skipped,'final_train':final,'test':test,'test_freeze':str(freeze.date())}

def baseline_fit(rows):
    residuals=[]
    for history in rows.aged_history:
        a=np.array(history); fitted=np.polyval(np.polyfit(np.arange(len(a)),a,1),np.arange(len(a)))
        residuals.extend(a-fitted)
    return {'sigma':max(2.0,float(np.std(residuals))), 'description':'Six-month least-squares aged-stock slope; maximum of next three projected monthly shares. Historical training residual scale maps distance to the threshold into a heuristic score; not a calibrated probability.'}

def baseline_predict(rows,fitted,config=SETTINGS):
    maximum=np.max(np.column_stack([rows.current_aged_stock+rows.aged_slope*h for h in range(1,config['horizon_months']+1)]),axis=1)
    z=np.clip((np.clip(maximum,0,100)-config['event_threshold_pct'])/(fitted['sigma']*math.sqrt(config['horizon_months'])),-30,30)
    return 1/(1+np.exp(-z))

def report(y,p,rows,threshold):
    from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,brier_score_loss
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);flag=p>=threshold
    top=[]
    for month,group in rows.assign(score=p,event=y,flag=flag).groupby('month'):
        flagged=group[group.flag].sort_values(['score','dealer_id'],ascending=[False,True]).head(10)
        top.append({'month':str(pd.Timestamp(month).date()),'flagged_count':int(group.flag.sum()),'top_count':len(flagged),'precision':float(flagged.event.mean()) if len(flagged) else None})
    defined=[r['precision'] for r in top if r['precision'] is not None]
    return {'samples':len(y),'events':int(y.sum()),'prevalence':float(y.mean()) if len(y) else None,
        'average_precision':float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
        'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        'precision':float(precision_score(y,flag,zero_division=0)),'recall':float(recall_score(y,flag,zero_division=0)),
        'precision_top10_flagged':float(np.mean(defined)) if defined else None,'months_without_flags':sum(r['top_count']==0 for r in top),
        'brier':float(brier_score_loss(y,p)), 'alert_threshold':float(threshold),'top10_months':top}

def reliability(y,p):
    table=pd.DataFrame({'p':p,'y':y}); table['bin']=pd.cut(table.p,np.linspace(0,1,6),include_lowest=True)
    return [{'mean_score':float(g.p.mean()),'observed_rate':float(g.y.mean()),'samples':len(g)} for _,g in table.groupby('bin',observed=True)]

def train(df,config=SETTINGS,artifact_dir=None,force=False):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import fbeta_score
    directory=Path(artifact_dir or ROOT/'artifacts/inventory_warning'); directory.mkdir(parents=True,exist_ok=True)
    fp=fingerprint(df,config);path=directory/(fp['artifact_fingerprint']+'.joblib')
    if path.exists() and not force:return joblib.load(path)['report']
    data=dataset(df,config);plan=splits(data,config)
    if not plan['available']:return {**fp,**plan,'exclusions':data['exclusions']}
    rows=data['rows'];x=data['x']; y=rows.label
    def fitted(indices):
        lr=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('model',LogisticRegression(C=.2,max_iter=1500,random_state=config['seed']))])
        rf=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('model',RandomForestClassifier(n_estimators=180,max_depth=4,min_samples_leaf=12,max_features=.6,n_jobs=1,random_state=config['seed']))])
        lr.fit(x[indices],y.iloc[indices]);rf.fit(x[indices],y.iloc[indices])
        return {BASELINE:baseline_fit(rows.iloc[indices]),LR:lr,RF:rf}
    def predict(models,name,indices):
        return baseline_predict(rows.iloc[indices],models[name],config) if name==BASELINE else models[name].predict_proba(x[indices])[:,1]
    pooled={name:[] for name in [BASELINE,LR,RF]}; validation_indices=[];fold_reports=[]
    def split_report(indices):
        r=rows.iloc[indices];return {'from':str(r.month.min().date()),'to':str(r.month.max().date()),'samples':len(r),'events':int(r.label.sum()),'prevalence':float(r.label.mean()),'latest_label_available':str(r.label_available_at.max().date())}
    for fold in plan['folds']:
        models=fitted(fold['train']);va=fold['validation'];validation_indices.extend(va.tolist())
        for name in pooled:pooled[name].extend(predict(models,name,va).tolist())
        fold_reports.append({'freeze_at':fold['freeze_at'],'train':split_report(fold['train']),'validation':split_report(va)})
    validation_rows=rows.iloc[validation_indices].reset_index(drop=True); vy=validation_rows.label.to_numpy(dtype=int)
    validation={};thresholds={}
    for name,scores in pooled.items():
        threshold=max(config['alert_grid'],key=lambda t:(fbeta_score(vy,np.asarray(scores)>=t,beta=2,zero_division=0),t))
        thresholds[name]=threshold;validation[name]=report(vy,scores,validation_rows,threshold)
    winner=max(validation,key=lambda name:validation[name]['average_precision'] if validation[name]['average_precision'] is not None else -1)
    improvement=(validation[winner]['average_precision'] or 0)-(validation[BASELINE]['average_precision'] or 0)
    chosen=winner if winner!=BASELINE and improvement>=config['meaningful_ap_improvement'] else BASELINE
    models=fitted(plan['final_train']);test_rows=rows.iloc[plan['test']].reset_index(drop=True); ty=test_rows.label.to_numpy(dtype=int)
    tests={};bins={}
    for name in models:
        scores=predict(models,name,plan['test']); tests[name]=report(ty,scores,test_rows,thresholds[name]);bins[name]=reliability(ty,scores)
    result={**fp,'available':True,'chosen':chosen,'alert_threshold':thresholds[chosen],'config':config,'features':data['feature_names'],
        'selection':'Pooled purged-validation AP; challenger requires at least +0.02 absolute AP over trend baseline. Test results never select the model.',
        'validation':validation,'test':tests,'reliability':bins,'folds':fold_reports,'skipped_folds':plan['skipped_folds'],
        'final_train':split_report(plan['final_train']),'test_freeze':plan['test_freeze'],'exclusions':data['exclusions'],'limitations':LIMITS}
    joblib.dump({'report':result,'models':models},path)
    path.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

def load(df,config=SETTINGS,artifact_dir=None):
    fp=fingerprint(df,config);path=Path(artifact_dir or ROOT/'artifacts/inventory_warning')/(fp['artifact_fingerprint']+'.joblib')
    if not path.exists():return {**fp,'available':False,'reason':'No matching inventory-warning artifact. Train explicitly in Data & Methodology or with scripts/train_inventory_warning.py.'},None
    try:
        import joblib
        saved=joblib.load(path)
        if saved['report']['artifact_fingerprint']!=fp['artifact_fingerprint']:raise ValueError('Fingerprint mismatch')
        return saved['report'],saved['models']
    except Exception as exc:return {**fp,'available':False,'reason':f'Inventory artifact could not be loaded: {exc}. No prediction substituted.'},None

def forecasts(df,result,models,config=SETTINGS):
    data=dataset(df,config);predictions={}
    if not result.get('available') or not models:return predictions
    eligible=data['rows'][data['rows'].prediction_date>=pd.Timestamp(result['test_freeze'])]
    if eligible.empty:return predictions
    indices=eligible.index.to_numpy(); name=result['chosen']
    values=baseline_predict(eligible,models[name],config) if name==BASELINE else models[name].predict_proba(data['x'][indices])[:,1]
    for (_,row),score in zip(eligible.iterrows(),values):
        predictions[(row.dealer_id,row.month)]=float(score)
    return predictions

def warning(row,result,predictions,config=SETTINGS):
    aged=row.stock_over_90_days_pct;month=pd.Timestamp(row.month);score=predictions.get((row.dealer_id,month))
    status='Current breach—inventory review required' if pd.notna(aged) and aged>config['event_threshold_pct'] else 'At threshold—inventory review required' if pd.notna(aged) and aged==config['event_threshold_pct'] else 'Insufficient history' if score is None else 'Emerging inventory concern' if score>=result['alert_threshold'] else 'No inventory flag'
    if pd.isna(aged): status='Missing aged-stock observation'
    elif aged<config['event_threshold_pct'] and score is None and result.get('available') and month<pd.Timestamp(result['test_freeze']): status='Historical month—outside forecast window'
    if score is None and pd.notna(aged) and aged<config['event_threshold_pct'] and not result.get('available'):status='Forecast unavailable'
    return {'status':status,'score':score,'flagged':status=='Emerging inventory concern','event_threshold_pct':config['event_threshold_pct'],
        'threshold_status':config['threshold_status'],'prediction_date':str((month+pd.offsets.MonthEnd(0)).date()),
        'horizon_start':str((month+pd.offsets.MonthBegin(1)).date()),'horizon_end':str((month+pd.offsets.MonthBegin(config['horizon_months'])+pd.offsets.MonthEnd(0)).date()),
        'event_definition':f"Any of the next {config['horizon_months']} calendar monthly aged-stock shares strictly above {config['event_threshold_pct']:g}%; all future months required for evaluation.",
        'model':result.get('chosen'),'model_version':result.get('model_version'),'data_fingerprint':result.get('data_fingerprint'),
        'artifact_fingerprint':result.get('artifact_fingerprint'),'alert_threshold':result.get('alert_threshold'),'evidence_id':'FORECAST:inventory','limitations':LIMITS}
