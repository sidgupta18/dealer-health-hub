"""Six-month forecasting experiment. No training or torch import at app startup."""
import hashlib
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, brier_score_loss, f1_score
from modules.data import METRICS, COLUMNS
from modules.scoring import CONFIG
from modules.analysis import SETTINGS
ROOT=Path(__file__).resolve().parents[1]
ARTIFACT_DIR=ROOT/'artifacts'
PERSISTENCE='Persistence baseline'
LOGISTIC='Logistic regression'
TREE='Random forest'
GRU='GRU challenger'

def labels(df):
    d=df.sort_values(['dealer_id','month']).copy()
    d['next_month']=d.groupby('dealer_id').month.shift(-1)
    d['next_delay']=d.groupby('dealer_id').avg_payment_delay_days.shift(-1)
    consecutive=d.next_month.eq(d.month+pd.offsets.MonthBegin(1))
    d['label']=np.where(consecutive & d.next_delay.notna(),(d.next_delay>CONFIG['payment_delay_target_days']).astype(int),np.nan)
    d['label_available_at']=d.next_month+pd.offsets.MonthEnd(0)
    d.loc[d.label.isna(),'label_available_at']=pd.NaT
    return d

def sequences(df,window=None):
    window=window or SETTINGS['sequence_months']; d=labels(df); inputs=[]; rows=[]; skipped_gap=0; skipped_missing=0
    for dealer,g in d.groupby('dealer_id',sort=True):
        g=g.sort_values('month')
        for end in range(window-1,len(g)):
            block=g.iloc[end-window+1:end+1]; row=block.iloc[-1]
            expected=pd.date_range(block.month.iloc[0],periods=window,freq='MS')
            if not np.array_equal(block.month.to_numpy(),expected.to_numpy()):
                skipped_gap+=1; continue
            x=block[METRICS].to_numpy(dtype=float)
            if not np.isfinite(x).all(): skipped_missing+=1; continue
            inputs.append(x)
            rows.append({'dealer_id':dealer,'month':row.month,'prediction_date':row.month+pd.offsets.MonthEnd(0),'label':row.label,'label_available_at':row.label_available_at,'window_start':block.month.iloc[0],'current_delay':row.avg_payment_delay_days})
    x=np.asarray(inputs,dtype=float) if inputs else np.empty((0,window,len(METRICS)))
    metadata=pd.DataFrame(rows,columns=['dealer_id','month','prediction_date','label','label_available_at','window_start','current_delay'])
    features=[f'{k}__lag{lag}' for lag in range(window-1,-1,-1) for k in METRICS]
    return {'x':x,'tabular':x.reshape(len(x),window*len(METRICS)),'meta':metadata,'features':features,'excluded_gaps':skipped_gap,'excluded_missing_windows':skipped_missing}

def fingerprint(df):
    h=hashlib.sha256(df[COLUMNS].sort_values(['dealer_id','month']).to_csv(index=False,date_format='%Y-%m-%d').encode())
    h.update(json.dumps(CONFIG,sort_keys=True).encode()); h.update(json.dumps(SETTINGS,sort_keys=True).encode())
    for name in ['model.py','gru.py','analysis.py']:
        p=ROOT/'modules'/name
        if p.exists(): h.update(p.read_bytes())
    return h.hexdigest()[:24]

def split_plan(meta,settings=SETTINGS):
    known=meta[meta.label.notna()]; months=sorted(known.month.unique())
    size=settings['test_months']; foldsize=settings['validation_months_per_fold']
    if len(months)<settings['minimum_train_months']+foldsize+size+2:
        return {'available':False,'reason':'Need at least 14 eligible forecast months after six-month window construction.'}
    test_start=pd.Timestamp(months[-size]); test_idx=known.index[known.month>=test_start].to_numpy()
    # Selection labels must be available before the first test reporting month starts.
    pretest=known[known.label_available_at<test_start]
    valmonths=sorted(pretest.month.unique()); possible=len(valmonths)-settings['minimum_train_months']-1
    folds=min(settings['max_validation_folds'],max(0,possible//foldsize))
    if folds<1: return {'available':False,'reason':'Insufficient observable history for expanding validation and an untouched test.'}
    validation_months=valmonths[-folds*foldsize:]; plans=[]; skipped=[]
    for i in range(folds):
        block=validation_months[i*foldsize:(i+1)*foldsize]; start=pd.Timestamp(block[0])
        train_idx=known.index[(known.month<start)&(known.label_available_at<start)].to_numpy()
        val_idx=known.index[known.month.isin(block)].to_numpy()
        y=meta.loc[train_idx,'label']; v=meta.loc[val_idx,'label']
        if len(y)<30 or y.nunique()<2 or len(v)<10:
            skipped.append({'fold':i+1,'reason':'Fewer than 30 training / 10 validation sequences or single-class training.'}); continue
        plans.append({'fold':i+1,'train':train_idx,'validation':val_idx,'freeze_at':str(start.date())})
    final_train=pretest.index.to_numpy()
    if not plans or meta.loc[final_train,'label'].nunique()<2 or len(test_idx)<10:
        return {'available':False,'reason':'No usable validation folds, insufficient test sequences or single-class training.','skipped_folds':skipped}
    if not any(meta.loc[p['validation'],'label'].nunique()>1 for p in plans):
        return {'available':False,'reason':'All validation folds are single-class; average-precision model selection is undefined.','skipped_folds':skipped}
    return {'available':True,'folds':plans,'test':test_idx,'final_train':final_train,'test_start':test_start,'skipped_folds':skipped}

def top_k_precision(y,p,meta,k=10):
    table=meta[['dealer_id','month']].copy(); table['y']=np.asarray(y); table['p']=p
    rows=[]
    for month,g in table.groupby('month'):
        chosen=g.sort_values(['p','dealer_id'],ascending=[False,True]).head(min(k,len(g)))
        rows.append({'month':str(pd.Timestamp(month).date()),'flagged':len(chosen),'events':int(chosen.y.sum()),'precision':float(chosen.y.mean())})
    return float(np.mean([r['precision'] for r in rows])) if rows else None,rows

def metrics(y,p,threshold=.5,meta=None):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); both=len(np.unique(y))>1
    result={'samples':len(y),'events':int(y.sum()),'positive_rate':float(y.mean()),'Average precision':float(average_precision_score(y,p)) if both else None,'ROC-AUC':float(roc_auc_score(y,p)) if both else None,'precision':float(precision_score(y,p>=threshold,zero_division=0)),'recall':float(recall_score(y,p>=threshold,zero_division=0)),'Brier':float(brier_score_loss(y,p)),'alert_threshold':float(threshold)}
    if meta is not None:
        result['Precision@10/month'],result['top10_months']=top_k_precision(y,p,meta)
    return result

def choose_threshold(y,p):
    # Grid fixed before inspecting test results; highest threshold wins ties.
    return max(SETTINGS['threshold_grid'],key=lambda t:(f1_score(y,p>=t,zero_division=0),t))

def reliability(y,p):
    y=np.asarray(y); p=np.asarray(p); bins=np.minimum((p*5).astype(int),4); rows=[]
    for b in range(5):
        mask=bins==b
        if mask.any(): rows.append({'Bin':f'{b/5:.1f}–{(b+1)/5:.1f}','Samples':int(mask.sum()),'Mean prediction':float(p[mask].mean()),'Observed event rate':float(y[mask].mean())})
    return rows

def make_tabular(name):
    clf=LogisticRegression(max_iter=2000,C=.5,random_state=SETTINGS['seed']) if name==LOGISTIC else RandomForestClassifier(n_estimators=160,max_depth=5,min_samples_leaf=10,random_state=SETTINGS['seed'],n_jobs=1)
    return Pipeline([('scale',StandardScaler()),('classifier',clf)])

def describe_split(meta,idx):
    d=meta.loc[idx]
    return {'from':str(d.month.min().date()),'to':str(d.month.max().date()),'samples':len(d),'events':int(d.label.sum()),'positive_rate':float(d.label.mean()),'latest_label_available':str(d.label_available_at.max().date())}

def gru_inner_split(meta,train_indices):
    training=meta.loc[train_indices]; months=sorted(training.month.unique())
    if len(months)<4: return None
    stop_months=months[-2:]; start=pd.Timestamp(stop_months[0])
    fit_idx=training.index[training.label_available_at<start].to_numpy()
    stop_idx=training.index[training.month.isin(stop_months)].to_numpy()
    if len(fit_idx)<30 or meta.loc[fit_idx,'label'].nunique()<2 or len(stop_idx)<10: return None
    return fit_idx,stop_idx

def run_experiment(df,include_gru=True,settings=SETTINGS):
    data=sequences(df); meta=data['meta']; plan=split_plan(meta,settings)
    if not plan['available']: return {**plan,'fingerprint':fingerprint(df),'exclusions':{k:data[k] for k in ['excluded_gaps','excluded_missing_windows']}}
    names=[PERSISTENCE,LOGISTIC,TREE]; blockers=[]; gru_module=None
    if include_gru:
        try:
            from modules import gru as gru_module
            gru_module.check_backend()
            if all(gru_inner_split(meta,f['train']) is not None for f in plan['folds']): names.append(GRU)
            else: blockers.append('GRU unavailable: insufficient observable history for chronological inner early stopping in every outer fold.')
        except (ImportError,OSError,RuntimeError) as exc:
            blockers.append('GRU unavailable: '+str(exc))
    else: blockers.append('GRU was explicitly excluded from this training run.')
    collected={n:[] for n in names}; indices=[]; fold_reports=[]; best_epochs=[]
    x=data['tabular']; seq=data['x']
    for fold in plan['folds']:
        tr=fold['train']; va=fold['validation']; y=meta.loc[tr,'label'].to_numpy(dtype=int); v=meta.loc[va,'label'].to_numpy(dtype=int)
        predictions={PERSISTENCE:(meta.loc[va,'current_delay'].to_numpy()>CONFIG['payment_delay_target_days']).astype(float)}
        for name in [LOGISTIC,TREE]:
            pipe=make_tabular(name); pipe.fit(x[tr],y); predictions[name]=pipe.predict_proba(x[va])[:,1]
        if GRU in names:
            inner_fit,inner_stop=gru_inner_split(meta,tr)
            _,_,epoch=gru_module.fit(seq[inner_fit],meta.loc[inner_fit,'label'].to_numpy(dtype=int),seq[inner_stop],meta.loc[inner_stop,'label'].to_numpy(dtype=int),settings['gru'],settings['seed']+fold['fold'])
            model=gru_module.refit(seq[tr],y,settings['gru'],settings['seed']+fold['fold'],epoch)
            predictions[GRU]=gru_module.predict(model,seq[va]); best_epochs.append(epoch)
        indices.extend(va.tolist())
        fold_reports.append({'fold':fold['fold'],'freeze_at':fold['freeze_at'],'train':describe_split(meta,tr),'validation':describe_split(meta,va),'metrics':{n:metrics(v,p) for n,p in predictions.items()},'gru_best_epoch':best_epochs[-1] if GRU in names else None,'gru_inner_fit':describe_split(meta,inner_fit) if GRU in names else None,'gru_inner_stop':describe_split(meta,inner_stop) if GRU in names else None})
        for n in names: collected[n].extend(predictions[n].tolist())
    valmeta=meta.loc[indices]; yval=valmeta.label.to_numpy(dtype=int); pooled={n:np.array(v) for n,v in collected.items()}
    thresholds={n:choose_threshold(yval,p) for n,p in pooled.items()}
    validation={n:metrics(yval,p,thresholds[n],valmeta) for n,p in pooled.items()}
    chosen=max(names,key=lambda n:(validation[n]['Average precision'] if validation[n]['Average precision'] is not None else -1,-names.index(n)))
    # Refit on all labels observable before the untouched test starts.
    tr=plan['final_train']; te=plan['test']; ytrain=meta.loc[tr,'label'].to_numpy(dtype=int); ytest=meta.loc[te,'label'].to_numpy(dtype=int)
    fitted={}; test_predictions={PERSISTENCE:(meta.loc[te,'current_delay'].to_numpy()>CONFIG['payment_delay_target_days']).astype(float)}
    for name in [LOGISTIC,TREE]:
        pipe=make_tabular(name); pipe.fit(x[tr],ytrain); fitted[name]=pipe; test_predictions[name]=pipe.predict_proba(x[te])[:,1]
    if GRU in names:
        final_epochs=max(1,int(np.median(best_epochs)))
        fitted[GRU]=gru_module.refit(seq[tr],ytrain,settings['gru'],settings['seed'],final_epochs)
        test_predictions[GRU]=gru_module.predict(fitted[GRU],seq[te])
    test_reports={n:metrics(ytest,p,thresholds[n],meta.loc[te]) for n,p in test_predictions.items()}
    importance={}
    if chosen in [LOGISTIC,TREE]:
        clf=fitted[chosen].named_steps['classifier']; imp=clf.coef_[0] if chosen==LOGISTIC else clf.feature_importances_
        importance=dict(zip(data['features'],imp.tolist()))
    result={'available':True,'fingerprint':fingerprint(df),'version':settings['version'],'chosen':chosen,'selection_metric':settings['selection_metric'],'threshold':thresholds[chosen],'thresholds':thresholds,'validation':validation,'test':test_reports[chosen],'test_comparison':test_reports,'folds':fold_reports,'skipped_folds':plan['skipped_folds'],'splits':{'train':describe_split(meta,tr),'validation':describe_split(meta,indices),'test':describe_split(meta,te)},'test_start':str(plan['test_start'].date()),'freeze_at':str(plan['test_start'].date()),'trained_until':str(meta.loc[tr,'month'].max().date()),'importance':importance,'importance_kind':'standardized signed lag coefficients' if chosen==LOGISTIC else 'global impurity importance' if chosen==TREE else 'no feature attribution implemented','local_attribution':'No genuine local attribution is implemented. Observed KPI values are evidence, not model attributions.','reliability':{n:reliability(ytest,p) for n,p in test_predictions.items()},'blockers':blockers,'exclusions':{k:data[k] for k in ['excluded_gaps','excluded_missing_windows']},'gru_epochs':best_epochs,'final_gru_epochs':final_epochs if GRU in names else None,'fitted_models':fitted,'predictor_features':data['features']}
    return result

def artifact_path(df): return ARTIFACT_DIR/(fingerprint(df)+'.joblib')
def train(df,include_gru=True,force=False):
    """Explicit training entry point; never invoked by prepare()/page reruns."""
    path=artifact_path(df)
    if path.exists() and not force: return joblib.load(path)
    result=run_experiment(df,include_gru)
    ARTIFACT_DIR.mkdir(exist_ok=True); temporary=path.with_suffix('.joblib.tmp'); joblib.dump(result,temporary); temporary.replace(path)
    # Human-readable audit summary without estimator objects.
    summary={k:v for k,v in result.items() if k!='fitted_models'}
    path.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    return result

def forecast(df,result):
    risk=pd.Series(np.nan,index=df.index)
    if not result.get('available'): return risk,None
    data=sequences(df); meta=data['meta']; eligible=meta.month>=pd.Timestamp(result['test_start']); positions=meta.index[eligible].to_numpy()
    if not len(positions): return risk,None
    name=result['chosen']; err=None
    if name==PERSISTENCE:
        p=(meta.loc[positions,'current_delay'].to_numpy()>CONFIG['payment_delay_target_days']).astype(float)
    elif name==GRU:
        try:
            from modules.gru import predict
            p=predict(result['fitted_models'][GRU],data['x'][positions])
        except (ImportError,OSError,RuntimeError) as exc: return risk,'Selected GRU cannot run in this environment: '+str(exc)
    else: p=result['fitted_models'][name].predict_proba(data['tabular'][positions])[:,1]
    lookup={(r.dealer_id,r.month):prob for (_,r),prob in zip(meta.loc[positions].iterrows(),p)}
    for i,row in df.iterrows(): risk.loc[i]=lookup.get((row.dealer_id,row.month),np.nan)
    return risk,err

def load_results(df):
    path=artifact_path(df)
    if not path.exists(): return {'available':False,'reason':'No matching trained artifact. Run the training script or use the explicit training control in Data & Methodology.','fingerprint':fingerprint(df)}
    result=joblib.load(path)
    if result.get('fingerprint')!=fingerprint(df): return {'available':False,'reason':'Artifact fingerprint mismatch; retrain explicitly.'}
    result=dict(result); result['risk'],result['prediction_error']=forecast(df,result)
    return result
