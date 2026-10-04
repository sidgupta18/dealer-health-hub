"""Fixed-index deterioration: explicit training, purged evaluation, saved inference."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from modules.data import COLUMNS
from modules.scoring import CONFIG,score_data
from modules.inventory_warning import splits,report as inventory_report
ROOT=Path(__file__).resolve().parents[1]
SETTINGS=json.loads((ROOT/'config/overall_deterioration.json').read_text())
TREND='Recent health trend';PREVALENCE='Training prevalence';LR='Regularised logistic regression';RF='Constrained random forest'
DEFINITION='Persistent health-score deterioration over the next three months: raw fixed business-weighted health score at least 10 points below the current score in at least two of the next three calendar months; the two months need not be consecutive.'
LIMITS='Prototype index-decline rule, not bankruptcy, business failure or proven causes. Scores are not calibrated percentage chances. Synthetic evidence concerns later months of the existing network, not unseen dealers; repeated dealers and overlapping horizons reduce independence.'
SHOWCASES={'D001':'Aarav Motors','D002':'Neel Auto','D003':'Surya Mobility','D004':'Kiran Wheels'}

def fingerprint(df,config=SETTINGS):
 f=df[COLUMNS].copy().sort_values(['dealer_id','month']);f['month']=pd.to_datetime(f.month).dt.strftime('%Y-%m-%d')
 data=hashlib.sha256(f.to_csv(index=False,float_format='%.10g').encode()).hexdigest()
 source=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ['modules/overall_deterioration.py','modules/scoring.py','modules/inventory_warning.py'])).hexdigest()
 scoring=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
 token=hashlib.sha256((data+source+scoring+json.dumps(config,sort_keys=True)).encode()).hexdigest()[:24]
 return dict(data_fingerprint=data,implementation_fingerprint=source,scoring_version=scoring,artifact_fingerprint=token,model_version=config['version'])

def definition(config=SETTINGS):
 return f"Persistent health-score deterioration over the next three months: raw fixed business-weighted health score at least {config['drop_points']:g} points below the current score in at least {config['qualifying_months']} of the next three calendar months; the two months need not be consecutive."

def label(scores,month,config=SETTINGS):
 dates=pd.date_range(pd.Timestamp(month)+pd.offsets.MonthBegin(1),periods=3,freq='MS')
 if month not in scores.index or not dates.isin(scores.index).all():return np.nan
 values=scores.loc[[month,*dates]]
 if not np.isfinite(values.to_numpy(dtype=float)).all():return np.nan
 return float((values.iloc[1:]<=values.iloc[0]-config['drop_points']).sum()>=config['qualifying_months'])

def dataset(df,config=SETTINGS):
 if config['input_months']!=6 or config['horizon_months']!=3 or not 1<=config['qualifying_months']<=3 or config['drop_points']<=0:raise ValueError('Target configuration requires six input months, three future months, positive drop and one to three qualifying months.')
 d=df[COLUMNS].copy();d.month=pd.to_datetime(d.month).dt.to_period('M').dt.to_timestamp()
 if d.duplicated(['dealer_id','month']).any():raise ValueError('Duplicate dealer-month observations.')
 d=score_data(d);keys=list(CONFIG['kpis'])+['raw_index']+[p+'_score' for p in CONFIG['weights']]
 names=[f'{k}__lag{lag}' for lag in range(5,-1,-1) for k in keys]+[f'{k}__{change}' for change in ['change1','change3','trend6'] for k in keys]
 inputs=[];rows=[];excluded={'insufficient_or_gapped_history':0,'unknown_future_labels':0}
 for dealer,g in d.groupby('dealer_id'):
  g=g.set_index('month').sort_index()
  for month,row in g.iterrows():
   dates=pd.date_range(end=month,periods=6,freq='MS')
   if not dates.isin(g.index).all() or g.reindex(dates).raw_index.isna().any():excluded['insufficient_or_gapped_history']+=1;continue
   a=g.loc[dates,keys].to_numpy(float);trend=np.polyfit(np.arange(6),a,1)[0]
   inputs.append(np.concatenate([a.ravel(),a[-1]-a[-2],a[-1]-a[-4],trend]))
   y=label(g.raw_index,month,config);future=pd.date_range(month+pd.offsets.MonthBegin(1),periods=3,freq='MS')
   if pd.isna(y):excluded['unknown_future_labels']+=1
   health=g.loc[dates,'raw_index'].to_numpy(float)
   rows.append(dict(dealer_id=dealer,month=month,label=y,label_available_at=future[-1]+pd.offsets.MonthEnd(0) if pd.notna(y) else pd.NaT,current_health=float(health[-1]),health_history=health.tolist(),health_slope=float(np.polyfit(np.arange(3),health[-3:],1)[0]),showcase=SHOWCASES.get(dealer)==row.dealer_name))
 return dict(x=np.array(inputs).reshape(-1,len(names)),rows=pd.DataFrame(rows,columns=['dealer_id','month','label','label_available_at','current_health','health_history','health_slope','showcase']),feature_names=names,exclusions=excluded)

def baseline_fit(rows):
 residual=[]
 for h in rows.health_history:
  a=np.array(h[-3:]);residual.extend(a-np.polyval(np.polyfit(np.arange(3),a,1),np.arange(3)))
 return dict(sigma=max(2.,float(np.std(residual))))
def baseline_predict(rows,model,config=SETTINGS):
 future=np.clip(rows.current_health.to_numpy()[:,None]+rows.health_slope.to_numpy()[:,None]*np.arange(1,4),0,100)
 second=np.sort(future,axis=1)[:,config['qualifying_months']-1]
 z=np.clip((rows.current_health.to_numpy()-config['drop_points']-second)/(model['sigma']*math.sqrt(3)),-30,30)
 return 1/(1+np.exp(-z))
def metrics(y,p,rows,threshold):
 from sklearn.metrics import confusion_matrix
 r=inventory_report(y,p,rows,threshold);r['confusion_matrix']=confusion_matrix(y,np.asarray(p)>=threshold,labels=[0,1]).tolist();return r

def train(df,config=SETTINGS,artifact_dir=None):
 import joblib
 from sklearn.pipeline import make_pipeline
 from sklearn.impute import SimpleImputer
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import fbeta_score
 fp=fingerprint(df,config);directory=Path(artifact_dir or ROOT/'artifacts/overall_deterioration');directory.mkdir(parents=True,exist_ok=True);path=directory/(fp['artifact_fingerprint']+'.joblib')
 if path.exists():return joblib.load(path)['report']
 full=dataset(df,config)
 if full['rows'].empty:return {**fp,'available':False,'reason':'Insufficient history: no complete six-month input windows.','eligible_labelled_rows':0,'positive_events':0,'exclusions':full['exclusions']}
 keep=~full['rows'].showcase.to_numpy(dtype=bool);rows=full['rows'].loc[keep].reset_index(drop=True);x=full['x'][keep];data={**full,'rows':rows,'x':x};plan=splits(data,config)
 if not plan['available']:return {**fp,**plan,'eligible_labelled_rows':int(rows.label.notna().sum()),'positive_events':int(rows.label.sum()),'exclusions':full['exclusions']}
 def fit(ix):
  models={TREND:baseline_fit(rows.iloc[ix]),PREVALENCE:float(rows.iloc[ix].label.mean()),LR:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.2,max_iter=1500,random_state=config['seed'])),RF:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(n_estimators=180,max_depth=4,min_samples_leaf=12,max_features=.6,n_jobs=1,random_state=config['seed']))}
  for name in [LR,RF]:models[name].fit(x[ix],rows.iloc[ix].label)
  return models
 def predict(models,name,ix):
  return baseline_predict(rows.iloc[ix],models[name],config) if name==TREND else np.full(len(ix),models[name]) if name==PREVALENCE else models[name].predict_proba(x[ix])[:,1]
 def support(ix):
  r=rows.iloc[ix];return dict(samples=len(r),events=int(r.label.sum()),prevalence=float(r.label.mean()),start=str(r.month.min().date()),end=str(r.month.max().date()),latest_label_available=str(r.label_available_at.max().date()))
 pooled={n:[] for n in [TREND,PREVALENCE,LR,RF]};indices=[];folds=[]
 for fold in plan['folds']:
  models=fit(fold['train']);va=fold['validation'];indices.extend(va.tolist())
  for name in pooled:pooled[name].extend(predict(models,name,va).tolist())
  folds.append(dict(freeze_at=fold['freeze_at'],train=support(fold['train']),validation=support(va)))
 vr=rows.iloc[indices];vy=vr.label.to_numpy(int);validation={};thresholds={}
 for name,p in pooled.items():
  threshold=max(config['alert_grid'],key=lambda t:(fbeta_score(vy,np.asarray(p)>=t,beta=2,zero_division=0),t));thresholds[name]=float(threshold);validation[name]=metrics(vy,p,vr,threshold)
 bestbaseline=max([TREND,PREVALENCE],key=lambda n:validation[n]['average_precision']);bestml=max([LR,RF],key=lambda n:validation[n]['average_precision'])
 chosen=bestml if int(vy.sum())>=config['minimum_validation_events'] and validation[bestml]['average_precision']>=validation[bestbaseline]['average_precision']+config['meaningful_ap_improvement'] else bestbaseline
 models=fit(plan['final_train']);test={n:metrics(rows.iloc[plan['test']].label,predict(models,n,plan['test']),rows.iloc[plan['test']],thresholds[n]) for n in models}
 result={**fp,'available':True,'chosen':chosen,'alert_threshold':thresholds[chosen],'config':config,'target_definition':definition(config),'features':full['feature_names'],'validation':validation,'test':test,'folds':folds,'skipped_folds':plan['skipped_folds'],'final_train':support(plan['final_train']),'test_freeze':plan['test_freeze'],'exclusions':full['exclusions'],'showcase_exclusions':SHOWCASES,'excluded_showcase_rows':int((~keep).sum()),'selection':'Validation AP; ML requires +0.02 AP over the better baseline and at least five validation events. Threshold maximises validation F2 (recall weighted four times precision), ties higher threshold. Test never selects models or thresholds.','limitations':LIMITS}
 joblib.dump(dict(report=result,models=models),path);path.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result

def load(df,config=SETTINGS,artifact_dir=None):
 fp=fingerprint(df,config);directory=Path(artifact_dir or ROOT/'artifacts/overall_deterioration');path=directory/(fp['artifact_fingerprint']+'.joblib')
 if not path.exists():return {**fp,'available':False,'reason':'Artifact stale: data, scoring or implementation changed. Train explicitly.' if directory.exists() and list(directory.glob('*.joblib')) else 'No overall-health artifact. Run scripts/train_overall_deterioration.py.'},None
 try:
  import joblib
  saved=joblib.load(path)
  if any(saved['report'].get(k)!=v for k,v in fp.items()):raise ValueError('Artifact fingerprint mismatch')
  return saved['report'],saved['models']
 except Exception as exc:return {**fp,'available':False,'reason':'Model artifact error: '+str(exc)},None

def forecasts(df,result,models):
 if not result.get('available') or models is None:return {}
 d=dataset(df,result['config']);r=d['rows'];ix=r.index[r.month>=pd.Timestamp(result['test_freeze'])].to_numpy();r=r.loc[ix];name=result['chosen']
 if not len(ix):return {}
 p=baseline_predict(r,models[name],result['config']) if name==TREND else np.full(len(r),models[name]) if name==PREVALENCE else models[name].predict_proba(d['x'][ix])[:,1]
 return {(row.dealer_id,row.month):float(score) for (_,row),score in zip(r.iterrows(),p)}

def outlook(row,result=None,predictions=None):
 result=result or {};month=pd.Timestamp(row.month);start=month+pd.offsets.MonthBegin(1);end=month+pd.offsets.MonthBegin(3);score=(predictions or {}).get((row.dealer_id,month));available=score is not None
 reason='' if available else result.get('reason','Insufficient history: six consecutive months with valid fixed health scores required.')
 if result.get('available') and not available and pd.isna(row.get('raw_index',np.nan)):reason='Current health-score inputs missing; no overall forecast generated.'
 if result.get('available') and month<pd.Timestamp(result['test_freeze']):reason='Historical month precedes the model freeze; retrospective forecast suppressed.'
 if score is not None and (not np.isfinite(score) or not 0<=score<=1):score=None;available=False;reason='Model inference error: non-finite or invalid model score.'
 status=('Deterioration flagged' if score>=result['alert_threshold'] else 'No deterioration flag') if available else 'Model error' if 'error' in reason.lower() else 'Insufficient history' if result.get('available') else 'Artifact stale' if 'stale' in reason.lower() else 'Model not trained'
 return dict(status=status,available=available,flagged=available and score>=result['alert_threshold'],score=score,reason=reason,window=start.strftime('%B')+'–'+end.strftime('%B %Y'),prediction_date=str((month+pd.offsets.MonthEnd(0)).date()),horizon_start=str(start.date()),horizon_end=str((end+pd.offsets.MonthEnd(0)).date()),event_definition=result.get('target_definition',definition()),drop_points=result.get('config',SETTINGS)['drop_points'],qualifying_months=result.get('config',SETTINGS)['qualifying_months'],model=result.get('chosen'),model_version=result.get('model_version'),artifact_fingerprint=result.get('artifact_fingerprint'),scoring_version=result.get('scoring_version'),alert_threshold=result.get('alert_threshold'),training_cutoff=result.get('final_train',{}).get('latest_label_available'),evidence_id='FORECAST:overall',limitations=LIMITS)
