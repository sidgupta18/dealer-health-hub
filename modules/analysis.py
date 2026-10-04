"""Frozen PCA weighting, correlation diagnostics and reproducible sensitivity."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from modules.scoring import CONFIG, STATUS_ORDER, score_data
SETTINGS = json.loads((Path(__file__).resolve().parents[1]/'config/analysis.json').read_text())
BUSINESS='Business weights'
EQUAL='Equal pillar weights'
PCA_NAME='PCA-derived variance weights · experimental'

def health_matrix(df, config=CONFIG):
    d=df.copy(); d['target_attainment_pct']=100*d.sales_units/d.target_units
    return pd.DataFrame({k:((d[k]-s['bad'])/(s['good']-s['bad'])*100).clip(0,100) for k,s in config['kpis'].items()},index=d.index)

def business_indicator_weights(config=CONFIG):
    return {k:config['weights'][s['pillar']]/sum(v['pillar']==s['pillar'] for v in config['kpis'].values()) for k,s in config['kpis'].items()}

def fit_pca(df, config=CONFIG, settings=SETTINGS):
    months=sorted(df.month.unique())
    if len(months)<=settings['pca_reference_months']:
        return {'available':False,'reason':'PCA needs a historical reference of 12 months and at least one later month.'}
    end=pd.Timestamp(months[settings['pca_reference_months']-1]); aligned=health_matrix(df,config)
    reference=aligned.loc[df.month<=end].dropna()
    report={'available':False,'reference_from':str(pd.Timestamp(months[0]).date()),'reference_to':str(end.date()),'reference_rows':len(reference),'excluded_missing_reference_rows':int((df.month<=end).sum()-len(reference))}
    if len(reference)<settings['pca_min_complete_rows']:
        return {**report,'reason':'Too few complete reference rows for PCA.'}
    constant=[k for k in reference if reference[k].std(ddof=0)<=1e-10]
    varying=[k for k in reference if k not in constant]
    if len(varying)<2:
        return {**report,'constant_features':constant,'reason':'PCA requires at least two varying reference indicators.'}
    scaler=StandardScaler().fit(reference[varying]); transformed=scaler.transform(reference[varying])
    pca=PCA(svd_solver='full').fit(transformed)
    retained=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),settings['pca_variance_threshold'])+1)
    importance=(pca.components_[:retained]**2 * pca.explained_variance_ratio_[:retained,None]).sum(axis=0)
    importance/=importance.sum()
    weights={k:0.0 for k in aligned}
    weights.update(dict(zip(varying,importance.tolist())))
    pillar={p:sum(weights[k] for k,s in config['kpis'].items() if s['pillar']==p) for p in config['weights']}
    return {**report,'available':True,'indicator_weights':weights,'pillar_weights':pillar,'constant_features':constant,'varying_features':varying,'retained_components':retained,'retained_variance':float(pca.explained_variance_ratio_[:retained].sum()),'component_variance':pca.explained_variance_ratio_.tolist(),'scaler':scaler,'pca':pca,'correlation':reference.corr()}

def score_methods(df, pca=None):
    result={BUSINESS:score_data(df)}
    equal=copy.deepcopy(CONFIG); equal['weights']={p:.25 for p in CONFIG['weights']}
    result[EQUAL]=score_data(df,equal,method=EQUAL)
    if pca and pca['available']:
        result[PCA_NAME]=score_data(df,indicator_weights=pca['indicator_weights'],method=PCA_NAME)
    return result

def compare_methods(methods):
    base=methods[BUSINESS]; rows=[]
    for name,data in methods.items():
        rank=data.groupby('month').raw_index.rank(ascending=False,method='average')
        rows.append(pd.DataFrame({'dealer_id':data.dealer_id,'dealer_name':data.dealer_name,'month':data.month,'method':name,'raw_index':data.raw_index,'rank':rank,'raw_status':data.raw_status,'final_status':data.status,'score_difference':data.raw_index-base.raw_index,'status_disagreement':data.status.ne(base.status),'raw_disagreement':data.raw_status.ne(base.raw_status)}))
    return pd.concat(rows,ignore_index=True)

def correlation_diagnostics(df,pca,settings=SETTINGS):
    if 'reference_to' not in pca: return pd.DataFrame(),pd.DataFrame()
    ref=health_matrix(df.loc[df.month<=pd.Timestamp(pca['reference_to'])]).dropna()
    corr=ref.corr(); flags=[]
    for i,k in enumerate(corr):
        for other in corr.columns[i+1:]:
            r=corr.loc[k,other]
            if pd.notna(r) and abs(r)>=settings['correlation_flag_abs']:
                flags.append({'Metric 1':k,'Metric 2':other,'Correlation':r})
    return corr,pd.DataFrame(flags,columns=['Metric 1','Metric 2','Correlation'])

def severity(scores,config=CONFIG):
    result=np.full(scores.shape,3,dtype=int)
    for name,threshold in sorted(config['classes'].items(),key=lambda pair:pair[1]):
        result[scores>=threshold]=STATUS_ORDER.index(name)
    result[~np.isfinite(scores)]=4
    return result

def sensitivity(df,config=CONFIG,settings=SETTINGS):
    d=score_data(df,config); aligned=health_matrix(d,config)
    baseline=d.raw_index.to_numpy(); n=settings['sensitivity_scenarios']; rng=np.random.default_rng(settings['sensitivity_seed'])
    pillar_names=list(config['weights']); base_weights=np.array(list(config['weights'].values()))
    perturb=rng.uniform(-settings['pillar_weight_relative_perturbation'],settings['pillar_weight_relative_perturbation'],size=(n,len(pillar_names)))
    weights=base_weights*(1+perturb); weights/=weights.sum(axis=1,keepdims=True)
    pillars=np.column_stack([aligned[[k for k,s in config['kpis'].items() if s['pillar']==p]].mean(axis=1,skipna=False) for p in pillar_names])
    weight_scores=weights@pillars.T
    # Move good and bad anchors independently by <=5% of original span.
    # Original span cannot reverse: minimum resulting span is 90% of original.
    anchor_scores=np.zeros((n,len(d))); anchor_samples=[]
    iw=business_indicator_weights(config)
    for k,s in config['kpis'].items():
        span=abs(s['good']-s['bad']); shifts=rng.uniform(-1,1,(n,2))*settings['anchor_perturbation_fraction_of_span']*span
        good=s['good']+shifts[:,0]; bad=s['bad']+shifts[:,1]
        score=np.clip((d[k].to_numpy()[None,:]-bad[:,None])/(good-bad)[:,None]*100,0,100)
        anchor_scores+=iw[k]*score
        anchor_samples.append({'metric':k,'good_min':float(good.min()),'good_max':float(good.max()),'bad_min':float(bad.min()),'bad_max':float(bad.max())})
    missing=aligned.isna().any(axis=1).to_numpy()
    weight_scores[:,missing]=np.nan; anchor_scores[:,missing]=np.nan
    base_raw=severity(baseline,config); base_final=np.array([STATUS_ORDER.index(s) if s in STATUS_ORDER else 4 for s in d.status])
    floor=np.zeros(len(d),dtype=int)
    ops={'>=':np.greater_equal,'>':np.greater,'<=':np.less_equal,'<':np.less}
    for rule in config['overrides']:
        hit=ops[rule['operator']](d[rule['metric']].to_numpy(),rule['threshold'])
        floor[hit]=np.maximum(floor[hit],STATUS_ORDER.index(rule['status']))
    rows=[]
    for name,cube in [('Pillar weights ±20%',weight_scores),('Anchors ±5% of span',anchor_scores)]:
        # Include the operational baseline in the displayed sensitivity envelope.
        cube=np.vstack([baseline,cube]); raw=severity(cube,config); final=np.maximum(raw,floor[None,:]); final[:,missing]=4
        ranks=np.full(cube.shape,np.nan)
        for _,indices in d.groupby('month').groups.items():
            positions=d.index.get_indexer(indices)
            ranks[:,positions]=rankdata(-cube[:,positions],axis=1,method='average',nan_policy='omit')
        mins=np.full(len(d),np.nan); maxs=mins.copy(); rankmove=mins.copy()
        complete=~missing
        mins[complete]=cube[:,complete].min(axis=0); maxs[complete]=cube[:,complete].max(axis=0)
        rankmove[complete]=np.abs(ranks[:,complete]-ranks[0,complete]).max(axis=0)
        raw_stability=(raw[1:]==base_raw).mean(axis=0); raw_stability[missing]=np.nan
        final_stability=(final[1:]==base_final).mean(axis=0)
        rows.append(pd.DataFrame({'dealer_id':d.dealer_id,'dealer_name':d.dealer_name,'month':d.month,'scenario':name,'raw_min':mins,'raw_max':maxs,'raw_class_stability':raw_stability,'final_status_stability':final_stability,'max_rank_change':rankmove,'baseline_raw_status':d.raw_status,'baseline_final_status':d.status}))
    return {'table':pd.concat(rows,ignore_index=True),'weight_draws':weights,'anchor_draw_ranges':anchor_samples,'scenario_count':n,'seed':settings['sensitivity_seed']}
