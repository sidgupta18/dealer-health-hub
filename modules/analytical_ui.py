"""Exploration stays on Methodology; operational assessments remain business-weighted."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from modules.analysis import (SETTINGS,BUSINESS,EQUAL,PCA_NAME,fit_pca,score_methods,compare_methods,
                              sensitivity,correlation_diagnostics,business_indicator_weights,health_matrix)
from modules.scoring import CONFIG
from modules.data import COLUMNS,generate,generate_v2
from modules.ui import label,chart

@st.cache_data(show_spinner='Preparing index comparisons and sensitivity scenarios…')
def analyse(df):
    pca=fit_pca(df); methods=score_methods(df,pca)
    corr,flags=correlation_diagnostics(df,pca)
    return pca,compare_methods(methods),sensitivity(df),corr,flags

def metric_report(reports):
    cols=['Average precision','ROC-AUC','precision','recall','Precision@10/month','Brier','alert_threshold']
    table=pd.DataFrame(reports).T
    table=table[[k for k in cols if k in table]]
    return table.rename(columns={'samples':'Samples','events':'Events','positive_rate':'Event rate','precision':'Precision','recall':'Recall','Brier':'Brier score','alert_threshold':'Alert threshold'}).astype(float).round(3)

def display_report(reports):
    table=metric_report(reports)
    st.dataframe(table,width='stretch',column_config={k:st.column_config.NumberColumn(format='%.2f' if k=='Alert threshold' else '%.3f') for k in table})

def render(df,model,raw,clear_callback):
    data=df[COLUMNS].copy()
    st.title('Data & methodology')
    st.caption('Active index: business weights. Alternatives are exploratory and never rewrite action evidence.')
    with st.expander('Hosted AI setup'):
        st.caption('Credentials remain server-side. Configure .streamlit/secrets.toml using the example file, or environment variables. No .env loading is implied.')
        st.code('AI_BASE_URL = \"https://api.groq.com/openai/v1\"\nAI_MODEL_ID = \"openai/gpt-oss-20b\"\nAI_API_KEY = \"YOUR_API_KEY\"\nAI_FREE_PLAN_CONFIRMED = false',language='toml')
        st.caption('Verify your account/model free-plan entitlement before setting AI_FREE_PLAN_CONFIRMED=true. Never paste keys into chat. Restart after configuring local secrets. No model/provider substitution or billing changes.')
        st.link_button('Groq free-plan limits','https://console.groq.com/docs/rate-limits')
    data_tab,index_tab,overall_tab,inventory_tab,predict_tab=st.tabs(['Data Quality','Index Methods','Health outlook','Inventory warning','Exploratory forecasting'])
    with overall_tab:
        from modules.overall_ui import evaluation
        evaluation(data)
    with inventory_tab:
        st.caption('Superseded inventory forecast · preserved for technical review, outside the manager workflow. This predicts aged-stock breaches, not overall health deterioration.')
        from modules.inventory_ui import evaluation
        evaluation(data)
    with index_tab:
        compare_tab,sensitivity_tab=st.tabs(['Weighting comparison','Sensitivity'])
        with compare_tab:
            pca,comparison,robust,corr,flags=analyse(data)
            st.subheader('Weighting methods')
            st.write('Business weights remain the operational default. Equal pillars and PCA-derived variance weights are alternatives for review, not automatic replacements.')
            if pca['available']:
                st.caption(f"PCA reference: {pca['reference_from']} to {pca['reference_to']} · {pca['reference_rows']} complete rows · {pca['retained_components']} components explain {pca['retained_variance']:.1%} variance. Parameters and weights are frozen for later months.")
                if pca['constant_features']: st.info('Constant reference indicators have zero PCA weight: '+', '.join(label(k) for k in pca['constant_features'])+'. They remain required scoring inputs.')
            else: st.warning(pca['reason'])
            c1,c2=st.columns([1,2])
            month=c1.selectbox('Comparison month',sorted(data.month.unique(),reverse=True),format_func=lambda m:m.strftime('%b %Y'),key='comparison_month')
            ids=['All dealers']+data.dealer_id.unique().tolist()
            dealer=c2.selectbox('Comparison dealer',ids,format_func=lambda d:d if d=='All dealers' else data.loc[data.dealer_id==d,'dealer_name'].iloc[0]+' · '+d)
            view=comparison[comparison.month==month]
            if dealer!='All dealers': view=view[view.dealer_id==dealer]
            if pca.get('available') and month<=pd.Timestamp(pca['reference_to']):
                st.info('This month is inside the PCA reference period: its comparison is retrospective. Only later months use a fully historical frozen reference.')
            counts=view.groupby('method').agg(assessed=('raw_index','count'),raw_disagreements=('raw_disagreement','sum'),final_disagreements=('status_disagreement','sum')).reset_index()
            st.dataframe(counts.rename(columns={'method':'Method','assessed':'Assessed','raw_disagreements':'Raw-class disagreements','final_disagreements':'Final-status disagreements'}),hide_index=True,width='stretch')
            st.caption('Disagreements are measured against the business index. Overrides can suppress a raw-class disagreement in the final status.')
            table=view[['dealer_name','method','raw_index','rank','raw_status','final_status','score_difference']].rename(columns={'dealer_name':'Dealer','method':'Method','raw_index':'Raw score /100','rank':'Rank','raw_status':'Raw class','final_status':'Final status','score_difference':'Score Δ vs business'})
            table['Method']=table['Method'].replace({BUSINESS:'Business',EQUAL:'Equal pillars',PCA_NAME:'PCA variance · experimental'})
            st.dataframe(table,hide_index=True,width='stretch',height=300,column_config={k:st.column_config.NumberColumn(format='%+.1f' if k=='Score Δ vs business' else '%.1f') for k in ['Raw score /100','Rank','Score Δ vs business']})
            with st.expander('Indicator weights, implied pillars & contributions'):
                weights=pd.DataFrame({'Metric':[label(k) for k in CONFIG['kpis']],BUSINESS:list(business_indicator_weights().values()),EQUAL:[.25/sum(s['pillar']==CONFIG['kpis'][k]['pillar'] for s in CONFIG['kpis'].values()) for k in CONFIG['kpis']]})
                if pca['available']: weights[PCA_NAME]=list(pca['indicator_weights'].values())
                for column in weights.columns[1:]: weights[column]=(weights[column]*100).round(2)
                st.write('**Indicator weights (%)**'); st.dataframe(weights,hide_index=True,width='stretch')
                pillars=pd.DataFrame({'Pillar':list(CONFIG['weights']),BUSINESS:[w*100 for w in CONFIG['weights'].values()],EQUAL:[25]*4})
                if pca['available']: pillars[PCA_NAME]=[pca['pillar_weights'][p]*100 for p in CONFIG['weights']]
                st.write('**Implied pillar weights (%)**'); st.dataframe(pillars.round(2),hide_index=True,width='stretch')
                st.caption('PCA importance = Σ retained component variance ratio × squared component coefficient; normalize across indicators. Positive variance weights are not proven business importance. Constant indicators get zero weight; missing required inputs still yield Unassessed.')
                if pca['available'] and dealer!='All dealers':
                    selected=data[(data.month==month)&(data.dealer_id==dealer)]
                    health=health_matrix(selected).iloc[0]
                    contributions={p:sum(health[k]*pca['indicator_weights'][k] for k,s in CONFIG['kpis'].items() if s['pillar']==p) for p in CONFIG['weights']}
                    st.dataframe(pd.DataFrame({'Pillar':list(contributions),'PCA score contribution (pts)':list(contributions.values())}).round(2),hide_index=True,width='stretch')
                    st.caption('Grouped indicator contributions sum to the PCA raw index when all required inputs exist.')
            with st.expander('Correlation diagnostics · historical reference'):
                st.caption(f'Pearson correlations on complete, direction-aligned reference scores; |r| ≥{SETTINGS["correlation_flag_abs"]:.2f} flags possible redundancy. No indicators are deleted automatically.')
                if not flags.empty:
                    table=flags.copy(); table['Metric 1']=table['Metric 1'].map(label); table['Metric 2']=table['Metric 2'].map(label)
                    st.dataframe(table.round(3),hide_index=True,width='stretch')
                else: st.info('No pairs exceed the review threshold, or insufficient usable reference data.')
                if not corr.empty:
                    heat=corr.rename(index=label,columns=label)
                    fig=px.imshow(heat,zmin=-1,zmax=1,color_continuous_scale=['#214F87','#FFFFFF','#725A8E'],labels={'color':'Correlation'},aspect='auto')
                    fig.update_xaxes(tickangle=-45); chart(fig,height=480)
        with sensitivity_tab:
            st.subheader('How stable is the business index?')
            st.caption(f'{robust["scenario_count"]} reproducible draws per scenario family · seed {robust["seed"]}. These are sensitivity ranges, not confidence intervals.')
            st.write('Weights: each pillar changes by up to ±20% relative, then weights renormalize to sum to one. Anchors: each good/bad anchor shifts independently by up to 5% of its original span; direction is preserved. Overrides stay fixed.')
            smonth=st.selectbox('Sensitivity month',sorted(data.month.unique(),reverse=True),format_func=lambda m:m.strftime('%b %Y'))
            scenario=st.selectbox('Scenario family',['Pillar weights ±20%','Anchors ±5% of span'])
            stable=robust['table']; stable=stable[(stable.month==smonth)&(stable.scenario==scenario)].copy()
            table=stable[['dealer_name','raw_min','raw_max','raw_class_stability','final_status_stability','max_rank_change','baseline_raw_status','baseline_final_status']].rename(columns={'dealer_name':'Dealer','raw_min':'Raw minimum','raw_max':'Raw maximum','raw_class_stability':'Raw-class stability','final_status_stability':'Final-status stability','max_rank_change':'Max rank move','baseline_raw_status':'Raw class','baseline_final_status':'Final status'})
            for k in ['Raw-class stability','Final-status stability']: table[k]=(table[k]*100).round(1)
            st.dataframe(table,hide_index=True,width='stretch',height=340,column_config={k:st.column_config.NumberColumn(format='%.1f%%' if 'stability' in k else '%.1f') for k in ['Raw minimum','Raw maximum','Raw-class stability','Final-status stability','Max rank move']})
            st.caption('Stability is the percentage of perturbed scenarios retaining the baseline class/status. Raw envelopes include the operational baseline. Missing inputs have no raw range or rank; their final status remains Unassessed. Ranks compare eligible dealers within each month.')
            with st.expander('Exact scenario draws & anchor ranges'):
                st.dataframe(pd.DataFrame(robust['weight_draws'],columns=CONFIG['weights']).round(4),hide_index=True,width='stretch')
                st.dataframe(pd.DataFrame(robust['anchor_draw_ranges']).round(3),hide_index=True,width='stretch')
    with predict_tab:
        st.subheader('Exploratory forecasting')
        st.caption('Technical appendix · superseded in the core workflow. Payment target requires business validation; results do not inform health priority or operational recommendations.')
        if st.checkbox('Load experimental forecasting results',value=False):
            from modules.model import load_results,train,PERSISTENCE
            model=load_results(data)
        else:
            model={'available':False,'reason':'Experimental artifacts are not loaded. Select the control above to inspect the preserved results.'}
        st.caption('Forecast made at reporting-month end after that month’s metrics are available. Target: next calendar month’s reported average payment delay >20 days. Label becomes available at next month end; no reporting lag is assumed.')
        with st.expander('Training controls & evaluation protocol'):
            st.write('Every model uses the same six consecutive calendar months. Dealer IDs are metadata only. Validation expands chronologically; preprocessing fits inside each training window. Final selection and alert thresholds use validation only.')
            if st.button('Train / reuse matching predictive artifact',disabled='train' not in locals()):
                try:
                    with st.spinner('Training six-month baselines and optional GRU; this is an explicit training run…'):
                        train(data)
                    clear_callback(); st.rerun()
                except Exception as exc: st.error(f'Training failed: {exc}. Current scoring and saved actions remain available.')
            st.caption('Training is never invoked on a normal rerun. Artifacts are keyed by data, scoring, experiment configuration and implementation fingerprints.')
        if model['available']:
            st.write(f'**Selected model: {model["chosen"]}** · highest pooled expanding-validation average precision · alert threshold {model["threshold"]:.2f}')
            if model.get('prediction_error'): st.warning(model['prediction_error'])
            for blocker in model.get('blockers',[]): st.warning(blocker)
            st.write('**Persistence baseline**: predict next month’s breach status equals this month’s (>20 days). Its outputs are binary 0/1, not calibrated probabilities.')
            validation_panel,test_panel,reliability_panel=st.tabs(['Validation','Held-out Test','Reliability'])
            with validation_panel:
                st.write('**Validation comparison · selection data**')
                vm=model['validation'][model['chosen']]
                st.caption(f'{vm["samples"]} common sequences · {vm["events"]} events · {vm["positive_rate"]:.1%} event rate. Metrics are on a 0–1 scale.')
                display_report(model['validation'])
            with test_panel:
                st.write('**Held-out test · same eligible sequences for every model**')
                tm=model['test']
                st.caption(f'{tm["samples"]} common sequences · {tm["events"]} events · {tm["positive_rate"]:.1%} event rate. Model and thresholds were fixed before test evaluation.')
                display_report(model['test_comparison'])
                st.caption('Average precision / ROC-AUC are undefined on single-class evaluations. Precision@10 averages monthly precision among the 10 highest-risk dealers (all eligible dealers if fewer than 10); ties break by dealer ID. Alert threshold maximizes pooled-validation F1 on a fixed grid, ties favor the higher threshold.')
            with reliability_panel:
                selected=PERSISTENCE if PERSISTENCE in model['reliability'] else model['chosen']
                name=st.selectbox('Reliability plot model',list(model['reliability']),index=list(model['reliability']).index(selected))
                bins=pd.DataFrame(model['reliability'][name])
                fig=go.Figure(); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Perfect-reliability reference',line=dict(color='#8796AB',dash='dash')))
                fig.add_trace(go.Scatter(x=bins['Mean prediction'],y=bins['Observed event rate'],mode='lines+markers',name=name,customdata=bins['Samples'],hovertemplate='Mean prediction: %{x:.2f}<br>Observed event rate: %{y:.2f}<br>Samples: %{customdata}<extra></extra>',line=dict(color='#214F87')))
                fig.update_layout(xaxis_title='Mean predicted score (five bins)',yaxis_title='Observed breach rate'); fig.update_xaxes(range=[0,1]); fig.update_yaxes(range=[0,1]); chart(fig)
                st.caption('Diagnostic on the untouched test set, not a claim that probabilities are calibrated. Empty bins are omitted.')
            with st.expander('Fold dates, label availability, event counts & exclusions'):
                rows=[]
                for fold in model['folds']:
                    for split in ['train','validation']:
                        rows.append({'Fold':fold['fold'],'Split':split,'Frozen before':fold['freeze_at'],**fold[split]})
                st.dataframe(rows,hide_index=True,width='stretch')
                st.dataframe(pd.DataFrame(model['splits']).T,width='stretch')
                st.json(model['exclusions']); st.json(model['skipped_folds'])
                st.json([{'fold':f['fold'],'inner_fit':f['gru_inner_fit'],'inner_early_stop':f['gru_inner_stop']} for f in model['folds']])
                st.caption(f"GRU best epochs by fold: {model['gru_epochs']}; final refit budget: {model['final_gru_epochs']}. Final GRU uses the median validation-selected epoch count; no test-based early stopping.")
                st.write('Monthly top-10 results'); st.dataframe(model['test']['top10_months'],hide_index=True,width='stretch')
            with st.expander('Global importance & explanation limits'):
                st.write('Observed dealer metrics are evidence, not local attribution. No genuine local attribution is implemented.')
                if model['importance']:
                    def lag_label(k):
                        metric,lag=k.split('__lag'); return label(metric)+f' · lag {lag} months'
                    imp=pd.DataFrame([(lag_label(k),v) for k,v in model['importance'].items()],columns=['Feature','Importance'])
                    imp=imp.assign(magnitude=imp.Importance.abs()).nlargest(12,'magnitude').sort_values('Importance')
                    chart(px.bar(imp,x='Importance',y='Feature',orientation='h',color_discrete_sequence=['#214F87']),height=420)
                    st.caption(model['importance_kind']+' · global associations, not causal or dealer-specific explanations.')
                else: st.info('No feature attribution is reported for persistence or the GRU. No GRU explanations have been invented.')
            st.caption('This evaluates future months for existing dealers. It does not establish new-dealer generalization or production validity. Synthetic results depend on the generator assumptions.')
        else: st.warning(model['reason'])
    with data_tab:
        st.subheader('Data quality')
        st.write(f'**Schema accepted** · {len(data):,} monthly records · {data.dealer_id.nunique()} dealers')
        missing=data.isna().sum(); missing=missing[missing>0]
        for k,n in missing.items(): st.warning(f'{label(k)}: {n} missing values. Affected health rows are Unassessed.')
        st.download_button('Download sample CSV · v2',generate_v2().to_csv(index=False),'sample_dealers_v2.csv','text/csv')
        st.download_button('Download original sample · v1',generate().to_csv(index=False),'sample_dealers.csv','text/csv')
        st.caption('The versioned v2 demo adds seasonality and random shocks. The original v1 CSV and SQLite actions are preserved. Uploaded data stays in session.')
        with st.expander('Scoring, experiment assumptions & limitations'):
            from pathlib import Path
            st.markdown((Path(__file__).resolve().parents[1]/'METHODOLOGY.md').read_text())
        with st.expander('Central configuration'):
            st.json(CONFIG); st.json(SETTINGS)
