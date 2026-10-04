"""Compact inventory prediction panel and transparent evaluation controls."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from modules.inventory_warning import SETTINGS,LIMITS,train,load,forecasts,warning,fingerprint
from modules.ui import chart

@st.cache_resource(show_spinner='Loading matching inventory-warning artifact…')
def _cached_inventory(df,artifact_fingerprint):
    result,models=load(df)
    return result,forecasts(df,result,models)

def prepare_inventory(df):
    return _cached_inventory(df,fingerprint(df)['artifact_fingerprint'])

prepare_inventory.clear=_cached_inventory.clear

def panel(row,history,result,predictions):
    current=warning(row,result,predictions)
    with st.container(border=True,key='card_inventory_warning'):
        st.subheader('Early warning · inventory')
        st.markdown('**'+current['status']+'**')
        if current['score'] is not None:
            st.caption(f"Model score {current['score']:.1%} · alert threshold {current['alert_threshold']:.2f} · not established as calibrated")
        st.caption(f"Prediction date {current['prediction_date']} · horizon {current['horizon_start']} to {current['horizon_end']} · {current['model'] or 'No matching model'}")
        st.caption(current['event_definition']+' '+current['threshold_status']+'. Separate from current health and readiness.')
        trend=history.tail(6)
        fig=px.line(trend,x='month',y='stock_over_90_days_pct',markers=True,color_discrete_sequence=['#0066CC'],labels={'month':'Reporting month','stock_over_90_days_pct':'Stock older than 90 days (%)'})
        fig.add_hline(y=SETTINGS['event_threshold_pct'],line_dash='dot',line_color='#737375',annotation_text=f"{SETTINGS['event_threshold_pct']:g}% · provisional threshold")
        chart(fig,210)
        st.caption('Observed inventory movement supports investigation; it is not a causal or dealer-specific model explanation.')
        with st.expander('Forecast evidence & limitations'):
            st.caption(LIMITS);st.json(current)
    return current

def evaluation(df):
    st.subheader('Inventory early warning')
    st.caption(f"Future aged-stock share >{SETTINGS['event_threshold_pct']:g}% in any of the next three calendar months. Provisional demo threshold; business validation required.")
    if st.button('Train / reuse inventory-warning artifact',key='train_inventory_warning'):
        try:
            with st.spinner('Training purged inventory baselines and evaluating the untouched test period…'):
                result=train(df)
            prepare_inventory.clear()
            if result['available']:st.rerun()
            else:st.warning(result['reason'])
        except Exception as exc:st.error('Inventory training failed: '+str(exc)+'. Health scoring and saved actions remain available.')
    result,_=prepare_inventory(df)
    if not result['available']:
        st.caption(result['reason']);return
    st.markdown(f"**Operational model: {result['chosen']}** · alert threshold {result['alert_threshold']:.2f}")
    st.caption(result['selection']+' '+SETTINGS['alert_threshold_rule']+'.')
    va,te,rel=st.tabs(['Inventory validation','Inventory test','Inventory reliability'])
    columns={'samples':'Samples','events':'Events','prevalence':'Event prevalence','average_precision':'Average precision','roc_auc':'ROC-AUC','precision':'Precision','recall':'Recall','precision_top10_flagged':'Precision@10 flagged/month','brier':'Brier score','alert_threshold':'Alert threshold'}
    for tab,key in [(va,'validation'),(te,'test')]:
        with tab:
            table=pd.DataFrame(result[key]).T[list(columns)].rename(columns=columns)
            st.dataframe(table,width='stretch',column_config={k:st.column_config.NumberColumn(format='%.0f' if k in ['Samples','Events'] else '%.1f%%' if k=='Event prevalence' else '%.3f') for k in table.columns if k!='Event prevalence'}|{'Event prevalence':st.column_config.NumberColumn(format='percent')})
            st.caption('Precision@10 considers at most 10 eligible flagged dealers each month. Months with no flags are excluded from that average and disclosed in the audit. Repeated dealers and overlapping horizons reduce independence.')
    with rel:
        name=st.selectbox('Inventory reliability model',list(result['reliability']))
        bins=pd.DataFrame(result['reliability'][name]);fig=go.Figure()
        fig.add_scatter(x=[0,1],y=[0,1],mode='lines',name='Reference',line=dict(color='#999999',dash='dot'))
        if not bins.empty:fig.add_scatter(x=bins.mean_score,y=bins.observed_rate,mode='lines+markers',name=name,customdata=bins.samples,hovertemplate='Mean score %{x:.2f}<br>Observed event rate %{y:.2f}<br>Samples %{customdata}<extra></extra>',line=dict(color='#0066CC'))
        fig.update_layout(xaxis_title='Mean model score',yaxis_title='Observed three-month event rate');fig.update_xaxes(range=[0,1]);fig.update_yaxes(range=[0,1]);chart(fig,260)
        st.caption('Five-bin held-out reliability diagnostic; no calibrated-probability claim.')
    with st.expander('Purged splits, features & artifact audit'):
        st.json({k:result[k] for k in ['folds','final_train','test_freeze','skipped_folds','exclusions','features','model_version','data_fingerprint','artifact_fingerprint']})
        st.caption(LIMITS)
