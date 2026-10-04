from datetime import datetime,timezone,date, timedelta
import hashlib
import json
import io
from html import escape
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.data import generate_v2, validate, METRICS
from modules.scoring import CONFIG, COLORS, score_data, normalize
from modules.ai import settings, evidence, demo, request_review, validate_review
from modules.storage import save, records, update
from modules.manager_ui import outlook,forecast_panel,override_reason,readiness_summary,action_title
from modules.readiness import assess, prerequisites, KINDS, CHECKS, business_today
from modules.readiness_ui import context as field_context, render as render_readiness
from modules.ui import LABELS, label, value, index, risk, priority, concern, chart, evidence_table, badge, health_change, style_status_table

ROOT = Path(__file__).resolve().parent
PAGES = ['Portfolio', 'Dealer Diagnosis', 'AI Review', 'Action Tracker', 'Data & Methodology']
st.set_page_config(page_title='Dealer Health & Action Hub', page_icon='◈', layout='wide')
st.html('<style>'+ (ROOT / 'assets/ui.css').read_text() +'</style>')
@st.cache_data(show_spinner='Validating data and preparing portfolio…')
def sample():
    return generate_v2()
@st.cache_data(show_spinner='Loading saved health outlook…')
def saved_outlook(raw,token):
    from modules.overall_deterioration import load,forecasts
    result,models=load(raw)
    try: predictions=forecasts(raw,result,models)
    except Exception as exc: result={**result,'available':False,'reason':'Model inference error: '+str(exc)};predictions={}
    return result,predictions

def prepare(raw):
    df, errors, warnings = validate(raw)
    if errors:
        return None, None, errors, warnings
    scored = score_data(df)
    from modules.overall_deterioration import fingerprint
    result,predictions=saved_outlook(df,fingerprint(df)['artifact_fingerprint'])
    return scored, {'overall':result,'predictions':predictions}, errors, warnings

prepare.clear=lambda: None

def navigate(page, dealer=None, month=None):
    st.session_state['workspace'] = page
    if dealer is not None:
        st.session_state['selected_dealer'] = dealer
        st.session_state['dealer_picker_'+page] = dealer
    if month is not None:
        st.session_state['selected_month'] = pd.Timestamp(month)
        if dealer is not None:
            st.session_state['month_picker_'+page+'_'+dealer] = pd.Timestamp(month)

def status_message(row):
    title, detail = concern(row)
    message = f'**{row.status} · {title}**\n\n{detail}'
    if row.status == 'Critical' or row.status == 'Unassessed':
        st.error(message)
    elif row.status in ['At Risk', 'Watchlist']:
        st.warning(message)
    else:
        st.success(message)

def dealer_selection(df):
    ids = df.dealer_id.unique().tolist()
    if st.session_state.get('selected_dealer') not in ids:
        st.session_state['selected_dealer'] = ids[0]
    a, b = st.columns([3, 1])
    dealer = a.selectbox('Dealer', ids, index=ids.index(st.session_state['selected_dealer']),
                         format_func=lambda d: df.loc[df.dealer_id == d, 'dealer_name'].iloc[0]+' · '+d,
                         key='dealer_picker_'+page)
    months = sorted(df.loc[df.dealer_id == dealer, 'month'], reverse=True)
    remembered = st.session_state.get('selected_month')
    selected_index = months.index(remembered) if remembered in months else 0
    month = b.selectbox('Reporting month', months, index=selected_index,
                       format_func=lambda m: m.strftime('%b %Y'), key='month_picker_'+page+'_'+dealer)
    st.session_state['selected_dealer'] = dealer
    st.session_state['selected_month'] = month
    row = df[(df.dealer_id == dealer) & (df.month == month)].iloc[0]
    history = df[(df.dealer_id == dealer) & (df.month <= month)]
    return row, history

st.session_state.setdefault('workspace', 'Portfolio')
page = st.session_state['workspace']
st.sidebar.markdown('## DealerHub')
st.sidebar.caption('HEALTH & ACTIONS')
icons = ['dashboard', 'monitor_heart', 'description', 'checklist', 'database']
for destination, icon in zip(PAGES, icons):
    st.sidebar.button(destination, icon=f':material/{icon}:', type='primary' if destination == page else 'tertiary', width='stretch', key='nav_'+destination, on_click=navigate, args=(destination,))
st.sidebar.divider()

def import_controls():
    st.markdown('#### Data source')
    upload = st.file_uploader('Import dealer CSV', type='csv', key='csv_import', help='Invalid imports block analysis; saved approvals remain unchanged.')
    if upload is not None:
        st.session_state['import_bytes'] = upload.getvalue()
        st.session_state['import_name'] = upload.name
    if st.session_state.get('import_bytes') is not None:
        def clear_import():
            for key in ['import_bytes', 'import_name', 'csv_import']:
                st.session_state.pop(key, None)
        st.button('Use synthetic demo', on_click=clear_import)
    st.caption('Import applies across every workspace. Required fields and validation are shown below.')

raw = sample()
source = 'Synthetic demo v2 · fictional dealers'
# The import widget lives here so a rejected file can always be corrected.
if page == 'Data & Methodology':
    with st.expander('Import & data source', expanded=st.session_state.get('import_bytes') is not None):
        import_controls()
if st.session_state.get('import_bytes') is not None:
    source = 'Uploaded · '+st.session_state['import_name']
    try:
        raw = pd.read_csv(io.BytesIO(st.session_state['import_bytes']))
    except Exception as exc:
        st.error(f'Cannot read this CSV: {exc}')
        st.button('Open data source', on_click=navigate, args=('Data & Methodology',))
        st.stop()
df, model, errors, warnings = prepare(raw)
st.sidebar.caption(source)
if errors:
    st.title('Check your import')
    st.error('Import rejected. Correct the issues below or use the synthetic demo in Data & Methodology.')
    for error in errors:
        st.write('• '+error)
    st.button('Open data source', on_click=navigate, args=('Data & Methodology',))
    st.stop()
st.sidebar.caption(f'{df.dealer_id.nunique()} dealers · {len(df):,} records')
st.sidebar.caption('Business-weighted health index')

if page == 'Portfolio':
    heading, period = st.columns([4, 1])
    heading.title('Portfolio')
    heading.caption('Current health. Clear priorities. Accountable follow-through.')
    months = sorted(df.month.unique(), reverse=True)
    remembered = st.session_state.get('selected_month')
    month = period.selectbox('Reporting month', months, index=months.index(remembered) if remembered in months else 0, format_func=lambda m: pd.Timestamp(m).strftime('%b %Y'), key='portfolio_month')
    st.session_state['selected_month'] = pd.Timestamp(month)
    with st.container(border=True, key='card_toolbar'):
        c1, c2 = st.columns(2)
        region = c1.selectbox('Region', ['All regions']+sorted(df.region.unique()))
        status = c2.selectbox('Health status', ['All statuses']+list(COLORS))
    view = df[df.month == month].copy()
    if region != 'All regions': view = view[view.region == region]
    if status != 'All statuses': view = view[view.status == status]
    view = view.sort_values(['review_priority', 'raw_index']).reset_index(drop=True)
    if view.empty:
        st.info('No dealers match these filters. Choose another region or health status.')
        st.stop()
    urgent = int(view.status.isin(['At Risk', 'Critical']).sum())
    missing = int(view.status.eq('Unassessed').sum())
    strong = view[(view.target_attainment_pct >= 100) & (view.Financial_score < 50)]
    featured = strong.iloc[0] if not strong.empty else view.iloc[0]
    title, detail = concern(featured)
    left, right = st.columns([2, 1], gap='medium')
    with left, st.container(border=True, key='card_attention'):
        st.markdown(f'### {urgent:,} dealers need a health review')
        if missing:st.caption(f'{missing:,} dealer(s) also need data verification.')
        st.markdown(f'**{escape(featured.dealer_name)} · {title}**')
        st.caption(detail)
        st.html('Current health · '+badge(featured.status))
        if override_reason(featured): st.caption(override_reason(featured))
    with right, st.container(border=True, key='card_distribution'):
        st.caption(f'{len(view):,} DEALERS IN VIEW')
        st.html('<div class=status-summary>'+''.join(f'<span>{badge(health)} <strong>{int(view.status.eq(health).sum())}</strong></span>' for health in COLORS if view.status.eq(health).any())+'</div>')
    include_demo = source.startswith('Synthetic')
    field_scopes = {d: st.session_state.get('selected_scopes',{}).get(d, {'D001':'Atlas SUV','D002':'Service workshop','D003':'Atlas SUV','D004':'Atlas SUV'}.get(d,'') if include_demo else '') for d in view.dealer_id}
    st.caption('Prototype outlook · as of '+str((pd.Timestamp(month)+pd.offsets.MonthEnd(0)).date())+' · '+outlook(view.iloc[0],model['overall'],model['predictions'])['window'] if len(view) else 'No dealers match these filters.')
    st.caption('No deterioration flag does not mean healthy; Critical dealers remain urgent.')
    cutoff=min(pd.Timestamp(month)+pd.offsets.MonthEnd(0),pd.Timestamp(business_today())).date()
    field_indicators = {d: assess(d,field_scopes[d],cutoff,include_demo=include_demo)['indicator'] for d in view.dealer_id}
    with st.container(border=True, key='card_queue'):
        st.subheader('Review queue')
        st.caption('Ordered by review priority, then current health score. Select a dealer to investigate.')
        ids = view.dealer_id.tolist()
        remembered_dealer = st.session_state.get('selected_dealer')
        a, b = st.columns([4, 1])
        chosen = a.selectbox('Choose dealer', ids, index=ids.index(remembered_dealer) if remembered_dealer in ids else ids.index(featured.dealer_id), format_func=lambda d: view.loc[view.dealer_id == d, 'dealer_name'].iloc[0]+' · '+d, label_visibility='collapsed')
        b.button('Review dealer →', type='primary', width='stretch', on_click=navigate, args=('Dealer Diagnosis', chosen, month))
        sorting,details_control=st.columns([3,1])
        detailed=details_control.checkbox('Show queue details',key='queue_details')
        sort=sorting.selectbox('Sort review queue',['Review priority','Health score','Dealer name'],label_visibility='collapsed',key='queue_sort')
        rows=view.sort_values(['review_priority','raw_index']) if sort=='Review priority' else view.sort_values('raw_index') if sort=='Health score' else view.sort_values('dealer_name')
        table=pd.DataFrame({'Dealer':rows.dealer_name,'Current status':rows.status,'Main concern':[concern(r)[0] for _,r in rows.iterrows()],'Three-month outlook':[outlook(r,model['overall'],model['predictions'])['status'] for _,r in rows.iterrows()],'Review priority':rows.review_priority.map(priority)})
        if detailed:
            table['Region']=rows.region;table['Health score']=rows.raw_index.round(1);table['Change (pts)']=rows.trajectory.round(1)
            table['Status basis']=[override_reason(r) or ('Missing inputs: '+r.missing_inputs if r.status=='Unassessed' else 'Score-based status') for _,r in rows.iterrows()]
            table['Action scope']=rows.dealer_id.map(field_scopes).replace('','Dealer-wide / model not specified')
            table['Readiness']=rows.dealer_id.map(field_indicators)
        st.dataframe(style_status_table(table, ['Current status','Three-month outlook','Readiness'], 'Change (pts)'),hide_index=True,width='stretch',height=280,column_config={'Health score':st.column_config.NumberColumn(format='%.1f'),'Change (pts)':st.column_config.NumberColumn('Change vs previous available month (pts)',format='%+.1f'),'Three-month outlook':st.column_config.TextColumn(help='At least a 10-point raw health-score drop in two of the next three calendar months. No flag does not mean healthy.')})
    with st.expander('Priority rules, data quality & growth opportunity'):
        st.write('P1: critical or missing data. P2: at risk or index decline ≥8 points. P3: watchlist. P4: healthy. Critical overrides and observed deterioration affect health review priority. Operational blockers are shown separately; field observations never change the index.')
        st.write(f'{view.missing_inputs.ne("").sum()} dealer(s) have missing scoring inputs in this view. Unassessed dealers need data verification first.')
        growth = view[['dealer_name', 'growth_opportunity']].rename(columns={'dealer_name':'Dealer', 'growth_opportunity':'Growth hypothesis /100'})
        st.dataframe(growth, hide_index=True, column_config={'Growth hypothesis /100':st.column_config.NumberColumn(format='%.1f')}, width='stretch')
        st.caption('Growth combines target shortfall and customer satisfaction. It is separate from health and is not a forecast.')

elif page in ['Dealer Diagnosis', 'AI Review']:
    if page == 'AI Review':
        st.title('Prepare dealer review')
        st.caption('Evidence first. Actions saved only with manager approval.')
    with st.expander('Dealer & reporting month'):
        row, history = dealer_selection(df)
        field_snapshot = field_context(row.dealer_id,row.month,source.startswith('Synthetic'))
    bundle = evidence(row, history, readiness=field_snapshot)
    bundle['sustained_deterioration_forecast']=outlook(row,model['overall'],model['predictions'])
    if page == 'Dealer Diagnosis':
        st.html(f'<div class="identity"><span>{escape(row.dealer_name)} <small>{escape(row.dealer_id)} · {escape(row.region)} · {row.month.strftime("%b %Y")}</small></span>{badge(row.status)}</div>')
        title, detail = concern(row)
        health_area,outlook_area=st.columns([1.7,1],gap='medium')
        with health_area,st.container(border=True, key='card_health'):
            st.caption('HEALTH SCORE')
            previous=history.iloc[-2].month.strftime('%b %Y') if len(history)>1 else None
            score = f'{row.raw_index:.1f}' if pd.notna(row.raw_index) else '—'
            st.html(f'<div class="health-score">{score}<span> / 100</span>{health_change(row.trajectory, 'previous available month ('+previous+')' if previous else None)}</div>')
            if override_reason(row): st.write('**'+override_reason(row)+'**')
            st.markdown('### '+title)
            st.write(detail)
            st.caption('Financial counterparties, overdue denominator and payment-obligation definitions need confirmation. These indicators do not establish insolvency.')
            st.button('Prepare dealer review →', type='primary', on_click=navigate, args=('AI Review', row.dealer_id, row.month))
        with outlook_area:forecast_panel(row,model['overall'],model['predictions'])
        left, right = st.columns([1, 1.6], gap='medium')
        with left, st.container(border=True, key='card_pillars'):
            st.subheader('Pillar profile')
            meters = []
            for pillar in CONFIG['weights']:
                score = row[pillar+'_score']
                percent = float(score) if pd.notna(score) else 0
                state=' · needs attention' if pd.notna(score) and score<65 else ' · not assessed' if pd.isna(score) else ''
                meters.append(f'<div class="pillar-meter"><div><strong>{escape(pillar+state)}</strong><span>{index(score)}</span></div><div class="meter-track"><span style="width:{percent}%"></span></div></div>')
            st.html(''.join(meters))
        with right, st.container(border=True, key='card_trajectory'):
            st.subheader('Recent trajectory')
            trend = history.tail(12).rename(columns={'month':'Month', 'raw_index':'Health index'})
            fig = px.line(trend, x='Month', y='Health index', markers=True, color_discrete_sequence=['#0066CC'], labels={'Health index':'Health score /100'})
            fig.update_yaxes(range=[0,100]); chart(fig, 245)
        readiness_summary(field_snapshot)
        with st.expander('Assessment details & recording'):render_readiness(field_snapshot)
        with st.expander('Observed drivers & status overrides'):
            drivers = [{'Metric':label(k), 'Observed':value(k,row[k]), 'KPI score /100':normalize(row[k], spec) if pd.notna(row[k]) else None} for k,spec in CONFIG['kpis'].items()]
            st.dataframe(pd.DataFrame(drivers).sort_values('KPI score /100', na_position='first'), hide_index=True, width='stretch', column_config={'KPI score /100':st.column_config.NumberColumn(format='%.1f')})
            st.caption(f'Raw class: {row.raw_status}. Final status: {row.status}.')
            if row.override_evidence: st.write('Override evidence: '+row.override_evidence)
        with st.expander('KPI history & benchmark details'):
            metric = st.selectbox('Metric', METRICS, format_func=label)
            fig = px.line(history, x='month', y=metric, markers=True, color_discrete_sequence=['#214F87'],
                          labels={'month':'Reporting month', metric:label(metric)+' ('+LABELS[metric][1]+')'})
            chart(fig)
            benchmarks=[]
            for k,spec in CONFIG['kpis'].items():
                count=sum(v['pillar']==spec['pillar'] for v in CONFIG['kpis'].values())
                score=normalize(row[k],spec) if pd.notna(row[k]) else None
                benchmarks.append({'Metric':label(k),'Observed':value(k,row[k]),'Good anchor':value(k,spec['good']),
                                   'Bad anchor':value(k,spec['bad']), 'Index contribution':score*CONFIG['weights'][spec['pillar']]/count if score is not None else None})
            st.dataframe(benchmarks, hide_index=True, width='stretch', column_config={'Index contribution':st.column_config.NumberColumn(format='%.1f pts')})
            st.caption('Contributions describe the scoring arithmetic, not causes. An overall index requires every scoring input. Global model importance is available on Data & Methodology; it does not explain an individual forecast.')
    else:
        from modules.hosted_ai import configuration,configured,shared_evidence,context_key,session_context,begin_request,request as hosted_request,AIError
        from modules.chat_ui import render_chat
        import time
        try: cfg=configuration(dict(st.secrets))
        except Exception: cfg=configuration()
        connected=configured(cfg)
        live_enabled=connected and cfg['free_confirmed'] and cfg.get('live_enabled',True)
        bundle=shared_evidence(bundle,row,history,hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest())
        review_key=context_key(bundle,cfg)
        session_context(st.session_state,review_key)
        cached=st.session_state['ai_reviews'].get(review_key)
        st.session_state['review']=(review_key,cached,cached['metadata']['source']) if cached else None
        st.caption(f"{row.dealer_name} · {row.month.strftime('%b %Y')} · {field_snapshot['scope'] or 'Dealer-wide / model not specified'}")
        forecast = bundle['sustained_deterioration_forecast']
        st.html('<div class="status-summary">Current health '+badge(row.status)+' <span>Three-month outlook</span> '+badge(forecast['status'], forecast.get('reason'))+'</div>')
        if override_reason(row): st.caption(override_reason(row))
        if not forecast['available']: st.caption(forecast['reason'])
        verification=st.session_state.get('ai_verified') if st.session_state.get('ai_verified_key')==review_key else None
        if not cfg.get('live_enabled',True): st.caption('AI Demo · deterministic evidence-based briefing. Live review and chat are disabled pending verification.')
        elif not connected: st.caption('Not configured · AI Demo available; live review and chat need local setup.')
        elif st.session_state.get('ai_last_error'): st.caption('Last live request failed · '+st.session_state['ai_last_error'])
        elif verification: st.caption('Live request verified · '+pd.Timestamp(verification['generated_at']).strftime('%d %b %Y · %H:%M UTC'))
        else: st.caption('Configured, not yet verified'+(' · free entitlement pending' if not live_enabled else ''))
        if field_snapshot.get('later_evidence_used'): st.caption('Later field evidence · current intervention context, not proof of historical readiness.')
        stored = st.session_state.get('review')
        ready = stored and stored[0] == review_key
        title, detail = concern(row)
        if not ready:
            with st.container(border=True,key='card_review_start'):
                st.write('**'+title+'**')
                st.write(detail)
        controls=st.columns([1.4,1.1,1,1])
        controls[0].button('Regenerate review' if ready else 'Generate review',type='primary',key='generate_review',disabled=not live_enabled,width='stretch')
        use_demo=controls[1].button('AI Demo',key='use_demo_brief',type='primary' if not live_enabled else 'secondary',width='stretch')
        if not live_enabled:controls[2].button('AI setup →',key='ai_setup_link',on_click=navigate,args=('Data & Methodology',),width='stretch')
        mode='AI Demo' if not live_enabled else 'AI Demo' if cached and cached.get('metadata',{}).get('model') is None else 'Live AI' if live_enabled and not st.session_state.get('ai_last_error') else 'Unavailable'
        controls[3].html('<div class=mode>'+mode+'</div>')
        if (live_enabled and st.session_state.get('generate_review')) or use_demo:
            try:
                if use_demo:
                    review=demo(bundle)
                    review['actions']=review['actions'][:3]
                    review['metadata']={'source':'AI Demo—no live AI response','provider':None,'model':None,'generated_at':datetime.now(timezone.utc).isoformat(),'review_version':'deterministic-demo-v3','evidence_fingerprint':bundle['evidence_fingerprint']}
                else:
                    begin_request(st.session_state)
                    with st.spinner('Generating and validating the hosted dealer review…'):
                        review=hosted_request(bundle,cfg)
                    st.session_state.update(ai_verified=review['metadata'],ai_verified_key=review_key,ai_last_error='')
                problems=validate_review(review,bundle)
                if problems: raise ValueError('; '.join(problems))
                st.session_state['ai_reviews'][review_key]=review
                st.rerun()
            except ValueError as exc:
                if not use_demo:
                    st.session_state['ai_last_error']=str(exc)
                    st.session_state['ai_next_request']=max(st.session_state.get('ai_next_request',0),time.time()+getattr(exc,'retry_after',0))
                st.error(str(exc))
                if getattr(exc,'http_status',None)==413:st.caption('Deterministic fallback available: select AI Demo above. This is not a live AI response.')
        stored = st.session_state.get('review')
        if stored and stored[0] == review_key:
            review = stored[1]
            document, proposals = st.columns([1, 1.2], gap='large')
            with document:
                st.subheader('Main finding')
                generated=review.get('metadata',{}).get('generated_at','')
                st.caption('Previously generated · '+(pd.Timestamp(generated).strftime('%d %b %Y · %H:%M UTC') if generated else 'Time not recorded'))
                st.write(review['summary'])
                for finding in review.get('findings',[]):
                    st.markdown('**'+('Possible explanation' if finding['kind']=='hypothesis' else 'Observed finding')+'** · '+finding['text'])
                    from modules.hosted_ai import references
                    st.caption('Evidence · '+' · '.join(references(finding['evidence_ids'],bundle)))
                with st.container(border=True, key='card_review_evidence'):
                    st.markdown('**Key evidence**')
                    
                    for n, ref in enumerate(review['evidence'], 1):
                        st.markdown(f'**[{n}] {label(ref["metric"])}** · {value(ref["metric"], ref["value"])}')
                render_chat(bundle,cfg,review_key,live_enabled,review)
                for section in ['observed_facts','hypotheses','unknowns']:
                    if review.get(section):
                        with st.expander({'observed_facts':'Evidence details','hypotheses':'Possible explanations','unknowns':'What needs checking'}[section]):
                            for item in review[section]: st.write('• '+item)
                if bundle.get('inventory_warning'):
                    st.caption('Inventory early warning · '+bundle['inventory_warning']['status']+' · investigation only; not proof of future distress.')
                from modules.hosted_ai import references
                with st.expander('Evidence & generation details'):
                    st.caption(stored[2]+' · schema/reference checks do not prove every narrative claim.')
                    st.json(review.get('metadata',{}))
                    st.caption(' · '.join(dict.fromkeys(references(review.get('evidence_ids',[]),bundle))))
                st.caption('Uncertainty · '+review['uncertainty'])
            with proposals:
                st.subheader('Proposed actions')
                st.caption('Edit, assign an owner and deadline, then explicitly approve.')
                for i, proposal in enumerate(review['actions']):
                    with st.expander(action_title(proposal['action'])+' · '+proposal['priority'], expanded=i==0):
                        from modules.hosted_ai import references
                        st.caption('Evidence · '+' · '.join(references(proposal.get('evidence_ids',[]),bundle)))
                        if proposal.get('rationale'): st.write(proposal['rationale'])
                        st.caption('Scope: '+(proposal.get('scope') or 'Dealer-wide / model not specified'))
                        review_identity=hashlib.sha256(json.dumps(review,sort_keys=True).encode()).hexdigest()
                        with st.form(f'approve_{review_identity}_{i}'):
                            action=st.text_area('Recommended action',proposal['action'],height=100)
                            c1,c2,c3=st.columns([1.3,1.1,1])
                            intervention=st.selectbox('Intervention type',list(KINDS),index=list(KINDS).index(proposal.get('intervention','health_review')),format_func=KINDS.get)
                            site_dependent=st.checkbox('This action depends on showroom / site access',value=proposal.get('site_dependent',False))
                            gate=prerequisites({**proposal,'action':action,'intervention':intervention,'site_dependent':site_dependent},field_snapshot)
                            st.html('Prerequisites · '+badge(gate['status']))
                            for check in gate['checks']:
                                st.caption(check['label']+' · '+check['status']+' · '+check['reason'])
                            st.caption(gate['alternative'])
                            for alternative in gate['corrective_alternatives']:
                                st.caption('Separate alternative · '+alternative['action'])
                            st.caption('Edited action and current field evidence are rechecked when you submit.')
                            owner=c1.text_input('Assigned owner','')
                            due=c2.date_input('Deadline',business_today()+timedelta(days=14))
                            action_priority=c3.selectbox('Priority',['High','Medium','Low'],index=['High','Medium','Low'].index(proposal['priority']))
                            confirm=st.checkbox('I have reviewed the evidence and approve this action.')
                            if st.form_submit_button('Approve & save action',type='primary'):
                                try:
                                    action_id=save(bundle,review,{**proposal,'action':action,'owner':owner,'priority':action_priority,'intervention':intervention,'site_dependent':site_dependent,'submission_identity':review_identity+':'+str(i)},due,confirm,stored[2])
                                    st.success(f'Action #{action_id} saved for {owner}. View it in Action Tracker.')
                                except ValueError as exc:
                                    st.error(str(exc))
            if not review['actions']:
                st.info('No action proposals in this review. Use the follow-up questions to clarify the evidence.')
            st.button('Open Action Tracker →',on_click=navigate,args=('Action Tracker',))
        else:
            st.caption('Generate a brief to review the evidence and edit proposed actions. Saving requires explicit approval.')
        with st.expander('Can the proposed action proceed? · assessment details'):
            render_readiness(field_snapshot)
        with st.expander('Review workflow & full evidence'):
            st.write('Health and field evidence → validated review → action-specific prerequisites → manager edits → current evidence rechecked → explicit approval → saved snapshot. No autonomous approval.')
            st.json(bundle)

elif page == 'Action Tracker':
    st.title('Action tracker')
    st.caption('Manager-approved actions with accountable owners and visible deadlines.')
    actions, hist = records()
    if actions.empty:
        with st.container(border=True, key='card_empty'):
            st.html('<div class=empty-icon aria-hidden=true>✓</div>')
            st.subheader('Ready for your first action')
            st.write('Prepare a dealer review and approve a proposal with an owner and deadline.')
            st.button('Prepare a dealer review →',type='primary',on_click=navigate,args=('AI Review',))
    else:
        today = business_today()
        actions['overdue'] = pd.to_datetime(actions.due_date).dt.date.lt(today) & ~actions.status.isin(['Completed','Cancelled'])
        for col,name,count in zip(st.columns(3),['Active actions','Past deadline','Completed'],
                                  [(~actions.status.isin(['Completed','Cancelled'])).sum(),actions.overdue.sum(),actions.status.eq('Completed').sum()]):
            col.metric(name,f'{count:,}')
        status_filter=st.selectbox('Action status',['All statuses','Open','In Progress','Completed','Cancelled'])
        visible=actions if status_filter=='All statuses' else actions[actions.status==status_filter]
        if visible.empty:
            st.info('No actions with this status. Choose All statuses to see the full tracker.')
        for _, action in visible.iterrows():
            with st.container(border=True, key='card_action_'+str(action.id)):
                dealer_name = df.loc[df.dealer_id==action.dealer_id,'dealer_name']
                dealer_name = dealer_name.iloc[0] if len(dealer_name) else action.dealer_id
                approved=json.loads(action.evidence_json).get('approved_proposal',{})
                title_col,owner_col,date_col,status_col=st.columns([2.6,1.2,1.1,1])
                title_col.markdown('**'+action_title(action.action)+'**')
                title_col.caption(dealer_name+' · '+action.priority+' priority')
                owner_col.write('**Owner:** '+action.owner)
                date_col.write('**Deadline:** '+pd.Timestamp(action.due_date).strftime('%d %b %Y'))
                status_col.html(badge(action.status))
                if action.overdue:status_col.html(badge('Past deadline'))
                with st.expander('Description, evidence & status'):
                    st.write(action.action)
                    with st.form('status_'+str(action.id)):
                        new_status=st.selectbox('Status',['Open','In Progress','Completed','Cancelled'],index=['Open','In Progress','Completed','Cancelled'].index(action.status))
                        if st.form_submit_button('Save status'):
                            update(int(action.id),new_status); st.rerun()
                    st.caption(f'Approved from {action.source} · reporting month {pd.Timestamp(action.month).strftime("%b %Y")}')
                    st.dataframe(hist[hist.action_id==action.id][['status','changed_at']].rename(columns={'status':'Status','changed_at':'Updated at'}),hide_index=True,width='stretch')
                    with st.expander('Saved approval evidence'):
                        st.json(json.loads(action.evidence_json))

else:
    from modules.analytical_ui import render
    render(df,model,raw,prepare.clear)
    from modules.demo_controls import render as render_demo_controls
    render_demo_controls()
