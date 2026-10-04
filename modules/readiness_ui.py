"""Native Streamlit field-assessment controls; evidence is kept separate from health."""
from datetime import date, timedelta
from html import escape
import pandas as pd
import streamlit as st
from modules.readiness import CHECKS, assess, assessment_history, record_assessment, business_today, recorded_date
from modules.ui import badge, style_status_table

DEFAULT_SCOPES={'D001':'Atlas SUV','D002':'Service workshop','D003':'Atlas SUV','D004':'Atlas SUV'}

def context(dealer, month, include_demo):
    scopes=st.session_state.setdefault('selected_scopes',{})
    default=scopes.get(dealer,DEFAULT_SCOPES.get(dealer,'') if include_demo else '')
    scope=st.text_input('Model / campaign / service scope',value=default,key='scope_'+dealer,
        help='Use the exact same scope as the field assessments. Blank means dealer-wide checks; model not specified; it never verifies a different model or campaign.')
    scopes[dealer]=scope.strip()
    cutoff=min((pd.Timestamp(month)+pd.offsets.MonthEnd(0)).date(),business_today())
    current=st.checkbox('Use current field evidence for this review',key='current_field_'+dealer,
        help='When the reporting month is historical, later field evidence is explicitly labelled and is not treated as contemporaneous.')
    snapshot=assess(dealer,scope,business_today() if current else cutoff,include_demo=include_demo)
    snapshot['later_evidence_used']=bool(current and business_today()>cutoff)
    snapshot['reporting_cutoff']=str(cutoff)
    return snapshot

def render(snapshot):
    st.subheader('Can the proposed action proceed?')
    st.caption(f"{snapshot['indicator']} · {snapshot['scope'] or 'Dealer-wide / model not specified'} · evidence as of {snapshot['as_of']}. Coverage and blockers are not a health score.")
    if snapshot.get('later_evidence_used'):
        st.caption('Later evidence · assessed for the current intervention, not a retrospective conclusion about the reporting month.')
    blocked=[c['label'] for c in snapshot['checks'] if c['status']=='Blocked']
    unknown=[c['label'] for c in snapshot['checks'] if c['status']=='Not assessed']
    dates=sorted({r['assessed_on'] for c in snapshot['checks'] for r in c['records']})
    if dates: st.caption('Latest applicable assessment: '+dates[-1]+'. Synthetic records are labelled; no actual site visits are implied.')
    if blocked: st.markdown('**Blockers:** '+', '.join(blocked))
    if unknown: st.caption('Verify: '+', '.join(unknown))
    with st.expander('Readiness checks, evidence & history'):
        st.html(''.join(f'<div class="distribution"><span>{escape(c["label"])}</span>{badge(c["status"])}</div>' for c in snapshot['checks']))
        for check in snapshot['checks']:
            st.markdown('**'+check['label']+'** · '+check['reason'])
            st.caption(check['evidence_id'])
            for record in check['records']:
                st.write(record['observation'])
                st.caption(f"{record['evidence_id']} · {record['source']} · {record['assessor']} · assessed {record['assessed_on']} · review by {record['expires_on'] or '30-day maximum age'}")
                st.caption('Evidence: '+record['evidence_note'])
        if snapshot['later_evidence']:
            st.markdown('**Later evidence · excluded from this historical assessment**')
            for record in snapshot['later_evidence']:
                st.caption(f"{record['evidence_id']} · assessed {record['assessed_on']} · recorded {recorded_date(record)} ({snapshot['business_timezone']}) · {record['status']}")
        history=assessment_history(snapshot['dealer_id'],include_demo=snapshot['include_demo'])
        if history:
            st.dataframe(style_status_table(pd.DataFrame(history)[['evidence_id','scope','check_key','status','assessed_on','expires_on','assessor','supersedes_id','source']], ['status']),hide_index=True,width='stretch')
        else: st.caption('No field assessments recorded for this dealer.')
    with st.expander('Record field assessment',expanded=st.session_state.pop('open_field_form',False)):
        st.caption('User-entered assessment · each revision creates a new immutable version. Synthetic scenarios are separately labelled.')
        history=assessment_history(snapshot['dealer_id'],include_demo=snapshot['include_demo'])
        versions={r['evidence_id']:r for r in history if r['scope'].strip().casefold()==snapshot['scope'].strip().casefold()}
        superseded={r.get('supersedes_id') for r in history}
        options=['New independent assessment']+[k for k in versions if k not in superseded]
        version=st.selectbox('Assessment version',options,help='Choose an existing version to explicitly revise it. Independent contradictory assessments require verification.')
        prior=versions.get(version)
        with st.form('field_assessment_'+snapshot['dealer_id']+'_'+version):
            key=st.selectbox('Readiness check',list(CHECKS),index=list(CHECKS).index(prior['check_key']) if prior else 0,format_func=CHECKS.get,disabled=bool(prior))
            scope=st.text_input('Assessment scope',value=prior['scope'] if prior else snapshot['scope'],disabled=bool(prior))
            status=st.selectbox('Assessment status',['Not assessed','Ready','Blocked'],index=['Not assessed','Ready','Blocked'].index(prior['status']) if prior else 0)
            observation=st.text_area('Concrete observation',value=prior['observation'] if prior and not prior['source'].startswith('Synthetic') else '',height=90)
            note=st.text_input('Evidence note / reference',value=prior['evidence_note'] if prior and not prior['source'].startswith('Synthetic') else '')
            assessor=st.text_input('Assessor',value='')
            a,b=st.columns(2)
            assessed=a.date_input('Assessment date',business_today(),max_value=business_today())
            expiry=b.date_input('Review / expiry date',business_today()+timedelta(days=30))
            use_expiry=st.checkbox('Set an explicit review / expiry date',value=True,help='Without an explicit date, the 30-day maximum evidence age still applies.')
            if st.form_submit_button('Save field assessment',type='primary'):
                try:
                    evidence_id=record_assessment(snapshot['dealer_id'],scope,key,status,observation,note,assessor,assessed,expiry if use_expiry else None,
                        supersedes_id=prior['evidence_id'] if prior else None,include_demo=snapshot['include_demo'])
                    st.session_state['field_saved']=evidence_id
                    st.rerun()
                except ValueError as exc: st.error(str(exc))
    if st.session_state.get('field_saved'):
        st.success(st.session_state.pop('field_saved')+' saved. Use current field evidence to view assessments recorded after the reporting month.')
