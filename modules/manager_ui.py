"""Plain-language manager summaries. No analytic changes."""
import pandas as pd
import streamlit as st
from modules.ui import label,value,badge
from modules.scoring import CONFIG
from modules.readiness import prerequisites

from modules.overall_deterioration import outlook

def forecast_panel(row,result=None,predictions=None):
 f=outlook(row,result,predictions)
 with st.container(border=True,key='card_forecast'):
  st.subheader('Three-month outlook')
  st.html(badge(f['status'], f.get('reason')))
  st.caption('As of '+f['prediction_date']+' · '+f['window'])
  st.caption('Experimental prediction of persistent index decline. No flag does not mean healthy; current concerns remain urgent.')
  if not f['available']:st.caption(f['reason'])
  with st.expander('Forecast details'):
   st.write(f['event_definition'])
   st.json(f)
   if result and result.get('available'):
    st.write('Validation and untouched test performance')
    st.json({k:result[k][result['chosen']] for k in ['validation','test']})

def override_reason(row):
 if row.status==row.raw_status:return ''
 triggered=[]
 for rule in CONFIG['overrides']:
  actual=row[rule['metric']]
  if pd.isna(actual):continue
  hit=actual>=rule['threshold'] if rule['operator']=='>=' else actual>rule['threshold'] if rule['operator']=='>' else actual<=rule['threshold'] if rule['operator']=='<=' else actual<rule['threshold']
  if hit and rule['status']==row.status:triggered.append(label(rule['metric'])+' '+value(rule['metric'],actual)+' (rule: '+rule['operator']+' '+value(rule['metric'],rule['threshold'])+')')
 return row.status+': '+ '; '.join(triggered)+' overrides the score-based status ('+row.raw_status+').' if triggered else ''

def readiness_summary(snapshot):
 scope=snapshot['scope'] or 'model not specified'
 # Evaluate named interventions independently; do not imply all marketing is blocked.
 campaign=prerequisites({'intervention':'model_campaign','action':'','site_dependent':False},snapshot)
 blocked=[c for c in campaign['checks'] if c['status']=='Blocked']
 demo=next((c for c in blocked if c['check_key']=='demo_vehicle'),None)
 with st.container(border=True,key='card_readiness'):
  st.subheader('Can the proposed action proceed?')
  st.html('Readiness · '+badge('Blocked' if blocked else 'Not assessed' if campaign['status']=='Verification required' else 'Ready'))
  if demo:st.markdown('**Test-drive campaign blocked: usable '+scope+' demo vehicle unavailable.**')
  elif blocked:st.markdown('**Model campaign blocked · '+', '.join(c['label'].lower() for c in blocked)+'**')
  elif campaign['status']=='Verification required':st.write('Model campaign: verify the relevant prerequisites before proceeding.')
  else:st.write('Model campaign prerequisites satisfied; manager approval still required.')
  other=[c for c in snapshot['checks'] if c['status']=='Blocked' and c not in campaign['checks']]
  for c in other:st.caption('Affected intervention needs review · '+c['label']+' · '+c['reason'])
  dates=sorted({r['assessed_on'] for c in snapshot['checks'] for r in c['records']})
  freshness='Stale evidence — reassess' if any('Stale' in c['reason'] for c in snapshot['checks']) else 'Verification needed' if any(c['status']=='Not assessed' for c in snapshot['checks']) else 'Current scoped evidence'
  st.caption(freshness)
  st.caption('Scope: '+(snapshot['scope'] or 'Dealer-wide / model not specified')+' · evidence as of '+snapshot['as_of']+(' · assessed '+dates[-1] if dates else ' · no applicable assessment'))
  if snapshot.get('later_evidence_used'):st.caption('Current field evidence; does not establish historical readiness.')
  if blocked:st.caption(campaign['alternative'])

def action_title(text):
 sentence=str(text).split('. ')[0].strip().rstrip('.')
 return sentence if len(sentence)<=100 else sentence[:97]+'…'
