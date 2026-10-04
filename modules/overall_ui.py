import pandas as pd
import streamlit as st
from modules.overall_deterioration import load,DEFINITION,LIMITS

def evaluation(data):
 result,_=load(data)
 st.subheader('Persistent health-score deterioration')
 st.write(result.get('target_definition',DEFINITION))
 st.caption('Configurable prototype assumption · six consecutive monthly inputs across all four pillars · fixed business weights and anchors · no field assessments or dealer IDs as predictors.')
 if not result.get('available'):
  st.warning(result['reason'])
  if 'eligible_labelled_rows' in result:st.caption(f"Eligible labelled rows: {result['eligible_labelled_rows']} · Positive events: {result['positive_events']}")
  st.code('.venv/bin/python scripts/train_overall_deterioration.py')
  st.caption('For imported data, add --csv /path/to/dealers.csv. Training runs outside Streamlit startup.');return
 st.write('Operational outlook: **'+result['chosen']+'**')
 st.caption(result['selection'])
 selected=result['test'][result['chosen']]
 if selected['recall']==0:st.warning('The frozen threshold caught none of the '+str(selected['events'])+' final-test events. Treat this as an experimental outlook, not a validated early-warning system.')
 a,b=st.tabs(['Validation & test','Artifact & temporal audit'])
 with a:
  for title,key in [('Purged validation','validation'),('Untouched final test','test')]:
   st.markdown('**'+title+'**')
   table=pd.DataFrame(result[key]).T
   st.dataframe(table[['samples','events','prevalence','average_precision','roc_auc','brier','precision','recall','alert_threshold']].astype(float).round(3),width='stretch')
   with st.expander(title+' confusion matrices (rows observed 0/1; columns predicted 0/1)'):
    st.json({name:r['confusion_matrix'] for name,r in result[key].items()})
  st.caption(LIMITS)
 with b:
  st.write('Scripted showcases excluded from training, validation and test: '+', '.join(result['showcase_exclusions'].values())+'. They remain eligible for inference.')
  st.json({k:result[k] for k in ['model_version','artifact_fingerprint','data_fingerprint','scoring_version','final_train','test_freeze','folds','skipped_folds','exclusions','excluded_showcase_rows']})
  st.code('.venv/bin/python scripts/train_overall_deterioration.py')
