"""Session-isolated read-only manager conversation."""
import time
import streamlit as st
from modules.hosted_ai import request,begin_request,AIError,references
STARTERS=['Why is this dealer flagged?','What changed over the last three months?','What must be verified before the proposed intervention?']
def _render_chat(bundle,cfg,key,enabled,review):
 st.subheader('Ask about this dealer')
 st.caption('Selected dealer only · chat cannot approve or save actions.')
 columns=st.columns(3)
 for i,q in enumerate(STARTERS):
  text=['Why?','Changes','Verify'][i]
  if columns[i].button(text,help=q,key=key+'starter'+str(i),disabled=not enabled):st.session_state.update(ai_pending=q,ai_submit=True)
 for message in st.session_state['ai_chat']:
  with st.chat_message(message['role']):
   st.write(message['text'])
   if message.get('answer'):
    st.caption('Evidence · '+' · '.join(references(message['answer']['evidence_ids'],bundle)))
    for u in message['answer']['uncertainties']:st.caption(u)
 pending=st.session_state.get('ai_pending','')
 if pending:
  st.caption('Pending question · '+pending)
  if st.button('Retry pending question',key=key+'retry',disabled=not enabled):st.session_state['ai_submit']=True
 question=st.chat_input('Ask a question about the selected dealer',max_chars=600,disabled=not enabled,key=key+'chat_input')
 if question:st.session_state.update(ai_pending=question,ai_submit=True)
 if st.session_state.pop('ai_submit',False) and enabled:
  question=st.session_state['ai_pending']
  try:
   begin_request(st.session_state)
   with st.spinner('Checking selected dealer evidence…'):
    history=[{'role':m['role'],'text':m['text']} for m in st.session_state['ai_chat'][-6:]]
    answer=request(bundle,cfg,question,history+[{'role':'assistant','text':'Current review: '+review['summary']}])
   st.session_state['ai_chat']=(st.session_state['ai_chat']+[{'role':'user','text':question},{'role':'assistant','text':answer['answer'],'answer':answer}])[-12:]
   st.session_state.update(ai_pending='',ai_last_error='')
   if answer['metadata']['source']=='Live hosted AI': st.session_state.update(ai_verified=answer['metadata'],ai_verified_key=key)
   st.rerun()
  except AIError as e:
   st.session_state['ai_last_error']=str(e)
   st.session_state['ai_next_request']=max(st.session_state.get('ai_next_request',0),time.time()+e.retry_after)
   st.error(str(e))
   if getattr(e,'http_status',None)==413:
    from modules.hosted_ai import deterministic_guidance
    st.caption('Deterministic fallback — no live AI response')
    st.write(deterministic_guidance(bundle))
 if not enabled:st.caption('Live chat is unavailable. Use the clearly labelled AI Demo briefing to review this dealer’s evidence and proposed actions.')

def render_chat(bundle,cfg,key,enabled,review):
 with st.container(border=True,key="card_manager_chat"):
  _render_chat(bundle,cfg,key,enabled,review)
