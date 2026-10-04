"""Local/demo-only action reset. Server-side opt-in; never enabled by URL/UI."""
import os
import sqlite3
import streamlit as st
from modules.storage import records,reset_actions

def reset_enabled(secrets=None):
 s=secrets or {}
 return str(os.environ.get('DEMO_RESET_ENABLED',s.get('DEMO_RESET_ENABLED','false'))).strip().lower()=='true'

def clear_action_state(state):
 for key in list(state):
  if key in ['review','ai_context','ai_reviews','ai_chat','ai_pending','ai_submit','ai_last_error','ai_verified','ai_verified_key','ai_next_request'] or key.startswith(('approve_','approved_','FormSubmitter:approve_','status_')):del state[key]

def render():
 with st.expander('Demo controls'):
  try:enabled=reset_enabled(dict(st.secrets))
  except Exception:enabled=reset_enabled()
  if not enabled:
   st.caption('Action reset is disabled. For a private local recording workspace only, set DEMO_RESET_ENABLED=true server-side. Keep it disabled in hosted/shared deployments.')
   return
  actions,_=records();count=len(actions)
  st.write(f'Reset demo actions will clear all {count:,} tracked actions and their associated action history, including manually approved actions. A recoverable action-only backup is created first.')
  st.caption('Dealer data, field assessments, scoring rules, artifacts, secrets and application configuration are preserved. Review/chat/action drafts in this session are cleared; other sessions cannot reuse an outdated reset preview.')
  confirmed=st.checkbox(f'I confirm deletion of these {count:,} tracked actions and their associated action history.',key='confirm_demo_reset_'+str(st.session_state.get('reset_epoch',0))+'_'+str(count))
  if st.button('Reset demo actions',disabled=not confirmed or not count,type='primary'):
   try:
    backup=reset_actions(confirmed=True,enabled=enabled,expected_count=count)
    clear_action_state(st.session_state)
    st.session_state['demo_reset_notice']='Actions cleared. Recoverable backup: '+str(backup)
    st.session_state['reset_epoch']=st.session_state.get('reset_epoch',0)+1
    st.rerun()
   except (ValueError,OSError,sqlite3.Error) as exc:st.error('Reset failed; records were preserved. '+str(exc))
 if st.session_state.get('demo_reset_notice'):st.success(st.session_state.pop('demo_reset_notice'))
