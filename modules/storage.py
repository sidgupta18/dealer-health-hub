import json
import hashlib
import os
from datetime import datetime,timezone
import sqlite3
from pathlib import Path
from modules.ai import validate_review
DEFAULT_DB=Path(__file__).resolve().parents[1]/'data/actions.sqlite3'
def connect(path=DEFAULT_DB):
    conn=sqlite3.connect(path)
    conn.execute('CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY, dealer_id TEXT, month TEXT, action TEXT, owner TEXT, due_date TEXT, priority TEXT, status TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, evidence_json TEXT, source TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, action_id INTEGER, status TEXT, changed_at TEXT DEFAULT CURRENT_TIMESTAMP)')
    conn.execute('CREATE TABLE IF NOT EXISTS approval_submissions (submission_key TEXT PRIMARY KEY, action_id INTEGER NOT NULL)')
    return conn

def save(bundle,review,proposal,due_date,confirmed,source,path=DEFAULT_DB):
    if not confirmed: raise ValueError('Explicit manager confirmation required.')
    if bundle.get('decision_version') != 2: raise ValueError('Generate a current health-and-readiness review before approving a new action. Historical snapshots remain unchanged.')
    errors=validate_review(review,bundle)
    if errors: raise ValueError('Invalid review: '+' '.join(errors))
    if review.get('review_version')=='hosted-review-v3':
        from modules.hosted_ai import validate_output
        if validate_output(review,bundle): raise ValueError('Invalid hosted review structure or prerequisites.')
    if not isinstance(proposal.get('owner'),str) or not proposal['owner'].strip(): raise ValueError('Enter an assigned owner before approval.')
    checked={**review,'actions':[proposal]}
    if validate_review(checked,bundle): raise ValueError('Invalid edited proposal.')
    if bundle.get('decision_version') == 2:
        from modules.readiness import connect as field_connect
        with field_connect(path): pass
    with connect(path) as c:
        c.execute('BEGIN IMMEDIATE')
        identity=proposal.get('submission_identity')
        submission_key=hashlib.sha256(json.dumps({'identity':identity,'proposal':proposal,'due_date':str(due_date),'review':review},sort_keys=True).encode()).hexdigest() if identity else None
        if submission_key:
            prior=c.execute('SELECT a.id FROM approval_submissions s JOIN actions a ON a.id=s.action_id WHERE s.submission_key=?',(submission_key,)).fetchone()
            if prior:return prior[0]
        approval_readiness=None
        gate=None
        if bundle.get('decision_version') == 2:
            from modules.readiness import assess,prerequisites
            approval_readiness=assess(bundle['dealer_id'],proposal['scope'],path=path,include_demo=bundle['readiness'].get('include_demo',False),connection=c)
            gate=prerequisites(proposal,approval_readiness)
            if gate['status'] != 'Eligible for approval':
                raise ValueError(gate['status']+': '+gate['alternative'])
        cursor=c.execute('INSERT INTO actions (dealer_id,month,action,owner,due_date,priority,status,evidence_json,source) VALUES (?,?,?,?,?,?,?,?,?)',(bundle['dealer_id'],bundle['month'],proposal['action'],proposal['owner'],str(due_date),proposal['priority'],'Open',json.dumps({'bundle':bundle,'review':review,'approved_proposal':proposal,'approval_readiness':approval_readiness,'approval_prerequisites':gate}),source))
        c.execute('INSERT INTO history(action_id,status) VALUES (?,?)',(cursor.lastrowid,'Open'))
        if submission_key:c.execute('INSERT INTO approval_submissions VALUES (?,?)',(submission_key,cursor.lastrowid))
        return cursor.lastrowid

def update(action_id,status,path=DEFAULT_DB):
    if status not in ['Open','In Progress','Completed','Cancelled']: raise ValueError('Invalid status')
    with connect(path) as c:
        if c.execute('UPDATE actions SET status=? WHERE id=?',(status,action_id)).rowcount!=1: raise ValueError('Action not found')
        c.execute('INSERT INTO history(action_id,status) VALUES (?,?)',(action_id,status))
def records(path=DEFAULT_DB):
    import pandas as pd
    with connect(path) as c: return pd.read_sql_query('SELECT * FROM actions ORDER BY id DESC',c),pd.read_sql_query('SELECT * FROM history ORDER BY id DESC',c)


def reset_actions(confirmed=False,enabled=False,path=DEFAULT_DB,backup_dir=None,expected_count=None):
    """Back up action-only tables under the write lock, then delete transactionally."""
    if not enabled:raise ValueError('Demo reset is disabled by server configuration.')
    if not confirmed:raise ValueError('Explicit reset confirmation required.')
    directory=Path(backup_dir or Path(path).parent/'action_backups')
    directory.mkdir(parents=True,exist_ok=True)
    backup=directory/('actions-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.sqlite3')
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        count=conn.execute('SELECT COUNT(*) FROM actions').fetchone()[0]
        if expected_count is not None and count!=expected_count:raise ValueError('Action count changed. Review the new count and confirm again.')
        # Only action records, their history and submission keys; never settings/secrets/assessments.
        with sqlite3.connect(backup) as saved:
            for table,query in [('actions','SELECT * FROM actions'),('history','SELECT * FROM history WHERE action_id IN (SELECT id FROM actions)'),('approval_submissions','SELECT * FROM approval_submissions WHERE action_id IN (SELECT id FROM actions)')]:
                saved.execute(conn.execute('SELECT sql FROM sqlite_master WHERE name=?',(table,)).fetchone()[0])
                rows=conn.execute(query).fetchall()
                if rows:saved.executemany('INSERT INTO '+table+' VALUES ('+','.join('?' for _ in rows[0])+')',rows)
        os.chmod(backup,0o600)
        with sqlite3.connect(backup) as saved:
            if saved.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or saved.execute('SELECT COUNT(*) FROM actions').fetchone()[0]!=count:raise ValueError('Backup verification failed. No actions deleted.')
        conn.execute('DELETE FROM history WHERE action_id IN (SELECT id FROM actions)')
        conn.execute('DELETE FROM approval_submissions WHERE action_id IN (SELECT id FROM actions)')
        conn.execute('DELETE FROM actions')
    return backup
