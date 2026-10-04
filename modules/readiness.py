"""Versioned field evidence and deterministic, action-specific prerequisite checks."""
import json
import hashlib
import re
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'data/actions.sqlite3'
RULES = json.loads((ROOT / 'config/readiness.json').read_text())
CHECKS = RULES['checks']
KINDS = {'health_review':'Review observed health', 'lead_growth':'Increase leads',
         'model_campaign':'Run a model / campaign promotion', 'service_growth':'Increase service demand',
         'stock_change':'Change stock mix', 'site_intervention':'Site-dependent intervention',
         'verification':'Verify field evidence', 'remediation':'Resolve an operational blocker'}

def business_today():
    return datetime.now(ZoneInfo(RULES['timezone'])).date()

def recorded_date(record):
    stamp=datetime.fromisoformat(record['recorded_at'])
    if stamp.tzinfo is None: stamp=stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(ZoneInfo(RULES['timezone'])).date()

def connect(path=DEFAULT_DB):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS field_assessments (
        id INTEGER PRIMARY KEY, dealer_id TEXT NOT NULL, scope TEXT NOT NULL,
        check_key TEXT NOT NULL, status TEXT NOT NULL, observation TEXT NOT NULL,
        evidence_note TEXT NOT NULL, assessor TEXT NOT NULL, assessed_on TEXT NOT NULL,
        expires_on TEXT, recorded_at TEXT NOT NULL, supersedes_id TEXT)''')
    return conn

def demo_records():
    return json.loads((ROOT / 'data/sample_field_assessments.json').read_text())

def assessment_history(dealer_id=None, path=DEFAULT_DB, include_demo=False, connection=None):
    def read(conn):
        cursor=conn.execute('SELECT * FROM field_assessments ORDER BY id DESC')
        columns=[c[0] for c in cursor.description]
        return [dict(zip(columns,r)) for r in cursor.fetchall()]
    if connection is not None:
        rows=read(connection)
    else:
        with connect(path) as conn:
            rows=read(conn)
    for row in rows:
        row['evidence_id'] = 'FIELD-'+str(row['id'])
        row['source'] = 'User-entered field assessment'
    if include_demo:
        rows += demo_records()
    return [r for r in rows if dealer_id is None or r['dealer_id'] == dealer_id]

def record_assessment(dealer_id, scope, check_key, status, observation, evidence_note,
                      assessor, assessed_on, expires_on=None, supersedes_id=None,
                      path=DEFAULT_DB, include_demo=False):
    if check_key not in CHECKS or status not in ['Ready', 'Blocked', 'Not assessed']:
        raise ValueError('Choose a valid check and status.')
    if not all(isinstance(x,str) and x.strip() for x in [dealer_id, observation, evidence_note, assessor]):
        raise ValueError('Dealer, concrete observation, evidence note and assessor are required.')
    assessed = date.fromisoformat(str(assessed_on))
    expiry = date.fromisoformat(str(expires_on)) if expires_on else None
    if assessed > business_today(): raise ValueError('Assessment date cannot be in the future.')
    if expiry and expiry < assessed: raise ValueError('Review / expiry date must not precede the assessment.')
    scope = scope.strip()
    with connect(path) as conn:
        # BEGIN IMMEDIATE keeps an edit from overwriting a concurrently revised version.
        conn.execute('BEGIN IMMEDIATE')
        if supersedes_id:
            rows = assessment_history(dealer_id,path,include_demo,connection=conn)
            prior = next((r for r in rows if r['evidence_id'] == supersedes_id),None)
            if not prior or (prior['scope'],prior['check_key']) != (scope,check_key):
                raise ValueError('A revision must match the dealer, scope and check of its original version.')
            if any(r.get('supersedes_id') == supersedes_id for r in rows):
                raise ValueError('This version has already been revised. Reload the latest assessment.')
        cursor = conn.execute('''INSERT INTO field_assessments
            (dealer_id,scope,check_key,status,observation,evidence_note,assessor,assessed_on,
             expires_on,recorded_at,supersedes_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (dealer_id,scope,check_key,status,observation.strip(),evidence_note.strip(),assessor.strip(),
             str(assessed),str(expiry) if expiry else None,
             datetime.now(timezone.utc).isoformat(timespec='microseconds'),supersedes_id))
        return 'FIELD-'+str(cursor.lastrowid)

def assess(dealer_id, scope='', as_of=None, path=DEFAULT_DB, include_demo=False, rows=None, connection=None):
    """Never use later-recorded evidence retroactively; exact scope matching only."""
    cutoff = date.fromisoformat(str(as_of or business_today()))
    rows = assessment_history(dealer_id,path,include_demo,connection=connection) if rows is None else rows
    applicable = [r for r in rows if r['dealer_id']==dealer_id and r['scope'].strip().casefold()==scope.strip().casefold()]
    later = [r for r in applicable if date.fromisoformat(r['assessed_on']) > cutoff or recorded_date(r) > cutoff]
    known = [r for r in applicable if r not in later]
    superseded = {r.get('supersedes_id') for r in known if r.get('supersedes_id')}
    leaves = [r for r in known if r['evidence_id'] not in superseded]
    results=[]
    for key,label in CHECKS.items():
        candidates = [r for r in leaves if r['check_key']==key]
        candidates.sort(key=lambda r:(r['assessed_on'],r['recorded_at'],r['evidence_id']),reverse=True)
        fresh = [r for r in candidates if (cutoff-date.fromisoformat(r['assessed_on'])).days <= RULES['max_age_days'] and (not r['expires_on'] or cutoff <= date.fromisoformat(r['expires_on']))]
        statuses = {r['status'] for r in fresh}
        if len(statuses)>1: status,reason='Not assessed','Conflicting current evidence; verify before intervention.'
        elif fresh and fresh[0]['status']!='Not assessed': status,reason=fresh[0]['status'],'Current scoped evidence'
        elif fresh: status,reason='Not assessed','The check is explicitly not assessed.'
        elif candidates: status,reason='Not assessed','Stale evidence; reassessment required.'
        else: status,reason='Not assessed','No applicable evidence for this scope and date.'
        results.append({'check_key':key,'label':label,'status':status,'reason':reason,
            'evidence_id':f'READINESS:{dealer_id}:{scope}:{key}:{cutoff}',
            'records':fresh if fresh else candidates[:1]})
    indicator = 'Blocker identified' if any(c['status']=='Blocked' for c in results) else 'Verified' if all(c['status']=='Ready' for c in results) else 'Verification needed'
    return {'dealer_id':dealer_id,'scope':scope,'as_of':str(cutoff),'business_timezone':RULES['timezone'],'indicator':indicator,'checks':results,
            'policy_fingerprint':hashlib.sha256(json.dumps(RULES,sort_keys=True).encode()).hexdigest()[:16],'later_evidence':later,'include_demo':include_demo,'limitation':'Exact scope only. Coverage and blockers are not a health score. Later evidence is excluded from historical conclusions.'}

# Infer additional requirements from edited prose; metadata cannot bypass a gate.
TEXT_RULES = {
 'lead_growth':r'\b(increas\w*|boost\w*|generat\w*|expand\w*|driv\w*|attract\w*|acquir\w*|buy\w*|collect\w*|seek\w*|source\w*|grow\w*|scal\w*)\b.{0,45}\b(leads?|prospects?|sales demand)\b',
 'model_campaign':r'\b(run\w*|launch\w*|start\w*|fund\w*|expand\w*|execute\w*|increas\w*)\b.{0,45}\b(campaign|advertis\w*|promotion|marketing|lead generation)\b',
 'service_growth':r'\b(increas\w*|boost\w*|expand\w*|driv\w*|attract\w*|promot\w*)\b.{0,45}\b(service demand|service bookings?|workshop visits?|service customers?)\b',
 'stock_change':r'\b(increas\w*|reduc\w*|chang\w*|rebalanc\w*|order\w*|buy\w*|shift\w*)\b.{0,45}\b(stock|inventory|variants?|model mix)\b',
 'site_intervention':r'\b(run\w*|host\w*|hold\w*|organis\w*|organiz\w*|launch\w*)\b.{0,45}\b(event|showroom|on.site|test.drive day)\b'
}

def prerequisites(proposal, readiness):
    kind = proposal.get('intervention','health_review')
    if kind not in KINDS: raise ValueError('Unknown intervention type.')
    text = proposal.get('action','')
    inferred = {k for k,pattern in TEXT_RULES.items() if re.search(pattern,text,re.I|re.S)}
    kinds = inferred | {kind}
    keys = set(k for k in kinds for k in RULES['prerequisites'][k])
    if proposal.get('site_dependent') and (kind not in ['verification','remediation'] or inferred): keys.add('facility_access')
    checks = [c for c in readiness['checks'] if c['check_key'] in keys]
    blocked = [c for c in checks if c['status']=='Blocked']
    unknown = [c for c in checks if c['status']=='Not assessed']
    scope_missing=bool(kinds.intersection(RULES['requires_scope']) and not readiness['scope'].strip())
    state = 'Blocked' if blocked else 'Verification required' if unknown or scope_missing else 'Eligible for approval'
    alternative = 'Resolve demo-vehicle availability before increasing leads for this model.' if any(c['check_key']=='demo_vehicle' for c in blocked) else 'Resolve the identified prerequisites before proceeding.' if blocked else 'Approve a separate verification action first; reassess before intervention.' if unknown else 'Prerequisites satisfied for this action; manager approval still required.'
    if scope_missing: alternative='Specify the relevant model / campaign / service scope and verify its prerequisites before approval.'
    corrective = {'demo_vehicle':'Arrange a usable relevant demo vehicle',
        'sales_staff':'Assign the required trained sales staff',
        'service_capacity':'Restore relevant equipment, technician coverage and essential parts',
        'demand_fit':'Verify local model / variant demand and stock fit',
        'facility_access':'Resolve the relevant customer access or facility condition'}
    alternatives=[{'intervention':'remediation' if c['status']=='Blocked' else 'verification',
        'action':(corrective[c['check_key']] if c['status']=='Blocked' else 'Verify '+c['label'].lower())+' for '+(readiness['scope'] or 'the dealership')+'. Record a new assessment before the intervention.',
        'evidence_id':c['evidence_id']} for c in checks if c['status']!='Ready']
    return {'status':state,'checks':checks,'alternative':alternative,'corrective_alternatives':alternatives,'scope_required':scope_missing,'policy_fingerprint':hashlib.sha256(json.dumps(RULES,sort_keys=True).encode()).hexdigest()[:16],'inferred_interventions':sorted(inferred)}

def evidence_ids(readiness):
    return {c['evidence_id'] for c in readiness['checks']} | {r['evidence_id'] for c in readiness['checks'] for r in c['records']}
