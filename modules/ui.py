"""Business-facing labels and small presentation helpers."""
import pandas as pd
import streamlit as st
from modules.scoring import CONFIG, normalize
LABELS = {
 'sales_units':('Vehicle sales','units'), 'target_units':('Sales target','units'),
 'target_attainment_pct':('Sales vs target','%'), 'lead_conversion_pct':('Lead conversion','%'),
 'operating_margin_pct':('Operating margin','%'), 'overdue_receivables_pct':('Overdue receivables','%'),
 'avg_payment_delay_days':('Payment delay','days'), 'inventory_days':('Inventory cover','days'),
 'stock_over_90_days_pct':('Stock older than 90 days','%'),
 'customer_satisfaction':('Customer satisfaction','/100'),
 'service_retention_pct':('Service retention','%'),
 'complaint_resolution_days':('Complaint resolution','days')}
def label(key): return LABELS.get(key,(key.replace('_',' ').title(),''))[0]
def value(key,number):
 if number is None or pd.isna(number): return 'Missing'
 unit=LABELS.get(key,('', ''))[1]
 if unit=='units': return f'{number:,.0f} units'
 if unit=='%': return f'{number:.1f}%'
 if unit=='/100': return f'{number:.1f} / 100'
 return f'{number:.1f} {unit}'.strip()
def index(number): return f'{number:.1f} / 100' if pd.notna(number) else 'Unassessed'
def risk(number): return f'{number:.1%}' if pd.notna(number) else 'Unavailable'
def priority(text):
 return {'P1 • Verify data':'P1 · Verify data','P1 • Urgent':'P1 · Urgent','P2 • Review':'P2 · Review','P3 • Monitor':'P3 · Monitor','P4 • Routine':'P4 · Routine'}[text]
def concern(row):
 if row.missing_inputs: return 'Verify missing inputs before assessing health', 'Missing: '+', '.join(label(k) for k in row.missing_inputs.split(', '))+'.'
 if row.target_attainment_pct>=100 and row['Financial_score']<50:
  return 'Strong sales, weak financial indicators', f"Sales are {value('target_attainment_pct',row.target_attainment_pct)} of target, while overdue receivables are {value('overdue_receivables_pct',row.overdue_receivables_pct)} and payment delay is {value('avg_payment_delay_days',row.avg_payment_delay_days)}."
 weakest=min(CONFIG['weights'],key=lambda p:row[p+'_score'])
 metrics=[k for k,s in CONFIG['kpis'].items() if s['pillar']==weakest]
 k=min(metrics,key=lambda k:normalize(row[k],CONFIG['kpis'][k]))
 if row.status=='Healthy': return 'Keep an eye on '+label(k).lower(), f"{label(k)} is {value(k,row[k])}. The {weakest.lower()} pillar scores {index(row[weakest+'_score'])}."
 return weakest+' needs attention', f"{label(k)} is {value(k,row[k])}. The {weakest.lower()} pillar scores {index(row[weakest+'_score'])}."
def chart(fig,height=280):
 fig.update_layout(height=height,margin=dict(l=12,r=12,t=32,b=12),font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif',size=12,color='#1D1D1F'),paper_bgcolor='#FFFFFF',plot_bgcolor='#FFFFFF',legend_title_text='',hovermode='x unified')
 fig.update_xaxes(showgrid=False,title_font_size=12)
 fig.update_yaxes(gridcolor='#E8E8ED',title_font_size=12,zerolinecolor='#E8E8ED')
 st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
def evidence_table(refs):
 return pd.DataFrame([{'Observed metric':label(r['metric']),'Verified value':value(r['metric'],r['value'])} for r in refs])


# Presentation only: these tokens never participate in scoring or prioritisation.
TONES = {
 'green': ('#157347', '#EDF7F0', '#CDE6D6'),
 'amber': ('#805D00', '#FBF5E7', '#E8DDB9'),
 'orange': ('#995000', '#FFF3E5', '#EBD3B3'),
 'red': ('#B42332', '#FCEEF0', '#EBCBD0'),
 'grey': ('#626268', '#EFEFF2', '#DCDCE2'),
 'slate': ('#475569', '#F1F5F9', '#D7DFE8'),
}
STATUS_TONES = {
 'Healthy':'green', 'Watchlist':'amber', 'At Risk':'orange', 'Critical':'red',
 'Ready':'green', 'Blocked':'red', 'Not assessed':'grey',
 'Deterioration flagged':'orange', 'No deterioration flag':'slate',
 'Completed':'green', 'Past deadline':'orange',
 'Verified':'green', 'Blocker identified':'red', 'Verification needed':'grey',
 'Eligible for approval':'green', 'Verification required':'grey',
}
def status_style(status):
 return TONES[STATUS_TONES.get(str(status), 'grey')]

def badge(status, detail=None):
 from html import escape
 color, background, border = status_style(status)
 title = ' title="'+escape(str(detail), quote=True)+'"' if detail else ''
 return f'<span class="status-badge"{title} style="color:{color};background:{background};border:1px solid {border}">{escape(str(status))}</span>'

def change_color(number):
 return TONES['grey' if pd.isna(number) or number == 0 else 'green' if number > 0 else 'red'][0]

def health_change(number, period):
 from html import escape
 text = f'{number:+.1f} pts vs {period}' if pd.notna(number) and period else 'No previous available month'
 return f'<small style="color:{change_color(number)}">{escape(text)}</small>'

def style_status_table(table, status_columns, change_column=None):
 """Native interactive table; tint individual status cells, never whole rows."""
 def cell(status):
  fg, bg, _ = status_style(status)
  return f'color: {fg}; background-color: {bg}; font-weight: 600'
 styled = table.style
 for column in status_columns:
  if column in table: styled = styled.map(cell, subset=[column])
 if change_column and change_column in table:
  styled = styled.map(lambda v: 'color: '+change_color(v), subset=[change_column])
 return styled
