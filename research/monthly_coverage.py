"""Census monthly follow-up; select mature secondary months by coverage, not AP."""
import argparse
import pandas as pd
from research.experiments import OUT, read_json
from src.prepare import file_sha256
from src.utils import write_json


def audit(source):
    assert file_sha256(source)==read_json('data/processed/manifest.json')['source_sha256'], 'Source fingerprint differs from the modeling data.'
    path=OUT/'source_monthly_status_counts.csv'
    OUT.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        pieces=[]
        for raw in pd.read_csv(source,usecols=['id','issue_d','term','loan_status'],dtype=str,chunksize=200000):
            raw['issue_month']=pd.to_datetime(raw.issue_d,format='%b-%Y',errors='coerce').dt.strftime('%Y-%m')
            raw['term']=raw.term.str.strip()
            raw=raw[raw.id.notna()&raw.issue_month.notna()]
            pieces.append(raw.groupby(['issue_month','term','loan_status'],dropna=False).size().reset_index(name='n'))
        pd.concat(pieces).groupby(['issue_month','term','loan_status'],dropna=False).n.sum().reset_index().to_csv(path,index=False)
    counts=pd.read_csv(path)
    rows=[]
    for (month,term),g in counts.groupby(['issue_month','term']):
        n=int(g.n.sum());done=int(g.loc[g.loan_status.isin(['Fully Paid','Charged Off']),'n'].sum())
        bad=int(g.loc[g.loan_status.eq('Charged Off'),'n'].sum())
        rows.append({'issue_month':month,'term':term,'accepted_n':n,'terminal_n':done,
                     'terminal_fraction':done/n,'nonterminal_n':n-done,
                     'conditional_default_rate':bad/done if done else None,
                     'all_accepted_default_lower_bound':bad/n,
                     'all_accepted_default_upper_bound':(bad+n-done)/n})
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'monthly_outcome_coverage.csv',index=False)
    eligible=frame[frame.issue_month.str.startswith('2016')&frame.term.eq('36 months')&frame.terminal_fraction.ge(.995)]
    write_json(OUT/'secondary_mature_months.json',{'rule':'2016 36-month source cohorts with terminal coverage >= 0.995; no model performance used to choose months',
               'months':eligible.issue_month.tolist(),'coverage_threshold':.995,
               'independent_pristine_test':False,'source_population_before_initial_sample_exclusion':True})
    print(frame[frame.issue_month.str.startswith('2016')&frame.term.eq('36 months')].to_string(index=False),flush=True)
    print('Mature secondary months:',eligible.issue_month.tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='data/accepted_2007_to_2018Q4.csv.gz')
    audit(parser.parse_args().source)
