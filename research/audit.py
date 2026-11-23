"""Streaming audit of the source; do not change the previous prepared cohorts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import RAW_PREDICTORS, BASE_NUMERIC, BUREAU_NUMERIC, FeatureBuilder
from src.prepare import file_sha256
from src.utils import write_json, utc_now

DATA = Path('data/research_v2')
OUT = Path(os.environ.get('LENDING_RESEARCH_OUTPUT','outputs/research_v2'))
STATUS = ['Fully Paid', 'Charged Off']
CORE = ['annual_inc', 'dti', 'loan_amnt', 'fico_range_low', 'revol_util', 'open_acc', 'total_acc']


def distribution_distance(reference, sample, column):
    p = reference[column].fillna('MISSING').value_counts(normalize=True)
    q = sample[column].fillna('MISSING').value_counts(normalize=True)
    p, q = p.align(q, fill_value=0)
    return float(np.abs(p-q).sum()/2)


def audit(source):
    expected=json.loads(Path('data/processed/manifest.json').read_text(encoding='utf-8'))['source_sha256']
    source_digest=file_sha256(source)
    assert source_digest==expected, 'Source differs from the original experiment; use a separately designed dataset/protocol.'
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = DATA/'through_2015_all_status.parquet'
    required=['source_status_counts.csv','missing_by_year.csv','invalid_by_year.csv','source_audit_manifest.json']
    if not cache.exists() or any(not (OUT/name).exists() for name in required):
        import pyarrow as pa
        import pyarrow.parquet as pq
        columns = list(dict.fromkeys(['id', 'loan_status', 'grade', 'sub_grade', 'int_rate', 'last_pymnt_d']+RAW_PREDICTORS))
        counts, missing, invalid, writer, total, max_payment = [], [], [], None, 0, None
        building = cache.with_suffix('.building.parquet')
        try:
            for chunk in pd.read_csv(source, usecols=columns, dtype=str, chunksize=75000):
                total += len(chunk)
                chunk['issue_date'] = pd.to_datetime(chunk.issue_d, format='%b-%Y', errors='coerce')
                chunk['issue_year'] = chunk.issue_date.dt.year.astype('Int16')
                chunk['term'] = chunk.term.str.strip()
                chunk = chunk[chunk.id.notna() & chunk.issue_date.notna()].copy()
                pay = pd.to_datetime(chunk.last_pymnt_d, format='%b-%Y', errors='coerce').max()
                if pd.notna(pay) and (max_payment is None or pay > max_payment):
                    max_payment = pay
                chunk['terminal'] = chunk.loan_status.isin(STATUS)
                counts.append(chunk.groupby(['issue_year', 'term', 'loan_status'], dropna=False).size().reset_index(name='n'))
                for c in BASE_NUMERIC+BUREAU_NUMERIC+['int_rate']:
                    chunk[c] = pd.to_numeric(chunk[c], errors='coerce').astype('float32')
                for year, g in chunk.groupby('issue_year'):
                    for col in BASE_NUMERIC+BUREAU_NUMERIC:
                        missing.append({'year': int(year), 'feature': col, 'n': len(g), 'missing_n': int(g[col].isna().sum())})
                    rules = {'income_nonpositive': g.annual_inc.le(0), 'dti_negative': g.dti.lt(0),
                             'fico_outside_300_850': ~g.fico_range_low.between(300, 850)&g.fico_range_low.notna(),
                             'fico_low_exceeds_high': g.fico_range_low.gt(g.fico_range_high),
                             'revol_util_negative': g.revol_util.lt(0), 'loan_amnt_nonpositive': g.loan_amnt.le(0)}
                    for name, mask in rules.items():
                        invalid.append({'year': int(year), 'rule': name, 'n': len(g), 'invalid_n': int(mask.sum())})
                keep = chunk[chunk.issue_year.le(2015)].drop(columns=['last_pymnt_d']).copy()
                if not keep.empty:
                    # Explicit categorical string types avoid chunk-dependent Arrow null schemas.
                    for col in keep.select_dtypes(include=['object', 'string']).columns:
                        keep[col] = keep[col].astype('string')
                    table = pa.Table.from_pandas(keep, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(building, table.schema, compression='zstd')
                    writer.write_table(table.cast(writer.schema))
                if total % 375000 == 0:
                    print(f'Audit source rows: {total:,}', flush=True)
        finally:
            if writer is not None:
                writer.close()
        building.replace(cache)
        pd.concat(counts).groupby(['issue_year', 'term', 'loan_status'], dropna=False).n.sum().reset_index().to_csv(OUT/'source_status_counts.csv', index=False)
        m = pd.DataFrame(missing).groupby(['year', 'feature'])[['n', 'missing_n']].sum().reset_index()
        m['missing_fraction'] = m.missing_n/m.n
        m.to_csv(OUT/'missing_by_year.csv', index=False)
        v = pd.DataFrame(invalid).groupby(['year', 'rule'])[['n', 'invalid_n']].sum().reset_index()
        v['invalid_fraction'] = v.invalid_n/v.n
        v.to_csv(OUT/'invalid_by_year.csv', index=False)
        write_json(OUT/'source_audit_manifest.json', {'created_utc': utc_now(), 'source_rows': total,
                   'source_sha256': source_digest, 'cache_sha256': file_sha256(cache),
                   'latest_recorded_payment_month': str(max_payment),
                   'payment_date_used_as_predictor': False, 'new_post_2015_individual_outcomes_opened_for_modeling': False})
    raw = pd.read_parquet(cache)
    assert raw.id.is_unique
    raw['target'] = raw.loan_status.map({'Fully Paid': 0, 'Charged Off': 1}).astype('Int8')
    assert raw.loc[~raw.terminal, 'target'].isna().all()
    terminal = raw[raw.terminal].copy()
    coverage = pd.read_csv(OUT/'source_status_counts.csv')
    coverage['charged'] = coverage.loan_status.eq('Charged Off')
    coverage['terminal'] = coverage.loan_status.isin(STATUS)
    rows = []
    for (year, term), g in coverage.groupby(['issue_year', 'term']):
        total = int(g.n.sum()); done = int(g.loc[g.terminal, 'n'].sum()); bad = int(g.loc[g.charged, 'n'].sum())
        rows.append({'year': int(year), 'term': term, 'accepted_n': total, 'resolved_n': done,
                     'nonterminal_n': total-done, 'terminal_fraction': done/total,
                     'conditional_default_rate': bad/done if done else None,
                     'all_accepted_default_lower_bound': bad/total,
                     'all_accepted_default_upper_bound': (bad+total-done)/total})
    pd.DataFrame(rows).to_csv(OUT/'outcome_coverage_bounds.csv', index=False)
    # Compare the initial random terminal sample to its actual sampling population.
    sample = pd.read_csv('data/lending_club_muestra.csv.gz', dtype=str)
    sample['issue_year'] = pd.to_datetime(sample.issue_d, format='%b-%Y').dt.year.astype('Int16')
    sample['term'] = sample.term.str.strip()
    represent = []
    for col in ['issue_year', 'term', 'loan_status']:
        sampled = sample[col].value_counts(normalize=True)
        census = coverage.groupby({'issue_year':'issue_year', 'term':'term', 'loan_status':'loan_status'}[col]).n.sum()
        if col == 'loan_status':
            census = census[census.index.isin(STATUS)]
        elif col == 'issue_year':
            census = coverage[coverage.terminal].groupby('issue_year').n.sum()
        else:
            census = coverage[coverage.terminal].groupby('term').n.sum()
        p, q = (census/census.sum()).align(sampled, fill_value=0)
        represent.append({'population': 'all_terminal_2007_2018', 'dimension': col, 'TVD': float(np.abs(p-q).sum()/2)})
        if col != 'loan_status':
            all_counts = coverage.groupby(col).n.sum()
            p, q = (all_counts/all_counts.sum()).align(sampled, fill_value=0)
            represent.append({'population': 'all_accepted_2007_2018', 'dimension': col, 'TVD': float(np.abs(p-q).sum()/2)})
    pd.DataFrame(represent).to_csv(OUT/'sample_representativeness.csv', index=False)
    # Distribution shifts and outcome-selection associations are descriptive, not causal.
    summaries, selection_bias = [], []
    for (year, term, resolved), g in raw.groupby(['issue_year', 'term', 'terminal']):
        for col in CORE:
            values = pd.to_numeric(g[col], errors='coerce')
            summaries.append({'year': int(year), 'term': term, 'terminal': bool(resolved), 'feature': col,
                              'n': len(g), 'mean': values.mean(), 'sd': values.std(), 'median': values.median(),
                              'q10': values.quantile(.1), 'q90': values.quantile(.9), 'missing': values.isna().mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'population_features_by_year_term_status.csv', index=False)
    for (year, term, col), g in summary.groupby(['year', 'term', 'feature']):
        if set(g.terminal) != {True, False}:
            continue
        a = g[g.terminal].iloc[0]; b = g[~g.terminal].iloc[0]
        pooled = np.sqrt((a.sd*a.sd+b.sd*b.sd)/2)
        selection_bias.append({'year': int(year), 'term': term, 'feature': col,
                               'SMD_terminal_minus_nonterminal': float((a['mean']-b['mean'])/pooled) if pooled>0 else 0,
                               'terminal_n': int(a.n), 'nonterminal_n': int(b.n)})
    pd.DataFrame(selection_bias).to_csv(OUT/'selection_bias_smd.csv', index=False)
    legacy_train = terminal[terminal.issue_date.le('2014-06-30')]
    features = FeatureBuilder().fit_transform(legacy_train)
    unlearnable = features.columns[(features.nunique()<=1)|(features.isna().mean()>.95)].tolist()
    write_json(OUT/'audit_findings.json', {'unlearnable_or_95pct_missing_in_old_training': unlearnable,
        'count': len(unlearnable), 'hypothesis_status': {
            'missingness': 'Structural feature availability verified; predictive harm requires ablation experiments.',
            'active_loans': 'Unequal terminal coverage verified. Eventual active-loan default cannot be identified from this snapshot.',
            'temporal_shift': 'Rates and covariates vary; censoring and population changes must be separated.',
            'representativeness': 'Evaluate the random sample against terminal loans separately from all accepted loans; rejected applicants unavailable.'},
        'pristine_mature_test_available_in_current_source': False,
        'scope_of_new_model_comparison': 'exploratory repeated retrospective benchmark, not a fresh final test'})
    print(pd.DataFrame(rows).query('year >= 2013').to_string(index=False), flush=True)
    print(pd.DataFrame(represent).to_string(index=False), flush=True)
    print('Unlearnable / >95% missing old-training features:', len(unlearnable), flush=True)
    print('Audit completed; no target assigned to active loans.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', default='data/accepted_2007_to_2018Q4.csv.gz')
    audit(p.parse_args().source)
