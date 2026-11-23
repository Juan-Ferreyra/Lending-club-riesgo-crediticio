"""Identifier-preserving application-context joins and chronological data recipes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.prepare import file_sha256, ids_sha256
from src.train import sample_development
from src.utils import write_json, utc_now

CACHE = Path('data/optimization/application_context.parquet')
SOURCE_SHA256 = '55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d'


def prepare_context(source='data/accepted_2007_to_2018Q4.csv.gz'):
    manifest = CACHE.with_suffix('.json')
    if CACHE.exists():
        record = json.loads(manifest.read_text(encoding='utf-8'))
        assert file_sha256(CACHE) == record['cache_sha256']
        assert record['source_sha256'] == SOURCE_SHA256
        return record
    assert file_sha256(source) == SOURCE_SHA256, 'Unexpected source snapshot'
    partitions = ['development', 'holdout_2015', 'holdout_2016']
    ids = set()
    for name in partitions:
        ids.update(pd.read_parquet(f'data/processed/{name}.parquet', columns=['id']).id.astype(str))
    chunks = []
    for chunk in pd.read_csv(source, usecols=['id', 'emp_title', 'zip_code'],
                             dtype='string', chunksize=150000, low_memory=False):
        rows = chunk[chunk.id.isin(ids)]
        if len(rows):
            chunks.append(rows)
    extra = pd.concat(chunks, ignore_index=True)
    assert not extra.id.duplicated().any()
    assert set(extra.id) == ids
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    extra.to_parquet(CACHE, index=False)
    record = {'created_utc': utc_now(), 'source_sha256': SOURCE_SHA256,
              'cache_sha256': file_sha256(CACHE), 'rows': len(extra),
              'ids_sha256': ids_sha256(extra.id), 'fields': ['emp_title', 'zip_code'],
              'labels_extracted': False,
              'purpose': 'Application fields only; joining evaluation inputs does not use their labels for selection.'}
    write_json(manifest, record)
    return record


def load_partition(name='development'):
    raw = pd.read_parquet(f'data/processed/{name}.parquet')
    extra = pd.read_parquet(CACHE)
    raw['id'] = raw.id.astype(str)
    extra['id'] = extra.id.astype(str)
    result = raw.merge(extra, on='id', how='left', sort=False, validate='one_to_one', indicator=True)
    assert result['_merge'].eq('both').all()
    assert result.id.tolist() == raw.id.tolist()
    return result.drop(columns='_merge')


def train_rows(df, end, start='2012-01-01', cap=None):
    mask = df.issue_date.between(start, end) & df.term.str.strip().eq('36 months')
    frame = df.loc[mask].copy()
    return sample_development(frame, cap) if cap else frame


def valid_rows(df, start, end, cap=None):
    frame = df.loc[df.issue_date.between(start, end) & df.term.str.strip().eq('36 months')].copy()
    return sample_development(frame, cap) if cap else frame


def recency_weights(frame, end, half_life=0):
    if not half_life:
        return None
    months = (pd.Timestamp(end) - frame.issue_date).dt.days.to_numpy() / 30.4375
    assert (months >= 0).all()
    weights = np.power(2.0, -months / half_life)
    return weights / weights.mean()


if __name__ == '__main__':
    print(prepare_context(), flush=True)
