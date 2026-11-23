"""Rebuild the recorded V3 finalists without repeating the adaptive search."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from optimization.data import prepare_context, load_partition
from optimization.evaluation import freeze, benchmark, read_json
from src.prepare import file_sha256
from src.utils import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    parser.add_argument('--recorded-output', default='outputs/research_v3')
    parser.add_argument('--v2-output', default=os.environ.get('LENDING_V2_OUTPUT', 'outputs/research_v2'))
    args = parser.parse_args()
    destination, recorded, previous = Path(args.destination), Path(args.recorded_output), Path(args.v2_output)
    if destination.resolve() == recorded.resolve():
        raise ValueError('Fixed replay must have a different destination from the published experiment')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use an empty destination; replay never overwrites an experiment')
    destination.mkdir(parents=True, exist_ok=True)
    protocol_path = Path('experiments/protocol_v3.json')
    protocol = read_json(protocol_path)
    selection = read_json(recorded/'selection.json')
    assert selection['protocol_sha256'] == file_sha256(protocol_path)
    write_json(destination/'selection.json', selection)
    write_json(destination/'ensemble_mappings.json', read_json(recorded/'ensemble_mappings.json'))
    write_json(destination/'context_manifest.json', prepare_context())
    df = load_partition()
    with threadpool_limits(limits=3):
        freeze(df, protocol, destination, previous, protocol_path)
        del df
        benchmark(protocol, destination, previous, protocol_path)
    expected = pd.read_csv(recorded/'benchmark_metrics.csv').set_index(['cohort', 'model'])
    actual = pd.read_csv(destination/'benchmark_metrics.csv').set_index(['cohort', 'model'])
    assert actual.index.equals(expected.index)
    columns = ['raw_AP', 'raw_ROC_AUC', 'Brier', 'recall_at_top20']
    differences = (actual[columns]-expected[columns]).abs()
    max_error = float(differences.max().max())
    matches = bool(np.isclose(actual[columns], expected[columns], atol=1e-8, rtol=0).all())
    write_json(destination/'reproduction_comparison.json', {
        'recorded_output': recorded.as_posix(), 'max_absolute_metric_difference': max_error,
        'matches_at_1e-8': matches,
        'new_independent_test': False,
        'note': 'Reconstruction of frozen specifications; no new model or hyperparameter selection.'})
    assert matches, f'Fixed replay differs from recorded metrics: max error={max_error:.12g}'
    print(f'Fixed replay completed; maximum metric difference: {max_error:.12g}', flush=True)


if __name__ == '__main__':
    main()
