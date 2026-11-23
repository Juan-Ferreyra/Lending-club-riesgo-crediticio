"""Freeze development-selected strategies before opening reused temporal benchmarks."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import average_precision_score, roc_auc_score

from optimization.data import load_partition, train_rows, valid_rows
from optimization.ensemble import PositiveSigmoid, ensemble_score
from src.evaluate import metrics, choose_thresholds, paired_bootstrap
from src.prepare import file_sha256, ids_sha256
from src.utils import write_json, utc_now


BASE = Path(os.environ.get('LENDING_BASE_OUTPUT', 'outputs'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def december_split(frame):
    """Split an already date-filtered December frame by stable identifier hash.

    Even SHA256 parity calibrates; odd parity sets decision policies. This is a
    disjoint same-month split, not a claim of chronological separation within December.
    """
    if frame.id.astype(str).duplicated().any():
        raise ValueError('December split requires unique loan IDs')
    parity = frame.id.astype(str).map(lambda loan_id: hashlib.sha256(loan_id.encode('utf-8')).digest()[-1] % 2)
    calibration = frame.loc[parity.eq(0)].copy()
    policy = frame.loc[parity.eq(1)].copy()
    assert set(calibration.id).isdisjoint(policy.id)
    assert len(calibration) + len(policy) == len(frame)
    return calibration, policy


def inherited_integrity(V2):
    """Verify the previously frozen research, source, data, and model artifacts."""
    V2 = Path(V2)
    original = read_json(BASE / 'selection_lock.json')
    previous = read_json(V2 / 'selection_lock.json')
    previous_receipt = read_json(V2 / 'benchmark_receipt.json')
    original_receipt = read_json(BASE / 'final' / 'evaluation_receipt.json')
    expected = dict(original['code_sha256']) | dict(previous['code_sha256'])
    expected.update({
        str(BASE / 'models' / 'frozen_bundle.joblib'): original['model_sha256'],
        str(V2 / 'models' / 'frozen_bundle.joblib'): previous['model_sha256'],
        'experiments/protocol.json': original['protocol_sha256'],
        'experiments/protocol_v2.json': previous['protocol_sha256'],
        str(BASE / 'selection.json'): original['selection_sha256'],
        str(V2 / 'selection.json'): previous['selection_sha256'],
        str(BASE / 'final' / 'metrics.json'): original_receipt['metrics_sha256'],
        str(V2 / 'benchmark_metrics.json'): previous_receipt['metrics_sha256'],
        str(V2 / 'selection_lock.json'): previous_receipt['lock_sha256'],
        str(BASE / 'selection_lock.json'): original_receipt['lock_sha256'],
        'data/processed/manifest.json': original['data_manifest_sha256'],
    })
    manifest = read_json('data/processed/manifest.json')
    for name, entry in manifest['partitions'].items():
        expected[f'data/processed/{name}.parquet'] = entry['file_sha256']
    expected = {str(Path(path)).replace('\\', '/'): digest for path, digest in expected.items()}
    for path, digest in expected.items():
        assert file_sha256(path) == digest, f'Previously frozen artifact changed: {path}'
    return expected


def _training_code_hashes():
    return {f'optimization/{name}.py': file_sha256(f'optimization/{name}.py')
            for name in ['data', 'features', 'models', 'ensemble', 'experiments']}


def _verify_lock(OUT, V2, PROTOCOL):
    OUT, V2, PROTOCOL = Path(OUT), Path(V2), Path(PROTOCOL)
    lock = read_json(OUT / 'selection_lock.json')
    assert file_sha256(PROTOCOL) == lock['protocol_sha256'], 'Frozen protocol changed'
    assert file_sha256(OUT / 'selection.json') == lock['selection_sha256'], 'Frozen selection changed'
    assert file_sha256(OUT / 'models' / 'frozen_bundle.joblib') == lock['model_sha256'], 'Frozen model changed'
    for group in ['code_sha256', 'model_files_sha256', 'inherited_sha256', 'input_sha256']:
        for path, digest in lock[group].items():
            assert file_sha256(path) == digest, f'Frozen artifact changed: {path}'
    return lock


def _metric_row(y, score, p, threshold):
    result = metrics(y, p, threshold)
    result['raw_AP'] = float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None
    result['raw_ROC_AUC'] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    result['calibration_AP_preserved'] = (result['AP'] is None or
                                         np.isclose(result['AP'], result['raw_AP'], rtol=0, atol=1e-12))
    return result


def freeze(df, protocol, OUT, V2, PROTOCOL):
    """Refit through November, calibrate on half of December, and freeze all choices."""
    # Runtime import avoids a circular dependency when the CLI is run as a module.
    from optimization.experiments import fit_model

    OUT, V2, PROTOCOL = Path(OUT), Path(V2), Path(PROTOCOL)
    for child in ['models', 'predictions']:
        (OUT / child).mkdir(parents=True, exist_ok=True)
    if (OUT / 'selection_lock.json').exists():
        _verify_lock(OUT, V2, PROTOCOL)
        print('V3 frozen bundle already present and verified.', flush=True)
        return
    assert not (OUT / 'benchmark_receipt.json').exists(), 'Cannot refit after opening the benchmark'
    inherited = inherited_integrity(V2)
    selection = read_json(OUT / 'selection.json')
    assert selection['protocol_sha256'] == file_sha256(PROTOCOL)
    followup = Path('experiments/development_followup_v3.json')
    if selection.get('development_followup_sha256'):
        assert file_sha256(followup) == selection['development_followup_sha256']
    preparation_followup = Path('experiments/preparation_followup_v3.json')
    if selection.get('preparation_followup_sha256'):
        assert file_sha256(preparation_followup) == selection['preparation_followup_sha256']
    assert not selection['new_benchmark_predictions_opened']
    assert protocol['calibration_start'] == protocol['policy_start']
    assert protocol['calibration_end'] == protocol['policy_end']
    december = valid_rows(df, protocol['calibration_start'], protocol['calibration_end'])
    cal, policy = december_split(december)
    assert len(cal) > 0 and len(policy) > 0
    assert pd.Timestamp(protocol['final_training_end']) < cal.issue_date.min()
    assert cal.issue_date.dt.month.eq(12).all() and policy.issue_date.dt.month.eq(12).all()
    split = {
        'rule': 'SHA256(str(id), UTF-8) parity: even calibration, odd policy; same month, disjoint groups',
        'calibration_n': len(cal), 'policy_n': len(policy), 'december_n': len(december),
        'calibration_ids_sha256': ids_sha256(cal.id), 'policy_ids_sha256': ids_sha256(policy.id),
        'december_ids_sha256': ids_sha256(december.id), 'overlap_n': 0,
    }
    write_json(OUT / 'calibration_split.json', split)
    models, calibrators, thresholds, training, rows = {}, {}, {}, {}, []
    cal_scores, policy_scores = {}, {}
    mapping_path = OUT / 'models' / 'ensemble_mappings.joblib'
    mappings = (joblib.load(mapping_path) if mapping_path.exists() else
                {name: PositiveSigmoid.from_dict(value)
                 for name, value in read_json(OUT / 'ensemble_mappings.json').items()})
    training_code = _training_code_hashes()
    for name in selection['finalists']:
        spec = selection['specs'][name]
        train = train_rows(df, protocol['final_training_end'], spec['start'])
        assert train.issue_date.max() < cal.issue_date.min()
        assert set(train.id).isdisjoint(december.id)
        destination = OUT / 'models' / f'final_{name}.joblib'
        metadata_path = destination.with_suffix('.json')
        metadata = {
            'spec': spec, 'train_n': len(train), 'train_ids_sha256': ids_sha256(train.id),
            'train_start': str(train.issue_date.min().date()),
            'train_end': str(train.issue_date.max().date()),
            'training_code_sha256': training_code,
        }
        started = time.time()
        if destination.exists():
            saved = read_json(metadata_path)
            assert all(saved[key] == value for key, value in metadata.items()), f'Stale checkpoint: {name}'
            assert saved['model_sha256'] == file_sha256(destination)
            model = joblib.load(destination)
        else:
            print(f'V3 fitting final {name}: n={len(train):,}, through November2014', flush=True)
            model = fit_model(spec, train)
            assert not model.convergence_warnings
            joblib.dump(model, destination, compress=3)
            write_json(metadata_path, metadata | {'model_sha256': file_sha256(destination)})
        models[name], training[name] = model, metadata
        cal_scores[name], policy_scores[name] = model.predict_score(cal), model.predict_score(policy)
        print(f'V3 final {name}: n={len(train):,}, {time.time() - started:.1f}s', flush=True)
        del train
        gc.collect()
    selected = selection['selected']
    if selected not in models:
        cal_scores[selected] = ensemble_score(cal_scores, selection['members'], mappings)
        policy_scores[selected] = ensemble_score(policy_scores, selection['members'], mappings)
    policy_predictions = policy[['id', 'target', 'issue_date']].copy()
    for name, score in cal_scores.items():
        calibration = PositiveSigmoid().fit(score, cal.target)
        p = calibration.predict(policy_scores[name])
        calibrators[name] = calibration
        thresholds[name] = choose_thresholds(policy.target, p)
        row = {'model': name, 'method': 'positive_sigmoid', 'calibration_n': len(cal),
               'sigmoid_slope': calibration.slope_, 'sigmoid_intercept': calibration.intercept_,
               **_metric_row(policy.target, policy_scores[name], p, thresholds[name]['f1'])}
        rows.append(row)
        policy_predictions[name] = p
        policy_predictions[f'score__{name}'] = policy_scores[name]
    bundle = {
        'models': models, 'calibrators': calibrators, 'thresholds': thresholds,
        'selected': selected, 'members': selection['members'], 'mappings': mappings,
        'specs': {name: selection['specs'][name] for name in selection['finalists']},
        'training': training, 'split': split,
        'control_note': 'v2_catboost uses V2 parameters but refits through November; v2_frozen remains the exact previous model.',
    }
    joblib.dump(bundle, OUT / 'models' / 'frozen_bundle.joblib', compress=3)
    write_json(OUT / 'calibration_selection.json', rows)
    write_json(OUT / 'calibrators.json', {name: calibration.to_dict()
                                        for name, calibration in calibrators.items()})
    pd.DataFrame(rows).drop(columns='confusion_matrix').to_csv(OUT / 'calibration_selection.csv', index=False)
    policy_predictions.to_parquet(OUT / 'predictions' / 'policy.parquet', index=False)
    write_json(OUT / 'training_manifest.json', training)
    assert training_code == _training_code_hashes(), 'Training code changed during fitting'
    inputs = [Path('data/optimization/application_context.parquet'),
              Path('data/optimization/application_context.json'),
              V2 / 'predictions' / 'benchmark_2015_36.parquet',
              V2 / 'predictions' / 'benchmark_2016_36.parquet']
    if selection.get('development_followup_sha256'):
        inputs.append(followup)
    if selection.get('preparation_followup_sha256'):
        inputs.append(preparation_followup)
    if (OUT / 'ensemble_mappings.json').exists():
        inputs.append(OUT / 'ensemble_mappings.json')
    lock = {
        'frozen_utc': utc_now(), 'selected': selected, 'members': selection['members'],
        'model_sha256': file_sha256(OUT / 'models' / 'frozen_bundle.joblib'),
        'protocol_sha256': file_sha256(PROTOCOL), 'selection_sha256': file_sha256(OUT / 'selection.json'),
        'code_sha256': {f'optimization/{name}.py': file_sha256(f'optimization/{name}.py')
                       for name in ['__init__', 'data', 'features', 'models', 'ensemble', 'experiments', 'evaluation']},
        'code_hash_policy': 'Freeze every module that prepares data, fits/selects/calibrates models, or computes benchmark metrics. Presentation, replay, and descriptive-analysis scripts receive separate hashes in final verification; they cannot alter this frozen prediction/evaluation path.',
        'model_files_sha256': {p.as_posix(): file_sha256(p) for p in sorted((OUT / 'models').glob('*.joblib'))},
        'inherited_sha256': inherited,
        'input_sha256': {p.as_posix(): file_sha256(p) for p in inputs},
        'training': {name: {key: value for key, value in entry.items() if key != 'training_code_sha256'}
                     for name, entry in training.items()},
        'split': split, 'benchmark_role': protocol['benchmark_role'],
        'new_pristine_final_test': False, 'new_benchmark_predictions_opened': False,
    }
    write_json(OUT / 'selection_lock.json', lock)
    print(f'Frozen V3 strategy: {selected}; benchmark remains unopened in this run.', flush=True)


def _segment_values(df):
    values = {name: df[name].fillna('__MISSING__').astype(str)
              for name in ['home_ownership', 'purpose', 'addr_state'] if name in df}
    if 'fico_range_low' in df:
        values['FICO'] = pd.cut(pd.to_numeric(df.fico_range_low, errors='coerce'),
                               [0, 660, 680, 700, 720, 760, np.inf], right=False).astype(str)
    if 'annual_inc' in df:
        values['annual_income'] = pd.cut(pd.to_numeric(df.annual_inc, errors='coerce'),
                                        [-np.inf, 40000, 60000, 80000, 120000, np.inf], right=False).astype(str)
    return values


def benchmark(protocol, OUT, V2, PROTOCOL):
    """Evaluate frozen choices on the exact prior cohorts; never select here."""
    OUT, V2, PROTOCOL = Path(OUT), Path(V2), Path(PROTOCOL)
    lock = _verify_lock(OUT, V2, PROTOCOL)
    receipt_path = OUT / 'benchmark_receipt.json'
    if receipt_path.exists() and read_json(receipt_path)['state'] == 'complete':
        receipt = read_json(receipt_path)
        assert receipt['lock_sha256'] == file_sha256(OUT / 'selection_lock.json')
        assert receipt['metrics_sha256'] == file_sha256(OUT / 'benchmark_metrics.json')
        print('V3 reused benchmark already complete and verified.', flush=True)
        return
    previous_receipt = read_json(receipt_path) if receipt_path.exists() else {}
    if previous_receipt:
        assert previous_receipt['lock_sha256'] == file_sha256(OUT / 'selection_lock.json'), 'Cannot resume a benchmark with a different frozen selection'
    started_utc = previous_receipt.get('started_utc', utc_now())
    write_json(receipt_path, {
        'state': 'in_progress', 'started_utc': started_utc, 'role': protocol['benchmark_role'],
        'lock_sha256': file_sha256(OUT / 'selection_lock.json'), 'selected': lock['selected'],
        'pristine_final_test': False,
    })
    bundle = joblib.load(OUT / 'models' / 'frozen_bundle.joblib')
    old_metrics = pd.DataFrame(read_json(V2 / 'benchmark_metrics.json'))
    rows, periods, errors, cohorts, uncertainty = [], [], [], {}, {}
    for year in [2015, 2016]:
        df = load_partition(f'holdout_{year}')
        df = df[df.term.str.strip().eq('36 months')].copy()
        old = pd.read_parquet(V2 / 'predictions' / f'benchmark_{year}_36.parquet')
        assert len(df) == {2015: 258856, 2016: 212482}[year]
        assert df.id.astype(str).tolist() == old.id.astype(str).tolist(), 'Reference IDs or order changed'
        assert np.array_equal(df.target.to_numpy(), old.target.to_numpy()), 'Reference outcomes changed'
        assert np.array_equal(df.issue_date.to_numpy(), old.issue_date.to_numpy()), 'Reference dates changed'
        assert not df.id.duplicated().any()
        cohorts[str(year)] = {'n': len(df), 'ids_sha256': ids_sha256(df.id),
                              'same_ids_targets_order_as_v2': True,
                              'reference_predictions_sha256': file_sha256(V2 / 'predictions' / f'benchmark_{year}_36.parquet')}
        scores, probabilities, thresholds = {}, {}, {}
        previous_row = old_metrics[(old_metrics.cohort == year) & old_metrics.model.eq('catboost') & old_metrics.scope.eq('36_month')].iloc[0]
        probabilities['v2_frozen'] = old.catboost.to_numpy()
        scores['v2_frozen'] = logit(np.clip(probabilities['v2_frozen'], np.finfo(float).eps, 1 - np.finfo(float).eps))
        thresholds['v2_frozen'] = float(previous_row['threshold'])
        previous_ap = average_precision_score(df.target, probabilities['v2_frozen'])
        assert np.isclose(previous_ap, previous_row['AP'], rtol=0, atol=1e-12)
        assert np.isclose(previous_ap, previous_row['raw_AP'], rtol=0, atol=1e-12), 'V2 rank preservation assumption failed'
        if 'original_selected' in old:
            probabilities['original_ensemble'] = old.original_selected.to_numpy()
            scores['original_ensemble'] = logit(np.clip(probabilities['original_ensemble'], np.finfo(float).eps, 1 - np.finfo(float).eps))
            thresholds['original_ensemble'] = read_json(BASE / 'selection_lock.json')['thresholds']['selected']['f1']
        for name, model in bundle['models'].items():
            scores[name] = model.predict_score(df)
            probabilities[name] = bundle['calibrators'][name].predict(scores[name])
            thresholds[name] = bundle['thresholds'][name]['f1']
        selected = bundle['selected']
        if selected not in bundle['models']:
            scores[selected] = ensemble_score(scores, bundle['members'], bundle['mappings'])
            probabilities[selected] = bundle['calibrators'][selected].predict(scores[selected])
            thresholds[selected] = bundle['thresholds'][selected]['f1']
        predictions = df[['id', 'target', 'issue_date']].copy()
        for name, score in scores.items():
            p = probabilities[name]
            result = {'cohort': year, 'scope': '36_month', 'model': name,
                      **_metric_row(df.target, score, p, thresholds[name]),
                      'selected_before_new_predictions': name == selected}
            result['delta_AP_vs_v2'] = result['raw_AP'] - previous_ap
            rows.append(result)
            predictions[name] = p
            predictions[f'score__{name}'] = score
            print(f'Reused benchmark {year} / {name}: AP={result["raw_AP"]:.6f}; delta V2={result["delta_AP_vs_v2"]:+.6f}', flush=True)
        quarter = df.issue_date.dt.to_period('Q').astype(str)
        for period in sorted(quarter.unique()):
            mask = quarter.eq(period).to_numpy()
            y = df.target.to_numpy()[mask]
            baseline_ap = average_precision_score(y, scores['v2_frozen'][mask])
            for name, score in scores.items():
                record = _metric_row(y, score[mask], probabilities[name][mask], thresholds[name])
                periods.append({'cohort': year, 'period': period, 'model': name,
                                **{key: value for key, value in record.items() if key != 'confusion_matrix'},
                                'delta_AP_vs_v2': record['raw_AP'] - baseline_ap})
        for dimension, labels in _segment_values(df).items():
            for segment in sorted(labels.unique()):
                mask = labels.eq(segment).to_numpy()
                if mask.sum() < 100:
                    continue
                for name in [selected, 'v2_frozen']:
                    record = _metric_row(df.target.to_numpy()[mask], scores[name][mask], probabilities[name][mask], thresholds[name])
                    errors.append({'cohort': year, 'dimension': dimension, 'segment': segment, 'model': name,
                                   **{key: value for key, value in record.items() if key != 'confusion_matrix'}})
        predictions.to_parquet(OUT / 'predictions' / f'benchmark_{year}_36.parquet', index=False)
        if year == 2015:
            # With observed numeric ties, AP must still use the original raw ranking.
            selected_row = next(r for r in reversed(rows) if r['cohort'] == year and r['model'] == selected)
            ap_preserved = bool(selected_row['calibration_AP_preserved'])
            primary_p = probabilities[selected] if ap_preserved else scores[selected]
            interval = paired_bootstrap(
                df.target.to_numpy(), primary_p, probabilities['v2_frozen'],
                df.issue_date.dt.strftime('%Y-%m').to_numpy(),
                reps=protocol.get('bootstrap_repetitions', 300), seed=protocol.get('seed', 42),
            )
            if not ap_preserved:
                # Raw-margin Brier is not a probability score; omit it rather than mislabel it.
                for method in interval.values():
                    method.pop('Brier', None)
            uncertainty['2015_36_vs_v2_frozen'] = {
                **interval, 'selected': selected, 'point_delta_AP': selected_row['delta_AP_vs_v2'],
                'raw_AP_primary': True, 'calibration_AP_preserved': ap_preserved,
                'interpretation': 'Conditional paired sampling uncertainty on a reused benchmark; does not restore independence.',
            }
        del df, old, predictions, scores, probabilities
        gc.collect()
    write_json(OUT / 'benchmark_metrics.json', rows)
    pd.DataFrame(rows).drop(columns='confusion_matrix').to_csv(OUT / 'benchmark_metrics.csv', index=False)
    pd.DataFrame(periods).to_csv(OUT / 'benchmark_periods.csv', index=False)
    pd.DataFrame(errors).to_csv(OUT / 'benchmark_segment_errors.csv', index=False)
    write_json(OUT / 'benchmark_cohorts.json', cohorts)
    write_json(OUT / 'benchmark_uncertainty.json', uncertainty)
    _verify_lock(OUT, V2, PROTOCOL)
    write_json(receipt_path, {
        'state': 'complete', 'started_utc': started_utc, 'completed_utc': utc_now(),
        'role': protocol['benchmark_role'], 'lock_sha256': file_sha256(OUT / 'selection_lock.json'),
        'selected': bundle['selected'], 'selected_model_unchanged': True,
        'metrics_sha256': file_sha256(OUT / 'benchmark_metrics.json'), 'cohorts': cohorts,
        'pristine_final_test': False,
        'outputs_sha256': {name: file_sha256(OUT / name) for name in [
            'benchmark_metrics.csv', 'benchmark_periods.csv', 'benchmark_segment_errors.csv',
            'benchmark_cohorts.json', 'benchmark_uncertainty.json']},
    })
