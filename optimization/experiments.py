"""Joint predictive search; development selection precedes reused benchmarks."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from optimization.data import load_partition, prepare_context, train_rows, valid_rows, recency_weights
from optimization.models import V3Model
from optimization.ensemble import PositiveSigmoid, ensemble_score
from research.models import ResearchModel
from src.prepare import file_sha256, ids_sha256
from src.evaluate import metrics, choose_thresholds, paired_bootstrap
from src.utils import write_json, utc_now

OUT = Path(os.environ.get('LENDING_OPTIMIZATION_OUTPUT', 'outputs/research_v3'))
V2 = Path(os.environ.get('LENDING_V2_OUTPUT', 'outputs/research_v2'))
PROTOCOL = Path('experiments/protocol_v3.json')
FOLLOWUP = Path('experiments/development_followup_v3.json')
PREPARATION_FOLLOWUP = Path('experiments/preparation_followup_v3.json')
TREE = {'lightgbm', 'catboost', 'xgboost'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def make_spec(family, params, **kwargs):
    return dict(family=family, params=dict(params), feature_mode='baseline', preparation='native',
                start='2012-01-01', half_life=0, seed=42) | kwargs


def fit_model(spec, train, validation=None):
    kwargs = {k: spec[k] for k in ['family', 'params', 'feature_mode', 'preparation', 'seed']}
    if spec.get('engine') == 'v2':
        model = ResearchModel(spec['family'], spec['params'],
                              {'mode': 'stable', 'start': spec['start'], 'term': '36 months'}, seed=spec['seed'])
        return model.fit(train, train.target, validation=validation)
    model = V3Model(**kwargs)
    result = model.fit(train, train.target, validation=validation,
                       sample_weight=recency_weights(train, train.issue_date.max(), spec['half_life']))
    if result.convergence_warnings:
        raise RuntimeError('; '.join(result.convergence_warnings))
    return result


def cv_score(df, spec, protocol, label, early_stop=True, trial=None):
    records, predictions = [], []
    for i, fold in enumerate(protocol['cv']):
        train = train_rows(df, fold['train_end'], spec['start'], protocol['search_train_cap'])
        valid = valid_rows(df, fold['validation_start'], fold['validation_end'], protocol['search_validation_cap'])
        assert train.issue_date.max() < valid.issue_date.min()
        assert set(train.id).isdisjoint(valid.id)
        started = time.time()
        model = fit_model(spec, train, (valid, valid.target) if early_stop and spec['family'] in TREE else None)
        scores = model.predict_score(valid)
        record = {'fold': i, 'AP': float(average_precision_score(valid.target, scores)),
                  'ROC_AUC': float(roc_auc_score(valid.target, scores)), 'train_n': len(train),
                  'validation_n': len(valid), 'train_ids_sha256': ids_sha256(train.id),
                  'validation_ids_sha256': ids_sha256(valid.id), 'best_iteration': model.best_iteration,
                  'features': len(model.features.columns), 'seconds': round(time.time()-started, 2)}
        records.append(record)
        prediction = valid[['id', 'target', 'issue_date']].copy()
        prediction['fold'], prediction['score'] = i, scores
        predictions.append(prediction)
        if trial:
            trial.set_user_attr('folds', records)
            trial.report(float(np.mean([r['AP'] for r in records])), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        del model, train, valid
        gc.collect()
    pd.concat(predictions, ignore_index=True).to_parquet(OUT/'predictions'/f'cv_{label}.parquet', index=False)
    return {'spec': spec, 'CV_AP': float(np.mean([r['AP'] for r in records])), 'folds': records,
            'prediction_label': label}


def ablations(df, protocol):
    path = OUT/'ablations.json'
    results = read_json(path) if path.exists() else []
    params = read_json(V2/'best_lightgbm.json')['params']
    variants = {'baseline': {}, 'financial': {'feature_mode': 'financial'},
                'context': {'feature_mode': 'context'}, 'combined': {'feature_mode': 'combined'},
                'clipping': {'preparation': 'clip'}, 'from_2013': {'start': '2013-01-01'},
                'recency12': {'half_life': 12}, 'recency24': {'half_life': 24}}
    for name, change in variants.items():
        if any(r['name'] == name for r in results):
            continue
        result = {'name': name, **cv_score(df, make_spec('lightgbm', params, **change), protocol,
                                         f'ablation_{name}', early_stop=False)}
        results.append(result); write_json(path, results)
        print(f"Ablation {name}: AP={result['CV_AP']:.6f}", flush=True)
    baseline = results[0]
    pd.DataFrame([{'name': r['name'], 'CV_AP': r['CV_AP'], 'delta_AP': r['CV_AP']-baseline['CV_AP'],
                  **{f'fold{i+1}_AP': f['AP'] for i, f in enumerate(r['folds'])}} for r in results]).to_csv(OUT/'ablations.csv', index=False)


def suggest(trial, family):
    feature_mode = trial.suggest_categorical('feature_mode', ['baseline', 'financial', 'context', 'combined'])
    prep = trial.suggest_categorical('preparation', ['native', 'clip'] if family in TREE else ['native', 'log_clip'])
    start = trial.suggest_categorical('start', ['2012-01-01', '2013-01-01'])
    half = trial.suggest_categorical('half_life', [0, 12, 24])
    if family == 'catboost':
        params = dict(iterations=1400, learning_rate=trial.suggest_float('learning_rate', .02, .07, log=True),
                      depth=trial.suggest_int('depth', 5, 7), l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 10, 100, log=True),
                      random_strength=trial.suggest_float('random_strength', .3, 3),
                      bagging_temperature=trial.suggest_float('bagging_temperature', .1, 1),
                      rsm=trial.suggest_float('rsm', .6, 1), max_ctr_complexity=trial.suggest_categorical('max_ctr_complexity', [1, 2]),
                      auto_class_weights=None)
    elif family == 'lightgbm':
        params = dict(n_estimators=1800, learning_rate=trial.suggest_float('learning_rate', .015, .07, log=True),
                      num_leaves=trial.suggest_int('num_leaves', 10, 50), min_child_samples=trial.suggest_int('min_child_samples', 80, 700, log=True),
                      reg_lambda=trial.suggest_float('reg_lambda', .1, 30, log=True), reg_alpha=trial.suggest_float('reg_alpha', .0001, 3, log=True),
                      colsample_bytree=trial.suggest_float('colsample_bytree', .6, 1), subsample=trial.suggest_float('subsample', .65, 1),
                      subsample_freq=1, scale_pos_weight=trial.suggest_categorical('scale_pos_weight', [1., 1.5, 2.]))
    elif family == 'xgboost':
        params = dict(n_estimators=1600, learning_rate=trial.suggest_float('learning_rate', .02, .09, log=True),
                      max_depth=trial.suggest_int('max_depth', 2, 5), min_child_weight=trial.suggest_float('min_child_weight', 5, 100, log=True),
                      reg_lambda=trial.suggest_float('reg_lambda', 1, 80, log=True), reg_alpha=trial.suggest_float('reg_alpha', .001, 10, log=True),
                      gamma=trial.suggest_float('gamma', 0, 2), subsample=trial.suggest_float('subsample', .65, 1),
                      colsample_bytree=trial.suggest_float('colsample_bytree', .6, 1), scale_pos_weight=trial.suggest_categorical('scale_pos_weight', [1., 1.5, 2.]))
    else:
        params = dict(C=trial.suggest_float('C', .0003, 3, log=True), class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']))
        if family == 'svm_nystroem':
            params.update(gamma=trial.suggest_float('gamma', .0003, .02, log=True),
                          n_components=trial.suggest_categorical('n_components', [300, 600]))
    return make_spec(family, params, feature_mode=feature_mode, preparation=prep, start=start, half_life=half)


def search(df, protocol):
    design_path = OUT/'search_design.json'
    dependencies = {f'optimization/{name}.py': file_sha256(f'optimization/{name}.py')
                    for name in ['data', 'features', 'models', 'ensemble']}
    design = {'protocol_sha256': file_sha256(PROTOCOL), 'dependencies_sha256': dependencies,
              'train_ids_sha256': ids_sha256(df.id), 'planned_trials': protocol['trials'],
              'benchmark_opened': False}
    if design_path.exists():
        prior = read_json(design_path)
        assert all(prior[k] == v for k, v in design.items()), 'Search inputs or model contract changed'
    else:
        write_json(design_path, design | {'started_utc': utc_now(), 'runner_sha256_at_start': file_sha256(__file__)})
    old = read_json(V2/'selection.json')['specs']
    for family, count in protocol['trials'].items():
        storage = f"sqlite:///{(OUT/'studies'/'joint_search.db').as_posix()}"
        study = optuna.create_study(study_name=family, storage=storage, load_if_exists=True, direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
        for interrupted in study.get_trials(states=[optuna.trial.TrialState.RUNNING]):
            study.tell(interrupted.number, state=optuna.trial.TrialState.FAIL)
        if not study.trials:
            warm = {k: v for k, v in old[family]['params'].items() if k not in ['iterations', 'n_estimators', 'auto_class_weights']}
            warm.update(feature_mode='baseline', preparation='native', start='2012-01-01', half_life=0)
            if family == 'catboost':
                warm['max_ctr_complexity'] = 1
            study.enqueue_trial(warm)
            study.enqueue_trial(warm | {'feature_mode': 'combined'})
            study.enqueue_trial(warm | {'feature_mode': 'financial', 'half_life': 12})
        def objective(trial):
            spec = suggest(trial, family)
            trial.set_user_attr('spec', spec)
            result = cv_score(df, spec, protocol, f'{family}_{trial.number}', trial=trial)
            trial.set_user_attr('result', result)
            return result['CV_AP']
        def callback(study, trial):
            study.trials_dataframe().to_csv(OUT/f'search_{family}.csv', index=False)
            print(f"Search {family} {trial.number}: {trial.state.name} AP={trial.value}", flush=True)
        remaining = remaining_trials(study, count)
        study.optimize(objective, n_trials=remaining, callbacks=[callback], catch=(RuntimeError, ValueError), gc_after_trial=True)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError(f'No completed configuration for {family}')
        top = sorted(complete, key=lambda t: t.value, reverse=True)[:2]
        write_json(OUT/f'best_{family}.json', [t.user_attrs['result'] | {'trial_number': t.number} for t in top])


def remaining_trials(study, count):
    completed_attempts = sum(t.state in [optuna.trial.TrialState.COMPLETE,
                                        optuna.trial.TrialState.PRUNED,
                                        optuna.trial.TrialState.FAIL] for t in study.trials)
    return max(0, count-completed_attempts)


def promoted_spec(result):
    spec = dict(result['spec']); spec['params'] = dict(spec['params'])
    if spec['family'] in TREE:
        key = 'iterations' if spec['family'] == 'catboost' else 'n_estimators'
        n = round(np.mean([r['best_iteration'] for r in result['folds']]) * 1.15)
        spec['params'][key] = min(spec['params'][key], max(1, n))
    return spec


def preparation_followup(df, protocol):
    path = OUT/'preparation_followup.json'
    results = read_json(path) if path.exists() else []
    registered = read_json(PREPARATION_FOLLOWUP)
    for family in registered['families']:
        if any(r['family'] == family for r in results):
            continue
        baseline = read_json(OUT/f'best_{family}.json')[0]
        spec = dict(baseline['spec']) | {'preparation': 'log_clip'}
        print(f'Preparation control {family}: log_clip on identical CV rows', flush=True)
        try:
            candidate = cv_score(df, spec, protocol, f'preparation_{family}', early_stop=False)
            row = {'family': family, 'status': 'COMPLETE', 'baseline_CV_AP': baseline['CV_AP'],
                   'candidate': candidate, 'delta_AP': candidate['CV_AP']-baseline['CV_AP'],
                   'promoted': candidate['CV_AP'] > baseline['CV_AP']}
        except (RuntimeError, ValueError) as error:
            row = {'family': family, 'status': 'FAIL', 'baseline_CV_AP': baseline['CV_AP'],
                   'candidate_spec': spec, 'error': str(error), 'promoted': False}
        results.append(row); write_json(path, results)
    pd.DataFrame([{'family': r['family'], 'status': r['status'], 'baseline_CV_AP': r['baseline_CV_AP'],
                  'log_clip_CV_AP': r.get('candidate', {}).get('CV_AP'), 'delta_AP': r.get('delta_AP'),
                  'promoted': r['promoted']} for r in results]).to_csv(OUT/'preparation_followup.csv', index=False)


def select(df, protocol):
    if (OUT/'selection.json').exists():
        return
    assert (OUT/'preparation_followup.json').exists(), 'Run the registered preparation controls before selection'
    old = read_json(V2/'selection.json')['specs']['catboost']
    specs = {'v2_catboost': make_spec('catboost', old['params'], engine='v2')}
    extra_blend_names = []
    if FOLLOWUP.exists():
        amendment = read_json(FOLLOWUP)
        assert amendment['new_benchmark_predictions_opened'] is False
        assert not (OUT/'benchmark_receipt.json').exists()
        for control in amendment['controlled_models']:
            specs[control['name']] = make_spec('catboost', old['params'], feature_mode=control['feature_mode'])
            extra_blend_names.append(control['name'])
    source_results = {}
    family_best = {}
    for family in protocol['trials']:
        best = read_json(OUT/f'best_{family}.json')
        if (OUT/'preparation_followup.json').exists():
            extra = [r['candidate'] for r in read_json(OUT/'preparation_followup.json')
                     if r['family'] == family and r['status'] == 'COMPLETE']
            best = sorted(best+extra, key=lambda r: r['CV_AP'], reverse=True)
        for index, result in enumerate(best[:2] if family in ['catboost', 'lightgbm'] else best[:1]):
            name = f'{family}_joint' + ('_runnerup' if index else '')
            specs[name] = promoted_spec(result); source_results[name] = result
            if index == 0:
                family_best[family] = name
    ablation = max([r for r in read_json(OUT/'ablations.json') if r['name'] in ['financial', 'context', 'combined']], key=lambda r: r['CV_AP'])
    specs['lightgbm_feature_control'] = ablation['spec']; source_results['lightgbm_feature_control'] = ablation
    specs['catboost_seed2026'] = specs[family_best['catboost']] | {'seed': 2026}
    valid = valid_rows(df, protocol['selection_start'], protocol['selection_end'])
    scores, mappings, rows = {}, {}, []
    for name, spec in specs.items():
        path = OUT/'models'/f'candidate_{name}.joblib'
        train = train_rows(df, protocol['candidate_training_end'], spec['start'])
        started = time.time()
        print(f"Q3 fitting {name}: n={len(train)}, features={spec['feature_mode']}, preparation={spec['preparation']}", flush=True)
        if name == 'v2_catboost':
            model = joblib.load(V2/'models'/'candidate_catboost.joblib')
        else:
            model = joblib.load(path) if path.exists() else fit_model(spec, train)
            if not path.exists():
                joblib.dump(model, path, compress=3)
        scores[name] = model.predict_score(valid)
        if name == 'v2_catboost':
            recorded = pd.read_parquet(V2/'predictions'/'selection.parquet').set_index('id')
            recorded.index = recorded.index.astype(str)
            reference = recorded.loc[valid.id]
            assert np.array_equal(reference.target.to_numpy(), valid.target.to_numpy())
            np.testing.assert_allclose(scores[name], reference.catboost.to_numpy(), atol=1e-12, rtol=0)
        if spec['family'].startswith('svm'):
            cv = pd.read_parquet(OUT/'predictions'/f"cv_{source_results[name]['prediction_label']}.parquet")
            mappings[name] = PositiveSigmoid().fit(cv.score, cv.target)
        rows.append({'model': name, 'type': 'single', 'AP': float(average_precision_score(valid.target, scores[name])),
                     'ROC_AUC': float(roc_auc_score(valid.target, scores[name])), 'train_n': len(train),
                     'validation_n': len(valid), 'train_ids_sha256': ids_sha256(train.id),
                     'validation_ids_sha256': ids_sha256(valid.id), 'seconds': round(time.time()-started, 2)})
        print(f"Q3 {name}: AP={rows[-1]['AP']:.6f}, ntrain={len(train)}", flush=True)
        del model; gc.collect()
    ensembles = {}
    for name in list(family_best.values())+extra_blend_names:
        for weight in [.25, .5, .75]:
            ensembles[f'blend_v2_{name}_{int(weight*100)}'] = {'v2_catboost': weight, name: 1-weight}
    ensembles['catboost_two_seeds'] = {family_best['catboost']: .5, 'catboost_seed2026': .5}
    ensembles['three_tree_families'] = {family_best[f]: 1/3 for f in ['catboost', 'lightgbm', 'xgboost']}
    for name, members in ensembles.items():
        scores[name] = ensemble_score(scores, members, mappings)
        rows.append({'model': name, 'type': 'ensemble', 'AP': float(average_precision_score(valid.target, scores[name])),
                     'ROC_AUC': float(roc_auc_score(valid.target, scores[name])), 'validation_n': len(valid),
                     'validation_ids_sha256': ids_sha256(valid.id)})
    table = pd.DataFrame(rows).sort_values('AP', ascending=False)
    table.to_csv(OUT/'selection_leaderboard.csv', index=False)
    predictions = valid[['id', 'target', 'issue_date']].copy()
    for name, score in scores.items():
        predictions[name] = score
    predictions.to_parquet(OUT/'predictions'/'selection.parquet', index=False)
    winner = table.iloc[0]['model']
    members = ensembles.get(winner, {winner: 1.})
    finalists = sorted(set(family_best.values()) | set(members) | {'v2_catboost'})
    joblib.dump(mappings, OUT/'models'/'ensemble_mappings.joblib')
    write_json(OUT/'ensemble_mappings.json', {name: obj.to_dict() for name, obj in mappings.items()})
    write_json(OUT/'selection.json', {'selected': winner, 'members': members, 'specs': specs,
               'family_best': family_best, 'finalists': finalists, 'ensembles': ensembles,
               'selection_AP': float(table.iloc[0]['AP']), 'selected_utc': utc_now(),
               'protocol_sha256': file_sha256(PROTOCOL), 'previous_benchmark_results_known': True,
               'development_followup_sha256': file_sha256(FOLLOWUP) if FOLLOWUP.exists() else None,
               'preparation_followup_sha256': file_sha256(PREPARATION_FOLLOWUP),
               'additional_development_candidates': 8 if extra_blend_names else 0,
               'new_benchmark_predictions_opened': False})
    print(f'Selected on Q3: {winner}, AP={table.iloc[0]["AP"]:.6f}', flush=True)


def stability(df, protocol):
    """Fixed selected strategy on earlier vintages; diagnostic, never reselection."""
    path = OUT/'temporal_stability.csv'
    if path.exists():
        return
    selection = read_json(OUT/'selection.json')
    names = sorted(set(selection['members']) | {'v2_catboost'})
    rows = []
    if len(selection['members']) > 1 and any(selection['specs'][n]['family'].startswith('svm') for n in names):
        write_json(OUT/'temporal_stability_limitation.json', {
            'reason': 'Selected SVM blend mapping learned through June2014; cannot reuse it in earlier vintages without future-label leakage.',
            'alternative': 'Full paired quarter-level benchmark stability will be reported; no early-fold ensemble claim.'})
        return
    for i, fold in enumerate(protocol['cv']):
        valid = valid_rows(df, fold['validation_start'], fold['validation_end'], protocol['search_validation_cap'])
        scores = {}
        for name in names:
            spec = selection['specs'][name]
            train = train_rows(df, fold['train_end'], spec['start'], protocol['search_train_cap'])
            predpath = OUT/'predictions'/f'stability_{i}_{name}.parquet'
            if predpath.exists():
                cached = pd.read_parquet(predpath)
                assert cached.id.tolist() == valid.id.tolist()
                scores[name] = cached.score.to_numpy()
            else:
                print(f"Temporal diagnostic fold{i+1}: fitting {name}, n={len(train)}", flush=True)
                model = fit_model(spec, train)
                scores[name] = model.predict_score(valid)
                pd.DataFrame({'id': valid.id, 'score': scores[name]}).to_parquet(predpath, index=False)
                del model; gc.collect()
        members = selection['members']
        selected = next(iter(scores[n] for n in members)) if len(members) == 1 else ensemble_score(scores, members, {})
        baseap = float(average_precision_score(valid.target, scores['v2_catboost']))
        ap = float(average_precision_score(valid.target, selected))
        rows.append({'fold': i+1, 'period': fold['validation_start']+' / '+fold['validation_end'],
                     'model': selection['selected'], 'n': len(valid), 'AP': ap,
                     'v2_AP': baseap, 'delta_AP_vs_v2': ap-baseap, 'validation_ids_sha256': ids_sha256(valid.id),
                     'role': 'Reused development folds; selected after seeing Q3; diagnostic only'})
        print(f"Fixed strategy temporal fold{i+1}: deltaAP={ap-baseap:+.6f}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'ablations', 'search', 'preparation', 'select', 'stability', 'freeze', 'benchmark', 'all'], default='all')
    args = parser.parse_args()
    for sub in ['', 'predictions', 'models', 'studies', 'logs']:
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    protocol = read_json(PROTOCOL)
    if args.stage == 'prepare':
        write_json(OUT/'context_manifest.json', prepare_context()); return
    if args.stage != 'benchmark':
        assert not (OUT/'benchmark_receipt.json').exists(), 'Cannot tune after opening this benchmark run'
    df = None if args.stage == 'benchmark' else load_partition()
    with threadpool_limits(limits=3):
        for name, fn in [('ablations', ablations), ('search', search), ('preparation', preparation_followup), ('select', select), ('stability', stability)]:
            if args.stage in [name, 'all']:
                fn(df, protocol)
        if args.stage in ['freeze', 'all']:
            from optimization.evaluation import freeze
            freeze(df, protocol, OUT, V2, PROTOCOL)
        if args.stage in ['benchmark', 'all']:
            from optimization.evaluation import benchmark
            benchmark(protocol, OUT, V2, PROTOCOL)


if __name__ == '__main__':
    main()
