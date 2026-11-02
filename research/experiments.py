"""Controlled V2 experiments; 2015/2016 are explicitly reused benchmarks."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '3')
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from research.models import ResearchModel, ScoreCalibrator, RECIPES
from src.train import sample_development
from src.evaluate import metrics, choose_thresholds, paired_bootstrap
from src.prepare import file_sha256, ids_sha256
from src.utils import write_json, utc_now

OUT = Path(os.environ.get('LENDING_RESEARCH_OUTPUT','outputs/research_v2'))
BASE = Path(os.environ.get('LENDING_BASE_OUTPUT','outputs'))
PROTOCOL = Path('experiments/protocol_v2.json')
FAMILIES = ['lightgbm', 'catboost', 'svm_linear', 'svm_rbf', 'svm_nystroem']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def train_rows(df, recipe, end, cap=None):
    mask = df.issue_date.le(end)
    if recipe['start']:
        mask &= df.issue_date.ge(recipe['start'])
    if recipe['term'] != 'all':
        mask &= df.term.str.strip().eq(recipe['term'])
    result = df.loc[mask].copy()
    return sample_development(result, cap) if cap else result


def valid_rows(df, start, end, cap=None):
    result = df[df.issue_date.between(start, end)&df.term.str.strip().eq('36 months')].copy()
    return sample_development(result, cap) if cap else result


def cv_score(df, recipe, family, params, protocol, trial=None):
    scores, aucs, iterations, seconds, records = [], [], [], [], []
    for i, fold in enumerate(protocol['cv']):
        cap = protocol['exact_rbf_train_cap'] if family == 'svm_rbf' else protocol['search_train_cap']
        vcap = protocol['exact_rbf_validation_cap'] if family == 'svm_rbf' else protocol['search_validation_cap']
        train = train_rows(df, recipe, fold['train_end'], cap)
        valid = valid_rows(df, fold['validation_start'], fold['validation_end'], vcap)
        assert train.issue_date.max() < valid.issue_date.min()
        assert set(train.id).isdisjoint(valid.id)
        t0 = time.time()
        model = ResearchModel(family, params, recipe).fit(train, train.target,
                    validation=(valid, valid.target) if family in ['lightgbm', 'catboost'] else None)
        prediction = model.predict_score(valid)
        if model.convergence_warnings:
            raise RuntimeError('SVM/linear optimization did not converge: '+'; '.join(model.convergence_warnings))
        scores.append(float(average_precision_score(valid.target, prediction)))
        aucs.append(float(roc_auc_score(valid.target, prediction)))
        iterations.append(model.best_iteration)
        seconds.append(round(time.time()-t0, 2))
        records.append({'fold': i, 'train_n': len(train), 'validation_n': len(valid),
                        'train_ids_sha256': ids_sha256(train.id), 'validation_ids_sha256': ids_sha256(valid.id),
                        'feature_count': len(model.features.columns), 'AP': scores[-1], 'ROC_AUC': aucs[-1]})
        if trial is not None:
            trial.set_user_attr('fold_AP', scores)
            trial.set_user_attr('fold_ROC_AUC', aucs)
            trial.set_user_attr('fold_iterations', iterations)
            trial.set_user_attr('fold_seconds', seconds)
            trial.set_user_attr('fold_records', records)
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        del model, train, valid
        gc.collect()
    return {'CV_AP': float(np.mean(scores)), 'fold_AP': scores, 'fold_ROC_AUC': aucs,
            'fold_iterations': iterations, 'fold_seconds': seconds, 'fold_records': records}


def screen(df, protocol):
    path = OUT/'recipe_screen.json'
    results = read_json(path) if path.exists() else []
    params = read_json(BASE/'best_lightgbm.json')['params']
    for recipe in RECIPES:
        if any(r['recipe']['name']==recipe['name'] for r in results):
            continue
        row = {'recipe': recipe, **cv_score(df, recipe, 'lightgbm', params, protocol)}
        results.append(row)
        write_json(path, results)
        print(f"Recipe {recipe['name']}: CV AP={row['CV_AP']:.5f}, folds={row['fold_AP']}", flush=True)
    winner = max(results, key=lambda r:r['CV_AP'])
    write_json(OUT/'selected_recipe.json', {'recipe': winner['recipe'], 'CV_AP': winner['CV_AP'],
                 'selection_source': 'development CV only', 'benchmark_reopened': False})
    pd.DataFrame([{'recipe': r['recipe']['name'], 'CV_AP': r['CV_AP'], 'fold1_AP': r['fold_AP'][0], 'fold2_AP': r['fold_AP'][1]} for r in results]).to_csv(OUT/'recipe_screen.csv', index=False)


def suggest(trial, family):
    if family == 'lightgbm':
        return {'n_estimators': 1600, 'learning_rate': trial.suggest_float('learning_rate', .015, .09, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 7, 95),
                'min_child_samples': trial.suggest_int('min_child_samples', 50, 600, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', .2, 60, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', .6, 1),
                'subsample': trial.suggest_float('subsample', .65, 1), 'subsample_freq': 1,
                'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1., 2.])}
    if family == 'catboost':
        return {'iterations': 1200, 'learning_rate': trial.suggest_float('learning_rate', .02, .09, log=True),
                'depth': trial.suggest_int('depth', 4, 7),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 60, log=True),
                'random_strength': trial.suggest_float('random_strength', .1, 4, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),
                'rsm': trial.suggest_float('rsm', .65, 1),
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None, 'Balanced'])}
    p = {'C': trial.suggest_float('C', .001, 10, log=True),
         'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])}
    if family in ['svm_rbf', 'svm_nystroem']:
        p['gamma'] = trial.suggest_float('gamma', .0005, .06, log=True)
    if family == 'svm_nystroem':
        p['n_components'] = trial.suggest_categorical('n_components', [200, 400, 600])
    return p


def search(df, protocol):
    recipe = read_json(OUT/'selected_recipe.json')['recipe']
    (OUT/'studies').mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in FAMILIES:
        study = optuna.create_study(direction='maximize', study_name='v2_'+family,
                 storage='sqlite:///'+(OUT/'studies'/'search.sqlite3').as_posix(), load_if_exists=True,
                 sampler=optuna.samplers.TPESampler(seed=42),
                 pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
        if len(study.trials)==0 and family in ['lightgbm', 'catboost']:
            # Include the previous best as an explicit warm-start comparator.
            p = read_json(BASE/f'best_{family}.json')['params']
            p.pop('n_estimators' if family=='lightgbm' else 'iterations', None)
            if family == 'catboost':
                p['l2_leaf_reg'] = float(np.clip(p['l2_leaf_reg'], 3, 60))
            study.enqueue_trial(p)
        if len(study.trials)==0 and family in ['svm_rbf','svm_nystroem']:
            probes=([{'C':10.0,'gamma':.0005,'class_weight':None},
                     {'C':10.0,'gamma':.005,'class_weight':None}] if family=='svm_rbf' else
                    [{'C':1.0,'gamma':.001,'n_components':600,'class_weight':None},
                     {'C':10.0,'gamma':.005,'n_components':600,'class_weight':None}])
            for candidate in probes:
                study.enqueue_trial(candidate,user_attrs={'manual_probe_reason':'Recorded low-gamma/high-C boundary candidate'},skip_if_exists=True)
        def objective(trial):
            params = suggest(trial, family)
            trial.set_user_attr('model_params', params)
            return cv_score(df, recipe, family, params, protocol, trial)['CV_AP']
        def counted_trials(study):
            return sum(t.state!=optuna.trial.TrialState.WAITING and not t.user_attrs.get('administrative_interruption',False) for t in study.trials)
        def callback(study, trial):
            study.trials_dataframe().to_csv(OUT/f'search_{family}.csv', index=False)
            best = max((t.value for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE), default=float('nan'))
            print(f"V2 {family} {counted_trials(study)}/{protocol['trials'][family]} {trial.state.name} AP={trial.value}; best={best:.5f}", flush=True)
        study.optimize(objective, n_trials=max(0, protocol['trials'][family]-counted_trials(study)), callbacks=[callback],
                       catch=(RuntimeError,), gc_after_trial=True)
        trial = study.best_trial
        params = dict(trial.user_attrs['model_params'])
        trees = [i for i in trial.user_attrs['fold_iterations'] if i is not None]
        if trees:
            key = 'iterations' if family=='catboost' else 'n_estimators'
            params[key] = min(params[key], max(50, int(np.median(trees)*1.15)))
        write_json(OUT/f'best_{family}.json', {'family': family, 'recipe': recipe, 'params': params,
                   'CV_AP': trial.value, 'trial_count': counted_trials(study),
                   'all_recorded_trials_including_administrative_interruptions':len(study.trials), **trial.user_attrs})


def fit_candidates(df, protocol):
    from research.repair_svm import repair
    repair(df,protocol)
    recipe = read_json(OUT/'selected_recipe.json')['recipe']
    valid = valid_rows(df, protocol['selection_start'], protocol['selection_end'])
    specs = {f: {'family':f, 'params':read_json(OUT/f'best_{f}.json')['params'], 'recipe': recipe} for f in FAMILIES}
    for family in ['lightgbm','catboost']:
        specs[family+'_previous_params'] = {'family':family,
            'params':read_json(BASE/f'best_{family}.json')['params'], 'recipe':recipe}
    for family in ['logistic', 'random_forest', 'xgboost']:
        specs[family] = {'family': family, 'params':read_json(BASE/f'best_{family}.json')['params'], 'recipe':recipe}
    complete_recipe={**recipe,'name':recipe['name']+'_from_2013','start':'2013-01-01'}
    # Availability audit: these bureau fields are ~52% absent in 2012 and complete in 2013.
    # Hold predictive parameters fixed; select on the same Q3 cohort, before the benchmark.
    for family in FAMILIES:
        params=(read_json(BASE/f'best_{family}.json')['params'] if family in ['lightgbm','catboost']
                else read_json(OUT/f'best_{family}.json')['params'])
        specs[family+'_from_2013']={'family':family,'params':params,'recipe':complete_recipe}
    audit_path=OUT/'availability_2013_cv.json'
    if not audit_path.exists():
        write_json(audit_path,{'recipe':complete_recipe,'fixed_previous_lightgbm_params':True,
                  **cv_score(df,complete_recipe,'lightgbm',read_json(BASE/'best_lightgbm.json')['params'],protocol)})
    rows = []
    (OUT/'models').mkdir(exist_ok=True)
    (OUT/'predictions').mkdir(exist_ok=True)
    predictions = valid[['id', 'target']].copy()
    for name, spec in specs.items():
        train = train_rows(df, spec['recipe'], protocol['candidate_training_end'], protocol['exact_rbf_train_cap'] if spec['family']=='svm_rbf' else None)
        path = OUT/'models'/f'candidate_{name}.joblib'
        start = time.time()
        model = joblib.load(path) if path.exists() else ResearchModel(**spec).fit(train, train.target)
        assert not model.convergence_warnings, model.convergence_warnings
        if not path.exists():
            joblib.dump(model, path, compress=3)
        score = model.predict_score(valid)
        predictions[name] = score
        row = {'name':name, 'AP': float(average_precision_score(valid.target, score)),
               'ROC_AUC':float(roc_auc_score(valid.target, score)), 'train_n':len(train), 'validation_n':len(valid),
               'fit_predict_seconds': round(time.time()-start, 2), 'n_features': len(model.features.columns)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/'selection_leaderboard.csv', index=False)
        print('V2 candidate:', row, flush=True)
        del model, train; gc.collect()
    winner = max(rows, key=lambda r:r['AP'])
    predictions.to_parquet(OUT/'predictions'/'selection.parquet', index=False)
    write_json(OUT/'selection.json', {'selected':winner['name'], 'specs':specs, 'recipe':specs[winner['name']]['recipe'], 'base_recipe':recipe,
               'selection_AP':winner['AP'], 'selection_utc':utc_now(),
               'previous_benchmark_results_known':True, 'new_benchmark_predictions_not_opened':True})


def freeze(df, protocol):
    path = OUT/'selection_lock.json'
    if path.exists():
        print('V2 frozen bundle already present.', flush=True); return
    selection = read_json(OUT/'selection.json')
    recipe = selection['recipe']
    cal = valid_rows(df, protocol['calibration_start'], protocol['calibration_end'])
    policy = valid_rows(df, protocol['policy_start'], protocol['policy_end'])
    models, calibrators, thresholds, rows = {}, {}, {}, []
    for name, spec in selection['specs'].items():
        train = train_rows(df, spec['recipe'], protocol['final_training_end'], protocol['exact_rbf_train_cap'] if spec['family']=='svm_rbf' else None)
        destination = OUT/'models'/f'final_{name}.joblib'
        start = time.time()
        model = joblib.load(destination) if destination.exists() else ResearchModel(**spec).fit(train, train.target)
        assert not model.convergence_warnings, model.convergence_warnings
        if not destination.exists():
            joblib.dump(model, destination, compress=3)
        cs, ps = model.predict_score(cal), model.predict_score(policy)
        options = []
        for method in ['sigmoid', 'isotonic']:
            c = ScoreCalibrator(method).fit(cs, cal.target)
            p = c.predict(ps)
            row = {'model':name, 'method':method, **metrics(policy.target, p)}
            rows.append(row); options.append((row, c, p))
        row, c, p = min(options, key=lambda item:(item[0]['Brier'],item[0]['log_loss']))
        models[name], calibrators[name], thresholds[name] = model, c, choose_thresholds(policy.target, p)
        print(f"V2 final {name}: n={len(train):,}, {time.time()-start:.1f}s; calibration={c.method}", flush=True)
    bundle = {'models':models, 'calibrators':calibrators, 'thresholds':thresholds,
              'selected':selection['selected'], 'recipe':recipe}
    joblib.dump(bundle, OUT/'models'/'frozen_bundle.joblib', compress=3)
    pd.DataFrame(rows).drop(columns='confusion_matrix').to_csv(OUT/'calibration_selection.csv', index=False)
    lock = {'frozen_utc':utc_now(), 'selected':selection['selected'], 'recipe':recipe,
            'model_sha256':file_sha256(OUT/'models'/'frozen_bundle.joblib'),
            'protocol_sha256':file_sha256(PROTOCOL), 'selection_sha256':file_sha256(OUT/'selection.json'),
            'code_sha256':{str(p).replace('\\','/'):file_sha256(p) for p in Path('research').glob('*.py')},
            'original_model_sha256':read_json(BASE/'selection_lock.json')['model_sha256'],
            'benchmark_role':protocol['benchmark_role'], 'new_pristine_final_test':False}
    write_json(path, lock)


def benchmark(protocol):
    receipt_path = OUT/'benchmark_receipt.json'
    if receipt_path.exists() and read_json(receipt_path)['state']=='complete':
        print('V2 reused benchmark already complete.', flush=True); return
    lock = read_json(OUT/'selection_lock.json')
    assert lock['protocol_sha256']==file_sha256(PROTOCOL)
    assert lock['model_sha256']==file_sha256(OUT/'models'/'frozen_bundle.joblib')
    assert all(file_sha256(p)==h for p,h in lock['code_sha256'].items())
    write_json(receipt_path, {'state':'in_progress', 'role':protocol['benchmark_role'], 'lock_sha256':file_sha256(OUT/'selection_lock.json')})
    bundle = joblib.load(OUT/'models'/'frozen_bundle.joblib')
    rows, common_rows, uncertainty, common_ids = [], [], {}, {}
    for year in [2015, 2016]:
        raw = pd.read_parquet(f'data/processed/holdout_{year}.parquet')
        df = raw[raw.term.str.strip().eq('36 months')].copy()
        common = sample_development(df, protocol['exact_rbf_benchmark_cap'])
        positions = df.index.get_indexer(common.index)
        assert (positions >= 0).all() and len(set(common.id)) == len(common)
        common_ids[str(year)] = {'n':len(common), 'ids_sha256':ids_sha256(common.id),
                                 'sampling':'same stratified seed-42 rows for every model'}
        old = pd.read_parquet(BASE/'predictions'/f'external_{year}.parquet').set_index('id').loc[df.id]
        predictions = df[['id','target','issue_date']].copy()
        for name in ['baseline19','selected']:
            label = 'original_'+name
            p = old[name].to_numpy()
            predictions[label] = p
            old_lock = read_json(BASE/'selection_lock.json')
            rows.append({'cohort':year,'scope':'36_month','model':label,
                         **metrics(df.target,p,old_lock['thresholds'][name]['f1']), 'raw_AP':float(average_precision_score(df.target,p))})
            common_rows.append({'cohort':year,'scope':'36_month_common20000','model':label,
                                **metrics(common.target,p[positions],old_lock['thresholds'][name]['f1'])})
        for name, model in bundle['models'].items():
            limited = model.family == 'svm_rbf' and name != bundle['selected']
            evaluated = common if limited else df
            score = model.predict_score(evaluated)
            p = bundle['calibrators'][name].predict(score)
            result = {'cohort':year,'scope':'36_month_common20000' if limited else '36_month','model':name,
                         **metrics(evaluated.target,p,bundle['thresholds'][name]['f1']),
                         'raw_AP':float(average_precision_score(evaluated.target,score)),
                         'raw_ROC_AUC':float(roc_auc_score(evaluated.target,score)),
                         'selected_before_new_predictions':name==bundle['selected']}
            if not limited:
                predictions[name] = p
                rows.append(result)
            common_rows.append({'cohort':year,'scope':'36_month_common20000','model':name,
                         **metrics(common.target,p if limited else p[positions],bundle['thresholds'][name]['f1']),
                         'selected_before_new_predictions':name==bundle['selected']})
            print(f"Reused benchmark {year} / {name} / n={len(evaluated)}: AP={result['AP']:.5f}, ROC={result['ROC_AUC']:.5f}", flush=True)
        predictions.to_parquet(OUT/'predictions'/f'benchmark_{year}_36.parquet', index=False)
        if year == 2015:
            uncertainty['2015_36_vs_original_ensemble'] = paired_bootstrap(df.target.to_numpy(), predictions[bundle['selected']].to_numpy(),
                       predictions.original_selected.to_numpy(), df.issue_date.dt.strftime('%Y-%m').to_numpy(), reps=300)
    write_json(OUT/'benchmark_metrics.json',rows)
    pd.DataFrame(rows).drop(columns='confusion_matrix').to_csv(OUT/'benchmark_metrics.csv',index=False)
    write_json(OUT/'common_budget_benchmark.json',common_rows)
    pd.DataFrame(common_rows).drop(columns='confusion_matrix').to_csv(OUT/'common_budget_benchmark.csv',index=False)
    write_json(OUT/'common_budget_ids.json',common_ids)
    write_json(OUT/'benchmark_uncertainty.json',uncertainty)
    write_json(receipt_path, {'state':'complete','completed_utc':utc_now(), 'role':protocol['benchmark_role'],
               'lock_sha256':file_sha256(OUT/'selection_lock.json'), 'selected':bundle['selected'],
               'metrics_sha256':file_sha256(OUT/'benchmark_metrics.json'), 'pristine_final_test':False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen','search','select','freeze','benchmark','all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = read_json(PROTOCOL)
    df = pd.read_parquet('data/processed/development.parquet')
    assert df.issue_date.max()<pd.Timestamp('2015-01-01')
    with threadpool_limits(limits=3):
        for stage, fn in [('screen',screen),('search',search),('select',fit_candidates),('freeze',freeze),('benchmark',None)]:
            if args.stage in [stage,'all']:
                if stage!='benchmark':
                    assert not (OUT/'benchmark_receipt.json').exists(), 'Do not retune this run after new benchmark predictions.'
                    fn(df,protocol)
                else:
                    benchmark(protocol)


if __name__=='__main__':
    main()
