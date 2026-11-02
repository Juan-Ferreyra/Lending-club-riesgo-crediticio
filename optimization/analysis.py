"""Development diagnostics and checks for the frozen third experiment.

These commands do not select candidates or change an estimator. Interpretation
uses the Q3 candidate fit through June, never the later final refit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from optimization.data import load_partition, train_rows, valid_rows
from optimization.ensemble import PositiveSigmoid, ensemble_score
from optimization.evaluation import _verify_lock, december_split, inherited_integrity
from optimization.experiments import OUT, V2, PROTOCOL
from optimization.features import CONTEXT_RAW, MISSING, _context_values
from src.prepare import file_sha256, ids_sha256
from src.train import sample_development
from src.utils import utc_now, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def context(protocol, out=OUT):
    """Audit inputs from development only; no held-out outcomes are read."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    development = load_partition('development')
    eligible = train_rows(development, '2014-12-31', '2012-01-01')
    availability, tops, shifts = [], [], []
    for year in [2012, 2013, 2014]:
        annual = eligible[eligible.issue_date.dt.year.eq(year)]
        for field in CONTEXT_RAW:
            values = _context_values(annual, field)
            counts = values.value_counts()
            availability.append({
                'year': year, 'field': field, 'n': len(annual),
                'missing_n': int(values.eq(MISSING).sum()),
                'missing_rate': float(values.eq(MISSING).mean()),
                'unique_nonmissing_normalized': int(values[values.ne(MISSING)].nunique()),
                'unique_fraction_nonmissing': float(values[values.ne(MISSING)].nunique() /
                                                    max(values.ne(MISSING).sum(), 1)),
            })
            for rank, (value, count) in enumerate(counts.head(10).items(), 1):
                tops.append({'year': year, 'field': field, 'rank': rank, 'normalized_value': value,
                             'count': int(count), 'fraction': float(count / len(annual))})
    for fold_id, fold in enumerate(protocol['cv'], 1):
        valid = valid_rows(development, fold['validation_start'], fold['validation_end'],
                           protocol['search_validation_cap'])
        for start in ['2012-01-01', '2013-01-01']:
            train = train_rows(development, fold['train_end'], start, protocol['search_train_cap'])
            for field in CONTEXT_RAW:
                a, b = _context_values(train, field), _context_values(valid, field)
                counts = a[a.ne(MISSING)].value_counts()
                ordered = counts.rename_axis('value').reset_index(name='count').sort_values(
                    ['count', 'value'], ascending=[False, True]
                ).set_index('value')['count']
                vocabulary = set(ordered[ordered.ge(50)].head(1500).index)
                known = set(counts.index)
                unseen = b.ne(MISSING) & ~b.isin(known)
                rare_known = b.isin(known) & ~b.isin(vocabulary)
                distribution_a, distribution_b = a.value_counts(normalize=True), b.value_counts(normalize=True)
                values = distribution_a.index.union(distribution_b.index)
                tvd = .5 * (distribution_a.reindex(values, fill_value=0) -
                            distribution_b.reindex(values, fill_value=0)).abs().sum()
                shifts.append({
                    'fold': fold_id, 'field': field, 'train_start': start,
                    'train_end': fold['train_end'], 'validation_start': fold['validation_start'],
                    'validation_end': fold['validation_end'], 'train_n': len(train), 'validation_n': len(valid),
                    'train_ids_sha256': ids_sha256(train.id), 'validation_ids_sha256': ids_sha256(valid.id),
                    'min_category_count': 50, 'max_categories': 1500, 'retained_category_n': len(vocabulary),
                    'train_missing_rate': float(a.eq(MISSING).mean()),
                    'validation_missing_rate': float(b.eq(MISSING).mean()),
                    'validation_unseen_nonmissing_rate': float(unseen.mean()),
                    'validation_known_rare_rate': float(rare_known.mean()),
                    'validation_retained_category_rate': float(b.isin(vocabulary).mean()),
                    'total_variation_distance': float(tvd),
                })
    pd.DataFrame(availability).to_csv(out / 'context_availability.csv', index=False)
    pd.DataFrame(tops).to_csv(out / 'context_top_values.csv', index=False)
    pd.DataFrame(shifts).to_csv(out / 'context_shift.csv', index=False)
    write_json(out / 'context_scope.json', {
        'created_utc': utc_now(), 'partition': 'development', 'years': [2012, 2013, 2014],
        'term': '36 months', 'fields': list(CONTEXT_RAW), 'labels_used_for_summary_or_encoding': False,
        'targets_used_only_for_shared_stratified_sampling': True,
        'normalization': 'Same _context_values implementation as the V3 feature contract.',
        'shift_denominator': 'All validation rows, including missing; missing/unseen/known-rare/retained are separate categories.',
        'interpretation': 'Changes in text representation can exist even when missingness rates are stable. These summaries do not date a platform-wide schema change.',
    })
    print('Development context availability and shift diagnostics saved.', flush=True)


PERMUTATION_GROUPS = {
    'loan_amount': ['loan_amnt'],
    'income': ['annual_inc', 'annual_inc_joint'],
    'FICO': ['fico_range_low', 'fico_range_high'],
    'dti': ['dti', 'dti_joint'],
    'revolving': ['revol_bal', 'revol_util', 'total_rev_hi_lim', 'bc_open_to_buy',
                  'bc_util', 'total_bc_limit', 'max_bal_bc', 'all_util', 'percent_bc_gt_75'],
    'bureau_accounts': ['open_acc', 'total_acc', 'tot_cur_bal', 'total_bal_ex_mort', 'tot_hi_cred_lim',
                        'avg_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m',
                        'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'inq_fi',
                        'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'mort_acc',
                        'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
                        'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
                        'num_sats', 'num_tl_op_past_12m', 'total_il_high_credit_limit', 'inq_last_6mths'],
    'credit_history': ['earliest_cr_line', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
                       'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_rcnt_il',
                       'mths_since_recent_bc', 'mths_since_recent_inq'],
    'delinquency': ['delinq_2yrs', 'pub_rec', 'mths_since_last_delinq', 'mths_since_last_record',
                    'mths_since_last_major_derog', 'collections_12_mths_ex_med', 'acc_now_delinq',
                    'tot_coll_amt', 'chargeoff_within_12_mths', 'delinq_amnt', 'mths_since_recent_bc_dlq',
                    'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_tl_120dpd_2m',
                    'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 'tax_liens'],
    'employment': ['emp_length'],
    'homepurpose': ['home_ownership', 'purpose', 'verification_status', 'application_type', 'addr_state'],
    'context_employer': ['emp_title'],
    'context_zip': ['zip_code'],
}


def interpret(protocol, out=OUT, previous=V2):
    """Grouped permutation on Q3 models, fixed before the final-period refit."""
    out, previous = Path(out), Path(previous)
    selection_path = out / 'selection.json'
    selection_hash = file_sha256(selection_path)
    selection = read_json(selection_path)
    development = load_partition('development')
    complete = valid_rows(development, protocol['selection_start'], protocol['selection_end'])
    valid = sample_development(complete, 6000, seed=42)
    members = selection['members']
    paths = {name: (previous / 'models' / 'candidate_catboost.joblib' if name == 'v2_catboost'
                    else out / 'models' / f'candidate_{name}.joblib') for name in members}
    model_hashes = {path.as_posix(): file_sha256(path) for path in paths.values()}
    models = {name: joblib.load(path) for name, path in paths.items()}
    mapping_json = out / 'ensemble_mappings.json'
    mappings = ({name: PositiveSigmoid.from_dict(value) for name, value in read_json(mapping_json).items()}
                if mapping_json.exists() else joblib.load(out / 'models' / 'ensemble_mappings.joblib'))

    def predict(frame):
        scores = {name: model.predict_score(frame) for name, model in models.items()}
        if len(members) == 1:
            return scores[next(iter(members))]
        return ensemble_score(scores, members, mappings)

    original_scores = predict(valid)
    saved = pd.read_parquet(out / 'predictions' / 'selection.parquet').set_index('id')
    saved.index = saved.index.astype(str)
    reference = saved.loc[valid.id.astype(str)]
    assert np.array_equal(reference.target.to_numpy(), valid.target.to_numpy())
    np.testing.assert_allclose(original_scores, reference[selection['selected']].to_numpy(), atol=1e-10, rtol=0)
    baseline = float(average_precision_score(valid.target, original_scores))
    # Identical permutations across groups reduce noise when comparing their effects.
    rng = np.random.default_rng(42)
    orders = [rng.permutation(len(valid)) for _ in range(3)]
    rows = []
    for name, configured in PERMUTATION_GROUPS.items():
        columns = [column for column in configured if column in valid]
        drops = []
        for order in orders:
            shuffled = valid.copy()
            # All fields within the group use one common permutation, preserving
            # FICO ranges and other internal combinations before derived features.
            for column in columns:
                shuffled[column] = valid[column].iloc[order].to_numpy()
            drops.append(baseline - float(average_precision_score(valid.target, predict(shuffled))))
        rows.append({'feature_group': name, 'mean_AP_drop': float(np.mean(drops)),
                     'permutation_sd': float(np.std(drops, ddof=1)), 'validation_n': len(valid),
                     'baseline_AP': baseline, 'permutations': len(orders),
                     'raw_columns': '|'.join(columns), **{f'AP_drop_{i+1}': value for i, value in enumerate(drops)}})
        print(f'Q3 permutation {name}: AP drop={np.mean(drops):+.6f}', flush=True)
    pd.DataFrame(rows).sort_values('mean_AP_drop', ascending=False).to_csv(out / 'interpretability.csv', index=False)
    assert file_sha256(selection_path) == selection_hash
    assert all(file_sha256(path) == digest for path, digest in model_hashes.items())
    write_json(out / 'interpretability_scope.json', {
        'created_utc': utc_now(), 'selected': selection['selected'], 'members': members,
        'model_stage': 'Q3 candidates trained through June 2014; final November models are not used.',
        'model_files_sha256': model_hashes, 'selection_sha256': selection_hash,
        'validation_start': protocol['selection_start'], 'validation_end': protocol['selection_end'],
        'complete_validation_n': len(complete), 'validation_n': len(valid), 'validation_ids_sha256': ids_sha256(valid.id),
        'sampling': 'Stratified sample_development, cap=6000, seed=42',
        'baseline_AP': baseline, 'permutations': 3, 'permutation_seed': 42,
        'grouping': 'One row permutation shared by all raw fields within a group; derived features recomputed.',
        'role': 'Post-selection descriptive interpretation on reused development; never used for new selection.',
        'limits': 'Importance is predictive and noncausal. Correlation between groups can substitute for shuffled information; combinations between shuffled and unshuffled groups may be unusual. SD across three shuffles is not a confidence interval.',
    })


def verify(protocol, out=OUT, previous=V2, protocol_path=PROTOCOL):
    """Recompute cohort, metric, budget, and artifact consistency checks."""
    out, previous = Path(out), Path(previous)
    checks, details = {}, {}

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail

    try:
        lock = _verify_lock(out, previous, protocol_path)
        check('frozen_artifact_hashes_match', True)
        inherited = inherited_integrity(previous)
        check('inherited_artifact_hashes_match', True, {'checked_artifacts': len(inherited)})
        selection = read_json(out / 'selection.json')
        candidate_contracts, candidate_identity = [], {}
        for name, spec in selection['specs'].items():
            candidate_path = (previous / 'models' / 'candidate_catboost.joblib' if name == 'v2_catboost'
                              else out / 'models' / f'candidate_{name}.joblib')
            candidate_model = joblib.load(candidate_path)
            identity_matches = (candidate_model.family == spec['family'] and
                                candidate_model.params == spec['params'] and candidate_model.seed == spec['seed'])
            if spec.get('engine') == 'v2':
                identity_matches = identity_matches and candidate_model.recipe['mode'] == 'stable' and \
                    candidate_model.recipe['start'] == spec['start'] and candidate_model.recipe['term'] == '36 months'
            else:
                identity_matches = identity_matches and candidate_model.feature_mode == spec['feature_mode'] and \
                    candidate_model.preparation == spec['preparation']
            if spec['family'] == 'catboost':
                identity_matches = identity_matches and candidate_model.estimator.tree_count_ == spec['params']['iterations']
            candidate_contracts.append(bool(identity_matches))
            candidate_identity[name] = {
                'path': candidate_path.as_posix(), 'sha256': file_sha256(candidate_path),
                'family': spec['family'], 'contract_matches': bool(identity_matches),
                'catboost_iteration_count_checked': spec['family'] == 'catboost',
            }
        check('all_Q3_candidate_contracts_match_selection_specs', bool(candidate_contracts) and all(candidate_contracts),
              {'candidate_count': len(candidate_contracts), 'matching_count': sum(candidate_contracts),
               'candidates': candidate_identity,
               'scope': 'Stored estimator parameters, seed and feature/preparation contract; predictions are not recomputed by this check.'})
        receipt = read_json(out / 'benchmark_receipt.json')
        check('benchmark_completed', receipt['state'] == 'complete')
        check('benchmark_receipt_matches_lock', receipt['lock_sha256'] == file_sha256(out / 'selection_lock.json'))
        check('benchmark_metrics_hash_matches', receipt['metrics_sha256'] == file_sha256(out / 'benchmark_metrics.json'))
        check('benchmark_output_hashes_match', all(file_sha256(out / name) == digest
                                                  for name, digest in receipt['outputs_sha256'].items()))
        bundle = joblib.load(out / 'models' / 'frozen_bundle.joblib')
        check('selected_strategy_unchanged', selection['selected'] == lock['selected'] == receipt['selected'] == bundle['selected'])
        check('selected_members_unchanged', selection['members'] == lock['members'] == bundle['members'])
        check('selected_before_freeze_before_benchmark',
              pd.Timestamp(selection['selected_utc']) <= pd.Timestamp(lock['frozen_utc']) <= pd.Timestamp(receipt['started_utc']))
        check('no_new_independent_test_claim', protocol['fresh_independent_test_available'] is False and
              lock['new_pristine_final_test'] is False and receipt['pristine_final_test'] is False and
              selection['previous_benchmark_results_known'] is True)
        check('development_selection_did_not_open_new_benchmark', selection['new_benchmark_predictions_opened'] is False)
        if (out / 'checkout_integrity.json').exists():
            checkout = read_json(out / 'checkout_integrity.json')
            check('checkout_filters_preserve_frozen_bytes', checkout['all_passed'] and all(checkout['checks'].values()))

        development = pd.read_parquet('data/processed/development.parquet', columns=['id', 'term', 'issue_date', 'target'])
        search_design = read_json(out / 'search_design.json')
        check('search_protocol_and_budget_unchanged',
              search_design['protocol_sha256'] == file_sha256(protocol_path) == selection['protocol_sha256'] and
              search_design['planned_trials'] == protocol['trials'])
        check('search_model_contract_unchanged', all(file_sha256(path) == digest
                                                   for path, digest in search_design['dependencies_sha256'].items()))
        check('search_development_population_unchanged', search_design['train_ids_sha256'] == ids_sha256(development.id))
        check('search_started_without_new_benchmark', search_design['benchmark_opened'] is False)
        details['search_runner_revision'] = (
            'The runner fingerprint at search start is historical provenance, not a required current-file match: '
            'the documented amendment changes later development selection only. The model contract, data, '
            'main protocol and HPO budget must still match search_design.json.'
        )
        december = valid_rows(development, protocol['calibration_start'], protocol['calibration_end'])
        cal, policy = december_split(december)
        split = read_json(out / 'calibration_split.json')
        check('calibration_and_policy_ids_disjoint', set(cal.id).isdisjoint(policy.id))
        check('calibration_policy_cover_december', set(cal.id) | set(policy.id) == set(december.id))
        check('calibration_split_hashes_match', split['calibration_ids_sha256'] == ids_sha256(cal.id) and
              split['policy_ids_sha256'] == ids_sha256(policy.id) and split['december_ids_sha256'] == ids_sha256(december.id))
        check('calibration_split_sizes_match', split['calibration_n'] == len(cal) and split['policy_n'] == len(policy) and
              split['december_n'] == len(december) and split['overlap_n'] == 0)
        train_manifest = read_json(out / 'training_manifest.json')
        train_checks = []
        for name, metadata in train_manifest.items():
            train = train_rows(development, protocol['final_training_end'], metadata['spec']['start'])
            train_checks.append(metadata['train_n'] == len(train) and metadata['train_ids_sha256'] == ids_sha256(train.id) and
                                train.issue_date.max() < cal.issue_date.min() and set(train.id).isdisjoint(december.id))
        check('all_final_training_rows_and_dates_match', bool(train_checks) and all(train_checks))
        check('final_training_metadata_matches_bundle', train_manifest == bundle['training'])
        check('extra_data_control_uses_expected_rows', train_manifest['v2_catboost']['train_n'] == 299588)
        old_spec = read_json(previous / 'selection.json')['specs']['catboost']
        new_spec = selection['specs']['v2_catboost']
        check('extra_data_control_keeps_v2_parameters', new_spec['params'] == old_spec['params'] and
              new_spec['engine'] == 'v2' and new_spec['start'] == old_spec['recipe']['start'] and new_spec['seed'] == 42)
        amendment_path = Path('experiments/development_followup_v3.json')
        if selection.get('development_followup_sha256'):
            amendment = read_json(amendment_path)
            amendment_hash = file_sha256(amendment_path)
            check('development_amendment_hash_matches_selection_and_lock',
                  amendment_hash == selection['development_followup_sha256'] ==
                  lock['input_sha256'].get(amendment_path.as_posix()))
            check('development_amendment_registered_before_selection_and_freeze',
                  pd.Timestamp(amendment['registered_utc']) <= pd.Timestamp(selection['selected_utc']) <=
                  pd.Timestamp(lock['frozen_utc']))
            check('development_amendment_before_new_benchmark', amendment['new_benchmark_predictions_opened'] is False and
                  pd.Timestamp(amendment['registered_utc']) <= pd.Timestamp(receipt['started_utc']))
            controls = {'catboost_context_control': 'context', 'catboost_combined_control': 'combined'}
            check('development_amendment_has_exact_two_controls',
                  {row['name']: row['feature_mode'] for row in amendment['controlled_models']} == controls and
                  len(amendment['controlled_models']) == 2)
            board = pd.read_csv(out / 'selection_leaderboard.csv').set_index('model')
            reference = board.loc['v2_catboost']
            control_specs_match, control_artifacts_match, control_rows_match = [], [], []
            extra_names = []
            blend_weights_match = []
            for name, mode in controls.items():
                spec = selection['specs'][name]
                expected_spec = {'family': 'catboost', 'params': old_spec['params'], 'feature_mode': mode,
                                 'preparation': 'native', 'start': '2012-01-01', 'half_life': 0, 'seed': 42}
                control_specs_match.append(spec == expected_spec)
                model = joblib.load(out / 'models' / f'candidate_{name}.joblib')
                control_artifacts_match.append(model.family == 'catboost' and model.params == old_spec['params'] and
                                               model.feature_mode == mode and model.preparation == 'native' and
                                               model.seed == 42 and model.estimator.tree_count_ == old_spec['params']['iterations'])
                row = board.loc[name]
                control_rows_match.append(row['train_n'] == reference['train_n'] == 215847 and
                                          row['validation_n'] == reference['validation_n'] == 40595 and
                                          row['train_ids_sha256'] == reference['train_ids_sha256'] and
                                          row['validation_ids_sha256'] == reference['validation_ids_sha256'])
                extra_names.append(name)
                for weight in [.25, .50, .75]:
                    blend_name = f'blend_v2_{name}_{int(weight * 100)}'
                    extra_names.append(blend_name)
                    blend_weights_match.append(selection['ensembles'][blend_name] ==
                                               {'v2_catboost': weight, name: 1 - weight} and
                                               board.loc[blend_name, 'validation_n'] == reference['validation_n'] and
                                               board.loc[blend_name, 'validation_ids_sha256'] == reference['validation_ids_sha256'])
            check('amendment_controls_change_only_declared_features', all(control_specs_match))
            check('amendment_control_artifacts_keep_exact_v2_parameters', all(control_artifacts_match))
            check('amendment_controls_use_identical_full_Q3_and_training_rows', all(control_rows_match))
            check('amendment_six_blends_keep_fixed_weights_and_Q3_rows', len(blend_weights_match) == 6 and all(blend_weights_match))
            check('amendment_all_eight_additional_candidates_retained',
                  amendment['additional_candidates'] == selection['additional_development_candidates'] == len(extra_names) == 8 and
                  set(extra_names).issubset(board.index) and not board.index.duplicated().any())
            details['development_amendment'] = {
                'registered_utc': amendment['registered_utc'], 'additional_candidates': extra_names,
                'role': amendment['role'],
                'interpretation': 'Adaptive development expansion; neither the original prespecified search nor independent confirmation.',
            }
        else:
            check('no_unregistered_development_amendment', not amendment_path.exists() and
                  selection.get('additional_development_candidates', 0) == 0)
        preparation_path = Path('experiments/preparation_followup_v3.json')
        preparation_records = []
        if selection.get('preparation_followup_sha256'):
            preparation_amendment = read_json(preparation_path)
            preparation_hash = file_sha256(preparation_path)
            check('preparation_amendment_hash_matches_selection_and_lock',
                  preparation_hash == selection['preparation_followup_sha256'] ==
                  lock['input_sha256'].get(preparation_path.as_posix()))
            check('preparation_amendment_registered_before_selection_and_freeze',
                  pd.Timestamp(preparation_amendment['registered_utc']) <= pd.Timestamp(selection['selected_utc']) <=
                  pd.Timestamp(lock['frozen_utc']))
            check('preparation_amendment_before_new_benchmark',
                  preparation_amendment['new_benchmark_predictions_opened'] is False and
                  pd.Timestamp(preparation_amendment['registered_utc']) <= pd.Timestamp(receipt['started_utc']))
            preparation_records = read_json(out / 'preparation_followup.json')
            preparation_families = {'logistic', 'svm_linear', 'svm_nystroem'}
            check('preparation_amendment_has_exact_three_family_controls',
                  set(preparation_amendment['families']) == preparation_families and
                  len(preparation_amendment['families']) == 3 and
                  {row['family'] for row in preparation_records} == preparation_families and
                  preparation_amendment['additional_CV_configurations'] == len(preparation_records) == 3)
            prep_specs, prep_rows, prep_metrics, prep_promotions, prep_failures, prep_slots = [], [], [], [], [], []
            for row in preparation_records:
                family = row['family']
                baseline = read_json(out / f'best_{family}.json')[0]
                expected_spec = baseline['spec'] | {'preparation': 'log_clip'}
                prep_metrics.append(np.isclose(row['baseline_CV_AP'], baseline['CV_AP'], rtol=0, atol=1e-12))
                if row['status'] == 'COMPLETE':
                    candidate = row['candidate']
                    prep_specs.append(baseline['spec']['preparation'] == 'native' and candidate['spec'] == expected_spec)
                    candidate_folds = {fold['fold']: fold for fold in candidate['folds']}
                    baseline_folds = {fold['fold']: fold for fold in baseline['folds']}
                    prep_rows.append(set(candidate_folds) == set(baseline_folds) == {0, 1})
                    for fold_id, original_fold in baseline_folds.items():
                        candidate_fold = candidate_folds[fold_id]
                        prep_rows.append(all(candidate_fold[key] == original_fold[key] for key in
                                             ['train_n', 'validation_n', 'train_ids_sha256', 'validation_ids_sha256']))
                    candidate_mean = float(np.mean([fold['AP'] for fold in candidate['folds']]))
                    prep_metrics.append(np.isclose(candidate_mean, candidate['CV_AP'], rtol=0, atol=1e-12) and
                                        np.isclose(row['delta_AP'], candidate['CV_AP'] - baseline['CV_AP'], rtol=0, atol=1e-12))
                    should_promote = candidate['CV_AP'] > baseline['CV_AP']
                    prep_promotions.append(row['promoted'] == should_promote)
                    expected_winner = candidate['spec'] if should_promote else baseline['spec']
                    cv_predictions = pd.read_parquet(out / 'predictions' / f"cv_{candidate['prediction_label']}.parquet")
                    for fold_id, candidate_fold in candidate_folds.items():
                        predicted_fold = cv_predictions[cv_predictions.fold.eq(fold_id)]
                        matching_outcomes = development.set_index('id').loc[predicted_fold.id, 'target'].to_numpy()
                        prep_rows.append(ids_sha256(predicted_fold.id) == candidate_fold['validation_ids_sha256'] and
                                         np.array_equal(predicted_fold.target.to_numpy(), matching_outcomes))
                        prep_metrics.append(np.isclose(average_precision_score(predicted_fold.target, predicted_fold.score),
                                                       candidate_fold['AP'], rtol=0, atol=1e-12))
                elif row['status'] == 'FAIL':
                    prep_specs.append(baseline['spec']['preparation'] == 'native' and row['candidate_spec'] == expected_spec)
                    prep_failures.append(bool(row.get('error')) and row['promoted'] is False and
                                         'candidate' not in row and 'delta_AP' not in row)
                    expected_winner = baseline['spec']
                else:
                    prep_failures.append(False)
                    expected_winner = baseline['spec']
                winner_name = selection['family_best'][family]
                prep_slots.append(winner_name == f'{family}_joint' and selection['specs'][winner_name] == expected_winner and
                                  sum(spec['family'] == family for spec in selection['specs'].values()) == 1)
            check('preparation_controls_change_only_native_to_log_clip', bool(prep_specs) and all(prep_specs))
            check('preparation_completed_controls_use_identical_two_fold_rows', all(prep_rows))
            check('preparation_CV_metrics_and_paired_deltas_recomputed', bool(prep_metrics) and all(prep_metrics))
            check('preparation_promotions_match_mean_CV_AP_rule', all(prep_promotions))
            check('preparation_failed_controls_retained_without_promotion', all(prep_failures),
                  {'failed_controls': sum(row['status'] == 'FAIL' for row in preparation_records)})
            check('preparation_controls_keep_one_existing_Q3_slot_per_family', bool(prep_slots) and all(prep_slots))
            details['preparation_amendment'] = {
                'registered_utc': preparation_amendment['registered_utc'],
                'family_status': {row['family']: row['status'] for row in preparation_records},
                'additional_CV_configurations': len(preparation_records), 'additional_Q3_slots': 0,
                'role': preparation_amendment['role'],
                'interpretation': 'The contrast estimates the combined clipping/logarithmic preparation change. It does not separately identify the effect of each operation.',
            }
        else:
            check('no_unregistered_preparation_amendment', not preparation_path.exists())
        calibrators = read_json(out / 'calibrators.json')
        check('sigmoid_slopes_positive_and_finite', all(np.isfinite([v['slope'], v['intercept']]).all() and
                                                      v['slope'] > 0 for v in calibrators.values()))

        records = read_json(out / 'benchmark_metrics.json')
        cohort_manifest = read_json(out / 'benchmark_cohorts.json')
        ap_flags, point_deltas, raw_metrics, probability_metrics = [], [], [], []
        numerical_ties = []
        for year, expected_n in [(2015, 258856), (2016, 212482)]:
            predicted = pd.read_parquet(out / 'predictions' / f'benchmark_{year}_36.parquet')
            old = pd.read_parquet(previous / 'predictions' / f'benchmark_{year}_36.parquet', columns=['id', 'target', 'issue_date', 'catboost'])
            source = pd.read_parquet(f'data/processed/holdout_{year}.parquet', columns=['id', 'target', 'issue_date', 'term'])
            source = source[source.term.str.strip().eq('36 months')]
            check(f'{year}_same_cohort_n', len(predicted) == len(old) == len(source) == expected_n)
            check(f'{year}_same_ids_targets_dates',
                  predicted.id.tolist() == old.id.tolist() == source.id.tolist() and
                  np.array_equal(predicted.target, old.target) and np.array_equal(predicted.target, source.target) and
                  np.array_equal(predicted.issue_date, old.issue_date) and np.array_equal(predicted.issue_date, source.issue_date))
            check(f'{year}_cohort_ids_hash_matches', ids_sha256(predicted.id) == cohort_manifest[str(year)]['ids_sha256'] ==
                  receipt['cohorts'][str(year)]['ids_sha256'])
            base_ap = float(average_precision_score(predicted.target, predicted.v2_frozen))
            check(f'{year}_frozen_reference_predictions_match', np.array_equal(predicted.v2_frozen.to_numpy(), old.catboost.to_numpy()))
            for row in [r for r in records if r['cohort'] == year]:
                name = row['model']
                ap = float(average_precision_score(predicted.target, predicted['score__' + name]))
                probability_ap = float(average_precision_score(predicted.target, predicted[name]))
                raw_metrics.append(np.isclose(ap, row['raw_AP'], rtol=0, atol=1e-12))
                probability_metrics.append(np.isclose(probability_ap, row['AP'], rtol=0, atol=1e-12))
                point_deltas.append(np.isclose(ap - base_ap, row['delta_AP_vs_v2'], rtol=0, atol=1e-12))
                preserved = bool(np.isclose(ap, probability_ap, rtol=0, atol=1e-12))
                ap_flags.append(preserved == row['calibration_AP_preserved'])
                if not preserved:
                    numerical_ties.append({'year': year, 'model': name, 'raw_AP': ap, 'probability_AP': probability_ap})
        check('all_reported_raw_AP_recomputed', bool(raw_metrics) and all(raw_metrics))
        check('all_reported_probability_AP_recomputed', bool(probability_metrics) and all(probability_metrics))
        check('all_paired_point_deltas_recomputed', bool(point_deltas) and all(point_deltas))
        check('calibration_rank_preservation_flags_accurate', bool(ap_flags) and all(ap_flags), numerical_ties)
        uncertainty = read_json(out / 'benchmark_uncertainty.json')['2015_36_vs_v2_frozen']
        selected_row = next(r for r in records if r['cohort'] == 2015 and r['model'] == selection['selected'])
        check('bootstrap_point_matches_selected_raw_delta', np.isclose(uncertainty['point_delta_AP'],
              selected_row['delta_AP_vs_v2'], rtol=0, atol=1e-12) and uncertainty['raw_AP_primary'] is True)

        expected_valid = {}
        for i, fold in enumerate(protocol['cv']):
            valid = valid_rows(development, fold['validation_start'], fold['validation_end'], protocol['search_validation_cap'])
            expected_valid[i] = ids_sha256(valid.id)
        all_fold_records = []
        for result in read_json(out / 'ablations.json'):
            all_fold_records.extend(result['folds'])
        for row in preparation_records:
            if row['status'] == 'COMPLETE':
                all_fold_records.extend(row['candidate']['folds'])
        budget_counts = {}
        for family, budget in protocol['trials'].items():
            study = optuna.load_study(study_name=family,
                                     storage=f"sqlite:///{(out / 'studies' / 'joint_search.db').as_posix()}")
            counts = {state.name: sum(t.state == state for t in study.trials) for state in optuna.trial.TrialState}
            finished = sum(counts[name] for name in ['COMPLETE', 'PRUNED', 'FAIL'])
            check(f'{family}_executed_budget_matches', finished == budget and counts['RUNNING'] == 0 and counts['COMPLETE'] > 0)
            budget_counts[family] = counts | {'expected_executed': budget}
            for trial in study.trials:
                all_fold_records.extend(trial.user_attrs.get('folds', []))
        details['executed_search_budget'] = budget_counts
        check('all_CV_validation_population_hashes_match', bool(all_fold_records) and
              all(row['validation_ids_sha256'] == expected_valid[row['fold']] for row in all_fold_records))
        check('all_CV_validation_cohorts_report_two_periods', {row['fold'] for row in all_fold_records} == {0, 1})

        interpretation = read_json(out / 'interpretability_scope.json')
        check('interpretation_uses_selected_Q3_candidates', interpretation['selected'] == selection['selected'] and
              interpretation['selection_sha256'] == file_sha256(out / 'selection.json') and
              all('/candidate_' in path.replace('\\', '/') for path in interpretation['model_files_sha256']))
        check('interpretation_candidate_hashes_match', all(file_sha256(path) == digest
                                                        for path, digest in interpretation['model_files_sha256'].items()))
        details['data_control_interpretation'] = (
            'Raw AP comparison of v2_catboost versus v2_frozen isolates the fixed V2 pipeline refitted on additional training rows. '
            'Brier/log-loss differences also include the changed calibration period and size, so they are not attributed solely to extra training data.'
        )
    except Exception as error:
        check('verification_completed_without_exception', False, f'{type(error).__name__}: {error}')
    auxiliary = [Path('.gitattributes'), Path('optimization/analysis.py'), Path('optimization/documentation.py'), *Path('optimization').glob('*notebook*.py'),
                 *Path('optimization').glob('*replay*.py'), *Path('tests').glob('test_optimization*.py'),
                 *Path('notebooks').glob('07*.ipynb')]
    auxiliary_hashes = {path.as_posix(): file_sha256(path) for path in auxiliary if path.exists()}
    result = {'verified_utc': utc_now(), 'checks': checks, 'details': details,
              'check_count': len(checks), 'all_checks_passed': bool(checks) and all(checks.values()),
              'auxiliary_sha256': auxiliary_hashes,
              'role': 'Artifact and calculation verification; it does not make reused benchmarks independent.'}
    write_json(out / 'verification.json', result)
    print(f"V3 verification: {sum(checks.values())}/{len(checks)} checks passed.", flush=True)
    if not result['all_checks_passed']:
        raise AssertionError('V3 verification failed; inspect verification.json')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['context', 'interpret', 'verify', 'all'], default='all')
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    with threadpool_limits(limits=3):
        for stage, function in [('context', context), ('interpret', interpret), ('verify', verify)]:
            if args.stage in [stage, 'all']:
                function(protocol)


if __name__ == '__main__':
    main()
