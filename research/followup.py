"""Post-selection hypothesis diagnostics; never change the selected predictor."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from lightgbm import LGBMClassifier
from research.models import ResearchModel, ScoreCalibrator
from research.experiments import OUT, BASE, read_json, train_rows, valid_rows
from src.features import FeatureBuilder, CATEGORICAL, BUREAU_NUMERIC
from src.train import sample_development
from src.models import LoanEnsemble
from src.evaluate import metrics
from src.utils import write_json


def factor_ablations(df, protocol):
    path = OUT/'one_factor_ablation.csv'
    if path.exists():
        return
    variants = ['original', 'domain_only', 'drop_unavailable_only', 'stable_missing_count_only',
                'deduplicate_only', 'median_indicators_only', 'clip_only']
    params = read_json(BASE/'best_lightgbm.json')['params']
    rows = []
    for i, fold in enumerate(protocol['cv']):
        train = train_rows(df, {'term':'all','start':None}, fold['train_end'], protocol['search_train_cap'])
        valid = valid_rows(df,fold['validation_start'],fold['validation_end'],protocol['search_validation_cap'])
        for variant in variants:
            X = FeatureBuilder().transform(train)
            V = FeatureBuilder().transform(valid)
            if variant=='domain_only':
                for c in ['open_acc','total_acc','delinq_2yrs','pub_rec','inq_last_6mths']:
                    X[c] = X[c].where(X[c]>=0); V[c] = V[c].where(V[c]>=0)
                for c in ['pct_tl_nvr_dlq','percent_bc_gt_75']:
                    X[c] = X[c].where(X[c].between(0,100)); V[c] = V[c].where(V[c].between(0,100))
            elif variant=='drop_unavailable_only':
                cols = X.columns[(X.nunique()>1)&(X.isna().mean()<=.95)]
                X,V = X[cols].copy(),V[cols].copy()
            elif variant=='stable_missing_count_only':
                cols = [c for c in BUREAU_NUMERIC if X[c].notna().mean()>=.05 and X[c].nunique()>1]
                X['missing_bureau_count'] = X[cols].isna().sum(axis=1).astype(float)
                V['missing_bureau_count'] = V[cols].isna().sum(axis=1).astype(float)
            elif variant=='deduplicate_only':
                X=X.drop(columns=['loan_to_monthly_income','term_months'])
                V=V.drop(columns=['loan_to_monthly_income','term_months'])
            cats=[c for c in CATEGORICAL if c in X]
            nums=[c for c in X if c not in cats]
            if variant=='clip_only':
                lo,hi=X[nums].quantile(.005),X[nums].quantile(.995)
                X[nums]=X[nums].clip(lo,hi,axis=1);V[nums]=V[nums].clip(lo,hi,axis=1)
            num_transform=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True) if variant=='median_indicators_only' else 'passthrough'
            prep=ColumnTransformer([('num',num_transform,nums),('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),cats)])
            Z=prep.fit_transform(X).astype('float32'); W=prep.transform(V).astype('float32')
            nnum=Z.shape[1]-len(cats)
            model=LGBMClassifier(**params,verbosity=-1,n_jobs=3,random_state=42,deterministic=True,force_col_wise=True)
            model.fit(Z,train.target,categorical_feature=list(range(nnum,Z.shape[1])))
            p=model.predict_proba(W)[:,1]
            rows.append({'fold':i,'variant':variant,'AP':float(average_precision_score(valid.target,p)),
                         'ROC_AUC':float(roc_auc_score(valid.target,p)), 'training_n':len(train),'validation_n':len(valid),
                         'fixed_hyperparameters':True})
        print('Single-factor audit fold complete:',i,flush=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def matched_budget(df,protocol,bundle):
    path=OUT/'matched_budget_validation.csv'
    selection=read_json(OUT/'selection.json'); recipe=bundle['recipe']
    specs={name:spec for name,spec in selection['specs'].items() if not name.endswith('_from_2013')}
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    if {r['family'] for r in rows}==set(specs):return
    train=train_rows(df,recipe,protocol['candidate_training_end'],protocol['exact_rbf_train_cap'])
    valid=valid_rows(df,protocol['selection_start'],protocol['selection_end'])
    for name,spec in specs.items():
        if any(row['family']==name for row in rows):continue
        start=time.time()
        model=ResearchModel(**spec).fit(train,train.target)
        assert not model.convergence_warnings,model.convergence_warnings
        p=model.predict_score(valid)
        rows.append({'family':name,'train_n':len(train),'validation_n':len(valid),
                     'AP':float(average_precision_score(valid.target,p)),
                     'ROC_AUC':float(roc_auc_score(valid.target,p)),'seconds':time.time()-start,
                     'same_training_rows_for_all_models':True})
        pd.DataFrame(rows).to_csv(path,index=False)
        print('Matched-budget validation:',name,rows[-1]['AP'],flush=True)
        del model;gc.collect()


def pricing_scope(protocol,bundle):
    path=OUT/'information_scope_sensitivity.csv'
    if path.exists() and (OUT/'platform_score_baselines.csv').exists():return
    raw=pd.read_parquet('data/research_v2/through_2015_all_status.parquet')
    raw=raw[raw.terminal & raw.issue_date.lt('2015-01-01')].copy()
    raw['target']=raw.loan_status.eq('Charged Off').astype('int8')
    recipe=bundle['recipe']
    train=train_rows(raw,recipe,protocol['candidate_training_end'],60000)
    valid=valid_rows(raw,protocol['selection_start'],protocol['selection_end'])
    params=read_json(OUT/'best_lightgbm.json')['params']
    # Establish how much signal is already supplied by the platform itself.
    rates=pd.to_numeric(valid.int_rate,errors='coerce').fillna(pd.to_numeric(train.int_rate,errors='coerce').median())
    table=train.groupby('sub_grade').target.agg(['sum','count'])
    table['risk']=(table['sum']+100*train.target.mean())/(table['count']+100)
    subgrade=valid.sub_grade.map(table.risk).fillna(train.target.mean())
    baselines=[]
    for label,score in [('LC_interest_rate_rank_only',rates),('training_subgrade_frequency_only',subgrade)]:
        baselines.append({'strategy':label,'AP':float(average_precision_score(valid.target,score)),
                          'ROC_AUC':float(roc_auc_score(valid.target,score)),
                          'scope':'after platform scoring, diagnostic only','validation_n':len(valid),
                          'eligible_for_primary_selection':False})
    pd.DataFrame(baselines).to_csv(OUT/'platform_score_baselines.csv',index=False)
    rows=[]
    for pricing in [False,True]:
        model=ResearchModel('lightgbm',params,recipe,pricing=pricing).fit(train,train.target)
        score=model.predict_score(valid)
        rows.append({'scope':'after_LC_grading_DIAGNOSTIC_ONLY' if pricing else 'independent_application_features',
                     'AP':float(average_precision_score(valid.target,score)),
                     'ROC_AUC':float(roc_auc_score(valid.target,score)),
                     'train_n':len(train),'validation_n':len(valid),
                     'eligible_for_primary_selection':False,
                     'validation_cohort':'2014 Q3, 36-month',
                     'added_inputs':'grade, sub_grade, int_rate' if pricing else ''})
        print('Information scope diagnostic:',rows[-1]['scope'],rows[-1]['AP'],flush=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def hyperparameter_controls(df,protocol,bundle):
    path=OUT/'hyperparameter_controls_validation.csv'
    if path.exists():return
    recipe=read_json(OUT/'selection.json').get('base_recipe',bundle['recipe'])
    train=train_rows(df,recipe,protocol['candidate_training_end'])
    valid=valid_rows(df,protocol['selection_start'],protocol['selection_end'])
    rows=[]
    for family in ['lightgbm','catboost']:
        for setting in ['previous_parameters','new_search_parameters']:
            if setting=='previous_parameters':
                candidate_path=OUT/'models'/f'candidate_{family}_previous_params.joblib'
                if candidate_path.exists():
                    model=joblib.load(candidate_path)
                else:
                    params=read_json(BASE/f'best_{family}.json')['params']
                    model=ResearchModel(family,params,recipe).fit(train,train.target)
            else:
                model=joblib.load(OUT/'models'/f'candidate_{family}.joblib')
            p=model.predict_score(valid)
            rows.append({'family':family,'setting':setting,'train_n':len(train),'validation_n':len(valid),
                         'AP':float(average_precision_score(valid.target,p)),
                         'ROC_AUC':float(roc_auc_score(valid.target,p)),
                         'same_features_and_training_rows':True,
                         'diagnostic_not_used_to_reselect_after_benchmark':True})
            print('Hyperparameter control:',family,setting,rows[-1]['AP'],flush=True)
            del model;gc.collect()
    pd.DataFrame(rows).to_csv(path,index=False)


def learning_curve(df,protocol,bundle):
    path=OUT/'training_size_validation.csv'
    if path.exists():return
    recipe=bundle['recipe'];params=read_json(OUT/'best_lightgbm.json')['params']
    full=train_rows(df,recipe,protocol['candidate_training_end'])
    valid=valid_rows(df,protocol['selection_start'],protocol['selection_end'])
    rows=[]
    for cap in sorted(set([min(n,len(full)) for n in [12000,30000,60000,len(full)]])):
        train=sample_development(full,cap)
        model=ResearchModel('lightgbm',params,recipe).fit(train,train.target)
        p=model.predict_score(valid)
        rows.append({'training_n':len(train),'validation_n':len(valid),
                     'AP':float(average_precision_score(valid.target,p)),
                     'ROC_AUC':float(roc_auc_score(valid.target,p)),
                     'same_features_parameters_validation':True,'not_used_to_reselect':True})
        print('Learning curve:',len(train),rows[-1]['AP'],flush=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def frozen_missingness_probe():
    path=OUT/'original_model_missingness_probe.csv'
    if path.exists():return
    bundle=joblib.load(BASE/'models'/'frozen_bundle.joblib')
    names=bundle['selected_weights']
    model=LoanEnsemble([bundle['models'][n] for n in names],list(names.values()))
    cal=bundle['calibrators']['selected'];threshold=bundle['thresholds']['selected']['f1']
    df=pd.read_parquet('data/processed/holdout_2015.parquet')
    df=sample_development(df[df.term.str.strip().eq('36 months')],20000)
    unavailable=[c for c in read_json(OUT/'audit_findings.json')['unlearnable_or_95pct_missing_in_old_training'] if c in BUREAU_NUMERIC]
    rows=[]
    for scenario in ['unchanged','mask_previously_unavailable_bureau']:
        raw=df.copy()
        if scenario!='unchanged':raw[unavailable]=np.nan
        p=cal.predict(model.predict_proba(raw)[:,1])
        rows.append({'scenario':scenario,'reused_benchmark_diagnostic_only':True,
                     **metrics(df.target,p,threshold)})
    pd.DataFrame(rows).drop(columns='confusion_matrix').to_csv(path,index=False)
    write_json(OUT/'missingness_probe_scope.json',{'masked_columns':unavailable,
               'model_and_calibration_unchanged':True,'not_used_for_new_model_selection':True,
               'interpretation':'Counterfactual input-availability sensitivity, not a validated deployment fix.'})


def final_diagnostics(bundle):
    from src.analyze import segments
    reference=pd.read_parquet('data/processed/development.parquet')
    reference=valid_rows(reference,'2014-10-01','2014-12-31')
    rows=[]
    segment_rows=[]
    for year in [2015,2016]:
        raw=pd.read_parquet(f'data/processed/holdout_{year}.parquet')
        raw=raw[raw.term.str.strip().eq('36 months')].copy()
        pred=pd.read_parquet(OUT/'predictions'/f'benchmark_{year}_36.parquet')
        assert raw.id.to_list()==pred.id.to_list()
        p=pred[bundle['selected']].to_numpy()
        old=pred.original_selected.to_numpy()
        for label,values in [('new_selected',p),('original_ensemble',old)]:
            rows.append({'year':year,'model':label,'n':len(raw),'mean_prediction':float(np.mean(values)),
                         'observed_rate':float(raw.target.mean()),
                         'observed_minus_prediction':float(raw.target.mean()-np.mean(values))})
            threshold=(bundle['thresholds'][bundle['selected']]['f1'] if label=='new_selected'
                       else read_json(BASE/'selection_lock.json')['thresholds']['selected']['f1'])
            for row in segments(raw,values,threshold,year):
                row['model']=label
                row['threshold_source']='development only; separate policy blocks for each experiment'
                segment_rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'benchmark_calibration_drift.csv',index=False)
    pd.DataFrame(segment_rows).drop(columns='confusion_matrix').to_csv(OUT/'benchmark_segment_errors.csv',index=False)
    # Rank-only simple term baseline shows why overall mixed-term AP differs from within-term AP.
    from sklearn.metrics import average_precision_score as ap
    training=pd.read_parquet('data/processed/development.parquet')
    rates=training[training.issue_date.le('2014-06-30')].groupby('term').target.mean()
    tests=[]
    for year in [2015,2016]:
        df=pd.read_parquet(f'data/processed/holdout_{year}.parquet')
        p=df.term.map(rates).to_numpy()
        tests.append({'year':year,'scope':'all_terminal','AP_term_only':float(ap(df.target,p)),
                      'prevalence':float(df.target.mean()),'interpretation':'Different lifetime horizons and selection; not comparable to the within-36-month task.'})
    pd.DataFrame(tests).to_csv(OUT/'term_only_baseline.csv',index=False)


def temporal_population_shift():
    from scipy.stats import ks_2samp
    raw=pd.read_parquet('data/research_v2/through_2015_all_status.parquet')
    raw=raw[raw.term.eq('36 months')]
    ref=raw[raw.issue_year.eq(2014)]
    new=raw[raw.issue_year.eq(2015)]
    rows=[]
    for feature in ['annual_inc','dti','loan_amnt','fico_range_low','revol_util','open_acc','total_acc']:
        a=pd.to_numeric(ref[feature],errors='coerce').dropna()
        b=pd.to_numeric(new[feature],errors='coerce').dropna()
        bins=np.unique(np.r_[-np.inf,a.quantile(np.linspace(.1,.9,9)),np.inf])
        pa=pd.cut(a,bins).value_counts(normalize=True,sort=False).to_numpy()
        pb=pd.cut(b,bins).value_counts(normalize=True,sort=False).to_numpy()
        pa,pb=np.maximum(pa,1e-5),np.maximum(pb,1e-5)
        pooled=np.sqrt((a.std()**2+b.std()**2)/2)
        rows.append({'feature':feature,'reference_year':2014,'comparison_year':2015,
                     'population':'all accepted 36-month loans, including active loans',
                     'reference_n':len(ref),'comparison_n':len(new),
                     'SMD':float((b.mean()-a.mean())/pooled),'KS_distance':float(ks_2samp(a,b).statistic),
                     'PSI_train_bins':float(np.sum((pb-pa)*np.log(pb/pa))),
                     'median_2014':float(a.median()),'median_2015':float(b.median()),
                     'missing_2014':float(ref[feature].isna().mean()),'missing_2015':float(new[feature].isna().mean())})
    pd.DataFrame(rows).to_csv(OUT/'temporal_population_shift.csv',index=False)


def mature_secondary(bundle):
    months=read_json(OUT/'secondary_mature_months.json')['months']
    rows=[]
    if months:
        raw=pd.read_parquet('data/processed/holdout_2016.parquet')
        raw=raw[raw.term.str.strip().eq('36 months')].copy()
        pred=pd.read_parquet(OUT/'predictions'/'benchmark_2016_36.parquet')
        assert raw.id.to_list()==pred.id.to_list()
        mask=raw.issue_date.dt.strftime('%Y-%m').isin(months).to_numpy()
        original=read_json(BASE/'selection_lock.json')
        names=['original_baseline19','original_selected']+list(bundle['models'])
        for name in [n for n in names if n in pred.columns]:
            threshold=(original['thresholds'][name.removeprefix('original_')]['f1'] if name.startswith('original_')
                       else bundle['thresholds'][name]['f1'])
            rows.append({'model':name,'cohort':2016,'scope':'coverage_selected_mature_months_36',
                         'months':','.join(months),**metrics(raw.target.to_numpy()[mask],pred[name].to_numpy()[mask],threshold)})
    result=pd.DataFrame(rows)
    if result.empty:
        result=pd.DataFrame(columns=['model','cohort','scope','months','n','AP','ROC_AUC','Brier','precision','recall'])
    else:
        result=result.drop(columns='confusion_matrix')
    result.to_csv(OUT/'mature_secondary_metrics.csv',index=False)


def interpretation(df,protocol,bundle):
    """Permutation sensitivity on development validation, never used for selection."""
    path=OUT/'interpretability_validation.csv'
    if path.exists():return
    name=bundle['selected']
    model=joblib.load(OUT/'models'/f'candidate_{name}.joblib')
    valid=valid_rows(df,protocol['selection_start'],protocol['selection_end'],5000)
    baseline=average_precision_score(valid.target,model.predict_score(valid))
    columns=['fico_range_low','fico_range_high','annual_inc','dti','loan_amnt',
             'revol_util','revol_bal','emp_length','home_ownership','purpose',
             'earliest_cr_line','total_acc','open_acc','tot_cur_bal','total_rev_hi_lim']
    groups={'FICO':['fico_range_low','fico_range_high']}
    groups.update({c:[c] for c in columns if c not in ['fico_range_low','fico_range_high'] and c in valid})
    rng=np.random.default_rng(42);rows=[]
    for label,cols in groups.items():
        drops=[]
        for repeat in range(3):
            shuffled=valid.copy();order=rng.permutation(len(valid))
            shuffled[cols]=valid[cols].iloc[order].to_numpy()
            drops.append(float(baseline-average_precision_score(valid.target,model.predict_score(shuffled))))
        rows.append({'model':name,'feature_group':label,'validation_n':len(valid),
                     'baseline_AP':float(baseline),'mean_AP_drop':float(np.mean(drops)),
                     'permutation_sd':float(np.std(drops,ddof=1)),'repetitions':3,
                     'scope':'2014 Q3 development validation; descriptive, not causal',
                     'correlated_predictors_can_share_importance':True})
    pd.DataFrame(rows).sort_values('mean_AP_drop',ascending=False).to_csv(path,index=False)
    print('Validation permutation interpretation complete.',flush=True)


def sample_feature_representativeness():
    """Compare the original random sample with its defined source population."""
    from scipy.stats import ks_2samp
    if (OUT/'sample_feature_representativeness.csv').exists():return
    raw=pd.read_parquet('data/research_v2/through_2015_all_status.parquet')
    raw=raw[raw.terminal].copy()
    sample=pd.read_csv('data/lending_club_muestra.csv.gz',dtype=str)
    sample['issue_date']=pd.to_datetime(sample.issue_d,format='%b-%Y')
    sample['term']=sample.term.str.strip()
    sample=sample[sample.issue_date.lt('2016-01-01')]
    rows=[]
    for scope in ['terminal_through_2015','2015_terminal_36_month']:
        a,b=raw,sample
        if scope=='2015_terminal_36_month':
            a=a[a.issue_date.dt.year.eq(2015)&a.term.eq('36 months')]
            b=b[b.issue_date.dt.year.eq(2015)&b.term.eq('36 months')]
        for feature in ['annual_inc','dti','loan_amnt','fico_range_low','revol_util','open_acc','total_acc']:
            x=pd.to_numeric(a[feature],errors='coerce').astype(float)
            y=pd.to_numeric(b[feature],errors='coerce').astype(float)
            pooled=np.sqrt((x.var()+y.var())/2)
            rows.append({'scope':scope,'feature':feature,'type':'numeric',
                         'source_n':len(a),'sample_n':len(b),
                         'SMD_sample_minus_source':float((y.mean()-x.mean())/pooled),
                         'KS_distance':float(ks_2samp(x.dropna(),y.dropna()).statistic),
                         'TVD':np.nan,'missing_source':float(x.isna().mean()),'missing_sample':float(y.isna().mean())})
        for feature in ['home_ownership','purpose','verification_status','application_type']:
            x=a[feature].fillna('__MISSING__').astype(str).str.strip().value_counts(normalize=True)
            y=b[feature].fillna('__MISSING__').astype(str).str.strip().value_counts(normalize=True)
            x,y=x.align(y,fill_value=0)
            rows.append({'scope':scope,'feature':feature,'type':'categorical',
                         'source_n':len(a),'sample_n':len(b),'TVD':float(abs(x-y).sum()/2)})
    pd.DataFrame(rows).to_csv(OUT/'sample_feature_representativeness.csv',index=False)
    print('Sample feature representativeness audit complete.',flush=True)


def main():
    assert read_json(OUT/'benchmark_receipt.json')['state']=='complete'
    protocol=read_json('experiments/protocol_v2.json')
    df=pd.read_parquet('data/processed/development.parquet')
    bundle=joblib.load(OUT/'models'/'frozen_bundle.joblib')
    with threadpool_limits(limits=3):
        factor_ablations(df,protocol)
        matched_budget(df,protocol,bundle)
        hyperparameter_controls(df,protocol,bundle)
        learning_curve(df,protocol,bundle)
        pricing_scope(protocol,bundle)
        frozen_missingness_probe()
        final_diagnostics(bundle)
        temporal_population_shift()
        mature_secondary(bundle)
        interpretation(df,protocol,bundle)
        sample_feature_representativeness()
    write_json(OUT/'followup_complete.json',{'state':'complete','selected_predictor_changed':False,
               'additional_diagnostics_used_for_retuning':False})
    print('Hypothesis follow-up diagnostics completed.',flush=True)


if __name__=='__main__':main()
