"""Retry nonconverged RBF configurations with a larger numerical solver budget."""
import time
import optuna
from research.experiments import OUT, read_json, cv_score
from src.utils import write_json, utc_now


def repair(df,protocol):
    path=OUT/'svm_solver_retries.json'
    if path.exists():return
    assert not (OUT/'selection_lock.json').exists()
    assert not (OUT/'benchmark_receipt.json').exists()
    study=optuna.load_study(study_name='v2_svm_rbf',storage='sqlite:///'+(OUT/'studies'/'search.sqlite3').as_posix())
    recipe=read_json(OUT/'selected_recipe.json')['recipe']
    best=read_json(OUT/'best_svm_rbf.json')
    retries=[]
    for trial in study.trials:
        if trial.state!=optuna.trial.TrialState.FAIL or trial.user_attrs.get('administrative_interruption',False):continue
        if 'model_params' not in trial.user_attrs:continue
        params={**trial.user_attrs['model_params'],'max_iter':100000}
        start=time.time()
        row={'original_trial_number':trial.number,'params':params,'same_predictive_parameters':True,'solver_iteration_limit':100000}
        try:
            score=cv_score(df,recipe,'svm_rbf',params,protocol)
            row.update(state='COMPLETE',**score)
            if score['CV_AP']>best['CV_AP']:
                best.update(params=params,model_params=params,**score,
                            chosen_from_convergence_retry=True,original_trial_number=trial.number)
            print('RBF convergence retry:',trial.number,'AP=',score['CV_AP'],flush=True)
        except RuntimeError as error:
            row.update(state='FAIL',reason=str(error))
            print('RBF still not converged after 100000 iterations:',trial.number,flush=True)
        row['seconds']=time.time()-start
        retries.append(row)
        write_json(OUT/'svm_solver_retries_progress.json',retries)
    best['params']['max_iter']=100000
    best['solver_retry_count']=len(retries)
    best['solver_limit_note']='Converged original fits are unchanged by a larger unused cap; incomplete configurations are retried before selection.'
    write_json(OUT/'best_svm_rbf.json',best)
    write_json(path,{'completed_utc':utc_now(),'original_configuration_budget':protocol['trials']['svm_rbf'],
               'additional_same_configuration_solver_retries':retries,
               'new_benchmark_predictions_accessed':False,'best_CV_AP':best['CV_AP']})


if __name__=='__main__':
    import pandas as pd
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=3):repair(pd.read_parquet('data/processed/development.parquet'),read_json('experiments/protocol_v2.json'))
