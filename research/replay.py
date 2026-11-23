"""Rebuild recorded V2 predictors without repeating adaptive hyperparameter search."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd
from src.utils import write_json


def replay(source_results,destination,reference_output):
    root=Path(__file__).resolve().parents[1]
    source=(root/source_results).resolve()
    dest=(root/destination).resolve()
    if source==dest or (dest.exists() and any(dest.iterdir())):
        raise FileExistsError('Choose a new, empty replay directory; existing results are preserved.')
    selection=source/'selection.json'
    if not selection.exists():raise FileNotFoundError(selection)
    (dest/'models').mkdir(parents=True,exist_ok=True)
    (dest/'predictions').mkdir(exist_ok=True)
    shutil.copy2(selection,dest/'selection.json')
    env=dict(os.environ,LENDING_RESEARCH_OUTPUT=str(dest),LENDING_BASE_OUTPUT=str((root/reference_output).resolve()))
    for stage in ['freeze','benchmark']:
        subprocess.run([sys.executable,'-u','-m','research.experiments','--stage',stage],cwd=root,env=env,check=True)
    published=pd.read_csv(source/'benchmark_metrics.csv')
    rebuilt=pd.read_csv(dest/'benchmark_metrics.csv')
    match=published.merge(rebuilt,on=['cohort','scope','model'],suffixes=('_published','_rebuilt'),validate='one_to_one')
    differences={metric:float((match[metric+'_published']-match[metric+'_rebuilt']).abs().max()) for metric in ['AP','ROC_AUC','Brier']}
    write_json(dest/'replay_comparison.json',{'same_recorded_model_specifications':True,
                'search_repeated':False,'metric_max_absolute_difference':differences,
                'benchmark_remains_reused':True,'reference_output':str(reference_output)})
    print('Fixed-configuration replay complete:',differences,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-results',default='outputs/research_v2')
    p.add_argument('--destination',default='outputs/research_fixed')
    p.add_argument('--reference-output',default='outputs')
    a=p.parse_args();replay(a.source_results,a.destination,a.reference_output)
