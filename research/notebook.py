"""Generate diagnostic figures and execute notebook 06 from completed V2 artifacts."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat as nbf
from nbclient import NotebookClient
from ipykernel.kernelspec import install
from research.experiments import OUT, read_json
from src.utils import write_json

FIG = OUT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':140,'savefig.facecolor':'white'})


def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/name,dpi=170,bbox_inches='tight')
    plt.close(fig)


def figures():
    FIG.mkdir(exist_ok=True)
    cov=pd.read_csv(OUT/'outcome_coverage_bounds.csv')
    frame=cov[cov.year.ge(2012)]
    fig,ax=plt.subplots(figsize=(9,4.5))
    for term,color in [('36 months','#096B68'),('60 months','#BA5A35')]:
        g=frame[frame.term.eq(term)]
        ax.plot(g.year,g.terminal_fraction,marker='o',label=term,color=color)
    ax.set(xlabel='Año de emisión',ylabel='Fracción con resultado terminal',ylim=(0,1.05),
           title='Cobertura observada: un año más reciente no implica mejor prueba')
    ax.legend();save(fig,'cobertura.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,term in zip(axes,['36 months','60 months']):
        g=frame[frame.term.eq(term)]
        ax.fill_between(g.year,g.all_accepted_default_lower_bound,g.all_accepted_default_upper_bound,
                        color='#096B68',alpha=.15,label='Límites entre todos los aceptados')
        ax.plot(g.year,g.conditional_default_rate,color='#BA5A35',marker='o',label='Tasa entre terminales')
        ax.set(xlabel='Año de emisión',ylabel='Proporción de incumplimiento',title=term,ylim=(0,1))
        ax.legend(fontsize=8)
    save(fig,'tasas_y_limites.png')
    missing=pd.read_csv(OUT/'missing_by_year.csv')
    top=read_json(OUT/'audit_findings.json')['unlearnable_or_95pct_missing_in_old_training']
    top=[f for f in top if f in set(missing.feature)][:16]
    pivot=missing[missing.feature.isin(top)&missing.year.ge(2012)].pivot(index='feature',columns='year',values='missing_fraction').reindex(top)
    fig,ax=plt.subplots(figsize=(10,6))
    im=ax.imshow(pivot.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(len(pivot.columns)),pivot.columns)
    ax.set_yticks(range(len(pivot.index)),pivot.index)
    ax.set(xlabel='Año de emisión',title='Disponibilidad de variables: fracción ausente por año')
    fig.colorbar(im,ax=ax,label='Fracción ausente')
    save(fig,'disponibilidad_variables.png')
    screen=pd.read_csv(OUT/'recipe_screen.csv').sort_values('CV_AP')
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(screen.recipe,screen.CV_AP,color='#096B68')
    for y,value in enumerate(screen.CV_AP):ax.text(value+.0005,y,f'{value:.4f}',va='center',fontsize=9)
    ax.set(xlabel='AP media de dos validaciones: misma población a 36 meses',xlim=(0,screen.CV_AP.max()*1.1),
           title='Recetas de datos con hiperparámetros de LightGBM fijos')
    save(fig,'recetas_cv.png')
    factors=pd.read_csv(OUT/'one_factor_ablation.csv').pivot(index='variant',columns='fold',values='AP')
    delta=factors.subtract(factors.loc['original'],axis=1).mean(axis=1).drop('original').sort_values()
    fig,ax=plt.subplots(figsize=(9,4))
    ax.barh(delta.index,delta,color=['#096B68' if x>=0 else '#BA5A35' for x in delta])
    ax.axvline(0,color='#555',linewidth=.8)
    ax.set(xlabel='Cambio medio de AP frente al original; dos cortes',title='Ablaciones de un factor: existencia de ausentes no implica daño')
    save(fig,'ablaciones_factor.png')
    match=pd.read_csv(OUT/'matched_budget_validation.csv').sort_values('AP')
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.barh(match.family,match.AP,color=['#BA5A35' if n.startswith('svm') else '#096B68' for n in match.family])
    ax.set(xlabel='AP en 2014 Q3: mismas 12.000 filas de entrenamiento',
           title='Control del presupuesto de datos: SVM y familias anteriores')
    save(fig,'presupuesto_igual.png')
    bench=pd.read_csv(OUT/'benchmark_metrics.csv')
    selected=read_json(OUT/'selection_lock.json')['selected']
    primary=bench.query("cohort==2015").sort_values('AP')
    fig,ax=plt.subplots(figsize=(10,max(5,.35*len(primary)+1.5)))
    ax.barh(primary.model,primary.AP,color=['#096B68' if n==selected else '#BA5A35' if n=='original_selected' else '#91A4AF' for n in primary.model])
    for y,value in enumerate(primary.AP):ax.text(value+.0005,y,f'{value:.4f}',va='center',fontsize=9)
    ax.set(xlabel='AP sobre los mismos préstamos a 36 meses',xlim=(0,primary.AP.max()*1.12),
           title='2015: benchmark reutilizado; no es una nueva prueba independiente')
    save(fig,'benchmark_maduro.png')
    scope=pd.read_csv(OUT/'information_scope_sensitivity.csv')
    fig,ax=plt.subplots(figsize=(8,3.5))
    labels=['Solicitud independiente','Después de clasificación de LC']
    ax.barh(labels,scope.AP,color=['#096B68','#BA5A35'])
    for y,value in enumerate(scope.AP):ax.text(value+.002,y,f'{value:.4f}',va='center')
    ax.set(xlabel='AP en validación 2014 Q3; mismas filas y parámetros',xlim=(0,scope.AP.max()*1.15),
           title='Sensibilidad al alcance: agregar grade/tasa cambia la información')
    save(fig,'alcance_informacion.png')
    chosen=bench[(bench.cohort==2015)&bench.model.eq(selected)].iloc[0]
    old=bench[(bench.cohort==2015)&bench.model.eq('original_selected')].iloc[0]
    uncertainty=read_json(OUT/'benchmark_uncertainty.json')['2015_36_vs_original_ensemble']
    controls=pd.read_csv(OUT/'hyperparameter_controls_validation.csv').pivot(index='family',columns='setting',values='AP')
    median_delta=float(factors.loc['median_indicators_only'].mean()-factors.loc['original'].mean())
    best_svm=primary[primary.model.str.startswith('svm')].sort_values('AP',ascending=False).iloc[0]
    write_json(OUT/'conclusions.json',{
        'selected_on_development':selected,
        'recipe':read_json(OUT/'selection.json')['recipe'],
        'screened_recipe':read_json(OUT/'selected_recipe.json')['recipe'],
        'primary_scope':'same 2015 36-month cohort, reused benchmark',
        'new_AP':float(chosen.AP),'original_ensemble_AP':float(old.AP),'paired_AP_delta':float(chosen.AP-old.AP),
        'new_ROC_AUC':float(chosen.ROC_AUC),'original_ROC_AUC':float(old.ROC_AUC),
        'new_Brier':float(chosen.Brier),'original_Brier':float(old.Brier),
        'new_precision':float(chosen.precision),'new_recall':float(chosen.recall),
        'new_recall_at_top20':float(chosen.recall_at_top20),'original_recall_at_top20':float(old.recall_at_top20),
        'median_imputation_mean_CV_AP_delta':median_delta,
        'hyperparameter_validation_AP_delta':{name:float(row['new_search_parameters']-row['previous_parameters']) for name,row in controls.iterrows()},
        'best_svm_on_full_reused_benchmark_descriptive_only':str(best_svm.model),
        'best_svm_reused_benchmark_AP':float(best_svm.AP),
        'month_block_conditional_CI_AP_delta':uncertainty['issue_month_block']['paired_AP_delta_vs_baseline'],
        'unlearnable_old_training_columns':read_json(OUT/'audit_findings.json')['count'],
        'sample_TVD_year_vs_terminal':float(pd.read_csv(OUT/'sample_representativeness.csv').query("population=='all_terminal_2007_2018' and dimension=='issue_year'").TVD.iloc[0]),
        'sample_TVD_year_vs_all_accepted':float(pd.read_csv(OUT/'sample_representativeness.csv').query("population=='all_accepted_2007_2018' and dimension=='issue_year'").TVD.iloc[0]),
        'validation_AP_gain_with_LC_assigned_information':float(scope.AP.iloc[1]-scope.AP.iloc[0]),
        'LC_information_eligible_for_main_model':False,
        'pristine_independent_final_test':False,
        'conditional_CI_does_not_account_for_benchmark_reuse':True,
        'original_predictor_and_final_metrics_modified':False,
    })
    sample=pd.read_csv(OUT/'sample_representativeness.csv')
    terminal_year_tvd=float(sample.query("population=='all_terminal_2007_2018' and dimension=='issue_year'").TVD.iloc[0])
    rates=cov[cov.term.eq('36 months')].set_index('year')
    selection=pd.read_csv(OUT/'selection_leaderboard.csv')
    best_svm_selection=selection[selection.name.str.startswith('svm')].AP.max()
    results=[
        {'hypothesis':'Faltantes estructurales afectan la información disponible',
         'finding':'Confirmado: 17 columnas constantes o >95% ausentes en entrenamiento anterior.',
         'predictive_test':f'Imputación con indicadores: cambio medio CV AP {median_delta:+.6f}. La estructura por sí sola no prueba daño predictivo.'},
        {'hypothesis':'Excluir préstamos activos limita la generalización',
         'finding':f"Confirmado: cobertura 2015/36 = {rates.loc[2015,'terminal_fraction']:.4%}; 2016/36 = {rates.loc[2016,'terminal_fraction']:.4%}.",
         'predictive_test':'No se identifica el resultado futuro de activos. Se define población madura; no se inventan etiquetas.'},
        {'hypothesis':'La frecuencia de incumplimiento cambia entre años maduros',
         'finding':f"Confirmado descriptivamente: 2013/36 = {rates.loc[2013,'conditional_default_rate']:.4%}; 2015/36 = {rates.loc[2015,'conditional_default_rate']:.4%}.",
         'predictive_test':'Se prueban fechas de entrenamiento y validación temporal. No se atribuye causalidad al año.'},
        {'hypothesis':'La muestra inicial está mal tomada de su población terminal',
         'finding':f'No respaldado por las marginales revisadas: TVD año = {terminal_year_tvd:.6f}.',
         'predictive_test':'No demuestra representatividad conjunta ni permite extrapolar a todos los aceptados o rechazados.'},
        {'hypothesis':'Ajustar hiperparámetros mejora con las mismas filas y entradas',
         'finding':'; '.join(f"{name}: AP {row['new_search_parameters']-row['previous_parameters']:+.6f}" for name,row in controls.iterrows()),
         'predictive_test':'Comparación controlada en 2014 Q3; no equivale a confirmación externa independiente.'},
        {'hypothesis':'SVM supera las otras estrategias en la selección temporal',
         'finding':f'Mejor SVM AP = {best_svm_selection:.6f}; mejor candidato AP = {selection.AP.max():.6f}.',
         'predictive_test':'Incluye controles con mismas 12000 filas y benchmark común de 20000; el costo de RBF queda explícito.'},
        {'hypothesis':'La nueva selección supera al ensamble anterior en la cohorte comparable',
         'finding':f"AP {old.AP:.6f} → {chosen.AP:.6f}; cambio {chosen.AP-old.AP:+.6f}.",
         'predictive_test':'2015/36 reutilizado, no prueba nueva independiente. Ver intervalo pareado y métricas de calibración.'},
    ]
    pd.DataFrame(results).to_csv(OUT/'hypothesis_results.csv',index=False)


def notebook():
    root=Path(__file__).resolve().parents[1]
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    def pic(name):return code("display(Image(filename=str(OUT/'figures'/"+repr(name)+")))")
    cells=[
        md("""# 06. Diagnóstico de datos, hiperparámetros y máquinas de vectores de soporte

## Pregunta de investigación

¿Qué parte del desempeño depende del tratamiento y selección de datos, de los cambios temporales y del ajuste de hiperparámetros, manteniendo una población de evaluación definida?

Esta fase contrasta hipótesis posteriores al primer experimento. **2015/2016 ya fueron observados**: sus nuevas predicciones son benchmarks retrospectivos reutilizados, no una nueva prueba final independiente. La población principal son préstamos aceptados a 36 meses; el archivo no representa directamente solicitantes rechazados. No se asignan etiquetas de pago/incumplimiento a préstamos activos. El informe académico sigue pendiente.

Este notebook lee resultados guardados. Su ejecución no vuelve a ajustar modelos ni a buscar configuraciones."""),
        code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nOUT=ROOT/"+repr(str(OUT))+"\npd.set_option('display.max_columns',16)\npd.set_option('display.max_rows',35)\nprotocol=json.loads((ROOT/'experiments'/'protocol_v2.json').read_text(encoding='utf-8'))\nconclusion=json.loads((OUT/'conclusions.json').read_text(encoding='utf-8'))"),
        md("""## 1. Faltantes y valores inválidos

La auditoría revisa ingreso no positivo, deuda/utilización negativas, FICO fuera del dominio o con límites inconsistentes y monto no positivo. Son reglas concretas, no una certificación de calidad de todas las columnas. Los valores plausibles extremos se estudian separadamente mediante clipping ajustado con entrenamiento.

Se identifican 17 columnas constantes o con más de 95% de ausencia en el entrenamiento anterior. Varias comenzaron a estar disponibles en años posteriores. Una columna completamente vacía no enseña una relación útil, pero puede afectar indirectamente a una variable derivada que cuenta ausentes. Esto motiva una ablación y una prueba de sensibilidad; no demuestra automáticamente una mejora predictiva.

La imputación por mediana con indicadores se compara con el tratamiento nativo de ausentes. No se supone que rellenar todo sea superior: ausencia de una mora previa y ausencia de un campo nuevo pueden tener significados distintos."""),
        code("invalid=pd.read_csv(OUT/'invalid_by_year.csv')\nsummary=invalid.query('year<=2015').groupby('rule')[['n','invalid_n']].sum()\nsummary['fraction']=summary.invalid_n/summary.n\ndisplay(summary)\nprint(json.loads((OUT/'audit_findings.json').read_text(encoding='utf-8'))['unlearnable_or_95pct_missing_in_old_training'])"),
        pic('disponibilidad_variables.png'),
        md("""## 2. Préstamos activos: cobertura y límites de identificación

Las coberturas siguientes corresponden al censo de la fuente, antes de excluir IDs de la muestra inicial; por eso difieren ligeramente de las coberturas de la prueba original.

La tasa entre préstamos terminados no identifica la tasa final de todos los emitidos. Para mostrar qué se sabe sin inventar etiquetas, se calculan límites: mínimo = incumplimientos conocidos / aceptados; máximo = (incumplimientos conocidos + no terminales) / aceptados. No son intervalos de confianza, sino extremos lógicos sin supuestos adicionales. Que un préstamo permanezca activo no significa que sea malo.

La cohorte 2015 a 36 meses tiene seguimiento casi completo. Las cohortes 2017/2018 no ofrecen una prueba madura para este objetivo. Seleccionar solamente sus resultados conocidos favorece casos que terminaron pronto."""),
        pic('cobertura.png'),
        code("coverage=pd.read_csv(OUT/'outcome_coverage_bounds.csv')\ndisplay(coverage.query('year>=2013'))"),
        pic('tasas_y_limites.png'),
        md("""La cobertura también se revisa por mes. Como sensibilidad secundaria se eligen meses de emisión de 2016 a 36 meses con cobertura terminal de al menos 99,5% en el censo de la fuente, antes de excluir la muestra inicial. La selección se basa en seguimiento, no en AP. Esto permite comparar cohortes más maduras sin convertirlas en pruebas nuevas independientes: sus préstamos ya estaban en el benchmark anterior."""),
        code("monthly=pd.read_csv(OUT/'monthly_outcome_coverage.csv')\ndisplay(monthly[monthly.issue_month.str.startswith('2016') & monthly.term.eq('36 months')])\nprint(json.loads((OUT/'secondary_mature_months.json').read_text(encoding='utf-8')))"),
        md("""## 3. Diferencias entre años y selección de población

La frecuencia de incumplimiento a 36 meses aumenta entre 2013 y 2015, años con cobertura completa o casi completa. Eso documenta diferencias entre cohortes, sin identificar su causa. En 2016 y posteriores, la censura impide interpretar la tasa condicionada a terminar como si fuera la tasa final de todos los préstamos.

Se comparan ingreso, deuda, FICO, utilización y cuentas por año, plazo y estado terminal. La diferencia media estandarizada (SMD) describe diferencias observables; no prueba causalidad ni corrige la selección. Las diferencias de seguimiento y de variables pueden coexistir."""),
        code("smd=pd.read_csv(OUT/'selection_bias_smd.csv')\ndisplay(smd.query(\"year==2015 and term=='60 months'\"))\nfeatures=pd.read_csv(OUT/'population_features_by_year_term_status.csv')\ndisplay(features.query(\"year>=2013 and term=='36 months' and terminal==True and feature in ['annual_inc','dti','fico_range_low']\")[[ 'year','feature','n','median','q10','q90','missing']])\nprint('Cambio de covariables entre todos los aceptados a 36 meses, sin condicionar a terminar:')\ndisplay(pd.read_csv(OUT/'temporal_population_shift.csv'))"),
        md("""## 4. ¿La muestra inicial era poco representativa?

Se separan dos poblaciones: todos los resultados terminales y todos los préstamos aceptados. La distancia de variación total (TVD) vale cero cuando las distribuciones coinciden; aumenta con su diferencia. Aquí se compara año, plazo y estado.

La muestra aleatoria de 115.000 reproduce de cerca las distribuciones de la población terminal de la que se extrajo. La diferencia frente a todos los aceptados es mucho mayor, especialmente por año. El problema no es necesariamente haber tomado una muestra aleatoria: es extrapolar una muestra de préstamos terminados a otra población. La TVD de unas pocas variables no demuestra representatividad conjunta completa."""),
        code("display(pd.read_csv(OUT/'sample_representativeness.csv'))"),
        md("""También se comparan ingresos, DTI, monto, FICO, utilización y cuentas, además de vivienda, propósito, verificación y tipo de solicitud, contra la población terminal de la fuente hasta 2015 y dentro de 2015 a 36 meses. Se muestran diferencias estandarizadas de medias (SMD), distancia KS y TVD categórica. La muestra es parte del censo de referencia: son distancias descriptivas, sin pruebas de significación que supongan muestras independientes. Estas variables amplían la auditoría marginal; no certifican representatividad de toda combinación de características."""),
        code("display(pd.read_csv(OUT/'sample_feature_representativeness.csv'))"),
        md("""## 5. Recetas y ablaciones de datos

Con parámetros de LightGBM fijos se comparan nueve recetas. La validación siempre contiene préstamos a 36 meses de las mismas fechas; cambian las filas de entrenamiento y el tratamiento de variables. Las recetas estables eliminan columnas no aprendibles, ajustan el conteo de ausentes a campos disponibles en entrenamiento y eliminan derivados redundantes. Algunas recetas combinan factores; por eso se añade una ablación separada que cambia un factor a la vez.

Los recortes por cuantiles, medianas y selección de columnas se aprenden exclusivamente con entrenamiento. Dos cortes permiten una comprobación inicial; diferencias pequeñas no deben describirse como evidencia universal o causal."""),
        md("""La auditoría adicional observó aproximadamente 52% de ausencia en 2012 para saldo total, límite revolvente y cuentas revolventes, frente a 0% en 2013. Antes de generar nuevas predicciones del benchmark se añadieron variantes de entrenamiento desde 2013 para contrastar esa disponibilidad. CatBoost y LightGBM conservan sus parámetros anteriores; las variantes SVM conservan sus parámetros ajustados. Se seleccionan sobre la misma validación Q3. Este control de fecha cambia filas de entrenamiento, no la población de evaluación, y no implica que usar menos datos sea siempre mejor."""),
        pic('recetas_cv.png'),
        code("display(pd.read_csv(OUT/'recipe_screen.csv').sort_values('CV_AP',ascending=False))\nprint('Receta base del cribado:',conclusion['screened_recipe'])\nprint('Receta del modelo elegido finalmente en validación:',conclusion['recipe'])\nprint('Control adicional desde 2013:',json.loads((OUT/'availability_2013_cv.json').read_text(encoding='utf-8')))"),
        pic('ablaciones_factor.png'),
        code("display(pd.read_csv(OUT/'one_factor_ablation.csv').pivot(index='variant',columns='fold',values='AP'))"),
        md("""## 6. Hiperparámetros y SVM

Se ajustan LightGBM y CatBoost, y se añaden tres estrategias: **LinearSVC**, **SVC con RBF exacto** y **Nyström RBF + LinearSVC**. La última aproxima un kernel; no es SVC RBF exacta. SVM utiliza imputación, escalado y codificación ajustados en entrenamiento. Se exploran C, ponderación de clases y, en RBF, gamma; en Nyström también el número de componentes.

`C` controla la penalización de errores o violaciones del margen: valores altos pueden ajustar más el entrenamiento, pero requieren comprobar generalización. En RBF, `gamma` controla qué tan local es la influencia de los ejemplos. La ponderación balanceada modifica penalizaciones por clase; no garantiza probabilidades calibradas. La función de decisión de SVM es una puntuación de margen, no una probabilidad de incumplimiento.

La SVC exacta usa hasta 12.000 filas por su costo. Su búsqueda valida hasta 8.000 filas por corte; las otras estrategias utilizan hasta 60.000/20.000. Los CV con distinto presupuesto no son una comparación definitiva. La selección posterior utiliza toda la cohorte 2014 Q3 a 36 meses. Se incorpora además una comparación de todas las familias con exactamente las mismas 12.000 filas de entrenamiento. Las probabilidades SVM se calibran en un bloque temporal separado.

La predicción RBF exacta de 40.595 préstamos tardó entre 146 y 163 segundos. Antes de abrir las nuevas predicciones externas se fijó otro control de costo: **todos los modelos** se comparan sobre los mismos 20.000 préstamos estratificados por año, con semilla 42. Las variantes RBF exactas no seleccionadas se evalúan solo en esa muestra común. Los demás modelos, y siempre el ganador seleccionado en desarrollo, también se evalúan en la cohorte completa. Sus tablas se muestran separadas: AP de muestras distintas no se compara directamente.

Los intentos podados no completan necesariamente ambos cortes. Se selecciona entre intentos completos y se registran advertencias de convergencia. La regresión logística, Random Forest y XGBoost permanecen como comparadores con sus parámetros anteriores y la nueva receta."""),
        code("families=['lightgbm','catboost','svm_linear','svm_rbf','svm_nystroem']\nbest=[json.loads((OUT/f'best_{f}.json').read_text(encoding='utf-8')) for f in families]\ndisplay(pd.DataFrame(best)[['family','CV_AP','trial_count','fold_AP']])\nfor row in best: print(row['family'],row['params'])\nretry=json.loads((OUT/'svm_solver_retries.json').read_text(encoding='utf-8'))\nprint('Repeticiones adicionales de las mismas configuraciones RBF con mayor límite del solver:',len(retry['additional_same_configuration_solver_retries']))\ndisplay(pd.read_csv(OUT/'selection_leaderboard.csv').sort_values('AP',ascending=False))"),
        pic('presupuesto_igual.png'),
        code("display(pd.read_csv(OUT/'matched_budget_validation.csv').sort_values('AP',ascending=False))\ndisplay(pd.read_csv(OUT/'hyperparameter_controls_validation.csv'))\nprint('Efecto del tamaño de entrenamiento: mismas variables, parámetros y validación')\ndisplay(pd.read_csv(OUT/'training_size_validation.csv'))"),
        md("""## 7. Benchmark maduro reutilizado y comparación pareada

El ajuste definitivo llega a septiembre de 2014, la calibración se ajusta en octubre-noviembre y su método/umbral se eligen en diciembre. Es una separación temporal retrospectiva: las fechas históricas de disponibilidad de etiquetas no se pueden reconstruir completamente.

La comparación principal usa los mismos IDs a 36 meses de 2015 que la sensibilidad madura anterior. **No se compara su AP con 0,4164**, que correspondía a mezclar ambos plazos. El nuevo modelo fue elegido en desarrollo antes de generar estas nuevas predicciones; aun así, los años ya habían sido observados. Un intervalo pareado no restaura independencia ni corrige toda la selección adaptativa.

La calibración isotónica puede introducir empates; `raw_AP` registra la discriminación anterior a calibrar. La receta, fechas de entrenamiento, calibración y modelo cambian en conjunto; la diferencia total no se atribuye exclusivamente a hiperparámetros."""),
        pic('benchmark_maduro.png'),
        code("common=pd.read_csv(OUT/'common_budget_benchmark.csv')\nprint('Comparación común: mismos 20.000 préstamos por año para todos, incluida SVM RBF exacta.')\ndisplay(common[['cohort','model','n','AP','ROC_AUC','Brier']].sort_values(['cohort','AP'],ascending=[True,False]))\nprint('La siguiente tabla usa las cohortes completas; omite RBF exactas no seleccionadas.')"),
        code("benchmark=pd.read_csv(OUT/'benchmark_metrics.csv')\ndisplay(benchmark[['cohort','model','n','AP','raw_AP','ROC_AUC','Brier','precision','recall','recall_at_top20']])\nprint('Seleccionado en desarrollo:',conclusion['selected_on_development'])\nprint('Cambio pareado de AP frente al ensamble anterior:',conclusion['paired_AP_delta'])\nprint('IC condicional por bloques:',conclusion['month_block_conditional_CI_AP_delta'])\ndisplay(pd.read_csv(OUT/'benchmark_calibration_drift.csv'))\nsegments=pd.read_csv(OUT/'benchmark_segment_errors.csv')\ndisplay(segments.query(\"cohort==2015 and dimension in ['FICO','income'] and low_support==False\")[[ 'model','dimension','segment','n','AP','FNR','FPR']])"),
        code("mature=pd.read_csv(OUT/'mature_secondary_metrics.csv')\nif mature.empty:\n    print('Ningún mes de 2016 cumplió el criterio de cobertura fijado.')\nelse:\n    display(mature[['model','months','n','prevalence','AP','ROC_AUC','Brier','precision','recall']])"),
        md("""### Interpretación del predictor seleccionado

Se permutan grupos de entradas originales en 5.000 préstamos de la validación temporal 2014 Q3, con tres repeticiones. La caída de AP describe cuánto depende el ordenamiento de cada grupo en ese conjunto; **no identifica causas del incumplimiento**. FICO bajo y alto se permutan juntos. Las variables derivadas se recalculan desde las entradas permutadas; predictores correlacionados pueden repartirse la importancia. Esta sensibilidad posterior no se usa para elegir ni reajustar el modelo."""),
        code("display(pd.read_csv(OUT/'interpretability_validation.csv')[['model','feature_group','validation_n','baseline_AP','mean_AP_drop','permutation_sd']])"),
        md("""## 8. Sensibilidad a la información disponible

Se comparan las mismas filas y parámetros de LightGBM en validación 2014 Q3, con y sin `grade`, `sub_grade` e `int_rate` asignados por Lending Club. Es un diagnóstico de información adicional después de que la plataforma evaluó/clasificó el préstamo. Una mejora aquí no debe presentarse como mejor predicción independiente usando únicamente la solicitud. Este experimento no puede ganar la selección principal.

Como referencia del valor ya contenido en la evaluación de la plataforma, se añaden un ranking por tasa y una frecuencia de incumplimiento por subgrado aprendida en entrenamiento y suavizada hacia la media con 100 observaciones previas equivalentes. Su objetivo es evitar atribuir al nuevo algoritmo una señal que ya viene incorporada en el puntaje/precio de Lending Club.

También se mantiene el ensamble original y se ocultan campos de bureau que estaban ausentes en su entrenamiento, sobre una submuestra fija del benchmark reutilizado. Esto investiga sensibilidad al cambio de disponibilidad y al conteo derivado de ausentes; no valida una política de ocultar datos en producción."""),
        pic('alcance_informacion.png'),
        code("display(pd.read_csv(OUT/'information_scope_sensitivity.csv'))\ndisplay(pd.read_csv(OUT/'platform_score_baselines.csv'))\ndisplay(pd.read_csv(OUT/'original_model_missingness_probe.csv')[['scenario','n','AP','Brier','recall_at_top20']])"),
        md("""## 9. Conclusiones y trabajo pendiente

Se distinguen problemas de validez de población de mejoras predictivas. La cobertura desigual y la disponibilidad temporal de variables están verificadas. Los errores de dominio comprobados son escasos; no se justifica prometer una mejora grande por corregirlos. La muestra representa razonablemente sus marginales terminales, pero no a todos los aceptados.

Las tablas anteriores determinan si las recetas, el ajuste y SVM mejoran realmente las métricas bajo un alcance comparable. Una hipótesis puede estar respaldada, no respaldada o seguir sin identificarse: no se fuerza una conclusión favorable.

La comparación con información de la plataforma cambia el problema y se mantiene separada. Para una nueva confirmación final hacen falta desenlaces maduros no utilizados, junto con fechas fiables de resultado. Sin ellos no se demuestra desempeño entre todos los préstamos activos ni se reconstruye un backtest operativo.

Fuentes y decisiones: [bitácora de esta fase](../experiments/research_v2_notes.md), [protocolo v2](../experiments/protocol_v2.json), [documentación SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [aproximación de kernels](https://scikit-learn.org/stable/modules/kernel_approximation.html)."""),
        code("with pd.option_context('display.max_colwidth',180):\n    display(pd.read_csv(OUT/'hypothesis_results.csv'))"),
        code("print('Seleccionado en desarrollo:',conclusion['selected_on_development'])\nprint(f\"AP en la cohorte madura reutilizada: {conclusion['original_ensemble_AP']:.4f} → {conclusion['new_AP']:.4f}; cambio {conclusion['paired_AP_delta']:+.4f}.\")\ninterval=conclusion['month_block_conditional_CI_AP_delta']\nprint(f\"Intervalo bootstrap condicional por meses: [{interval['low']:+.4f}, {interval['high']:+.4f}].\")\nif conclusion['paired_AP_delta']<=0:\n    print('La nueva selección no supera al ensamble anterior en AP sobre esta cohorte; no se justifica afirmar una mejora por ese criterio.')\nelif interval['low']<=0:\n    print('Hay mejora puntual, pero el intervalo condicional incluye cero. No se afirma una mejora robusta ni una confirmación independiente.')\nelse:\n    print('El cambio es positivo bajo este remuestreo condicional. La reutilización del benchmark y las decisiones previas limitan la confirmación externa.')\nprint('Cambios de AP por hiperparámetros, con mismas filas y variables:',conclusion['hyperparameter_validation_AP_delta'])\nprint('Cambio medio de AP al imputar con indicadores:',conclusion['median_imputation_mean_CV_AP_delta'])\nprint('Ganancia de AP al agregar información asignada por LC, solo en diagnóstico de validación:',conclusion['validation_AP_gain_with_LC_assigned_information'])\nprint('Ese cambio de información no es elegible como mejora del modelo independiente principal.')\ndisplay(pd.Series(conclusion,name='resultado'))\nprint(json.loads((OUT/'benchmark_receipt.json').read_text(encoding='utf-8')))")
    ]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'lending-club-research','display_name':'Lending Club Research','language':'python'},'language_info':{'name':'python','version':'3.12'}})
    install(prefix=sys.prefix,kernel_name='lending-club-research',display_name='Lending Club Research')
    os.environ['JUPYTER_PATH']=str(Path(sys.prefix)/'share'/'jupyter')
    NotebookClient(nb,timeout=180,kernel_name='lending-club-research',resources={'metadata':{'path':str(root)}}).execute()
    nbf.validate(nb);nbf.write(nb,root/'notebooks'/'06_diagnostico_datos_svm.ipynb')
    # Append an explicit later audit note without replacing the first experiment's outputs.
    original_path=root/'notebooks'/'02_limpieza_features.ipynb'
    original=nbf.read(original_path,as_version=4)
    marker='## Revisión posterior de calidad y disponibilidad — fase 2'
    note=md(marker+"\n\nLa auditoría del archivo original identificó **17 variables constantes o con más de 95% de ausencia** en el entrenamiento anterior. Varias se incorporan en años posteriores; se revisa también el conteo derivado de faltantes para que no confunda disponibilidad del esquema con riesgo. En las reglas de dominio comprobadas hasta 2015 se detectaron dos ingresos no positivos, sin evidencia de errores masivos en esos campos.\n\nEstas observaciones no implican que eliminar o imputar variables mejore automáticamente la predicción. Se contrastaron tratamiento nativo, mediana con indicadores, recorte de extremos, disponibilidad estable y selección de cohortes mediante validación temporal, además de SVM lineal y con kernel. El análisis completo, los resultados y sus límites están en [06_diagnostico_datos_svm.ipynb](06_diagnostico_datos_svm.ipynb).\n\nLos resultados anteriores de este notebook documentan el primer experimento; la revisión posterior se conserva separada. Las pruebas de 2015/2016 ya observadas se identifican como benchmarks reutilizados en la segunda fase.")
    existing=next((i for i,c in enumerate(original.cells) if c.cell_type=='markdown' and c.source.startswith(marker)),None)
    if existing is None:original.cells.append(note)
    else:original.cells[existing]=note
    nbf.validate(original);nbf.write(original,original_path)
    print('Executed notebook 06 saved.',flush=True)


if __name__=='__main__':
    assert read_json(OUT/'followup_complete.json')['state']=='complete'
    figures();notebook()
