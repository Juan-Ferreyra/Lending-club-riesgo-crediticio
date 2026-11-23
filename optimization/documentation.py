"""Update repository navigation and experiment notes from verified results."""
from pathlib import Path
import json
import nbformat
import pandas as pd

OUT = Path('outputs/research_v3')
START = '<!-- v3-results-start -->'
END = '<!-- v3-results-end -->'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    verification = read('verification.json')
    assert verification['all_checks_passed']
    selection = read('selection.json')
    metrics = pd.read_csv(OUT/'benchmark_metrics.csv')
    primary = metrics[metrics.cohort.eq(2015)].set_index('model')
    baseline, selected = primary.loc['v2_frozen'], primary.loc[selection['selected']]
    control = primary.loc['v2_catboost']
    leaderboard = pd.read_csv(OUT/'selection_leaderboard.csv')
    single_count = int(leaderboard.type.eq('single').sum())
    ensemble_count = int(leaderboard.type.eq('ensemble').sum())
    delta = selected.raw_AP-baseline.raw_AP
    uncertainty = read('benchmark_uncertainty.json')['2015_36_vs_v2_frozen']
    interval = uncertainty['issue_month_block']['paired_AP_delta_vs_baseline']
    states = {}
    for path in OUT.glob('search_*.csv'):
        for name, count in pd.read_csv(path).state.value_counts().items():
            states[name] = states.get(name, 0)+int(count)
    period = pd.read_csv(OUT/'benchmark_periods.csv')
    period = period[period.model.eq(selection['selected']) & period.cohort.eq(2015)]
    improved_periods = int(period.delta_AP_vs_v2.gt(0).sum())
    stability_path = OUT/'temporal_stability.csv'
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    if not stability.empty:
        stability = stability[stability.model.eq(selection['selected'])]
    if not stability.empty:
        improved_earlier = int(stability.delta_AP_vs_v2.gt(0).sum())
        earlier_text = (f'En la revisión de los cortes anteriores de desarrollo, AP mejora en '
                        f'**{improved_earlier} de {len(stability)}**; estos cortes también se reutilizaron.')
        if improved_earlier < len(stability):
            earlier_text += ' La mejora no es uniforme en toda la historia.'
    else:
        earlier_text = 'No se dispone de la revisión de los cortes anteriores; no se afirma estabilidad en ellos.'
    members = ', '.join(f'{weight:.0%} `{name}`' for name, weight in selection['members'].items())
    state_text = ', '.join(f"{states.get(state, 0)} {label}" for state, label in
                          [('COMPLETE', 'completos'), ('PRUNED', 'podados'), ('FAIL', 'fallidos')])
    status = ('La estrategia preseleccionada mejora AP en esta cohorte reutilizada.' if delta > 0 else
              'La estrategia preseleccionada no mejora AP en esta cohorte; se conserva el CatBoost V2 como referencia vigente.')
    section = f'''{START}
### Tercera fase: optimización conjunta de la predicción

El [notebook 07](notebooks/07_optimizacion_predictiva.ipynb) documenta la investigación ejecutada sobre datos, variables, preparación por algoritmo, hiperparámetros y ensambles. Se registraron ocho recetas del bloque de ablación, incluida la referencia, y **{sum(states.values())} intentos de búsqueda** en seis familias: {state_text}. Se añadieron tres controles de preparación y ocho candidatos de contexto/ensamble en ampliaciones adaptativas de desarrollo documentadas antes del benchmark. La selección compara **{len(leaderboard)} estrategias en la misma cohorte de 2014 Q3: {single_count} modelos individuales y {ensemble_count} ensambles**. La estrategia elegida fue: **{members}**. Los ensayos sin mejora, fallidos o descartados se conservan.

La referencia es el CatBoost V2, sobre los mismos **258.856 préstamos de 2015 a 36 meses**:

| Estrategia | AP | ROC-AUC | Recall al revisar el 20% |
|---|---:|---:|---:|
| CatBoost V2 congelado | {baseline.raw_AP:.6f} | {baseline.raw_ROC_AUC:.6f} | {baseline.recall_at_top20:.2%} |
| Parámetros V2, entrenamiento ampliado hasta noviembre | {control.raw_AP:.6f} | {control.raw_ROC_AUC:.6f} | {control.recall_at_top20:.2%} |
| Estrategia V3 seleccionada en desarrollo | {selected.raw_AP:.6f} | {selected.raw_ROC_AUC:.6f} | {selected.recall_at_top20:.2%} |

{status} El cambio frente al CatBoost V2 es **{delta:+.6f} AP** ({delta/baseline.raw_AP:+.2%} relativo), con intervalo bootstrap pareado por mes **[{interval['low']:+.6f}; {interval['high']:+.6f}]**. El cambio es positivo en {improved_periods} de los {len(period)} trimestres de 2015; se muestran todos los períodos, incluidos los negativos. {earlier_text} 2016 se mantiene como estrés afectado por desenlaces aún no observados.

El entrenamiento final desde 2012 llega hasta noviembre de 2014: 299.588 préstamos, 43.146 más que V2. Diciembre se divide mediante hash de ID en grupos disjuntos de calibración y umbral; son grupos del mismo mes. El control con parámetros V2 permite separar el aporte de las filas adicionales al ranking. Sus diferencias de Brier también incluyen el cambio de calibración. AP se calcula sobre puntuaciones originales y se contrasta con probabilidades de una sigmoide de pendiente positiva.

**No es una nueva prueba final independiente.** Los años 2015/2016 ya habían sido observados; ni el intervalo ni la congelación recuperan esa independencia. El archivo disponible tampoco proporciona una cohorte futura madura sin usar. El alcance, controles y límites se detallan en el [protocolo V3](experiments/protocol_v3.json), la [bitácora](experiments/research_v3_notes.md) y el [contrato de variables](experiments/feature_contract_v3.csv). No se compara este AP con 0,4164 de la población original que mezclaba plazos.

Para reconstruir los predictores con las especificaciones publicadas, se necesitan los datos preparados, el archivo original en `data/accepted_2007_to_2018Q4.csv.gz` y los artefactos de referencia V1. La referencia V2 debe ser su reconstrucción fija en `outputs/research_fixed`, obtenida con `python -m research.replay --destination outputs/research_fixed --reference-output outputs/reproduction`, como se explicó en la segunda fase. Una búsqueda nueva de V2 puede elegir otros parámetros y no sirve como referencia de coincidencia exacta.

```powershell
$env:LENDING_BASE_OUTPUT = 'outputs/reproduction'
.\\.venv\\Scripts\\python.exe -m optimization.replay --destination outputs/optimization_fixed --v2-output outputs/research_fixed
```

Para repetir la búsqueda completa de V3 en una salida nueva se utiliza la salida **completa** de V2, `outputs/research_reproduction`, que incluye sus candidatos Q3 y archivos de búsqueda. La reconstrucción fija de V2 solo genera los predictores finales y no contiene esos archivos de desarrollo:

```powershell
$env:LENDING_BASE_OUTPUT = 'outputs/reproduction'
$env:LENDING_V2_OUTPUT = 'outputs/research_reproduction'
$env:LENDING_OPTIMIZATION_OUTPUT = 'outputs/optimization_reproduction'
.\\.venv\\Scripts\\python.exe -m optimization.experiments --stage prepare
.\\.venv\\Scripts\\python.exe -m optimization.experiments --stage all
.\\.venv\\Scripts\\python.exe -m optimization.analysis --stage all
.\\.venv\\Scripts\\python.exe -m optimization.notebook --output-dir outputs/optimization_reproduction
```

La reconstrucción fija evita depender de la trayectoria adaptativa de TPE. Repetir los experimentos no crea datos independientes. Los modelos, predicciones individuales y bases de estudio permanecen fuera de Git; los resultados agregados, parámetros, figuras y verificaciones sí se publican. El informe final continúa pendiente.
{END}'''
    path = Path('README.md')
    original = path.read_text(encoding='utf-8')
    if START in original:
        before, rest = original.split(START, 1)
        _, after = rest.split(END, 1)
        updated = before+section+after
    else:
        marker = '\n---\n\n## Avance inicial de semana 1'
        assert marker in original
        updated = original.replace(marker, '\n'+section+'\n'+marker, 1)
    nav = '- `notebooks/07_optimizacion_predictiva.ipynb`: optimización conjunta, ensambles, controles y estabilidad temporal.\n'
    if nav not in updated:
        updated = updated.replace('- `outputs/`:', nav+'- `outputs/`:', 1)
    latest = '**Última fase de optimización:** [Notebook 07](notebooks/07_optimizacion_predictiva.ipynb), con comparación sobre préstamos a **36 meses**.'
    if latest not in updated:
        title, separator, history = updated.partition('\n')
        assert separator and title.startswith('# '), 'Expected the README title on its first line'
        updated = title+'\n\n'+latest+'\n\n'+history.lstrip('\n')
    path.write_text(updated, encoding='utf-8')
    note = '''### Continuación experimental: optimización predictiva

La investigación posterior está en [07_optimizacion_predictiva.ipynb](07_optimizacion_predictiva.ipynb). Compara nuevas variables, preparación específica, datos de entrenamiento, hiperparámetros y ensambles con la referencia CatBoost de la fase 2. Los resultados históricos de este notebook se conservan; 2015/2016 son benchmarks reutilizados en esa continuación, no una nueva prueba independiente. El informe final aún no se ha elaborado.'''
    for name in ['02_limpieza_features.ipynb', '05_explicacion.ipynb', '06_diagnostico_datos_svm.ipynb']:
        path = Path('notebooks')/name
        notebook = nbformat.read(path, as_version=4)
        if not any('### Continuación experimental: optimización predictiva' in c.source for c in notebook.cells):
            notebook.cells.append(nbformat.v4.new_markdown_cell(note))
            nbformat.write(notebook, path)
    print('README and historical notebook navigation updated from verified V3 results.')


if __name__ == '__main__':
    main()
