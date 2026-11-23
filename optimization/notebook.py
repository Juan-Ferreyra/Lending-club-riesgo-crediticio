"""Build notebook 07 from completed optimization artifacts; never rerun training.

Usage after benchmarking: ``python -m optimization.notebook``. The generated
notebook loads aggregate CSV/JSON results and makes figures. Historical notebooks
and frozen experiment code are not modified.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    source = dedent(source).strip()
    ast.parse(source)
    return nbf.v4.new_code_cell(source)


def build_notebook(output_directory="outputs/research_v3"):
    """Create cells in memory; artifact values are read only when executed."""
    cells = [
        md("""
        # 07. Optimización predictiva de Lending Club

        **Objetivo:** mejorar el ordenamiento del riesgo de incumplimiento del CatBoost de la fase 2, mediante datos de entrenamiento, variables, preparación, hiperparámetros y combinaciones de modelos. Esta es una bitácora reproducible de investigación y resultados; no constituye el informe final del curso.

        La clase positiva es **Charged Off** y la negativa **Fully Paid**. El alcance son préstamos **aceptados, a 36 meses, con desenlace conocido**. Los préstamos activos no se etiquetan como pagados ni como incumplidos. Tampoco se extrapolan los resultados a solicitantes rechazados.

        La métrica principal es **Average Precision (AP)** sobre las puntuaciones originales (`raw_AP`). AP resume qué tan bien aparecen los incumplimientos al principio del ranking. No es el porcentaje de préstamos correctamente clasificados. La evaluación siempre utiliza las mismas filas dentro de cada comparación; la prevalencia del incumplimiento afecta AP.

        **Límite de independencia:** 2015 y 2016 ya se observaron en fases anteriores. Son benchmarks temporales reutilizados. Congelar esta nueva selección antes de generar sus predicciones evita reajustarla con estos resultados, pero no convierte los años conocidos en una nueva prueba independiente.
        """),
        code("""
        import json
        import os
        from pathlib import Path
        import numpy as np
        import pandas as pd
        from IPython import get_ipython
        if get_ipython() is not None:
            get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown

        ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / 'experiments' / 'protocol_v3.json').exists())
        OUT = ROOT / OUTPUT_DIRECTORY
        FIG = OUT / 'figures'
        FIG.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'figure.dpi': 120, 'savefig.facecolor': 'white'})
        COLORS = {'positive': '#176D78', 'negative': '#B75335',
                  'reference': '#64748B', 'secondary': '#7962A5'}
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.max_colwidth', 100)
        pd.set_option('display.float_format', lambda x: f'{x:.6f}')

        def read_json(name, default=None):
            path = OUT / name
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

        def read_table(name):
            path = OUT / name
            if not path.exists() or path.stat().st_size == 0:
                print(f'No se generó el artefacto opcional {name}.')
                return pd.DataFrame()
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        def show_columns(frame, names):
            display(frame[[name for name in names if name in frame]].copy())

        def ap_column(frame):
            return 'raw_AP' if 'raw_AP' in frame else 'AP'

        def cohort_year(frame):
            return frame['cohort'].astype(str).str.extract(r'(20\\d{2})', expand=False)

        def save_show(fig, name, layout_rect=None):
            fig.tight_layout(rect=layout_rect)
            fig.savefig(FIG / name, dpi=170, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        def choose_model(frame, choices, description):
            names = set(frame['model'].astype(str))
            match = next((name for name in choices if name in names), None)
            if match is None:
                raise ValueError(f'No se identificó {description}; modelos disponibles: {sorted(names)}')
            return match

        protocol = json.loads((ROOT / 'experiments' / 'protocol_v3.json').read_text(encoding='utf-8'))
        selection = read_json('selection.json', {})
        benchmark = read_table('benchmark_metrics.csv')
        assert not benchmark.empty, 'Primero debe concluir la evaluación registrada.'
        reference_name = choose_model(benchmark,
            ['v2_frozen'],
            'el CatBoost V2 congelado')
        selected_name = choose_model(benchmark,
            ['selected', 'v3_selected', 'selected_v3', selection.get('selected', '')],
            'la estrategia seleccionada en desarrollo')
        benchmark['year'] = cohort_year(benchmark)
        primary = benchmark.loc[benchmark['year'].eq('2015')].copy()
        assert primary['n'].eq(258856).all(), 'La comparación principal debe conservar los 258.856 IDs.'
        assert primary['model'].nunique() == len(primary), 'Debe existir una fila por modelo en 2015.'
        print('Estrategia elegida en desarrollo:', selection.get('selected'))
        print('Referencia congelada:', reference_name)
        print('Evaluación principal: mismos 258.856 préstamos de 2015 a 36 meses.')
        print('No se ha creado una prueba final independiente nueva.')
        """.replace("OUTPUT_DIRECTORY", repr(output_directory))),
        md("""
        ## 1. Diseño temporal y comparabilidad

        La búsqueda usa dos cortes retrospectivos: entrenar hasta junio de 2013 y validar julio–diciembre de 2013; entrenar hasta diciembre de 2013 y validar enero–junio de 2014. Cada corte conserva sus IDs de validación entre configuraciones. Se limita el entrenamiento de búsqueda a 60.000 filas y la validación a 20.000 por corte, con semilla fija.

        Los candidatos promovidos se entrenan hasta junio de 2014 y se comparan en **todo 2014 Q3**. Ese bloque elige el modelo y los pesos de los ensambles. La elección no utiliza las nuevas predicciones de 2015/2016.

        El entrenamiento definitivo de V3 llega a **noviembre de 2014**. En la receta que comienza en 2012, esto incorpora **43.146 préstamos** frente al entrenamiento V2 que terminaba en septiembre. Un control conserva los parámetros y variables de V2 y cambia las fechas de entrenamiento para separar parte del aporte de más datos del aporte de nuevas configuraciones. Las recetas que comienzan en 2013 tienen su propio tamaño.

        Diciembre de 2014 se divide determinísticamente por la paridad de SHA256 del ID: una mitad calibra probabilidades y la otra fija umbrales. Son grupos disjuntos del **mismo mes**, no dos subperíodos cronológicos. El calibrador sigmoide exige pendiente positiva para conservar el ranking; la selección principal usa siempre puntuaciones sin calibrar.

        Las fechas de emisión no reconstruyen cuándo se conoció históricamente cada desenlace. Este diseño permite contrastar cohortes, pero no simula completamente una decisión en tiempo real.
        """),
        code("""
        display(pd.DataFrame(protocol['cv']))
        display(pd.DataFrame([
            {'etapa': 'Selección de candidatos', 'entrenamiento_hasta': protocol['candidate_training_end'],
             'evaluación_desde': protocol['selection_start'], 'evaluación_hasta': protocol['selection_end']},
            {'etapa': 'Modelo definitivo + calibración / umbral', 'entrenamiento_hasta': protocol['final_training_end'],
             'evaluación_desde': protocol['calibration_start'], 'evaluación_hasta': protocol['calibration_end']},
        ]))
        context_manifest = read_json('context_manifest.json', {})
        if context_manifest:
            display(pd.Series(context_manifest, name='Trazabilidad del contexto de solicitud'))
        training_manifest = read_json('training_manifest.json', {})
        if training_manifest:
            training_table = pd.DataFrame.from_dict(training_manifest, orient='index')
            show_columns(training_table, ['train_n', 'train_start', 'train_end', 'train_ids_sha256'])
        calibration_split = read_json('calibration_split.json', {})
        if calibration_split:
            display(pd.Series(calibration_split, name='Partición de diciembre'))
        calibration = read_table('calibration_selection.csv')
        if not calibration.empty:
            display(calibration)
        """),
        md("""
        ## 2. Qué se cambia y qué información se permite

        | Componente | Hipótesis contrastada | Control frente a fuga de información |
        |---|---|---|
        | Datos de entrenamiento | Empezar en 2012 o 2013; dar distinto peso a préstamos recientes | La validación no cambia ni recibe ponderaciones de entrenamiento |
        | Relaciones financieras | La capacidad disponible, composición de cuentas y carga relativa pueden añadir señal | Solo montos y variables del bureau observables al originar el préstamo |
        | Contexto de solicitud | Cargo/empleador declarado y área postal de tres dígitos pueden aportar contexto | Vocabulario, categorías raras y frecuencias se aprenden sin etiquetas y solo en entrenamiento |
        | Preparación por algoritmo | Los métodos lineales y kernels necesitan escalado; los árboles toleran faltantes de otra forma | Medianas, cuantiles y codificadores se ajustan exclusivamente en entrenamiento |
        | Hiperparámetros y ensambles | Regularización, capacidad y errores complementarios pueden mejorar AP | Configuraciones y pesos se eligen en desarrollo |

        Las veinte relaciones financieras nuevas incluyen préstamo/crédito disponible, cuentas activas/cuentas abiertas, morosidad histórica/cuentas totales y préstamo/ingreso residual estimado. Los denominadores inválidos producen faltantes, no infinitos. El indicador que suma DTI y amortización de principal es una **aproximación sin intereses**, no la cuota contractual.

        Se excluyen `grade`, `sub_grade`, `int_rate`, `installment`, pagos, recuperaciones, FICO posterior y datos de acuerdos de pago. Esto mantiene el objetivo de predecir desde la información de solicitud/bureau sin aprovechar la calificación o el precio ya asignados por Lending Club. El código también descarta columnas constantes o con más de 95% de ausencia en el entrenamiento correspondiente.

        El [contrato completo de las 24 variables nuevas](../experiments/feature_contract_v3.csv) registra entradas, fórmula y limitación de disponibilidad. A continuación se muestran cinco ejemplos. Un nombre intuitivo no sustituye su definición: el ingreso residual y el saldo hipotecario inferido son aproximaciones, y `emp_title` puede representar empleador o cargo según la cohorte. Una tasa baja de faltantes no garantiza que el significado del texto permanezca estable.
        """),
        code("""
        contract = pd.read_csv(ROOT / 'experiments' / 'feature_contract_v3.csv')
        intuitive = ['v3_loan_to_available_revolving', 'v3_active_revolving_share',
                     'v3_loan_to_annual_free_income', 'v3_principal_burden_plus_dti',
                     'v3_emp_title_category']
        display(contract.set_index('name').loc[intuitive].reset_index())
        if (OUT / 'context_availability.csv').exists():
            print('Disponibilidad del contexto en desarrollo: préstamos a 36 meses de 2012–2014.')
            display(read_table('context_availability.csv'))
        if (OUT / 'context_shift.csv').exists():
            print('Cambio entre entrenamiento y validación en los cortes de desarrollo.')
            context_shift = read_table('context_shift.csv')
            show_columns(context_shift, ['fold', 'field', 'train_start', 'train_end',
                'validation_start', 'validation_end', 'train_n', 'validation_n',
                'retained_category_n', 'train_missing_rate', 'validation_missing_rate',
                'validation_unseen_nonmissing_rate', 'validation_known_rare_rate',
                'validation_retained_category_rate', 'total_variation_distance'])
            print('Faltantes, textos no vistos, categorías raras conocidas y categorías retenidas usan todas las filas de validación como denominador.')
            print('La distancia de variación total describe cambios de distribución; no demuestra una causa ni fecha por sí sola un cambio de esquema.')
        """),
        md("""
        ## 3. Contrastes individuales y cambio combinado

        El bloque mantiene los parámetros V2 de LightGBM y las mismas filas de validación, sin parada anticipada. Incluye una referencia, seis contrastes que cambian **un factor a la vez** y la opción `combined`, que incorpora simultáneamente las relaciones financieras y el contexto. Los contrastes individuales permiten medir cada cambio respecto a la base. El resultado de `combined` corresponde al cambio conjunto de esos dos grupos y no se atribuye a una sola intervención.

        Una ganancia media puede ocultar que un período mejora y otro empeora. Por eso se conservan AP por corte, su diferencia frente a la base y los resultados negativos. Estas diferencias son descriptivas de los cortes utilizados; no prueban causalidad.
        """),
        code("""
        ablations = read_table('ablations.csv')
        if not ablations.empty:
            display(ablations.sort_values('CV_AP', ascending=False))
            base = ablations.loc[ablations['name'].eq('baseline')].iloc[0]
            delta_columns = []
            for col in [c for c in ablations if c.startswith('fold') and c.endswith('_AP')]:
                delta_col = col.replace('_AP', '_delta_AP')
                ablations[delta_col] = ablations[col] - base[col]
                delta_columns.append(delta_col)
            ablations['mejora_en_todos_los_cortes'] = ablations[delta_columns].gt(0).all(axis=1)
            show_columns(ablations, ['name', 'delta_AP', *delta_columns, 'mejora_en_todos_los_cortes'])
            ordered = ablations.loc[~ablations['name'].eq('baseline')].sort_values('delta_AP')
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.barh(ordered['name'], ordered['delta_AP'],
                    color=[COLORS['positive'] if x > 0 else COLORS['negative'] for x in ordered['delta_AP']])
            ax.axvline(0, color='#334155', linewidth=1)
            ax.set(xlabel='Cambio medio de AP frente a la base, mismas validaciones',
                   title='Cambios individuales y combinación de variables')
            save_show(fig, '01_ablaciones.png')
        """),
        md("""
        ## 4. Búsqueda conjunta y preparación específica

        Se ajustan **CatBoost, LightGBM, XGBoost, regresión logística, SVM lineal y Nyström RBF + SVM lineal**. Nyström aproxima el kernel; no es una SVC RBF exacta. La SVC RBF exacta ya se contrastó en la fase 2 con un presupuesto más reducido y no se repite aquí como si fuera un experimento nuevo.

        CatBoost trabaja con categorías nativas. LightGBM y XGBoost reciben categorías codificadas con un vocabulario aprendido en entrenamiento. Regresión logística y SVM incorporan imputación, escalado y codificación de categorías; una variante comprime magnitudes monetarias y recorta extremos con cuantiles del entrenamiento. `C` controla la regularización de los métodos lineales; `gamma` y el número de componentes controlan la aproximación RBF.

        La búsqueda combina estos cambios con los hiperparámetros y la receta temporal. Los árboles usan AP como criterio de parada anticipada sobre una validación sin ponderar. **AP no debe confundirse con un área trapezoidal interpolada bajo la curva precisión–recall.** Los pesos de clase o de recencia afectan el entrenamiento, no redefinen la población de evaluación.

        Los intentos podados y fallidos permanecen registrados. Un intento podado puede tener solo un corte y su valor parcial no se compara como si fuera el promedio final de los dos cortes. La búsqueda es finita y condicionada al presupuesto; no demuestra un óptimo global.

        **Cobertura adicional de preparación:** los **46 intentos HPO originales** finalizaron, pero los cuatro intentos de cada familia lineal/kernel usaron `native`; el muestreo no llegó a evaluar `log_clip` en ellas. Antes de abrir nuevas predicciones de benchmark se registraron [tres comparaciones adicionales de desarrollo](../experiments/preparation_followup_v3.json), una para regresión logística, SVM lineal y Nyström. Cada comparación conserva la mejor configuración HPO original de su familia y cambia únicamente la preparación a `log_clip`, sobre los mismos dos cortes e IDs.

        Se promueve la opción con mayor AP media al puesto de candidato Q3 ya reservado a esa familia, sin añadir puestos ni pesos de ensamble. Los resultados y fallos se conservan. La diferencia mide el **tratamiento conjunto de recorte y transformación logarítmica**, no el aporte aislado de cada operación. Una media positiva no implica que ambos cortes mejoren: se muestran las diferencias individuales después de comprobar los mismos IDs de entrenamiento y validación. Son tres pruebas adaptativas adicionales, separadas del conteo de 46 intentos originales; tampoco constituyen confirmación independiente.
        """),
        code("""
        trial_tables = []
        best_rows = []
        for family in protocol['trials']:
            frame = read_table(f'search_{family}.csv')
            if not frame.empty:
                frame['family'] = family
                trial_tables.append(frame)
            best = read_json(f'best_{family}.json', [])
            for rank, item in enumerate(best, start=1):
                spec = item['spec']
                best_rows.append({'family': family, 'rank': rank, 'trial': item.get('trial_number'),
                    'CV_AP': item['CV_AP'], 'feature_mode': spec['feature_mode'],
                    'preparation': spec['preparation'], 'start': spec['start'],
                    'half_life': spec['half_life'], 'params': spec['params'],
                    'fold_AP': [fold['AP'] for fold in item['folds']]})
        trials = pd.concat(trial_tables, ignore_index=True) if trial_tables else pd.DataFrame()
        best_table = pd.DataFrame(best_rows)
        if not trials.empty:
            states = pd.crosstab(trials['family'], trials['state'])
            display(states)
            print('Intentos HPO originales registrados:', len(trials), '; completos:', trials['state'].eq('COMPLETE').sum())
            rejected = trials.loc[~trials['state'].eq('COMPLETE')]
            if not rejected.empty:
                show_columns(rejected, ['family', 'number', 'state', 'value', 'duration',
                    'params_feature_mode', 'params_preparation', 'params_start', 'params_half_life'])
        if not best_table.empty:
            print('Mejores intentos de la búsqueda HPO original; las comparaciones adicionales de preparación se muestran por separado.')
            display(best_table.sort_values(['rank', 'CV_AP'], ascending=[True, False]))
            top = best_table.loc[best_table['rank'].eq(1)].sort_values('CV_AP')
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.2, 1]})
            axes[0].barh(top['family'], top['CV_AP'], color=COLORS['positive'])
            axes[0].set(xlabel='AP media de los dos cortes', title='Mejor intento HPO original por familia')
            if not trials.empty:
                states.plot.barh(stacked=True, ax=axes[1], color=['#176D78', '#B75335', '#C7CCD5'][:len(states.columns)])
                axes[1].set(xlabel='Intentos', ylabel='', title='Se conservan todos los estados')
                axes[1].legend(title='Estado', fontsize=8)
            save_show(fig, '02_busqueda.png')
        if (OUT / 'preparation_followup.csv').exists():
            preparation_followup = read_table('preparation_followup.csv')
            print('Comparaciones adicionales de preparación, fuera de los 46 intentos originales:')
            show_columns(preparation_followup, ['family', 'status', 'baseline_CV_AP',
                'log_clip_CV_AP', 'delta_AP', 'promoted'])
            print('Configuraciones adicionales registradas:', len(preparation_followup),
                  '; conservadas para promoción:', int(preparation_followup['promoted'].eq(True).sum()))
            preparation_fold_rows = []
            for result in read_json('preparation_followup.json', []):
                if result['status'] != 'COMPLETE' or 'candidate' not in result:
                    continue
                family = result['family']
                baseline_result = read_json(f'best_{family}.json', [])[0]
                baseline_folds = {fold['fold']: fold for fold in baseline_result['folds']}
                candidate_folds = {fold['fold']: fold for fold in result['candidate']['folds']}
                assert set(baseline_folds) == set(candidate_folds), f'Cortes distintos en {family}'
                for fold_id, candidate_fold in sorted(candidate_folds.items()):
                    baseline_fold = baseline_folds[fold_id]
                    for key in ['train_ids_sha256', 'validation_ids_sha256', 'train_n', 'validation_n']:
                        assert baseline_fold[key] == candidate_fold[key], f'{family}, corte {fold_id}: cambió {key}'
                    preparation_fold_rows.append({'family': family, 'fold': fold_id + 1,
                        'baseline_AP': baseline_fold['AP'], 'log_clip_AP': candidate_fold['AP'],
                        'delta_AP': candidate_fold['AP'] - baseline_fold['AP'],
                        'validation_n': candidate_fold['validation_n'], 'same_ids_verified': True})
            preparation_by_fold = pd.DataFrame(preparation_fold_rows)
            if not preparation_by_fold.empty:
                print('Diferencias por corte, con hashes de entrenamiento y validación coincidentes:')
                display(preparation_by_fold)
                mixed = preparation_by_fold.groupby('family')['delta_AP'].apply(lambda values: not values.gt(0).all())
                print('Familias con al menos un corte sin mejora positiva:', mixed.index[mixed].tolist())
                print('Promover por AP media no demuestra que la preparación mejore todos los períodos.')
        """),
        md("""
        ## 5. Selección posterior y ensambles

        Se promueven los mejores intentos a entrenamiento completo hasta junio de 2014. El número de árboles se fija mediante la media de las mejores iteraciones de los cortes, multiplicada por 1,15 y limitada por el máximo buscado. El control V2 conserva sus parámetros originales. La validación completa de Q3 decide entre ellos.

        **Ampliación adaptativa de desarrollo:** después de observar parte de la búsqueda y la auditoría del contexto se registró una [extensión al protocolo original](../experiments/development_followup_v3.json), antes de generar nuevas predicciones de 2015/2016. Añade **ocho candidatos**: dos controles CatBoost con los parámetros V2 y todas las mismas filas hasta junio —uno agrega contexto y otro contexto más relaciones financieras—, y seis combinaciones con el CatBoost V2, usando pesos V2 de 0,25, 0,50 y 0,75. Se mantiene el criterio de mayor AP sobre los mismos 40.595 préstamos de Q3.

        La presencia de empleadores entre los textos laborales antiguos y de cargos entre los recientes motivó explorar heterogeneidad temporal; **no demuestra que el contexto vaya a mejorar la predicción**. La pérdida observada de AP del contexto en los cortes tempranos permanece registrada. Esta extensión no formaba parte del protocolo original y aumenta el optimismo potencial de la selección de desarrollo; no es una confirmación independiente ni autoriza escoger pesos con 2015/2016.

        Un ensamble combina **predicciones**, no mezcla filas ni resultados reales. Si dos modelos asignan probabilidades distintas a un préstamo, una media ponderada produce una probabilidad combinada. Aquí se prueban pesos predefinidos, un promedio de tres familias de árboles y dos semillas de CatBoost. Los márgenes SVM se llevan a escala de probabilidad con una sigmoide aprendida a partir de predicciones de los cortes anteriores.

        Los pesos ganadores se eligen en Q3. Este bloque es de selección, por lo que su mejor puntuación puede ser optimista. Si la combinación no mejora, se conserva el resultado y no se justifica usarla solo por ser más compleja.
        """),
        code("""
        leaderboard = read_table('selection_leaderboard.csv')
        followup_path = ROOT / 'experiments' / 'development_followup_v3.json'
        if followup_path.exists():
            followup = json.loads(followup_path.read_text(encoding='utf-8'))
            display(pd.Series({key: followup.get(key) for key in [
                'registered_utc', 'role', 'observed_before_extension', 'new_benchmark_predictions_opened',
                'controlled_models', 'additional_candidates', 'selection_rule_unchanged']},
                name='Registro de la ampliación adaptativa de desarrollo'))
        if not leaderboard.empty:
            display(leaderboard.sort_values('AP', ascending=False))
            assert leaderboard['validation_n'].nunique() == 1
            single_count = int(leaderboard['type'].eq('single').sum())
            ensemble_count = int(leaderboard['type'].eq('ensemble').sum())
            print(f'Estrategias comparadas en Q3: {len(leaderboard)} = '
                  f'{single_count} modelos individuales + {ensemble_count} ensambles.')
            print('Filas de validación compartidas:', int(leaderboard['validation_n'].iloc[0]))
            print('Elegido antes de generar nuevas predicciones de benchmark:', selection['selected'])
            print('Pesos de los componentes:', selection['members'])
            chosen_specs = {name: selection['specs'][name] for name in selection['members']}
            display(pd.json_normalize(chosen_specs, sep='.').T.rename(columns={0: 'configuración'}))
            control_names = ['v2_catboost', 'catboost_context_control', 'catboost_combined_control']
            if set(control_names).issubset(set(leaderboard['model'])):
                controls = leaderboard.set_index('model').loc[control_names].copy()
                for key in ['train_ids_sha256', 'validation_ids_sha256', 'train_n', 'validation_n']:
                    assert controls[key].nunique(dropna=False) == 1, f'Los controles CatBoost difieren en {key}'
                base_spec = selection['specs']['v2_catboost']
                for name in control_names[1:]:
                    for key in ['family', 'params', 'start', 'half_life', 'seed', 'preparation']:
                        assert selection['specs'][name][key] == base_spec[key], f'{name} difiere del control en {key}'
                controls['delta_AP_vs_v2'] = controls['AP'] - controls.loc['v2_catboost', 'AP']
                print('Controles de variables: mismos parámetros CatBoost e IDs de entrenamiento y validación.')
                show_columns(controls.reset_index(), ['model', 'AP', 'delta_AP_vs_v2', 'train_n', 'validation_n'])
                print('El control combinado incorpora contexto y relaciones financieras juntos; su cambio total no se atribuye por separado a cada grupo ni a cada variable.')
            ensemble_rows = leaderboard.loc[leaderboard['type'].eq('ensemble')]
            if not ensemble_rows.empty:
                best_ensemble = ensemble_rows.loc[ensemble_rows['AP'].idxmax()]
                chosen_ap = float(leaderboard.loc[leaderboard['model'].eq(selection['selected']), 'AP'].iloc[0])
                ensemble_gap = chosen_ap - float(best_ensemble['AP'])
                print(f"Mejor ensamble en Q3: {best_ensemble['model']}, AP={best_ensemble['AP']:.8f}; "
                      f"AP de la estrategia elegida menos AP de ese ensamble: {ensemble_gap:+.8f}.")
                print('La diferencia es descriptiva del bloque de selección; no establece significación estadística.')
            top = leaderboard.nlargest(12, 'AP').sort_values('AP')
            fig, ax = plt.subplots(figsize=(11, 5.6))
            colors = [COLORS['positive'] if name == selection['selected'] else COLORS['reference'] for name in top['model']]
            ax.barh(top['model'], top['AP'], color=colors)
            old_row = leaderboard.loc[leaderboard['model'].eq('v2_catboost')]
            if not old_row.empty:
                ax.axvline(old_row['AP'].iloc[0], color=COLORS['negative'], linestyle='--', label='CatBoost V2, mismo Q3')
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), frameon=False)
            ax.set(xlabel='AP en validación completa 2014 Q3', title='Selección en desarrollo: doce primeras estrategias')
            save_show(fig, '03_seleccion_q3.png', layout_rect=(0, .04, 1, 1))
        """),
        md("""
        ## 6. Resultado principal: comparación sobre exactamente la misma población

        La tabla principal usa **258.856 préstamos de 2015 a 36 meses**, los mismos IDs del benchmark V2. Se compara contra el **CatBoost V2 congelado** y se mantiene separado el control que reutiliza sus parámetros con más meses de entrenamiento.

        **No se compara con AP = 0,4164**, porque ese resultado anterior mezclaba plazos de 36 y 60 meses. Tampoco se interpreta una AP mayor en 2016 como una mejora del modelo respecto a 2015: cambia la población y su prevalencia. En cada año se comparan modelos sobre sus mismas filas.

        `raw_AP` mide el ranking sin calibración; `AP` utiliza probabilidades calibradas cuando corresponda. `ROC_AUC` mide discriminación global y `Brier` evalúa el error de las probabilidades —menor es mejor—. `recall_at_top20` indica qué fracción de los incumplimientos aparece entre el 20% de préstamos con mayor riesgo estimado. Ninguna de estas cifras equivale directamente al beneficio económico de una política de crédito.
        """),
        code("""
        metric = ap_column(primary)
        primary = primary.sort_values(metric, ascending=False)
        show_columns(primary, ['model', 'n', 'prevalence', 'raw_AP', 'AP', 'ROC_AUC',
                               'Brier', 'precision', 'recall', 'recall_at_top20'])
        baseline_ap = float(primary.loc[primary['model'].eq(reference_name), metric].iloc[0])
        selected_ap = float(primary.loc[primary['model'].eq(selected_name), metric].iloc[0])
        delta_ap = selected_ap - baseline_ap
        relative_gain = 100 * delta_ap / baseline_ap
        print(f'CatBoost V2 congelado: AP={baseline_ap:.6f}')
        print(f'Selección V3: AP={selected_ap:.6f}; cambio={delta_ap:+.6f} ({relative_gain:+.2f}% relativo).')
        print('El conjunto completo de cambios incluye fechas de entrenamiento y calibración; no se atribuye todo a hiperparámetros.')
        fig, ax = plt.subplots(figsize=(11, max(4.5, .38 * len(primary))))
        ordered = primary.sort_values(metric)
        colors = [COLORS['positive'] if name == selected_name else
                  COLORS['negative'] if name == reference_name else COLORS['reference'] for name in ordered['model']]
        ax.barh(ordered['model'], ordered[metric], color=colors)
        ax.axvline(baseline_ap, color=COLORS['negative'], linestyle='--', linewidth=1)
        ax.set(xlabel='AP del ranking original · mismas 258.856 filas',
               title='Benchmark maduro reutilizado: 2015, plazo de 36 meses')
        save_show(fig, '04_benchmark_2015.png')
        print('Comparación secundaria de 2016: estrés temporal condicionado a tener desenlace.')
        show_columns(benchmark.loc[benchmark['year'].eq('2016')],
            ['model', 'n', 'prevalence', 'raw_AP', 'AP', 'ROC_AUC', 'Brier', 'recall_at_top20'])
        """),
        md("""
        ## 7. ¿La mejora se mantiene entre períodos?

        Se examinan las diferencias entre el modelo seleccionado y el CatBoost V2 **dentro de cada período**, sin comparar directamente AP entre poblaciones distintas. Las filas por período deben coincidir para ambos modelos. Una mejora total no garantiza que todos los meses o trimestres mejoren.

        Cuando está disponible, se vuelve a aplicar la estrategia ya elegida a los dos cortes anteriores con su configuración fijada. Es un diagnóstico sobre desarrollo reutilizado, posterior a la elección en Q3. Si un ensamble SVM necesita una transformación de probabilidades aprendida con etiquetas posteriores a esos cortes, no se reutiliza esa transformación para fingir una validación anterior: se registra la limitación.

        El bootstrap pareado vuelve a muestrear resultados de ambos modelos juntos; el remuestreo por mes conserva parte de la estructura temporal. Sus intervalos son **condicionales al benchmark reutilizado**: no corrigen la historia completa de decisiones adaptativas, no restauran independencia y no garantizan rendimiento futuro.

        La cohorte de 2016 tiene más préstamos sin desenlace en la fuente. Su resultado terminal se conserva como estrés temporal secundario y no se presenta como confirmación sobre todos los préstamos emitidos. Los años 2017/2018 tampoco crean una prueba madura por excluir sus préstamos activos.
        """),
        code("""
        stability = read_table('temporal_stability.csv')
        if not stability.empty:
            print('Estrategia ya seleccionada sobre cortes anteriores: diagnóstico, no selección nueva.')
            display(stability)
            print('Cortes anteriores con ΔAP positiva:', int(stability['delta_AP_vs_v2'].gt(0).sum()),
                  'de', len(stability))
        stability_limit = read_json('temporal_stability_limitation.json', {})
        if stability_limit:
            display(pd.Series(stability_limit, name='Límite de la revisión temporal anterior'))
        periods = read_table('benchmark_periods.csv')
        period_deltas = pd.DataFrame()
        if not periods.empty:
            display(periods)
            period_key = next((c for c in ['period', 'quarter', 'issue_quarter', 'issue_month', 'month'] if c in periods), None)
            if period_key is not None and {reference_name, selected_name}.issubset(set(periods['model'])):
                period_metric = ap_column(periods)
                index_cols = ['cohort', period_key] if 'cohort' in periods else [period_key]
                subset = periods.loc[periods['model'].isin([reference_name, selected_name])]
                values = subset.pivot(index=index_cols, columns='model', values=period_metric).dropna()
                counts = subset.pivot(index=index_cols, columns='model', values='n').loc[values.index]
                assert counts[reference_name].eq(counts[selected_name]).all()
                values['delta_AP'] = values[selected_name] - values[reference_name]
                period_deltas = values.reset_index()
                display(period_deltas)
                positive = int(period_deltas['delta_AP'].gt(0).sum())
                print(f'Mejora en {positive} de {len(period_deltas)} períodos registrados; no se ocultan cambios negativos.')
                if 'cohort' in period_deltas:
                    period_deltas['year'] = cohort_year(period_deltas)
                else:
                    period_deltas['year'] = period_deltas[period_key].astype(str).str[:4]
                years = sorted(period_deltas['year'].dropna().unique())
                fig, axes = plt.subplots(1, len(years), figsize=(max(8, 6 * len(years)), 4.5), squeeze=False)
                for ax, year in zip(axes[0], years):
                    group = period_deltas.loc[period_deltas['year'].eq(year)].sort_values(period_key)
                    ax.bar(group[period_key].astype(str), group['delta_AP'],
                           color=[COLORS['positive'] if x > 0 else COLORS['negative'] for x in group['delta_AP']])
                    ax.axhline(0, color='#334155', linewidth=1)
                    ax.tick_params(axis='x', rotation=60)
                    ax.set(ylabel='AP seleccionada − AP CatBoost V2', title=f'{year}: diferencias sobre las mismas filas')
                save_show(fig, '05_estabilidad_temporal.png')
        uncertainty = read_json('benchmark_uncertainty.json', {})
        paired_month_interval = None
        if uncertainty:
            display(pd.json_normalize(uncertainty, sep='.').T.rename(columns={0: 'valor'}))
            primary_uncertainty = uncertainty.get('2015_36_vs_v2_frozen', {})
            paired_month_interval = primary_uncertainty.get('issue_month_block', {}).get('paired_AP_delta_vs_baseline')
            if paired_month_interval is not None:
                print(f"IC bootstrap condicional del 95% de ΔAP, por bloques de mes: "
                      f"[{paired_month_interval['low']:+.6f}, {paired_month_interval['high']:+.6f}].")
        else:
            print('No se dispone de un intervalo de incertidumbre registrado; no se inventa uno.')
        """),
        md("""
        ## 8. Errores, umbrales y probabilidades

        Se revisan segmentos de FICO, ingresos y demás dimensiones registradas para detectar dónde se concentran los errores. **FNR** es la proporción de incumplimientos que el umbral deja pasar; **FPR** es la proporción de préstamos pagados que marca como riesgosos. Cambiar el umbral mueve ambos errores y no cambia el ranking subyacente.

        Los umbrales V3 se fijaron con la parte reservada de diciembre. El modelo V2 conserva su procedimiento anterior: una diferencia de FNR/FPR puede combinar cambios de ranking, calibración y política de umbral. Para el ranking se mantiene AP y la revisión del 20% con mayor riesgo. Segmentos pequeños o con pocos incumplimientos pueden producir estimaciones inestables; deben leerse junto con su tamaño.

        La calibración busca probabilidades más coherentes con la frecuencia de incumplimiento; no añade información al ranking. Una sigmoide de pendiente positiva conserva el orden salvo posibles empates numéricos. Por ello `raw_AP` permanece como criterio principal.
        """),
        code("""
        segments = read_table('benchmark_segment_errors.csv')
        if not segments.empty:
            focus = segments.loc[segments['model'].isin([reference_name, selected_name])].copy()
            if 'cohort' in focus:
                focus = focus.loc[cohort_year(focus).eq('2015')]
            show_columns(focus, ['cohort', 'model', 'dimension', 'segment', 'n', 'prevalence',
                'raw_AP', 'AP', 'precision', 'recall', 'FNR', 'FPR', 'low_support'])
            if 'low_support' in focus:
                print('Filas de segmentos marcadas con soporte insuficiente:', int(focus['low_support'].eq(True).sum()))
        if not calibration.empty:
            print('Calibración y umbrales: bloques y tamaños registrados, sin ajuste en el benchmark.')
            display(calibration)
        """),
        md("""
        ## 9. Interpretabilidad del predictor

        Esta importancia corresponde a los **modelos candidatos de la estrategia seleccionada, entrenados hasta junio de 2014**, evaluados en una muestra estratificada de **6.000 préstamos de Q3**, con semilla 42 y **tres permutaciones por grupo**. Su alcance es el predictor de desarrollo; el reajuste final hasta noviembre se evalúa en el benchmark posterior.

        Se altera un grupo de entradas y se mide cuánto cae AP manteniendo fijo el modelo. Todas las columnas de un grupo reciben la misma permutación y las relaciones financieras se recalculan a partir de las entradas modificadas. Una caída mayor indica mayor dependencia predictiva en este conjunto de desarrollo.

        Esta importancia **no demuestra que una variable cause el incumplimiento**. Predictores correlacionados pueden compartir señal, y una caída negativa pequeña puede aparecer por ruido. La tabla muestra el tamaño de evaluación y la dispersión de repeticiones cuando están disponibles. El análisis posterior no se utiliza para volver a seleccionar el modelo.
        """),
        code("""
        interpretation_scope = read_json('interpretability_scope.json', {})
        if interpretation_scope:
            display(pd.Series({key: interpretation_scope.get(key) for key in [
                'selected', 'model_stage', 'validation_start', 'validation_end',
                'complete_validation_n', 'validation_n', 'sampling', 'permutations', 'role', 'limits']},
                name='Alcance de la interpretación del predictor de desarrollo'))
        importance = read_table('interpretability.csv')
        if not importance.empty:
            display(importance)
            name_col = next((c for c in ['feature_group', 'feature', 'variable'] if c in importance), None)
            value_col = next((c for c in ['mean_AP_drop', 'AP_drop', 'importance_mean'] if c in importance), None)
            if name_col is not None and value_col is not None:
                plot = importance.copy()
                if 'model' in plot and selected_name in set(plot['model']):
                    plot = plot.loc[plot['model'].eq(selected_name)]
                plot = plot.nlargest(15, value_col).sort_values(value_col)
                error_col = next((c for c in ['permutation_sd', 'std_AP_drop', 'importance_std'] if c in plot), None)
                errors = plot[error_col].fillna(0) if error_col is not None else None
                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.barh(plot[name_col], plot[value_col], xerr=errors, color=COLORS['positive'], capsize=2)
                ax.axvline(0, color='#334155', linewidth=1)
                ax.set(xlabel='Caída de AP al permutar · barras: dispersión de repeticiones, si existe',
                       title='Dependencia predictiva de las variables; no efecto causal')
                save_show(fig, '06_interpretabilidad.png')
        """),
        md("""
        ## 10. Balance de la investigación y resultados que no mejoraron

        La conclusión se calcula a partir de los artefactos, sin asumir que una nueva familia o un ensamble debe ganar. Las ablaciones distinguen los cambios individuales del cambio conjunto `combined`; la selección Q3 decide la estrategia; el benchmark conocido contrasta el resultado posterior. Si los signos varían entre períodos, no se afirma una mejora uniforme.

        Para una **prueba final realmente independiente** hacen falta desenlaces maduros y no utilizados, además de información fiable sobre cuándo estuvieron disponibles. Volver a dividir préstamos ya observados, escoger los casos terminados de 2017/2018 o cambiar de población no resuelve esa necesidad.
        """),
        code("""
        if delta_ap > 0:
            print(f'La estrategia seleccionada supera puntualmente al CatBoost V2 en el benchmark 2015: ΔAP={delta_ap:+.6f}.')
            print('Es una mejora observada en una evaluación reutilizada, no una confirmación independiente nueva.')
        elif delta_ap < 0:
            print(f'La estrategia seleccionada no supera al CatBoost V2 en 2015: ΔAP={delta_ap:+.6f}.')
            print('Se conserva este resultado negativo. Elegir retrospectivamente otro finalista por esta tabla usaría de nuevo el benchmark para selección.')
        else:
            print('La estrategia seleccionada y el CatBoost V2 empatan en AP en esta evaluación.')
        if paired_month_interval is not None:
            if paired_month_interval['low'] > 0:
                print('El intervalo pareado por meses queda por encima de cero bajo este remuestreo condicional; la reutilización impide tratarlo como confirmación independiente.')
            elif paired_month_interval['high'] < 0:
                print('El intervalo pareado por meses queda por debajo de cero: este remuestreo condicional respalda una pérdida de AP frente a V2.')
            else:
                print('El intervalo pareado por meses incluye cero: no se distingue una ganancia positiva bajo este criterio de incertidumbre.')
        if not stability.empty:
            earlier_gains = stability['delta_AP_vs_v2'].gt(0)
            print(f"Cortes anteriores de desarrollo con ΔAP positiva: {earlier_gains.sum()} de {len(stability)}; son desarrollo reutilizado.")
        if not period_deltas.empty:
            gains = period_deltas['delta_AP'].gt(0)
            benchmark_years = ', '.join(sorted(period_deltas['year'].dropna().astype(str).unique()))
            if gains.all():
                print(f'ΔAP es positiva en los {len(gains)} trimestres evaluados del benchmark ({benchmark_years}); '
                      'esta afirmación se limita a esos trimestres y a sus poblaciones observadas.')
            else:
                print(f'En el benchmark ({benchmark_years}), {(~gains).sum()} de {len(gains)} trimestres '
                      'no presentan ganancia positiva de AP.')
        if not stability.empty and not earlier_gains.all():
            print(f'La revisión anterior conserva {(~earlier_gains).sum()} de {len(earlier_gains)} cortes '
                  'sin mejora positiva; el resultado no es uniforme en toda la historia.')
        if not ablations.empty:
            negatives = ablations.loc[~ablations['name'].eq('baseline') & ablations['delta_AP'].le(0)]
            print('Intervenciones que no mejoraron AP media en la ablación controlada:')
            display(negatives[['name', 'CV_AP', 'delta_AP']])
        if not leaderboard.empty:
            reference_q3 = leaderboard.loc[leaderboard['model'].eq('v2_catboost'), 'AP']
            if not reference_q3.empty:
                rejected_ensembles = leaderboard.loc[leaderboard['type'].eq('ensemble') &
                                                     leaderboard['AP'].le(reference_q3.iloc[0])]
                print('Ensambles que no superaron al control V2 en la validación Q3:')
                display(rejected_ensembles[['model', 'AP']])
        print('La tabla completa también conserva modelos perdedores, intentos podados y configuraciones fallidas.')
        print('No se ha demostrado el mejor modelo posible ni una mejora garantizada fuera de estas cohortes.')
        verification = read_json('verification.json', {})
        if verification:
            display(pd.Series({key: verification.get(key) for key in
                ['verified_utc', 'check_count', 'all_checks_passed', 'role']},
                name='Verificación científica registrada'))
            print('Las huellas actualizadas de los archivos se consultan en outputs/research_v3/verification.json; no se incrusta la huella del propio notebook.')
        receipt = read_json('benchmark_receipt.json', {})
        if receipt:
            display(pd.Series(receipt, name='Recibo de evaluación'))
        """),
        md("""
        ## Reproducción y referencias

        Este notebook lee los artefactos ya calculados; ejecutarlo no entrena modelos. El protocolo, los intentos y los archivos de selección conservan las decisiones. Los modelos y predicciones individuales se almacenan localmente conforme a `.gitignore`; los CSV/JSON agregados y este notebook documentan el resultado.

        Para repetir la investigación desde los datos preparados, revisar primero [el protocolo V3](../experiments/protocol_v3.json) y la interfaz `python -m optimization.experiments --help`. Cada etapa conserva trazabilidad. La protección del benchmark impide continuar ajustando sobre un directorio cuya evaluación ya se abrió; una reproducción debe usar otro directorio de salida. Repetir el entrenamiento completo exige tiempo de CPU y el archivo fuente local.

        - [Definición de Average Precision — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
        - [Calibración y conservación del ranking — scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).
        - [Métricas de clasificación — CatBoost](https://catboost.ai/docs/en/concepts/loss-functions-classification).
        - [CatBoost: tratamiento de categorías y sesgo de predicción — artículo original](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
        - [Parámetros de LightGBM — documentación oficial](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

        Estos fundamentos orientan las hipótesis; las tablas de este proyecto determinan cuáles aportaron mejora sobre sus datos.
        """),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "lending-club-research", "display_name": "Lending Club Research", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
    })


def generate(root, output_directory="outputs/research_v3", execute=True):
    root = Path(root).resolve()
    out = root / output_directory
    for name in ["selection.json", "benchmark_metrics.csv", "benchmark_receipt.json"]:
        if not (out / name).exists():
            raise FileNotFoundError(f"Finish and freeze the benchmark before generating notebook 07: {out / name}")
    receipt = json.loads((out / "benchmark_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("state") != "complete":
        raise ValueError("The benchmark receipt must be complete before generating notebook 07")
    notebook = build_notebook(output_directory)
    if execute:
        from ipykernel.kernelspec import install
        from nbclient import NotebookClient
        install(prefix=sys.prefix, kernel_name="lending-club-research", display_name="Lending Club Research")
        os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share" / "jupyter")
        NotebookClient(notebook, timeout=180, kernel_name="lending-club-research",
                       resources={"metadata": {"path": str(root)}}).execute()
    nbf.validate(notebook)
    target = root / "notebooks" / "07_optimizacion_predictiva.ipynb"
    nbf.write(notebook, target)
    print(json.dumps({"notebook": str(target), "executed": execute,
                      "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                      "embedded_figures": sum("image/png" in output.get("data", {})
                        for c in notebook.cells if c.cell_type == "code" for output in c.outputs)}, ensure_ascii=False))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.environ.get("LENDING_OPTIMIZATION_OUTPUT", "outputs/research_v3"))
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    generate(root, args.output_dir, execute=not args.no_execute)


if __name__ == "__main__":
    main()
