"""Build and execute artifact-reading notebooks; never reopens tests for tuning."""
import argparse
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from ipykernel.kernelspec import install


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def image(name):
    return code(f"display(Image(filename=str(OUT/'figures'/'{name}')))" )


def build(root, output):
    common = code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\nOUT = ROOT / "+repr(output)+"\npd.set_option('display.max_columns', 15)\npd.set_option('display.max_rows', 30)\nprotocol = json.loads((ROOT/'experiments'/'protocol.json').read_text(encoding='utf-8'))")
    intro = "Estos notebooks documentan experimentos ejecutados y leen sus artefactos. El entrenamiento completo se reproduce con `scripts/run_experiments.py`; las celdas de revisión no ajustan modelos ni cambian la selección. El informe final y la presentación corresponden a una etapa posterior."
    limitations = "**Alcance de la evaluación:** clasificación retrospectiva de `Charged Off` frente a `Fully Paid` entre préstamos aceptados con resultado registrado. El split es por fecha de emisión, con IDs externos a la muestra inicial en 2015 y 2016. El snapshot no reconstruye todas las fechas históricas de disponibilidad de las etiquetas. Deben leerse la censura y los plazos junto a las métricas."
    datasets = [
        ("02_limpieza_features.ipynb", [
            md("# Preparación, variables y contrato de evaluación\n\n"+intro), common,
            md("## Protocolo fijado antes de entrenar\n\n"+limitations),
            code("display(pd.DataFrame(protocol['cv']))\ndisplay(pd.DataFrame([{k: protocol[k] for k in ['candidate_training_end','selection_end','final_training_end','calibration_start','calibration_end','policy_start','policy_end']}]).T)"),
            md("## Trazabilidad de los datos\n\nEl archivo original no se añade a Git por su tamaño. El manifiesto contiene su SHA-256, las huellas de los IDs y el solapamiento con la muestra previa. El pipeline comprueba que los 115.000 IDs de esa muestra estén en el archivo original."),
            code("manifest = json.loads((OUT/'data_manifest.json').read_text(encoding='utf-8'))\ndisplay(pd.DataFrame(manifest['partitions']).T)\nprint('Filas de la fuente:', manifest['source_rows'])\nprint('SHA-256:', manifest['source_sha256'])\nprint('IDs adicionales excluidos por inspección de esquema:',manifest['schema_preview_excluded_ids'])"),
            md("## Auditoría de variables\n\nIngreso no positivo, FICO imposible y valores negativos incompatibles con el dominio se convierten a ausentes mediante reglas fijas. Se conservan extremos plausibles y se añaden razones, logaritmos e historia crediticia. Imputación, codificación y escalado se ajustan únicamente en entrenamiento. Los campos totalmente ausentes o constantes en entrenamiento se identifican aquí; disponer de más columnas en años posteriores no permite aprenderlas retrospectivamente."),
            code("audit = pd.read_csv(OUT/'analysis'/'feature_audit.csv')\nprint('Variables del conjunto ampliado:', len(audit))\nprint('No informativas en entrenamiento:', int(audit.uninformative_on_train.sum()))\ndisplay(audit.sort_values('missing_fraction_train', ascending=False))"),
            md("## Exclusiones y disponibilidad\n\nSe excluyen estados finales, pagos, recuperaciones, cobranza, hardship, settlement, últimas consultas/FICO posteriores al otorgamiento e identificadores. También grade, sub_grade, tasa, cuota y financiación asignados por Lending Club. La verificación se considera disponible al completar la evaluación de la solicitud. `issue_d` solo calcula la antigüedad crediticia; no entra directamente en el modelo. Los campos del bureau describen información previa y no la historia posterior de este préstamo."),
            md("## Cobertura de resultados terminales\n\nLa siguiente tabla se liberó después de congelar el modelo. La fracción no terminal excluida cuantifica por qué los resultados de cohortes recientes o a 60 meses requieren cautela."),
            code("display(pd.read_csv(OUT/'final'/'maturity_coverage.csv'))"),
            md("## Fuentes de investigación\n\nLa bitácora con referencias a artículos y decisiones está en `experiments/research_notes.md`. Las hipótesis de esos artículos se contrastan aquí con datos del proyecto; no se copian sus métricas como objetivos."),
        ]),
        ("03_modelos.ipynb", [
            md("# Comparación, hiperparámetros y evaluación congelada\n\n"+intro), common,
            md("## Familias y búsqueda temporal\n\nRegresión logística, bagging con Random Forest y boosting con LightGBM, XGBoost y CatBoost. Todas usan el mismo contrato de información previo al desembolso. La búsqueda utiliza exclusivamente los cortes temporales de desarrollo: hasta 80.000 filas de entrenamiento y 40.000 de validación; Random Forest explora hasta 30.000/20.000 por su costo computacional. La selección posterior reentrena todas las familias con las mismas filas completas y las compara sobre la misma cohorte. Los números de intentos incluyen intentos podados; los CSV preservan configuraciones y puntuaciones por corte."),
            code("families=['logistic','random_forest','lightgbm','xgboost','catboost']\nbest=[json.loads((OUT/f'best_{f}.json').read_text(encoding='utf-8')) for f in families]\ndisplay(pd.DataFrame(best)[['family','CV_AP','trial_count','fold_AP','fold_best_iterations']])\nfor row in best:\n    print(row['family'], row['params'])"),
            code("display(pd.read_csv(OUT/'analysis'/'model_inventory.csv'))"),
            md("## Selección independiente del ajuste\n\nLa selección de familia, semillas y pesos usa 2014 H1, sin acceso a pruebas 2015/2016. Se añade el baseline de 19 variables y una ablación de la mejor familia. Un ensamble requiere al menos 0,001 AP de mejora sobre el mejor modelo individual. La tabla también registra variación por semilla."),
            code("display(pd.read_csv(OUT/'selection_leaderboard.csv')[['name','family','AP','ROC_AUC','Brier']])\nselection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))\nprint('Elección previa a la prueba:', selection['selected'])\nprint('Pesos:', selection['weights'])"),
            image("comparacion_modelos.png"),
            md("## Calibración y política de umbral\n\nDespués de seleccionar, se reentrena hasta junio de 2014. Identidad, sigmoid e isotonic se ajustan en Q3 y se comparan por Brier en Q4. El umbral F1 se fija en Q4. No se presupone un costo real de rechazar o conceder créditos; los umbrales son referencias experimentales. La AP original del avance aleatorio no es una comparación pareada con estas cohortes."),
            code("cal=pd.read_csv(OUT/'calibration_selection.csv')\ndisplay(cal[['model','calibration','AP','ROC_AUC','Brier','log_loss']])\nlock=json.loads((OUT/'selection_lock.json').read_text(encoding='utf-8'))\nprint('Congelado UTC:', lock['frozen_utc'])\nprint('Modelo SHA-256:', lock['model_sha256'])\nprint('Calibrador seleccionado:', lock['calibration']['selected'])\nprint('Umbrales:', lock['thresholds']['selected'])"),
            image("calibracion_temporal.png"),
            md("## Evaluación externa por fecha\n\n"+limitations+"\n\nLas otras familias se muestran como comparadores congelados. El ganador sigue siendo el elegido en desarrollo aunque otra fila obtenga una métrica mayor en prueba. `AP_raw` permite distinguir discriminación original de empates introducidos por calibración isotónica."),
            code("test=pd.read_csv(OUT/'final'/'metrics.csv')\ndisplay(test.query(\"scope == 'all_resolved'\")[[ 'cohort','model','n','prevalence','AP','AP_raw','ROC_AUC','Brier','precision','recall','F1']])\ndisplay(test.query(\"model == 'selected'\")[[ 'cohort','scope','n','AP','ROC_AUC','Brier','recall_at_top20']])"),
            code("records=json.loads((OUT/'final'/'metrics.json').read_text(encoding='utf-8'))\nfor row in records:\n    if row['model']=='selected' and row['scope']=='all_resolved':\n        print('Matriz de confusión, cohorte', row['cohort'], 'umbral', round(row['threshold'],4))\n        display(pd.DataFrame(row['confusion_matrix'],index=['Real pagado','Real incumplido'],columns=['Predicho pagado','Predicho incumplido']))"),
            image("curvas_pr_roc.png"),
            md("## Incertidumbre pareada\n\nSe comparan las mismas filas con el baseline reentrenado. Los intervalos por bloques remuestrean meses de emisión completos; solo hay 12 bloques, por lo que no representan toda la incertidumbre macroeconómica. Los intervalos por filas tampoco cubren cambios de institución, política crediticia o población."),
            code("uncertainty=json.loads((OUT/'final'/'uncertainty.json').read_text(encoding='utf-8'))\nfor cohort, estimates in uncertainty.items():\n    print(cohort)\n    display(pd.DataFrame({kind: values for kind, values in estimates.items()}))\nprint(json.loads((OUT/'final'/'evaluation_receipt.json').read_text(encoding='utf-8')))"),
        ]),
        ("04_analisis_errores.ipynb", [
            md("# Errores, estabilidad e interpretación\n\n"+intro), common,
            md("## Estabilidad temporal\n\nEstos diagnósticos se calculan con el predictor congelado. Los resultados 2016 son una prueba de deterioro temporal, condicionados a que el préstamo ya tenga resultado terminal en el snapshot."),
            image("estabilidad_temporal.png"),
            code("display(pd.read_csv(OUT/'analysis'/'distribution_drift.csv'))"),
            md("## Lectura de los resultados observados\n\nUn buen ordenamiento de riesgo no garantiza que el nivel de probabilidad siga calibrado al cambiar de año. La siguiente celda cuantifica la diferencia entre riesgo medio predicho y frecuencia observada en cada cohorte, sin volver a ajustar el calibrador."),
            code("reliability=pd.read_csv(OUT/'analysis'/'reliability.csv')\nfor year, group in reliability.query(\"model == 'selected' and mode == 'calibrated'\").groupby('cohort'):\n    predicted=(group.mean_prediction*group.n).sum()/group.n.sum()\n    observed=(group.observed_rate*group.n).sum()/group.n.sum()\n    print(f'{year}: riesgo medio predicho {predicted:.2%}; incumplimiento observado {observed:.2%}; diferencia observado-predicho {observed-predicted:+.2%}.')\nprint('Estas medias describen préstamos con resultado terminal; la cobertura limita su extrapolación.')"),
            md("## Errores por segmentos\n\nSe usa el umbral fijado en desarrollo. Se muestran FNR/FPR con intervalos Wilson. `low_support` marca segmentos con menos de 300 préstamos o menos de 30 ejemplos de alguna clase. Diferencias geográficas o socioeconómicas son señales para investigar: faltan atributos protegidos y solicitantes rechazados para una evaluación de equidad completa."),
            code("errors=pd.read_csv(OUT/'analysis'/'segment_errors.csv')\ndisplay(errors.query(\"cohort == 2015 and dimension in ['term','FICO','income','home_ownership','application_type','verification_status']\")[[ 'dimension','segment','n','prevalence','AP','FNR','FPR','low_support']])\ndisplay(errors.query(\"cohort == 2015 and dimension == 'purpose' and low_support == False\").sort_values('FNR', ascending=False)[['segment','n','AP','FNR','FPR']])"),
            image("errores_segmentos.png"),
            code("term_errors=errors.query(\"cohort == 2015 and dimension == 'term'\")\nfor row in term_errors.itertuples():\n    print(f'{row.segment}: se omite {row.FNR:.1%} de los incumplimientos y se marca como riesgoso {row.FPR:.1%} de los préstamos pagados con el umbral común de desarrollo.')\nprint('El umbral común produce compromisos distintos por plazo. Estos resultados son diagnósticos; no se optimizan nuevos umbrales con la prueba.')"),
            md("## Importancia global del predictor final\n\nLa permutación se realiza sobre grupos de entradas originales; las variables derivadas se recalculan automáticamente. Se conservan correlaciones dentro del grupo permutado, pero se rompen dependencias con otros grupos. Su importancia no es un efecto causal."),
            image("importancia_permutacion.png"),
            code("importance=pd.read_csv(OUT/'analysis'/'grouped_permutation.csv')\ndisplay(importance.head(6))\nprint('Grupos cuya alteración más reduce AP:', ', '.join(importance.head(3)['group']))\nprint('La permutación mide dependencia predictiva en esta muestra; puede verse afectada por correlaciones entre grupos.')"),
            md("## Atribuciones globales y casos locales\n\nTree SHAP explica cada componente de boosting en log-odds. Se verifica su suma aditiva. La combinación de probabilidades y su calibración posterior no admiten sumar esas contribuciones como si fueran cambios de la probabilidad final. Los metadatos registran unidades, referencia y error de aditividad; la regresión usa descomposición lineal si fuera seleccionada."),
            image("shap_global.png"),
            code("metadata=json.loads((OUT/'analysis'/'explanation_metadata.json').read_text(encoding='utf-8'))\ndisplay(pd.DataFrame(metadata).T)\ncases=json.loads((OUT/'analysis'/'local_cases.json').read_text(encoding='utf-8'))\ndisplay(pd.DataFrame([{k:v for k,v in c.items() if k != 'top_contributions'} for c in cases]))"),
            image("casos_locales.png"),
            md("## Resistencia a entradas ausentes\n\nSe ocultan aleatoriamente campos al predecir, usando el mismo modelo congelado. Es una perturbación sintética y no una garantía frente a faltantes sistemáticos del mundo real."),
            code("display(pd.read_csv(OUT/'analysis'/'missing_input_stress.csv')[[ 'scenario','fraction','AP','Brier','AP_delta_vs_intact','Brier_delta_vs_intact']])\ndisplay(pd.read_csv(OUT/'final'/'threshold_policies.csv')[[ 'cohort','policy','threshold','flag_rate','precision','recall','F1']])"),
        ]),
    ]
    install(prefix=sys.prefix, kernel_name="lending-club-research", display_name="Lending Club Research (Python 3.12)")
    os.environ["JUPYTER_PATH"] = str(Path(sys.prefix)/"share"/"jupyter")
    for filename, cells in datasets:
        # Fresh cells avoid shared execution metadata between notebooks.
        cells = [nbf.from_dict(dict(c)) for c in cells]
        nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "lending-club-research", "display_name": "Lending Club Research", "language": "python"}, "language_info": {"name": "python", "version": "3.12"}})
        NotebookClient(nb, timeout=180, kernel_name="lending-club-research", resources={"metadata": {"path": str(root)}}).execute()
        nbf.validate(nb)
        nbf.write(nb, root/"notebooks"/filename)
        print(f"Executed notebook saved: {filename}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output)
