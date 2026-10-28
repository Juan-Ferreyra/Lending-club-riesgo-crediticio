"""Build an executed conceptual notebook from the frozen research artifacts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import nbformat as nbf
from nbclient import NotebookClient
from ipykernel.kernelspec import install


# Coordinates are schematic; no node represents an observed applicant.
DIAGRAMS = {
    "logistic": ("Regresión logística: de las variables a la probabilidad", 6, [
        ("income", 2, 4.8, "Ingresos\nestandarizados"),
        ("debt", 2, 3, "Deuda e\nhistorial crediticio"),
        ("category", 2, 1.2, "Categorías\ncodificadas"),
        ("score", 6, 3, "Combinación lineal\n$z = \\beta_0 + \\sum_j \\beta_j x_j$"),
        ("sigmoid", 10, 3, "Función logística\nConvierte z a [0, 1]"),
        ("prob", 14, 3, "Probabilidad\nde incumplimiento"),
    ], [("income", "score"), ("debt", "score"), ("category", "score"), ("score", "sigmoid"), ("sigmoid", "prob")]),
    "tree": ("Un árbol de decisión: particiones sucesivas de los datos", 7, [
        ("root", 8, 5.8, "¿Deuda / ingreso\nsupera un corte?"),
        ("left", 4, 3.8, "No: evaluar\nhistorial crediticio"),
        ("right", 12, 3.8, "Sí: evaluar\nplazo solicitado"),
        ("a", 2, 1.4, "Hoja A\nRiesgo estimado"),
        ("b", 6, 1.4, "Hoja B\nRiesgo estimado"),
        ("c", 10, 1.4, "Hoja C\nRiesgo estimado"),
        ("d", 14, 1.4, "Hoja D\nRiesgo estimado"),
    ], [("root", "left"), ("root", "right"), ("left", "a"), ("left", "b"), ("right", "c"), ("right", "d")]),
    "forest": ("Random Forest: diversidad de árboles y promedio de probabilidades", 6, [
        ("data", 2, 3, "Mismo préstamo\nVariables de entrada"),
        ("t1", 6.5, 4.8, "Árbol 1\np = 10%"),
        ("t2", 6.5, 3, "Árbol 2\np = 30%"),
        ("t3", 6.5, 1.2, "Árbol 3\np = 20%"),
        ("avg", 11, 3, "Promedio\n(10 + 30 + 20) / 3"),
        ("prob", 14.5, 3, "Probabilidad\ncombinada: 20%"),
    ], [("data", "t1"), ("data", "t2"), ("data", "t3"), ("t1", "avg"), ("t2", "avg"), ("t3", "avg"), ("avg", "prob")]),
    "xgboost": ("XGBoost: optimización secuencial con regularización", 6, [
        ("initial", 2, 4.3, "Puntuación inicial\nF anterior"),
        ("gradient", 6, 4.3, "Gradientes de la pérdida\n¿Qué debe mejorar?"),
        ("tree", 10, 4.3, "Nuevo árbol\ncon regularización"),
        ("update", 14, 4.3, "Actualizar puntuación\nF nuevo = F anterior\n+ tasa × árbol"),
        ("repeat", 10, 1.5, "Repetir con\nárboles sucesivos"),
        ("prob", 14, 1.5, "Función logística\nProbabilidad final"),
    ], [("initial", "gradient"), ("gradient", "tree"), ("tree", "update"), ("update", "repeat"), ("repeat", "prob")]),
    "lightgbm": ("LightGBM: histogramas y crecimiento por la hoja de mayor ganancia", 6, [
        ("raw", 2, 4.3, "Variables numéricas\ny categóricas"),
        ("bins", 6, 4.3, "Histogramas\nValores en intervalos"),
        ("splits", 10, 4.3, "Buscar divisiones\ncon mayor ganancia"),
        ("leaves", 14, 4.3, "Dividir la mejor hoja\nCrecimiento leaf-wise"),
        ("boost", 10, 1.5, "Añadir árboles\npara reducir la pérdida"),
        ("prob", 14, 1.5, "Puntuación acumulada\ny probabilidad final"),
    ], [("raw", "bins"), ("bins", "splits"), ("splits", "leaves"), ("leaves", "boost"), ("boost", "prob")]),
    "catboost": ("CatBoost: tratamiento de categorías y boosting con árboles simétricos", 6, [
        ("cat", 2, 4.5, "Categorías\nVivienda, propósito…"),
        ("numeric", 2, 1.5, "Variables numéricas\nIngresos, deuda, FICO…"),
        ("encode", 6, 4.5, "Representaciones\ncategóricas aprendidas"),
        ("trees", 10, 3, "Boosting\nÁrboles simétricos"),
        ("prob", 14, 3, "Puntuación acumulada\ny probabilidad final"),
    ], [("cat", "encode"), ("encode", "trees"), ("numeric", "trees"), ("trees", "prob")]),
    "ensemble": ("Arquitectura seleccionada: seis modelos y una salida calibrada", 8, [
        ("data", 2, 5, "Mismo préstamo\nContrato de variables"),
        ("cat", 6, 6.4, "3 modelos CatBoost\nSemillas 42, 43 y 44"),
        ("light", 6, 3.6, "3 modelos LightGBM\nSemillas 42, 43 y 44"),
        ("catavg", 10, 6.4, "Promedio CatBoost\np C"),
        ("lightavg", 10, 3.6, "Promedio LightGBM\np L"),
        ("blend", 14, 5, "Mezcla ponderada\n0,60 × p C + 0,40 × p L"),
        ("cal", 14, 1.3, "Calibración sigmoid\nProbabilidad final"),
    ], [("data", "cat"), ("data", "light"), ("cat", "catavg"), ("light", "lightavg"), ("catavg", "blend"), ("lightavg", "blend"), ("blend", "cal")]),
    "blend_example": ("Ejemplo numérico de la mezcla: antes de calibrar", 6, [
        ("cat", 2, 4.5, "CatBoost\np C = 30%"),
        ("light", 2, 1.5, "LightGBM\np L = 20%"),
        ("a", 6, 4.5, "Peso 0,60\nAporta 18 puntos"),
        ("b", 6, 1.5, "Peso 0,40\nAporta 8 puntos"),
        ("sum", 10, 3, "Suma ponderada\n18 + 8 = 26%"),
        ("final", 14, 3, "26% sin calibrar\nLuego se aplica sigmoid"),
    ], [("cat", "a"), ("light", "b"), ("a", "sum"), ("b", "sum"), ("sum", "final")]),
    "protocol": ("Separación temporal: cada bloque cumple una función diferente", 8, [
        ("search", 2, 6, "Validaciones 2012 / 2013\nHiperparámetros"),
        ("select", 6, 6, "2014 H1\nElegir familia y pesos"),
        ("fit", 10, 6, "Hasta junio de 2014\nEntrenamiento definitivo"),
        ("calfit", 14, 6, "2014 Q3\nAjustar calibradores"),
        ("policy", 14, 2.3, "2014 Q4\nElegir calibración y umbral"),
        ("freeze", 10, 2.3, "Congelar modelo\ny decisiones"),
        ("test15", 6, 2.3, "2015\nPrueba principal"),
        ("test16", 2, 2.3, "2016\nEstrés temporal"),
    ], [("search", "select"), ("select", "fit"), ("fit", "calfit"), ("calfit", "policy"), ("policy", "freeze"), ("freeze", "test15"), ("test15", "test16")]),
}


def draw_diagram(name, output):
    title, height, nodes, edges = DIAGRAMS[name]
    fig, ax = plt.subplots(figsize=(11.5, height * .64))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 16.5), ylim=(0, height))
    ax.axis("off")
    positions = {key: (x, y) for key, x, y, _ in nodes}
    width, box_height = 3.15, 1.25
    for start, end in edges:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        if abs(x2-x1) >= abs(y2-y1):
            direction = 1 if x2 > x1 else -1
            a, b = (x1 + direction * width/2, y1), (x2 - direction * width/2, y2)
        else:
            direction = 1 if y2 > y1 else -1
            a, b = (x1, y1 + direction * box_height/2), (x2, y2 - direction * box_height/2)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                    color="#677982", linewidth=1.2, zorder=1))
    for i, (key, x, y, label) in enumerate(nodes):
        fill = "#EDF5F4" if i < len(nodes)-1 else "#DCEEEB"
        ax.add_patch(FancyBboxPatch((x-width/2, y-box_height/2), width, box_height,
                                   boxstyle="round,pad=0.04,rounding_size=0.10",
                                   facecolor=fill, edgecolor="#A6BDBA", linewidth=.8, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                color="#183D3A", linespacing=1.35, zorder=3)
    fig.suptitle(title, fontsize=13, fontweight="semibold", color="#183D3A", y=1.01)
    fig.tight_layout(pad=1.2)
    directory = Path(output)/"figures"/"explanation"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def draw_results(comparison, output):
    frame = comparison.sort_values("AP").copy()
    colors = ["#096B68" if name == "selected" else "#B85C38" if name == "baseline19" else "#91A4AF"
              for name in frame.model]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    bars = ax.barh(frame.modelo, frame.AP, color=colors, height=.65)
    for bar, value in zip(bars, frame.AP):
        ax.text(value+.004, bar.get_y()+bar.get_height()/2,
                f"{value:.4f}", va="center", fontsize=11)
    ax.set(xlabel="Average Precision (AP): mayor es mejor", xlim=(0, frame.AP.max()*1.17))
    ax.set_title("Prueba 2015: comparación sobre los mismos 343.507 préstamos", fontsize=13, pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    path = Path(output)/"figures"/"explanation"/"resultados_2015.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def build(root, output="outputs"):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# Explicación de algoritmos, ensamble y resultados

Este notebook presenta el fundamento de los métodos utilizados y relaciona sus mecanismos con los resultados del proyecto **Predicción del riesgo de incumplimiento en préstamos personales de Lending Club**. Complementa los notebooks 02, 03 y 04.

Los diagramas son esquemas conceptuales. Los porcentajes de sus ejemplos son **ilustrativos**; las tablas identificadas como resultados del proyecto se obtienen de los artefactos de la evaluación congelada. Estas celdas no entrenan modelos ni seleccionan nuevas configuraciones.

## 1. Problema y función de los modelos

Cada observación corresponde a un préstamo aceptado. La variable objetivo distingue `Charged Off = 1` de `Fully Paid = 0`; los demás estados no se consideran negativos. El modelo utiliza información de solicitud y bureau disponible bajo el supuesto de evaluación después de la verificación y antes del desembolso.

Su salida es una estimación de riesgo. Convertir esa estimación en una alerta requiere un **umbral**; no equivale automáticamente a rechazar una solicitud. La evaluación corresponde a préstamos con desenlace registrado y debe interpretarse junto con su cobertura.

Los cinco métodos reciben el mismo contrato de información. La referencia de 19 entradas se compara además con modelos que incorporan variables ampliadas y derivadas."""),
        code("from pathlib import Path\nimport sys\nimport json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display\nROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\nfrom scripts.build_explanation_notebook import draw_diagram, draw_results\nOUT = ROOT / " + repr(output) + "\npd.set_option('display.max_columns', 15)\npd.set_option('display.max_rows', 30)\nmetrics = pd.read_csv(OUT/'final'/'metrics.csv')\nselection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))\nlock = json.loads((OUT/'selection_lock.json').read_text(encoding='utf-8'))"),
        md(r"""## 2. Regresión logística: una referencia lineal

La regresión logística combina las variables transformadas mediante coeficientes aprendidos. La puntuación lineal se convierte en probabilidad con la función logística:

$$z=\beta_0+\sum_j\beta_j x_j, \qquad p(y=1\mid x)=\frac{1}{1+e^{-z}}.$$

Un coeficiente positivo aumenta el **log-odds** de incumplimiento manteniendo las demás variables constantes; su significado depende de la transformación y de la escala. No debe interpretarse directamente como puntos porcentuales de riesgo ni como efecto causal.

**Implementación del proyecto.** Se ajustan imputación por mediana, indicadores de ausencia, estandarización y codificación one-hot usando únicamente entrenamiento. Se busca regularización mediante `C` y alternativas de ponderación de clases. El baseline conserva 19 entradas; la regresión ampliada utiliza el conjunto extendido.

**Ventaja:** referencia eficiente y relativamente interpretable. **Límite:** las interacciones y relaciones no lineales requieren transformaciones explícitas; los árboles pueden aprenderlas de manera más flexible."""),
        code("draw_diagram('logistic', OUT)"),
        md("""## 3. Random Forest: bagging y diversidad de árboles

### 3.1. Qué hace un árbol

Un árbol divide los datos mediante preguntas sucesivas sobre variables. Cada camino termina en una hoja, donde se estima la proporción de la clase positiva. Las variables, los cortes y los caminos del esquema siguiente son ilustrativos; no son reglas extraídas del modelo final."""),
        code("draw_diagram('tree', OUT)"),
        md(r"""### 3.2. Cómo se forma el bosque

Random Forest entrena árboles sobre muestras bootstrap y considera subconjuntos aleatorios de variables en sus divisiones. La diversidad reduce la dependencia de un único árbol. En clasificación probabilística se promedian sus estimaciones:

$$p_{RF}(x)=\frac{1}{T}\sum_{t=1}^{T}p_t(x).$$

Los árboles no corrigen secuencialmente los errores del árbol anterior: se ajustan por separado y comparten los datos de origen. Su aleatoriedad no implica independencia estadística perfecta.

**Implementación del proyecto.** Imputación, indicadores de ausencia y one-hot; representación densa float32. La búsqueda utilizó submuestras de hasta 30.000/20.000 filas por corte y 120 árboles por su costo computacional. La selección posterior y el ajuste definitivo utilizaron todas las filas correspondientes y 250 árboles. Se controlaron profundidad, tamaño mínimo de hoja, muestreo de filas y variables.

**Ventaja:** aprende interacciones y relaciones no lineales. **Límite:** puede ser costoso y sus probabilidades pueden requerir calibración."""),
        code("draw_diagram('forest', OUT)"),
        md(r"""## 4. XGBoost: boosting regularizado

XGBoost construye un modelo aditivo de árboles. Cada nuevo árbol utiliza información de los gradientes de la función de pérdida para mejorar el conjunto existente. Para clasificación binaria, una forma esquemática es:

$$F_t(x)=F_{t-1}(x)+\eta f_t(x),\qquad p(x)=\sigma(F_T(x)).$$

`η` es la tasa de aprendizaje; `f_t` es la aportación del nuevo árbol. Las correcciones se acumulan en la puntuación interna, **no sumando directamente porcentajes de probabilidad**. La regularización y los controles de complejidad penalizan soluciones excesivamente flexibles.

**Implementación del proyecto.** Se utilizó `hist` en CPU, valores numéricos ausentes nativos y categorías codificadas con entrenamiento y declaradas como categóricas al algoritmo. Las categorías nuevas se convierten en ausentes. Se ajustaron profundidad, regularización, tasa de aprendizaje y muestreo. El early stopping se aplicó dentro de las validaciones de desarrollo.

**Ventaja:** flexibilidad y controles de regularización. **Límite:** sensibilidad a la configuración y mayor complejidad interpretativa que un modelo lineal."""),
        code("draw_diagram('xgboost', OUT)"),
        md("""## 5. LightGBM: histogramas y crecimiento leaf-wise

LightGBM también es gradient boosting con árboles. Los histogramas agrupan valores numéricos en intervalos para reducir el costo de buscar cortes. Su crecimiento habitual divide la hoja cuya división ofrece mayor ganancia, en lugar de exigir un crecimiento uniforme por niveles.

Este crecimiento permite ajustar relaciones complejas, pero requiere limitar hojas y exigir suficientes observaciones por hoja para controlar sobreajuste. La ganancia se calcula a partir del objetivo de entrenamiento; no es simplemente la cantidad de errores de clasificación.

**Implementación del proyecto.** Se emplearon numéricas con ausentes nativos y categorías codificadas en entrenamiento y declaradas categóricas. Se buscaron número de hojas, tamaño mínimo de hoja, tasa de aprendizaje, regularización y muestreo. La configuración seleccionada utiliza 143 árboles y hasta 28 hojas por árbol.

**Ventaja:** eficiencia en datos tabulares. **Límite:** una estructura demasiado flexible puede sobreajustar; velocidad y calidad dependen de los datos y de la configuración."""),
        code("draw_diagram('lightgbm', OUT)"),
        md("""## 6. CatBoost: categorías y árboles simétricos

CatBoost es otra implementación de boosting. Se distingue por su tratamiento de variables categóricas mediante representaciones aprendidas, incluidas estadísticas construidas con mecanismos que evitan usar directamente la etiqueta de la propia fila. También utiliza variables numéricas.

Sus árboles simétricos habituales aplican la misma regla de división a todos los nodos de un nivel, lo que produce una estructura regular. Este rasgo es distinto del crecimiento leaf-wise descrito para LightGBM.

**Implementación del proyecto.** Se conservaron las categorías como texto. Se utilizó `boosting_type='Plain'`, con estadísticas categóricas y complejidad de combinaciones limitada. El modo Plain no equivale al algoritmo de ordered boosting descrito en la literatura; tampoco elimina el tratamiento categórico propio de CatBoost. La configuración seleccionada utiliza 740 árboles, profundidad 7 y regularización.

**Ventaja:** integración de variables numéricas y categóricas. **Límite:** mayor tiempo de ajuste en este equipo y necesidad de explicación posterior. Que resultara el mejor modelo individual aquí es un resultado empírico, no una propiedad garantizada."""),
        code("draw_diagram('catboost', OUT)"),
        md("""## 7. Diferencias principales

| Método | Estructura del aprendizaje | Tratamiento empleado en el proyecto |
|---|---|---|
| Regresión logística | Combinación lineal y función logística | Imputación, escalado y one-hot |
| Random Forest | Árboles ajustados por separado y promediados | Bagging y selección aleatoria de variables |
| XGBoost | Árboles sucesivos con objetivo regularizado | Histogramas y categorías nativas declaradas |
| LightGBM | Árboles sucesivos con crecimiento leaf-wise | Histogramas y controles de hojas |
| CatBoost | Árboles sucesivos, simétricos por defecto | Tratamiento propio de categorías |

XGBoost, LightGBM y CatBoost pertenecen a la familia de gradient boosting. Comparten varios recursos, como regularización y manejo de ausentes; la tabla destaca diferencias relevantes, no capacidades exclusivas de cada biblioteca."""),
        md(r"""## 8. Ensamble: combinar predicciones de modelos completos

Los modelos se entrenan por separado y reciben el mismo préstamo al predecir. No se unen sus árboles ni se utiliza la predicción de CatBoost como una nueva variable de LightGBM. Se realiza un **promedio ponderado de probabilidades**, también llamado blending o soft voting ponderado.

En cada familia se promedian tres semillas:

$$\bar p_C(x)=\frac{p_{C,42}(x)+p_{C,43}(x)+p_{C,44}(x)}{3},\qquad
\bar p_L(x)=\frac{p_{L,42}(x)+p_{L,43}(x)+p_{L,44}(x)}{3}.$$

Después se combinan las familias:

$$p_{mezcla}(x)=0.60\bar p_C(x)+0.40\bar p_L(x).$$

Por tanto, cada CatBoost tiene peso 0,20 y cada LightGBM peso aproximado 0,1333. Las semillas generan variaciones del entrenamiento; no son nuevas poblaciones ni garantizan independencia de errores. El promedio busca reducir variación y aprovechar diferencias entre modelos.

Los pesos se buscaron en incrementos de 0,1 utilizando **2014 H1**, antes de abrir la prueba. El protocolo exigía al menos 0,001 AP de mejora respecto al mejor modelo individual. La AP de selección fue 0,3401 para el ensamble y 0,3387 para el mejor individual. La diferencia es pequeña, aunque cumplió la regla fijada; el costo incluye mantener seis modelos."""),
        code("draw_diagram('ensemble', OUT)\ndisplay(pd.DataFrame([{'componente': k, 'peso': v} for k,v in selection['weights'].items()]))"),
        md("""### 8.1. Ejemplo numérico ilustrativo

Supongamos que el promedio de CatBoost estima 30% y el promedio de LightGBM 20%. La combinación sin calibrar es `0,60 × 0,30 + 0,40 × 0,20 = 0,26`, es decir, 26%.

El 60% y el 40% indican **influencia sobre el promedio**. No indican precisión, porcentaje de datos utilizados, proporción de préstamos incumplidos ni porcentaje de árboles que aciertan."""),
        code("draw_diagram('blend_example', OUT)\np_cat, p_light = 0.30, 0.20\np_blend = 0.60*p_cat + 0.40*p_light\nprint(f'Ejemplo ilustrativo: probabilidad mezclada sin calibrar = {p_blend:.1%}')"),
        md(r"""### 8.2. De la mezcla a la probabilidad calibrada y a la alerta

El proyecto compara identidad, sigmoid e isotonic. Los calibradores se ajustan en **2014 Q3** y el método se elige por Brier en **2014 Q4**. El elegido para el ensamble fue sigmoid. Su forma es:

$$p_{final}(x)=\sigma\!\left(a\,\operatorname{logit}(p_{mezcla}(x))+b\right).$$

`a` y `b` se aprenden en calibración. Así, **26% antes de calibrar no tiene por qué ser 26% después de calibrar**. La calibración intenta acercar las probabilidades a frecuencias observadas; no garantiza que sigan calibradas en otro año.

Después se compara `p_final` con un umbral fijado en desarrollo para producir una alerta. El umbral de F1 del ensamble es aproximadamente **0,2085**. La clasificación por umbral y la selección del 20% de mayor riesgo son políticas distintas; sus métricas no deben mezclarse."""),
        code("print('Calibración del ensamble:', lock['calibration']['selected'])\nprint('Umbrales fijados antes de probar:', lock['thresholds']['selected'])"),
        md("""## 9. Evaluación temporal y procedencia de los datos

El archivo original contiene 2.260.701 registros, pero no todos se utilizan para entrenar. El modelo definitivo se ajustó con **327.426 préstamos hasta junio de 2014**. Los bloques posteriores de 2014 se reservaron para calibración y política de umbral.

Las pruebas incluyen **343.507 préstamos de 2015** y **268.096 de 2016**, con IDs ajenos a la muestra inicial de 115.000 y exclusiones adicionales documentadas por inspección de esquema. La búsqueda utiliza dos validaciones crecientes de 2012/2013; la selección de modelos y pesos utiliza 2014 H1.

Se congelaron modelo, calibradores, umbrales y huellas de código/datos antes de abrir los desenlaces externos. Se trata de nuevas cohortes de la **misma fuente**, no de validación en otra institución o país."""),
        code("draw_diagram('protocol', OUT)"),
        md("""## 10. Resultados reales del experimento

La tabla siguiente corresponde a **2015, todos los resultados terminales**, y utiliza las mismas filas para todos los comparadores. `baseline19` es la referencia logística balanceada reentrenada bajo el nuevo protocolo; `logistic` es la regresión con variables ampliadas. No debe confundirse esta comparación con la AP 0,3610 del avance inicial, que pertenecía a otra partición.

Cada comparador fue calibrado con el mismo esquema temporal. Las métricas de umbral usan el umbral propio fijado en desarrollo. El ganador sigue siendo el seleccionado antes de probar; no se reelige usando esta tabla."""),
        code("names={'baseline19':'Referencia logística (19 entradas)','logistic':'Regresión logística ampliada','random_forest':'Random Forest','lightgbm':'LightGBM','xgboost':'XGBoost','catboost':'CatBoost','selected':'Ensamble seleccionado'}\ncomparison=metrics.query(\"cohort == 2015 and scope == 'all_resolved'\").copy()\ncomparison['modelo']=comparison.model.map(names)\ndisplay(comparison[['modelo','AP','ROC_AUC','Brier','precision','recall','F1']].round(4))\nbase=comparison.set_index('model').loc['baseline19']\nchosen=comparison.set_index('model').loc['selected']\nprint(f'Mejora de AP: {chosen.AP-base.AP:+.4f}; de ROC-AUC: {chosen.ROC_AUC-base.ROC_AUC:+.4f}.')"),
        code("draw_results(comparison, OUT)"),
        md("""### 10.1. Qué significa ROC-AUC = 0,7334

Mide discriminación: la capacidad de ordenar positivos por encima de negativos. Si se elige aleatoriamente un préstamo incumplido y uno pagado, el valor equivale a una probabilidad aproximada de **73,3%** de asignar mayor puntuación al incumplido, contando los empates con peso medio.

No significa 73,3% de exactitud, de incumplimientos detectados ni de probabilidades correctas. Una ROC-AUC de 0,5 representa un ordenamiento sin capacidad de discriminación; 1 representa discriminación perfecta.

### 10.2. Qué significa AP = 0,4164

Average Precision resume la precisión conseguida a medida que se recuperan más incumplimientos al variar el umbral. Se calcula como una media de precisiones ponderada por los incrementos de recall. En este proyecto se usa `average_precision_score`, **no integración trapezoidal de la curva PR**.

La prevalencia de incumplimiento de 2015 es aproximadamente 20,14%, referencia de un ordenamiento aleatorio en una población grande. La AP depende de esa prevalencia, por lo que no conviene comparar directamente AP de años o plazos con composiciones distintas.

AP 0,4164 no equivale a 41,64% de aciertos ni a la precisión de un único umbral. La mejora pareada frente al baseline es +0,0274; el intervalo bootstrap del 95% por bloques mensuales es aproximadamente [+0,0243; +0,0302]. Solo hay 12 bloques: ese intervalo no recoge toda la incertidumbre macroeconómica o institucional.

### 10.3. Qué significa en una revisión de préstamos

Al ordenar los préstamos de 2015 y revisar el 20% de mayor riesgo, se recupera aproximadamente **42,5% de los incumplimientos**. Dentro de ese grupo, aproximadamente **42,8% incumplió**. Como ilustración proporcional, entre 100 préstamos con unos 20 incumplimientos, revisar los 20 de mayor riesgo permitiría encontrar alrededor de 9, frente a unos 4 al elegirlos al azar.

Con el umbral F1 fijado en desarrollo se obtiene otro compromiso: **precisión 35,6% y recall 61,6%**. Esta política genera una cantidad de alertas distinta de revisar exactamente el 20%."""),
        code("print('Cohorte 2015:', int(chosen.n), 'préstamos')\nprint(f'Prevalencia: {chosen.prevalence:.2%}')\nprint(f'Recall entre el 20% de mayor riesgo: {chosen.recall_at_top20:.2%}')\nprint(f'Precisión entre el 20% de mayor riesgo: {chosen.precision_at_top20:.2%}')\nprint(f'Umbral F1: {chosen.threshold:.4f}; precisión: {chosen.precision:.2%}; recall: {chosen.recall:.2%}')\nrecords=json.loads((OUT/'final'/'metrics.json').read_text(encoding='utf-8'))\nrecord=next(r for r in records if r['cohort']==2015 and r['scope']=='all_resolved' and r['model']=='selected')\ndisplay(pd.DataFrame(record['confusion_matrix'],index=['Real pagado','Real incumplido'],columns=['Sin alerta','Con alerta']))"),
        md("""## 11. Estabilidad, interpretación y límites

El ensamble obtuvo la mayor AP y ROC-AUC entre los comparadores congelados en 2015 y 2016. Eso no implica que todas sus métricas sean las mejores ni que sea el mejor modelo posible fuera del experimento.

**Calibración temporal.** En 2016 el riesgo medio predicho fue aproximadamente 18,0%, frente a 23,3% observado entre préstamos con resultado terminal. La capacidad de ordenar riesgos puede mantenerse mientras las probabilidades subestiman su nivel.

**Maduración y censura.** En 2015, 99,94% de los préstamos elegibles a 36 meses tienen resultado terminal; a 60 meses, solo 65,09%. Excluir los todavía vigentes puede sesgar los resultados agregados. La mejora se mantiene en la prueba de 2015 a 36 meses: AP del ensamble 0,2718 frente a 0,2386 del baseline.

**Disponibilidad histórica de etiquetas.** El corte se hace por emisión, pero el snapshot no permite reconstruir fiablemente cuándo se conoció cada incumplimiento. Es una evaluación retrospectiva, no un backtest exacto de producción.

**Interpretabilidad.** La permutación por grupos explica la dependencia predictiva del ensamble; destacan monto/plazo, ingresos y FICO. Tree SHAP explica sus componentes en log-odds. Esas contribuciones no se suman directamente como porcentajes de la probabilidad calibrada final. Ninguna de estas técnicas demuestra causalidad.

**Población y equidad.** No se evalúan solicitantes rechazados, todos los préstamos vigentes ni otra institución. Los diagnósticos por segmentos no certifican equidad en ausencia de atributos protegidos.

**Implicación.** La mejora apoya la utilidad del enfoque para investigación y priorización experimental de revisiones. El uso operativo requiere seguimiento completo, nuevas pruebas, calibración vigente y evaluación de costos y disparidades."""),
        code("display(metrics.query(\"model in ['baseline19','selected']\")[[ 'cohort','scope','model','n','prevalence','AP','ROC_AUC','Brier']].round(4))\ndisplay(pd.read_csv(OUT/'final'/'maturity_coverage.csv'))"),
        md("""## 12. Síntesis para la exposición

> Se comparó una referencia lineal, un método de bagging y tres implementaciones de gradient boosting. Se seleccionó mediante validación temporal un promedio de probabilidades de CatBoost y LightGBM, con pesos 60/40 y tres semillas por familia. Tras congelar el predictor y calibrarlo con datos de desarrollo, la prueba de 2015 mostró AP 0,4164 y ROC-AUC 0,7334, superiores a la referencia reentrenada sobre las mismas filas. El análisis también identificó subestimación de riesgo en 2016 y límites de censura, disponibilidad histórica de etiquetas y generalización.

### Referencias y trazabilidad

- [Artículos y decisiones metodológicas](../experiments/research_notes.md).
- [Protocolo fijado antes de probar](../experiments/protocol.json).
- [Comparación experimental completa](03_modelos.ipynb).
- [Análisis de errores e interpretabilidad](04_analisis_errores.ipynb).
- [Métricas de la prueba](../outputs/final/metrics.csv) y [congelación del experimento](../outputs/selection_lock.json).

Para regenerar únicamente este material desde la raíz del repositorio: `python scripts/build_explanation_notebook.py`. Solo lee resultados ya guardados y genera diagramas; no vuelve a ajustar ni evaluar los modelos."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "lending-club-research", "display_name": "Lending Club Research", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    install(prefix=sys.prefix, kernel_name="lending-club-research", display_name="Lending Club Research")
    os.environ["JUPYTER_PATH"] = str(Path(sys.prefix)/"share"/"jupyter")
    NotebookClient(nb, timeout=180, kernel_name="lending-club-research",
                   resources={"metadata": {"path": str(root)}}).execute()
    nbf.validate(nb)
    path = root/"notebooks"/"05_explicacion.ipynb"
    nbf.write(nb, path)
    print(f"Executed notebook saved: {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output)
