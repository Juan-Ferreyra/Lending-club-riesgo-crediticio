# Historial de fases del proyecto

> Este documento conserva sin cambios el README anterior a la entrega final (avance de semana 1, V1, V2 y V3). El README principal ahora es una guía corta; el informe final está en [reports/informe_final.md](../reports/informe_final.md).


**Última fase de optimización:** [Notebook 07](../notebooks/07_optimizacion_predictiva.ipynb), con comparación sobre préstamos a **36 meses**.

## Fase experimental de la entrega final

El repositorio incorpora una fase de investigación y experimentos posterior al avance inicial. Incluye comparación temporal de regresión logística, Random Forest, LightGBM, XGBoost y CatBoost; búsqueda de hiperparámetros; calibración; evaluación congelada; errores por segmentos e interpretación. El informe final y la presentación se elaborarán después.

- `experiments/protocol.json`: fechas, objetivo, métricas, presupuesto y reglas fijadas antes de evaluar.
- `experiments/research_notes.md`: artículos consultados y decisiones técnicas.
- `notebooks/02_limpieza_features.ipynb`: procedencia, limpieza y auditoría de variables.
- `notebooks/03_modelos.ipynb`: búsqueda, selección y evaluación temporal.
- `notebooks/04_analisis_errores.ipynb`: errores, estabilidad, permutación y explicaciones locales.
- `notebooks/05_explicacion.ipynb`: fundamentos de los cinco algoritmos, diagramas del ensamble y lectura de los resultados.
- `notebooks/06_diagnostico_datos_svm.ipynb`: auditoría de población y faltantes, ablaciones, ajuste adicional y SVM lineal/RBF.
- `notebooks/07_optimizacion_predictiva.ipynb`: optimización conjunta, ensambles, controles y estabilidad temporal.
- `outputs/`: configuraciones, resultados, huellas y figuras. Los modelos y predicciones grandes se regeneran localmente y están ignorados por Git.

### Resultado del primer experimento ejecutado

Se completaron **96 intentos de búsqueda** en cinco familias. La selección de 2014 H1 eligió un ensamble de **60% CatBoost y 40% LightGBM**, promediando tres semillas por familia. Se reentrenó con 327.426 préstamos hasta junio de 2014 y se eligió calibración sigmoid en los bloques posteriores de desarrollo. La elección quedó congelada antes de abrir las pruebas siguientes.

| Prueba | Préstamos | AP baseline reentrenado | AP ensamble | ROC-AUC ensamble |
|---|---:|---:|---:|---:|
| 2015, todos los resultados terminales | 343.507 | 0,3890 | **0,4164** | 0,7334 |
| 2015, plazo de 36 meses | 258.856 | 0,2386 | **0,2718** | 0,6920 |
| 2016, estrés temporal, resultados terminales | 268.096 | 0,3958 | **0,4248** | 0,7116 |

En 2015, la mejora pareada de AP es **+0,0274**, con intervalo bootstrap del 95% por bloques mensuales de **[+0,0243; +0,0302]**. El Brier baja de 0,1455 a 0,1418. Los valores completos están en `outputs/final/metrics.csv`, y los intervalos en `outputs/final/uncertainty.json`.

La prueba de 2015 a 36 meses tiene desenlace terminal en el **99,94%** de su población elegible; la de 60 meses, solo en el **65,09%**. En 2016 esas coberturas son 69,98% y 52,57%. Por ello las cifras agregadas y las de estrés deben interpretarse junto con la cobertura: no estiman el desempeño sobre todos los préstamos vigentes. La separación es retrospectiva por fecha de emisión; no reconstruye la disponibilidad histórica de cada etiqueta. El ensamble fue el mejor por AP y ROC-AUC entre los comparadores congelados en ambas cohortes; esto no demuestra que sea el mejor modelo posible fuera de este experimento.

### Reproducir los experimentos completos

Se recomienda **Python 3.12**. `requirements-lock.txt` registra las versiones efectivamente ejecutadas; `requirements-experiments.txt` contiene rangos compatibles. En Windows, desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/run_experiments.py --source "RUTA/accepted_2007_to_2018Q4.csv.gz" --output outputs/reproduction
```

El archivo original procede de la versión 3 de Kaggle citada abajo. No se añade a Git. El proceso comprueba que contenga todos los IDs de la muestra inicial, registra SHA-256 y reserva préstamos externos de 2015/2016 excluyendo los IDs de esa muestra. La búsqueda usa tres hilos de CPU. El nombre de salida debe ser nuevo; para reanudar una búsqueda interrumpida antes de probar se pueden usar las etapas individuales.

```powershell
.\.venv\Scripts\python.exe -m src.prepare --source "RUTA/accepted_2007_to_2018Q4.csv.gz"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --stage search
.\.venv\Scripts\python.exe -m src.train --stage select
.\.venv\Scripts\python.exe -m src.train --stage freeze
.\.venv\Scripts\python.exe -m src.final_test --execute-final
.\.venv\Scripts\python.exe -m src.analyze
.\.venv\Scripts\python.exe scripts/build_research_notebooks.py
```

La búsqueda usa dos cortes temporales crecientes y selección en 2014 H1. El entrenamiento final llega hasta junio de 2014; julio-septiembre ajustan calibradores y octubre-diciembre seleccionan calibración/umbrales. Los artefactos y el código se congelan mediante hashes antes de abrir 2015 y 2016. Una evaluación completada bloquea nuevos ajustes sobre esa misma prueba. Repetir los comandos con un directorio nuevo sirve para reproducir el experimento fijado; no convierte una prueba ya observada en otra prueba independiente.

**Alcance temporal:** es una evaluación retrospectiva por fecha de emisión. El snapshot no permite reconstruir con fiabilidad la fecha de disponibilidad de cada etiqueta de incumplimiento. Los préstamos sin resultado terminal se excluyen del objetivo y su cobertura queda en `outputs/final/maturity_coverage.csv`. Se reporta separadamente 2015 a 36 meses para estudiar maduración. No se puede extrapolar la calibración directamente a todos los solicitantes ni a los préstamos vigentes. La comparación válida con el baseline usa las mismas filas de prueba; el resultado aleatorio del avance anterior pertenece a otra partición.

### Segunda fase: diagnóstico de datos, hiperparámetros y SVM

Esta fase investiga faltantes/valores inválidos, selección de resultados terminales, cambios entre años y representatividad. El alcance principal son préstamos aceptados **a 36 meses**, con seguimiento casi completo para 2015. Se comparan recetas de datos, hiperparámetros y tres estrategias SVM: lineal, RBF exacta y aproximación Nyström. También se controla el presupuesto de filas y se separa un diagnóstico con la clasificación/tasa de la plataforma, que cambia la información disponible.

El [protocolo v2](../experiments/protocol_v2.json) y la [bitácora](../experiments/research_v2_notes.md) distinguen las decisiones de las conclusiones. Los años 2015/2016 ya fueron observados: esta fase los identifica como **benchmarks retrospectivos reutilizados**, no como una nueva prueba final independiente. Los resultados se guardan en `outputs/research_v2/`, separados del primer experimento.

Se contrastan nueve recetas de datos y 80 configuraciones de hiperparámetros, más tres repeticiones de configuraciones RBF con mayor límite numérico del solver. La selección temporal compara 15 candidatos, incluidos controles con parámetros anteriores y entrenamiento desde 2013. Todos usan la misma validación 2014 Q3 a 36 meses. Una comparación adicional iguala las 12.000 filas de entrenamiento entre familias.

El benchmark principal utiliza los mismos 258.856 préstamos de 2015 a 36 meses de la sensibilidad anterior. Todos los modelos se comparan además sobre los mismos 20.000 préstamos por año; las SVM RBF exactas no seleccionadas se limitan a esa muestra por su costo de predicción. Los otros modelos y siempre el ganador preseleccionado usan también la cohorte completa. Las tablas de distinto tamaño se conservan separadas y no se compara directamente su AP. El resultado de esta fase tampoco se compara con AP 0,4164, que mezclaba plazos de 36 y 60 meses.

La selección de desarrollo eligió **CatBoost**, con variables de disponibilidad estable y entrenamiento de préstamos a 36 meses desde 2012. En la misma cohorte madura reutilizada:

| Modelo | AP 2015/36 meses | ROC-AUC |
|---|---:|---:|
| Ensamble anterior | 0,2718 | 0,6920 |
| CatBoost seleccionado en esta fase | 0,2752 | 0,6951 |

La mejora es pequeña: aproximadamente **+0,0034 AP**. No se presenta como un salto grande ni como confirmación en una nueva prueba independiente. Cambian conjuntamente receta, calendario de ajuste, calibración y predictor; esta diferencia total no se atribuye solo a hiperparámetros. El [notebook 06](../notebooks/06_diagnostico_datos_svm.ipynb) muestra controles, intervalos pareados, presupuestos SVM y resultados negativos de las hipótesis.

El intervalo bootstrap pareado por mes para ese cambio es **[+0,0018; +0,0048]** con 300 repeticiones. Es condicional a este benchmark reutilizado y no corrige su reutilización adaptativa. Al revisar el 20% de préstamos con mayor riesgo, el recall pasa de **38,20% a 38,66%**. La Brier baja de **0,118993 a 0,118769**; menor es mejor. Estos cambios respaldan una mejora modesta dentro de esta cohorte.

Para reproducir esta fase se necesitan los modelos/predicciones de la primera. En una copia nueva del repositorio, ejecutar primero la reproducción anterior en `outputs/reproduction`; después elegir una carpeta nueva para esta fase:

```powershell
$env:LENDING_BASE_OUTPUT = 'outputs/reproduction'
$env:LENDING_RESEARCH_OUTPUT = 'outputs/research_reproduction'
.\.venv\Scripts\python.exe -m research.audit --source "RUTA/accepted_2007_to_2018Q4.csv.gz"
.\.venv\Scripts\python.exe -m research.monthly_coverage --source "RUTA/accepted_2007_to_2018Q4.csv.gz"
.\.venv\Scripts\python.exe -m research.experiments --stage all
.\.venv\Scripts\python.exe -m research.followup
.\.venv\Scripts\python.exe -m research.notebook
```

En el entorno original, las rutas predeterminadas son `outputs` para la referencia y `outputs/research_v2` para esta fase. Una evaluación completada bloquea reajustes en esa misma salida. La calibración SVM utiliza un bloque temporal separado; sus puntuaciones de margen no se interpretan directamente como probabilidades.

Para reconstruir solamente los predictores y la evaluación de la segunda fase con sus configuraciones registradas, sin repetir la búsqueda adaptativa:

```powershell
.\.venv\Scripts\python.exe -m research.replay --destination outputs/research_fixed --reference-output outputs/reproduction
```

El comando requiere los datos preparados y los modelos/predicciones de referencia de la primera fase. Genera una salida nueva y compara sus métricas con las publicadas. Rehacer una búsqueda puede cambiar la trayectoria de TPE por reanudaciones y candidatos guiados; los parámetros realmente seleccionados quedan en `selection.json` y permiten reconstruir el resultado predictivo.

<!-- v3-results-start -->
### Tercera fase: optimización conjunta de la predicción

El [notebook 07](../notebooks/07_optimizacion_predictiva.ipynb) documenta la investigación ejecutada sobre datos, variables, preparación por algoritmo, hiperparámetros y ensambles. Se registraron ocho recetas del bloque de ablación, incluida la referencia, y **46 intentos de búsqueda** en seis familias: 36 completos, 10 podados, 0 fallidos. Se añadieron tres controles de preparación y ocho candidatos de contexto/ensamble en ampliaciones adaptativas de desarrollo documentadas antes del benchmark. La selección compara **39 estrategias en la misma cohorte de 2014 Q3: 13 modelos individuales y 26 ensambles**. La estrategia elegida fue: **100% `catboost_combined_control`**. Los ensayos sin mejora, fallidos o descartados se conservan.

La referencia es el CatBoost V2, sobre los mismos **258.856 préstamos de 2015 a 36 meses**:

| Estrategia | AP | ROC-AUC | Recall al revisar el 20% |
|---|---:|---:|---:|
| CatBoost V2 congelado | 0.275187 | 0.695059 | 38.66% |
| Parámetros V2, entrenamiento ampliado hasta noviembre | 0.277484 | 0.696322 | 38.88% |
| Estrategia V3 seleccionada en desarrollo | 0.280128 | 0.698253 | 39.09% |

La estrategia preseleccionada mejora AP en esta cohorte reutilizada. El cambio frente al CatBoost V2 es **+0.004941 AP** (+1.80% relativo), con intervalo bootstrap pareado por mes **[+0.003767; +0.006188]**. El cambio es positivo en 4 de los 4 trimestres de 2015; se muestran todos los períodos, incluidos los negativos. En la revisión de los cortes anteriores de desarrollo, AP mejora en **0 de 2**; estos cortes también se reutilizaron. La mejora no es uniforme en toda la historia. 2016 se mantiene como estrés afectado por desenlaces aún no observados.

El entrenamiento final desde 2012 llega hasta noviembre de 2014: 299.588 préstamos, 43.146 más que V2. Diciembre se divide mediante hash de ID en grupos disjuntos de calibración y umbral; son grupos del mismo mes. El control con parámetros V2 permite separar el aporte de las filas adicionales al ranking. Sus diferencias de Brier también incluyen el cambio de calibración. AP se calcula sobre puntuaciones originales y se contrasta con probabilidades de una sigmoide de pendiente positiva.

**No es una nueva prueba final independiente.** Los años 2015/2016 ya habían sido observados; ni el intervalo ni la congelación recuperan esa independencia. El archivo disponible tampoco proporciona una cohorte futura madura sin usar. El alcance, controles y límites se detallan en el [protocolo V3](../experiments/protocol_v3.json), la [bitácora](../experiments/research_v3_notes.md) y el [contrato de variables](../experiments/feature_contract_v3.csv). No se compara este AP con 0,4164 de la población original que mezclaba plazos.

Para reconstruir los predictores con las especificaciones publicadas, se necesitan los datos preparados, el archivo original en `data/accepted_2007_to_2018Q4.csv.gz` y los artefactos de referencia V1. La referencia V2 debe ser su reconstrucción fija en `outputs/research_fixed`, obtenida con `python -m research.replay --destination outputs/research_fixed --reference-output outputs/reproduction`, como se explicó en la segunda fase. Una búsqueda nueva de V2 puede elegir otros parámetros y no sirve como referencia de coincidencia exacta.

```powershell
$env:LENDING_BASE_OUTPUT = 'outputs/reproduction'
.\.venv\Scripts\python.exe -m optimization.replay --destination outputs/optimization_fixed --v2-output outputs/research_fixed
```

Para repetir la búsqueda completa de V3 en una salida nueva se utiliza la salida **completa** de V2, `outputs/research_reproduction`, que incluye sus candidatos Q3 y archivos de búsqueda. La reconstrucción fija de V2 solo genera los predictores finales y no contiene esos archivos de desarrollo:

```powershell
$env:LENDING_BASE_OUTPUT = 'outputs/reproduction'
$env:LENDING_V2_OUTPUT = 'outputs/research_reproduction'
$env:LENDING_OPTIMIZATION_OUTPUT = 'outputs/optimization_reproduction'
.\.venv\Scripts\python.exe -m optimization.experiments --stage prepare
.\.venv\Scripts\python.exe -m optimization.experiments --stage all
.\.venv\Scripts\python.exe -m optimization.analysis --stage all
.\.venv\Scripts\python.exe -m optimization.notebook --output-dir outputs/optimization_reproduction
```

La reconstrucción fija evita depender de la trayectoria adaptativa de TPE. Repetir los experimentos no crea datos independientes. Los modelos, predicciones individuales y bases de estudio permanecen fuera de Git; los resultados agregados, parámetros, figuras y verificaciones sí se publican. El informe final continúa pendiente.
<!-- v3-results-end -->

---

## Avance inicial de semana 1

Primera entrega (avance de semana 1) del proyecto de Machine Learning. Se busca estimar el riesgo de que un préstamo aceptado termine en `Charged Off`, utilizando información disponible al completar la evaluación de la solicitud y antes del otorgamiento. La [propuesta](../proposal.md) describe el objetivo, los riesgos y el plan de trabajo.

Este avance incluye la exploración de datos y un baseline de regresión logística, entrenado y evaluado, que corresponde a la tarea de clasificación binaria: `Charged Off = 1` y `Fully Paid = 0`. La comparación de otros modelos, la calibración y la evaluación final forman parte del plan de las siguientes semanas.

## Ejecutar en Google Colab

1. [Abrir el notebook en Colab](https://colab.research.google.com/github/Juan-Ferreyra/Lending-club-riesgo-crediticio/blob/main/notebooks/01_exploracion_inicial.ipynb).
2. Conectar el entorno de ejecución y elegir **Runtime → Run all** (o **Entorno de ejecución → Ejecutar todo**).

La primera celda utiliza la muestra incluida en el repositorio y la descarga automáticamente si no está en el entorno. No es necesario descargar el dataset original para ejecutar el notebook. Los resultados y las gráficas también están guardados en el archivo `.ipynb`.

Para ejecutarlo localmente, usar Python 3.10 o superior. Primero obtener esta versión del repositorio:

```bash
git clone --branch main --single-branch https://github.com/Juan-Ferreyra/Lending-club-riesgo-crediticio.git
cd Lending-club-riesgo-crediticio
```

Desde la raíz del repositorio, instalar las dependencias y abrir el notebook:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

## Datos y reproducibilidad

La [muestra utilizada](../data/lending_club_muestra.csv.gz) contiene **115 000 préstamos y 151 variables**: 91 959 `Fully Paid` y 23 041 `Charged Off`. Su [resumen](../data/sample_summary.json) detalla la distribución por estado, año y plazo. El archivo comprimido ocupa aproximadamente **20,3 MiB**, por lo que se incluye directamente en el repositorio.

Procede de [Lending Club Loan Data, versión 3](https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3). La ficha de Kaggle indica licencia **CC0: Public Domain**; descargar el archivo original puede requerir una cuenta. Se tomaron 115 000 préstamos al azar, sin reemplazo y con semilla 42, de los registros cuyo estado definitivo es `Fully Paid` o `Charged Off`. Las funciones de muestreo están en [`src/data.py`](../src/data.py) y se ejecutan mediante [`scripts/crear_muestra.py`](../scripts/crear_muestra.py).

Si se desea regenerar la muestra desde el ZIP original de Kaggle, colocarlo en `data/lending-club.zip` y ejecutar:

```bash
python -m pip install -r requirements.txt
python scripts/crear_muestra.py --archive data/lending-club.zip
```

También puede usarse el archivo `accepted_2007_to_2018Q4.csv.gz` extraído del ZIP:

```bash
python scripts/crear_muestra.py --source data/accepted_2007_to_2018Q4.csv.gz
```

El ZIP completo no se incluye por su tamaño. La muestra contiene solo préstamos aceptados con resultado definitivo; no representa a todos los solicitantes ni a todos los préstamos emitidos.

## Contenido del avance

- [`proposal.md`](../proposal.md): los 13 apartados de la propuesta, desde el título y los integrantes hasta el plan de trabajo.
- [`notebooks/01_exploracion_inicial.ipynb`](../notebooks/01_exploracion_inicial.ipynb): los 8 componentes de la exploración requerida: carga, dimensiones y tipos, faltantes, objetivo, seis visualizaciones, outliers, sesgos/leakage/limitaciones y baseline ejecutado.
- [`data/lending_club_muestra.csv.gz`](../data/lending_club_muestra.csv.gz): muestra utilizada en el análisis.
- [`data/sample_summary.json`](../data/sample_summary.json): composición de la muestra.
- [`src/data.py`](../src/data.py): funciones reutilizables de lectura y muestreo.
- [`scripts/crear_muestra.py`](../scripts/crear_muestra.py): comando para regenerar la muestra.
- [`requirements.txt`](requirements.txt): dependencias para la ejecución local.

## Validación y resultado inicial

La división aleatoria estratificada usa semilla 42: 80 500 préstamos para entrenamiento (70%), 17 250 para validación (15%) y 17 250 como reserva interna de prueba (15%). El pipeline ajusta imputación, escalado, codificación y regresión logística únicamente con entrenamiento. Utiliza 19 predictores anteriores al otorgamiento y excluye el estado final, identificadores y datos de pagos o cobranza.

En **validación**, la regresión logística obtuvo PR-AUC **0.3610**, frente a una prevalencia de incumplimiento de **0.2003**; ROC-AUC **0.6948**, precisión **0.3120**, recall **0.6418** y F1 **0.4199** con umbral 0.5. La columna PR-AUC corresponde a Average Precision (AP), calculada con `average_precision_score`. El baseline detecta 2 218 de los 3 456 incumplimientos de validación y genera 4 891 falsos positivos. Las probabilidades no se han calibrado; estas métricas describen el ordenamiento y la clasificación inicial del riesgo.

La reserva interna de prueba no se utiliza para ajustar ni evaluar el modelo, pero sus filas sí forman parte de la exploración descriptiva realizada sobre toda la muestra. Por ello no se presenta como una prueba completamente ajena a la exploración. Las métricas reportadas son preliminares y corresponden únicamente a validación. Para la entrega final se prevé separar, antes de explorarlos, registros del dataset original que no pertenezcan a esta muestra y estudiar una validación temporal, según el plan de la propuesta.
