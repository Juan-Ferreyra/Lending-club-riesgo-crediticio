# Propuesta de proyecto

## 1. Título del proyecto

Predicción del riesgo de incumplimiento en préstamos personales de Lending Club

## 2. Integrantes

**Grupo:** 7

- Marco Soto Maceda
- Marcelo Ferreyra
- Valentino Contreras
- Rayhan Matos
- Leonardo Panduro

## 3. Dataset elegido

Se utiliza el dataset **Lending Club Loan Data**, disponible en Kaggle:

https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3

La ficha de Kaggle declara la licencia **CC0: Public Domain**. La descarga del archivo original puede requerir iniciar sesión. La muestra incluida en `data/lending_club_muestra.csv.gz` se genera con `scripts/crear_muestra.py`, cuya lógica reutilizable está en `src/data.py`, a partir de `accepted_2007_to_2018Q4.csv.gz` de esa versión, con semilla 42.

El archivo original contiene 2,260,701 registros y 151 variables de préstamos emitidos entre 2007 y 2018. De ellos, 1,345,310 tienen estado definitivo `Fully Paid` o `Charged Off`. La muestra aleatoria sin reemplazo tiene 115,000 registros y 151 variables de 2007 a 2018, con plazos de 36 y 60 meses. Contiene 91,959 préstamos `Fully Paid` (79.96%) y 23,041 `Charged Off` (20.04%). La composición por año y plazo está en `data/sample_summary.json`.

Esta muestra representa solo préstamos con resultado definitivo; no representa a todos los solicitantes ni a todos los préstamos emitidos, especialmente los más recientes.

El volumen de datos, las 151 variables, los faltantes estructurales, los valores extremos, el desbalance y los cambios temporales justifican la complejidad del problema. La muestra permite ejecutar la exploración y el baseline en Colab sin cargar el dataset completo.

## 4. Pregunta predictiva

¿Cuál es la probabilidad de que un préstamo personal aceptado por Lending Club termine en incumplimiento, utilizando las características del solicitante, del préstamo solicitado y de su historial crediticio registradas antes del otorgamiento?

## 5. Variable objetivo

La variable objetivo se construye a partir de `loan_status`:

- `1`: préstamo en incumplimiento, correspondiente a `Charged Off`.
- `0`: préstamo pagado, correspondiente a `Fully Paid`.

Se trata de un problema de clasificación binaria supervisada. En este proyecto, incumplimiento se define operativamente como `Charged Off`; no se agrupan otros estados, como atrasos o préstamos todavía vigentes.

## 6. Unidad de predicción

La unidad de predicción es un préstamo personal aceptado por Lending Club. Cada fila representa un préstamo y la predicción se realiza antes de observar su comportamiento de pago.

## 7. Variables disponibles antes de la predicción

El baseline utiliza 19 variables: 12 numéricas y 7 categóricas. Se supone que la predicción se realiza al completar la evaluación de la solicitud y antes del otorgamiento; por ello `verification_status` solo sería utilizable si la verificación ya se realizó. Esta disponibilidad deberá confirmarse para cualquier uso operativo.

Variables numéricas:

- `loan_amnt`
- `annual_inc`
- `dti`
- `delinq_2yrs`
- `fico_range_low`
- `fico_range_high`
- `inq_last_6mths`
- `open_acc`
- `pub_rec`
- `revol_bal`
- `revol_util`
- `total_acc`

Variables categóricas:

- `term`
- `emp_length`
- `home_ownership`
- `verification_status`
- `purpose`
- `addr_state`
- `application_type`

## 8. Riesgos de leakage

Se excluyen de las entradas del modelo las variables creadas después de otorgar el préstamo o relacionadas directamente con su resultado. `loan_status` se utiliza únicamente para construir la etiqueta. Entre las variables excluidas se encuentran:

- `loan_status`
- `out_prncp`
- `out_prncp_inv`
- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_prncp`
- `total_rec_int`
- `total_rec_late_fee`
- `recoveries`
- `collection_recovery_fee`
- `last_pymnt_d`
- `last_pymnt_amnt`
- `next_pymnt_d`
- `last_credit_pull_d`
- `last_fico_range_high`
- `last_fico_range_low`
- Variables de hardship, settlement y cobranza.

También se excluyen `grade`, `sub_grade`, `int_rate` e `installment` para no depender de la evaluación o del precio asignados por Lending Club. Se trata de una decisión de alcance adicional al control de información posterior al otorgamiento. El código selecciona explícitamente las 19 entradas permitidas; tampoco utiliza identificadores, `target` ni las fechas derivadas como predictores. La imputación, la estandarización y la codificación se ajustan únicamente con entrenamiento, dentro del pipeline.

## 9. Métrica principal y métrica secundaria

La métrica principal es **Average Precision (AP)**, reportada como **PR-AUC** en las tablas del notebook y calculada con `average_precision_score`. Resume la relación entre precisión y recall sin integración trapezoidal. Es pertinente porque el incumplimiento es la clase minoritaria. La prevalencia de la clase positiva sirve como referencia de un ordenamiento sin señal predictiva, no como límite inferior de la métrica.

La métrica secundaria es **ROC-AUC**, porque mide la capacidad del modelo para ordenar los préstamos cumplidos e incumplidos a través de distintos umbrales.

Accuracy no se utiliza como métrica principal debido al desbalance de clases. También se reportan precisión, recall, F1 y matriz de confusión con umbral 0.5, fijado para esta referencia inicial.

## 10. Plan de validación

Para el baseline de este avance, la muestra se divide de forma estratificada y reproducible:

- 70% para entrenamiento: 80,500 préstamos.
- 15% para validación: 17,250 préstamos.
- 15% para una reserva interna de prueba: 17,250 préstamos.

La estratificación mantiene aproximadamente la misma proporción de `Fully Paid` y `Charged Off` en las tres particiones. Se utiliza una semilla aleatoria igual a 42.

El baseline y sus transformaciones se ajustan solo con entrenamiento; las métricas se calculan solo sobre validación. La reserva interna de prueba no se usa para ajustar ni evaluar el modelo. Sin embargo, la exploración descriptiva se realizó sobre los 115,000 registros antes de dividirlos, por lo que esta reserva sí está incluida en las gráficas y estadísticas iniciales y no constituye una prueba completamente ajena a la exploración.

Para la entrega final se planificará una evaluación independiente con registros del dataset original que no pertenezcan a esta muestra, separados antes de explorarlos o tomar nuevas decisiones de modelado. También se estudiará una validación temporal. Estas evaluaciones futuras todavía no se han ejecutado.

## 11. Modelo baseline

El baseline es una regresión logística implementada mediante un pipeline de scikit-learn. Corresponde al baseline mínimo de clasificación solicitado para la primera entrega. Su función es establecer una referencia cuantitativa; la comparación de familias de modelos corresponde al trabajo posterior.

El pipeline incluye:

- Imputación por mediana para variables numéricas.
- Indicadores de ausencia para variables numéricas.
- Estandarización de variables numéricas.
- Imputación por moda para variables categóricas.
- Codificación one-hot para variables categóricas, agrupando categorías con menos de 20 observaciones en entrenamiento e ignorando categorías desconocidas.
- Regresión logística con ponderación balanceada de clases.

En validación (17,250 préstamos), el baseline obtuvo PR-AUC (AP) de **0.3610** frente a una prevalencia de incumplimiento de **0.2003**, y ROC-AUC de **0.6948**. Con umbral 0.5, la precisión fue **0.3120**, el recall **0.6418** y el F1 **0.4199**. Detectó 2,218 de los 3,456 incumplimientos, con 4,891 falsos positivos. Esto muestra señal predictiva y también un costo importante en alertas incorrectas. Son resultados internos de validación; el alcance de la reserva de prueba se describe en la sección 10.

La ponderación de clases ayuda a tratar el desbalance, pero las salidas de `predict_proba` no se han calibrado. En este avance se evalúa principalmente el ordenamiento del riesgo; aún no se presentan esas salidas como probabilidades de incumplimiento calibradas para uso operativo.

## 12. Riesgos técnicos

- La muestra debe conservar la diversidad temporal y la distribución de la variable objetivo.
- El dataset contiene valores faltantes estructurales y variables con más del 95% de ausencia.
- Existen valores extremos en ingreso, deuda, utilización de crédito y otros indicadores financieros.
- La clase `Charged Off` es minoritaria.
- Las políticas crediticias y las condiciones económicas cambiaron entre 2007 y 2018.
- El dataset contiene únicamente préstamos aceptados, lo que genera sesgo de selección.
- Los préstamos con resultado definitivo de años recientes pueden no representar a todos los préstamos emitidos en esos años.
- Algunas variables geográficas y socioeconómicas podrían actuar como proxies de características sensibles.
- Los resultados corresponden a Lending Club en Estados Unidos y no se pueden generalizar directamente a otras entidades o países.
- El modelo identificará asociaciones históricas y no relaciones causales.

## 13. Plan de trabajo semanas restantes

1. Definir y separar los datos de evaluación final antes de explorarlos, excluyendo los identificadores de esta muestra; estudiar una partición temporal.
2. Revisar la calidad de datos y consolidar las reglas de limpieza.
3. Crear y evaluar nuevas variables predictivas.
4. Comparar la regresión logística con árboles de decisión, Random Forest y Gradient Boosting.
5. Realizar búsqueda de hiperparámetros usando entrenamiento y validación, sin consultar la prueba final.
6. Evaluar calibración de probabilidades y selección de umbral con datos de desarrollo.
7. Analizar errores por propósito, plazo, FICO, ingreso, vivienda y ubicación, y aplicar interpretabilidad global y local.
8. Documentar sesgos, limitaciones y consideraciones éticas.
9. Evaluar una sola vez la solución seleccionada en la prueba final y elaborar el informe y la presentación.

