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

Se utilizará el dataset **Lending Club Loan Data**, disponible en Kaggle:

https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3

La ficha de Kaggle declara la licencia **CC0: Public Domain**. La descarga del archivo original puede requerir iniciar sesión. La muestra incluida en `data/lending_club_muestra.csv.gz` se genera con `scripts/crear_muestra.py` a partir de `accepted_2007_to_2018Q4.csv.gz` de esa versión, con semilla 42.

El archivo original contiene 2,260,701 registros y 151 variables de préstamos emitidos entre 2007 y 2018. De ellos, 1,345,310 tienen estado definitivo `Fully Paid` o `Charged Off`. La muestra aleatoria sin reemplazo tiene 115,000 registros y 151 variables de 2007 a 2018, con plazos de 36 y 60 meses. Contiene 91,959 préstamos `Fully Paid` (79.96%) y 23,041 `Charged Off` (20.04%). La composición por año y plazo está en `data/sample_summary.json`.

Esta muestra representa solo préstamos con resultado definitivo; no representa a todos los solicitantes ni a todos los préstamos emitidos, especialmente los más recientes.

## 4. Pregunta predictiva

¿Cuál es la probabilidad de que un préstamo personal aceptado por Lending Club termine en incumplimiento, utilizando las características del solicitante, del préstamo solicitado y de su historial crediticio registradas antes del otorgamiento?

## 5. Variable objetivo

La variable objetivo se construirá a partir de `loan_status`:

- `1`: préstamo en incumplimiento, correspondiente a `Charged Off`.
- `0`: préstamo pagado, correspondiente a `Fully Paid`.

Se trata de un problema de clasificación binaria supervisada.

## 6. Unidad de predicción

La unidad de predicción es un préstamo personal aceptado por Lending Club. Cada fila representa un préstamo y la predicción se realiza antes de observar su comportamiento de pago.

## 7. Variables disponibles antes de la predicción

El baseline utilizará variables disponibles al momento de evaluar la solicitud.

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

Se excluirán las variables creadas después de otorgar el préstamo o relacionadas directamente con su resultado. Entre ellas se encuentran:

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

También se excluirán `grade`, `sub_grade`, `int_rate` e `installment` porque representan decisiones o evaluaciones realizadas por Lending Club y podrían transferir al modelo información de su sistema interno de riesgo.

## 9. Métrica principal y métrica secundaria

La métrica principal será **PR-AUC**, porque el incumplimiento es la clase minoritaria y esta métrica evalúa la relación entre precisión y recall.

La métrica secundaria será **ROC-AUC**, porque mide la capacidad del modelo para ordenar los préstamos cumplidos e incumplidos a través de distintos umbrales.

Accuracy no se utilizará como métrica principal debido al desbalance de clases. También se reportarán precisión, recall, F1 y matriz de confusión como medidas complementarias.

## 10. Plan de validación

La muestra se dividirá de forma estratificada y reproducible:

- 70% para entrenamiento.
- 15% para validación.
- 15% para prueba final.

La estratificación mantendrá aproximadamente la misma proporción de `Fully Paid` y `Charged Off` en las tres particiones. Se utilizará una semilla aleatoria igual a 42.

El conjunto de prueba se mantendrá separado y no se utilizará para seleccionar variables, umbrales o decisiones de modelado. En este avance se reportan métricas solo sobre validación. Más adelante se estudiará una validación temporal.

## 11. Modelo baseline

El baseline es una regresión logística implementada mediante un pipeline de scikit-learn.

El pipeline incluye:

- Imputación por mediana para variables numéricas.
- Indicadores de ausencia para variables numéricas.
- Estandarización de variables numéricas.
- Imputación por moda para variables categóricas.
- Codificación one-hot para variables categóricas.
- Regresión logística con ponderación balanceada de clases.

En validación (17,250 préstamos), el baseline obtuvo PR-AUC de **0.3610** frente a una prevalencia de incumplimiento de **0.2003** (1.80 veces), y ROC-AUC de **0.6948**. Con umbral 0.5, la precisión fue **0.3120** y el recall **0.6418**. Es una referencia inicial, no una estimación de desempeño futuro: falta validar estabilidad temporal. La partición de prueba no se consultó.

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

1. Revisar la calidad de datos y consolidar las reglas de limpieza.
2. Crear y evaluar nuevas variables predictivas.
3. Comparar la regresión logística con árboles de decisión, Random Forest y Gradient Boosting.
4. Realizar búsqueda de hiperparámetros sin utilizar el conjunto de prueba.
5. Evaluar calibración de probabilidades y selección de umbral.
6. Analizar errores por propósito, plazo, FICO, ingreso, vivienda y ubicación.
7. Aplicar técnicas de interpretabilidad global y local.
8. Documentar sesgos, limitaciones y consideraciones éticas.
9. Elaborar el informe final y la presentación.

