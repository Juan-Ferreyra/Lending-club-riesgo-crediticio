# Predicción del riesgo crediticio con Lending Club

Avance previo a la semana 1 del proyecto de Machine Learning. Se busca estimar la probabilidad de que un préstamo aceptado termine en `Charged Off`, utilizando únicamente información disponible al evaluar la solicitud. La [propuesta](proposal.md) describe el objetivo, los riesgos y el plan de trabajo.

## Ejecutar en Google Colab

1. [Abrir el notebook en Colab](https://colab.research.google.com/github/Juan-Ferreyra/Lending-club-riesgo-crediticio/blob/avance-semana-1-muestra-reproducible/notebooks/01_exploracion_inicial.ipynb).
2. Conectar el entorno de ejecución y elegir **Runtime → Run all** (o **Entorno de ejecución → Ejecutar todo**).

La primera celda utiliza la muestra incluida en el repositorio y la descarga automáticamente si no está en el entorno. No es necesario descargar el dataset original para ejecutar el notebook. Los resultados y las gráficas también están guardados en el archivo `.ipynb`.

Para ejecutarlo localmente desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

## Datos y reproducibilidad

La [muestra utilizada](data/lending_club_muestra.csv.gz) contiene **115 000 préstamos y 151 variables**: 91 959 `Fully Paid` y 23 041 `Charged Off`. Su [resumen](data/sample_summary.json) detalla la distribución por estado, año y plazo. El archivo comprimido ocupa aproximadamente **20,3 MiB**, por lo que se incluye directamente en el repositorio.

Procede de [Lending Club Loan Data, versión 3](https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3). La ficha de Kaggle indica licencia **CC0: Public Domain**; descargar el archivo original puede requerir una cuenta. Se tomaron 115 000 préstamos al azar, sin reemplazo y con semilla 42, de los registros cuyo estado definitivo es `Fully Paid` o `Charged Off`. El procedimiento está implementado en [`scripts/crear_muestra.py`](scripts/crear_muestra.py).

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

- [`proposal.md`](proposal.md): pregunta predictiva, variable objetivo, riesgos, validación y plan.
- [`notebooks/01_exploracion_inicial.ipynb`](notebooks/01_exploracion_inicial.ipynb): carga, exploración, visualizaciones, análisis de faltantes y valores extremos, limitaciones y baseline.
- [`data/lending_club_muestra.csv.gz`](data/lending_club_muestra.csv.gz): muestra utilizada en el análisis.
- [`data/sample_summary.json`](data/sample_summary.json): composición de la muestra.
- [`scripts/crear_muestra.py`](scripts/crear_muestra.py): procedimiento de muestreo.
- [`requirements.txt`](requirements.txt): dependencias para la ejecución local.

## Resultado inicial y alcance

En **validación** (17 250 préstamos), la regresión logística obtuvo PR-AUC **0.3610**, frente a una prevalencia de incumplimiento de **0.2003**; ROC-AUC **0.6948**, precisión **0.3120** y recall **0.6418** con umbral 0.5. El modelo discrimina mejor que la referencia basada en prevalencia, pero la precisión indica que aún hay falsos positivos. Son resultados preliminares de una división aleatoria y no prueban desempeño futuro: la evaluación temporal y el análisis de sesgos quedan para etapas posteriores. El conjunto de prueba permanece reservado.
