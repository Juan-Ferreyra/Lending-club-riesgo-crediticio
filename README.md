# Predicción del riesgo crediticio con Lending Club

Primera entrega (avance de semana 1) del proyecto de Machine Learning. Se busca estimar el riesgo de que un préstamo aceptado termine en `Charged Off`, utilizando información disponible al completar la evaluación de la solicitud y antes del otorgamiento. La [propuesta](proposal.md) describe el objetivo, los riesgos y el plan de trabajo.

Este avance incluye la exploración de datos y un baseline de regresión logística, entrenado y evaluado, que corresponde a la tarea de clasificación binaria: `Charged Off = 1` y `Fully Paid = 0`. La comparación de otros modelos, la calibración y la evaluación final forman parte del plan de las siguientes semanas.

## Ejecutar en Google Colab

1. [Abrir el notebook en Colab](https://colab.research.google.com/github/Juan-Ferreyra/Lending-club-riesgo-crediticio/blob/avance-semana-1-muestra-reproducible/notebooks/01_exploracion_inicial.ipynb).
2. Conectar el entorno de ejecución y elegir **Runtime → Run all** (o **Entorno de ejecución → Ejecutar todo**).

La primera celda utiliza la muestra incluida en el repositorio y la descarga automáticamente si no está en el entorno. No es necesario descargar el dataset original para ejecutar el notebook. Los resultados y las gráficas también están guardados en el archivo `.ipynb`.

Para ejecutarlo localmente, usar Python 3.10 o superior. Primero obtener esta versión del repositorio:

```bash
git clone --branch avance-semana-1-muestra-reproducible --single-branch https://github.com/Juan-Ferreyra/Lending-club-riesgo-crediticio.git
cd Lending-club-riesgo-crediticio
```

Desde la raíz del repositorio, instalar las dependencias y abrir el notebook:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

## Datos y reproducibilidad

La [muestra utilizada](data/lending_club_muestra.csv.gz) contiene **115 000 préstamos y 151 variables**: 91 959 `Fully Paid` y 23 041 `Charged Off`. Su [resumen](data/sample_summary.json) detalla la distribución por estado, año y plazo. El archivo comprimido ocupa aproximadamente **20,3 MiB**, por lo que se incluye directamente en el repositorio.

Procede de [Lending Club Loan Data, versión 3](https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3). La ficha de Kaggle indica licencia **CC0: Public Domain**; descargar el archivo original puede requerir una cuenta. Se tomaron 115 000 préstamos al azar, sin reemplazo y con semilla 42, de los registros cuyo estado definitivo es `Fully Paid` o `Charged Off`. Las funciones de muestreo están en [`src/data.py`](src/data.py) y se ejecutan mediante [`scripts/crear_muestra.py`](scripts/crear_muestra.py).

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

- [`proposal.md`](proposal.md): los 13 apartados de la propuesta, desde el título y los integrantes hasta el plan de trabajo.
- [`notebooks/01_exploracion_inicial.ipynb`](notebooks/01_exploracion_inicial.ipynb): los 8 componentes de la exploración requerida: carga, dimensiones y tipos, faltantes, objetivo, seis visualizaciones, outliers, sesgos/leakage/limitaciones y baseline ejecutado.
- [`data/lending_club_muestra.csv.gz`](data/lending_club_muestra.csv.gz): muestra utilizada en el análisis.
- [`data/sample_summary.json`](data/sample_summary.json): composición de la muestra.
- [`src/data.py`](src/data.py): funciones reutilizables de lectura y muestreo.
- [`scripts/crear_muestra.py`](scripts/crear_muestra.py): comando para regenerar la muestra.
- [`requirements.txt`](requirements.txt): dependencias para la ejecución local.

## Validación y resultado inicial

La división aleatoria estratificada usa semilla 42: 80 500 préstamos para entrenamiento (70%), 17 250 para validación (15%) y 17 250 como reserva interna de prueba (15%). El pipeline ajusta imputación, escalado, codificación y regresión logística únicamente con entrenamiento. Utiliza 19 predictores anteriores al otorgamiento y excluye el estado final, identificadores y datos de pagos o cobranza.

En **validación**, la regresión logística obtuvo PR-AUC **0.3610**, frente a una prevalencia de incumplimiento de **0.2003**; ROC-AUC **0.6948**, precisión **0.3120**, recall **0.6418** y F1 **0.4199** con umbral 0.5. La columna PR-AUC corresponde a Average Precision (AP), calculada con `average_precision_score`. El baseline detecta 2 218 de los 3 456 incumplimientos de validación y genera 4 891 falsos positivos. Las probabilidades no se han calibrado; estas métricas describen el ordenamiento y la clasificación inicial del riesgo.

La reserva interna de prueba no se utiliza para ajustar ni evaluar el modelo, pero sus filas sí forman parte de la exploración descriptiva realizada sobre toda la muestra. Por ello no se presenta como una prueba completamente ajena a la exploración. Las métricas reportadas son preliminares y corresponden únicamente a validación. Para la entrega final se prevé separar, antes de explorarlos, registros del dataset original que no pertenezcan a esta muestra y estudiar una validación temporal, según el plan de la propuesta.
