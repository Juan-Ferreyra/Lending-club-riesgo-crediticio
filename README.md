# Predicción del riesgo crediticio con Lending Club

Entrega previa de la semana 1 del proyecto de Machine Learning. El objetivo es estimar la probabilidad de que un préstamo aceptado termine en `Charged Off` usando información disponible antes del otorgamiento.

## Abrir en Google Colab

[Abrir el notebook de esta rama en Colab](https://colab.research.google.com/github/Juan-Ferreyra/Lending-club-riesgo-crediticio/blob/avance-semana-1-muestra-reproducible/notebooks/01_exploracion_inicial.ipynb).

Conectar el entorno y pulsar **Runtime → Run all**. La primera celda descarga automáticamente la muestra de esta rama si no está en `/content/`. Comprueba su SHA-256 para evitar mezclarla con la muestra anterior de Marcelo. No hace falta descargar el dataset completo ni crear un notebook de Kaggle para ejecutar este avance.

Si se abre el notebook de la rama `main`, se verá la versión anterior, que todavía pide subir manualmente otro archivo. Hasta que se apruebe esta rama, usar el enlace de arriba.

## Muestra y fuente

La muestra incluida en [`data/lending_club_muestra.csv.gz`](data/lending_club_muestra.csv.gz) tiene **115 000 filas y 151 columnas**: 91 959 `Fully Paid` y 23 041 `Charged Off`. Pesa aproximadamente **20,3 MiB** comprimida. No es la muestra anterior de Marcelo (92 191 / 22 809); los resultados de ambas no se deben mezclar. El resumen por clase, año y plazo está en `data/sample_summary.json`.

SHA-256 del archivo de muestra: `c863e3348c96428195f158bf456c8abc289515737d77f59981804c1f5eed82da`.

Fuente original: [Lending Club Loan Data, versión 3 en Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3), publicada con licencia **CC0: Public Domain** según la ficha de Kaggle. El archivo de origen es `accepted_2007_to_2018Q4.csv.gz`, incluido en el ZIP de Kaggle. La descarga de Kaggle puede requerir una cuenta.

La muestra se seleccionó al azar sin reemplazo, con semilla 42, entre los préstamos cuyo estado final es `Fully Paid` o `Charged Off`. El procedimiento exacto está en `scripts/crear_muestra.py`. Para regenerarla desde el ZIP original:

```bash
python -m pip install -r requirements.txt
python scripts/crear_muestra.py --archive data/lending-club.zip
```

Si se extrajo el archivo de aceptados:

```bash
python scripts/crear_muestra.py --source data/accepted_2007_to_2018Q4.csv.gz
```

El ZIP original no está en GitHub por su tamaño. La muestra **sí está en esta rama**, así que el profesor y el equipo pueden ejecutar el notebook sin una entrega adicional en Canvas. La muestra contiene solo préstamos aceptados con resultado definitivo; no representa a todos los solicitantes.

## Archivos

- `proposal.md`: pregunta, objetivo, riesgos, validación y plan.
- `notebooks/01_exploracion_inicial.ipynb`: exploración, seis gráficas y baseline ejecutado.
- `data/lending_club_muestra.csv.gz`: muestra exacta usada por el notebook.
- `data/sample_summary.json`: conteos verificables de la muestra.
- `scripts/crear_muestra.py`: reproducción de la selección.
- `requirements.txt`: dependencias para ejecución local.

## Resultado inicial

La regresión logística obtuvo en **validación** (17 250 préstamos) PR-AUC **0.3610** frente a una prevalencia de **0.2003**, ROC-AUC **0.6948**, precisión **0.3120** y recall **0.6418**. La partición de prueba quedó reservada. La división aleatoria sirve como línea base inicial; falta evaluar estabilidad temporal y sesgos en las semanas siguientes.
