# Datos y procedencia

La fuente es [Lending Club Loan Data, versión 3](https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3), que declara licencia CC0. El archivo original contiene 2.260.701 registros y 151 columnas. Su SHA-256 concreto se conserva en `outputs/data_manifest.json`. No se debe asumir que otra versión del dataset sea idéntica.

| Archivo | Uso | Git |
|---|---|---|
| `lending_club_muestra.csv.gz` | Muestra inicial de 115.000 préstamos resueltos, semilla 42 | Incluido |
| `sample_summary.json` | Origen y composición de esa muestra | Incluido |
| `accepted_2007_to_2018Q4.csv.gz` | Archivo original completo para los experimentos nuevos | Ignorado por tamaño |
| `processed/development.parquet` | Resultados terminales emitidos hasta 2014 | Regenerable, ignorado |
| `processed/holdout_2015.parquet` | Prueba por emisión, sin IDs de la muestra inicial | Regenerable, ignorado |
| `processed/holdout_2016.parquet` | Estrés temporal, sin IDs de la muestra inicial | Regenerable, ignorado |
| `processed/sealed_coverage.parquet` | Cobertura de todos los estados del snapshot | Se abre después de congelar |
| `processed/manifest.json` | Hashes de fuente, particiones, IDs y protocolo | Copia pública en `outputs/` |

La preparación verifica IDs únicos, particiones disjuntas y que todos los IDs de la muestra previa estén en la fuente. Solo `Fully Paid` y `Charged Off` integran el objetivo. Otros estados se conservan únicamente en estadísticas de cobertura. El muestreo de la búsqueda ocurre después de fijar los cortes temporales y utiliza exclusivamente desarrollo.

Se excluyen además los cuatro IDs de `experiments/previewed_source_ids.json`, vistos al inspeccionar el esquema y un ejemplo público. Su cobertura se conserva separadamente en `processed/sealed_preview_counts.parquet` para restarlos también del denominador de las cohortes externas.

El snapshot no reconstruye una fecha fiable de disponibilidad de cada etiqueta. Su evaluación por fecha de emisión es retrospectiva. Los préstamos recientes o de 60 meses pueden estar censurados; la cobertura por año/plazo y la sensibilidad de préstamos a 36 meses son necesarias para interpretar los resultados. Los préstamos rechazados no forman parte del problema definido.

Para regenerar los archivos derivados:

```powershell
.\.venv\Scripts\python.exe -m src.prepare --source data/accepted_2007_to_2018Q4.csv.gz
```

El comando se ejecuta desde la raíz del repositorio. No reemplaza unas cohortes preparadas existentes; para otro experimento se puede usar `--out` con un directorio nuevo.
