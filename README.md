# Predicción del riesgo de incumplimiento en préstamos de Lending Club

**Grupo 7.** Marco Soto Maceda, Marcelo Ferreyra, Valentino Contreras, Rayhan Matos y Leonardo Panduro.

Se estima la probabilidad de que un préstamo personal aceptado por Lending Club termine en `Charged Off` (1) en lugar de `Fully Paid` (0). Solo se usa información disponible al originar el préstamo. La métrica principal es la **Average Precision (AP)**, adecuada para clases desbalanceadas. Las métricas secundarias son ROC-AUC, Brier y recall al revisar el 20% de mayor riesgo.

- **Informe final:** [reports/informe_final.md](reports/informe_final.md)
- **Presentación:** [reports/presentacion.md](reports/presentacion.md)
- **Historial completo de las fases** (avance S1, V1, V2, V3): [docs/historial_de_fases.md](docs/historial_de_fases.md)

## Resultados principales

| Evaluación | Modelo | Préstamos | AP | ROC-AUC |
|---|---|---:|---:|---:|
| **Prueba confirmatoria V1**, 2015, todos los plazos (no reutilizada) | Ensamble 60% CatBoost + 40% LightGBM | 343.507 | **0,4164** (baseline 0,3890) | 0,7334 |
| Benchmark reutilizado, 2015, 36 meses | Ensamble V1 | 258.856 | 0,2718 | 0,6920 |
| Benchmark reutilizado, 2015, 36 meses | **CatBoost V3 (modelo recomendado)** | 258.856 | **0,2801** | 0,6983 |

La mejora de V1 sobre el baseline es de +0,0274 AP, con IC95 pareado [+0,0252; +0,0296]. Todas las familias probadas, desde la regresión logística hasta CatBoost, quedan en un rango estrecho de AP. El límite parece estar en la información disponible al originar el préstamo y no en el algoritmo (ver informe, §10–12).

## Estructura

```
data/          muestra de 115.000 préstamos (incluida) y README del dataset
notebooks/     01 exploración · 02 limpieza · 03 modelos · 04 errores · 05 explicación · 06 diagnóstico/SVM · 07 optimización V3
src/           código principal (datos, variables, modelos, evaluación, entrenamiento) y src/pipeline.py
research/      módulos de la fase V2 · optimization/ módulos de la fase V3 · fase4/ técnicas del curso (V4)
experiments/   protocolos fijados antes de evaluar (protocol*.json) y bitácoras
outputs/       resultados, figuras y verificaciones; outputs/metrics.json consolida las cifras
reports/       informe final, presentación y figures/
tests/         29 pruebas de integridad, leakage y protocolo
```

Los notebooks 02 a 07 **leen artefactos ya generados** y no entrenan modelos. El código de V1 a V3 está congelado con hashes SHA-256: no se edita, solo se agregan archivos nuevos.

## Instalación

Se requiere Python 3.12 o superior (probado con 3.12 en Windows y 3.13 en Linux).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate
pip install -r requirements-lock.txt # versiones exactas ejecutadas
python -m unittest discover -s tests # 29 pruebas, segundos
```

## Ejecución rápida (demo, menos de 1 minuto, sin descargar nada)

```bash
python -m src.pipeline --mode sample
```

Usa solo la muestra incluida. Entrena con préstamos hasta 2013, selecciona en 2014 entre regresión logística, LightGBM y CatBoost, y evalúa una sola vez en 2015. Guarda los resultados en `outputs/sample_demo/`. **Es una demostración:** con unas 20.000 filas de entrenamiento sus cifras son ruidosas y no reproducen las del archivo completo.

## Reproducción completa (horas de CPU)

1. Descargar `accepted_2007_to_2018Q4.csv.gz` de [Lending Club Loan Data v3 (Kaggle, CC0)](https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3) y colocarlo en `data/`. Su SHA-256 debe ser `55c16f75…62a2d` (registrado en `outputs/data_manifest.json`).
2. Ejecutar la evaluación V1 y la Fase 4:

```bash
python -m src.pipeline --mode full --source data/accepted_2007_to_2018Q4.csv.gz --output outputs/reproduction
```

3. Opcional: reproducir V2 y V3 con sus comandos, en [docs/historial_de_fases.md](docs/historial_de_fases.md).
4. Regenerar `outputs/metrics.json` y `reports/figures/`:

```bash
python scripts/build_reports.py
```

La Fase 4 también puede ejecutarse por separado (unos 15 minutos y no abre 2015/2016): `python -m fase4.experiment --output outputs/research_v4`.

Todas las semillas son 42. Una salida que ya contiene una evaluación final no puede reutilizarse para ajustar modelos: hay que usar un directorio nuevo.

## Datos

Las fuentes son [Kaggle, versión 3](https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3), con licencia CC0: Public Domain según la ficha. El archivo tiene 2.260.701 préstamos aceptados entre 2007 y 2018, con 151 variables. Solo se usan préstamos con desenlace terminal (`Fully Paid` / `Charged Off`). Los préstamos vigentes se excluyen, y la cobertura de cada cohorte se reporta en `outputs/final/maturity_coverage.csv`. Más detalles en [data/README.md](data/README.md).
