# Guía de navegación de notebooks

Este directorio contiene los notebooks del proyecto, organizados de forma incremental desde la exploración inicial hasta la fase final de investigación y contraste de hipótesis.

---

## 1. Mapa de correspondencia con la pauta de evaluación

La estructura recomendada de la pauta oficial (`instrucciones.md`) solicita cuatro notebooks principales (`01` a `04`). Para profundizar en la investigación técnica exigida en la rúbrica (comparación de familias, diagnóstico de hiperparámetros, interpretabilidad y técnicas del curso), el proyecto se extendió hasta 8 notebooks:

| Notebook | Fase | Propósito principal | Requisito de la pauta / rúbrica |
|---|:---:|---|---|
| **[01_exploracion_inicial.ipynb](01_exploracion_inicial.ipynb)** | S1 | Carga de muestra (115.000 filas), EDA completo, >5 visualizaciones, desbalance, análisis de outliers, riesgos de leakage y primer baseline logístico (AP 0,361). | Requisito de la Entrega Previa y EDA del informe (§5). |
| **[02_limpieza_features.ipynb](02_limpieza_features.ipynb)** | V1 | Contrato de variables de originación, lista negra `FORBIDDEN` contra fuga de datos, verificación de esquemas y reglas de dominio. | Limpieza, preparación y control de leakage (§4). |
| **[03_modelos.ipynb](03_modelos.ipynb)** | V1 | Comparación de 5 familias, búsqueda con Optuna (96 intentos), selección en 2014 S1, calibración y congelamiento previo a la prueba 2015. | Modelado, baselines y búsqueda de hiperparámetros (§8 y §9). |
| **[04_analisis_errores.ipynb](04_analisis_errores.ipynb)** | V1 | Desglose de FPR/FNR por segmentos (FICO, ingreso, plazo), fiabilidad de calibración, SHAP global y explicaciones locales. | Análisis de errores e interpretabilidad (§11 y §12). |
| **[05_explicacion.ipynb](05_explicacion.ipynb)** | V1/V2 | Fundamentos matemáticos de los algoritmos utilizados, diagramas de arquitectura del ensamble y lectura de límites. | Comunicación y fundamentación teórica (§8 y §10). |
| **[06_diagnostico_datos_svm.ipynb](06_diagnostico_datos_svm.ipynb)** | V2 | Diagnóstico de población terminal a 36 meses, auditoría de faltantes por época de buró y evaluación controlada de SVM (lineal, RBF y Nyström). | Complejidad del dataset y comparación justa de modelos (§3 y §8.2). |
| **[07_optimizacion_predictiva.ipynb](07_optimizacion_predictiva.ipynb)** | V3 | Optimización conjunta con 20 ratios financieros, variables de contexto (`zip_code`, `emp_title`), parada temprana por AP exacta y 26 ensambles. | Modelo recomendado V3 y demostración del techo predictivo (§8.3 y §10.2). |
| **[08_tecnicas_curso.ipynb](08_tecnicas_curso.ipynb)** | V4 | Contraste de técnicas del temario sobre desarrollo Q4 2014: Regularización L1, PCA, Naive Bayes, Árboles, ExtraTrees, MLP, HistGB, curva de aprendizaje y sesgo-varianza. | Rúbrica de modelado, familias adicionales y sesgo algorítmico (§8.4). |

---

## 2. Principio de diseño: Visualización instantánea y reproducibilidad

> [!NOTE]
> **Por qué los notebooks 02 al 08 no reentrenan modelos:**  
> Entrenar todas las familias, búsquedas bayesianas de Optuna y curvas de aprendizaje sobre cientos de miles de préstamos toma varias horas de cómputo.  
> Para permitir una **evaluación inmediata y sin bloqueos de hardware**, los notebooks `02` a `08` están diseñados para cargar, auditar y graficar los artefactos, tablas y métricas ya consolidados en el directorio `outputs/`.

- **Para inspección / corrección:** Abrir y ejecutar cualquier celda de los notebooks toma pocos segundos, ya que consumen los archivos serializados en `outputs/`.
- **Para reentrenamiento desde cero:** El código de entrenamiento completo reside en `src/` (`src/pipeline.py`, `src/train.py`, etc.) y puede ejecutarse siguiendo las instrucciones de [README.md](../README.md).
