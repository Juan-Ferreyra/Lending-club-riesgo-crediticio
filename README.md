# Predicción del riesgo crediticio con Lending Club


## Objetivo

Predecir la probabilidad de incumplimiento de un préstamo personal de Lending Club utilizando información disponible al momento de evaluar la solicitud.

## Estructura

```text
lending-club-riesgo-crediticio/
├── README.md
├── proposal.md
└── notebooks/
    └── 01_exploracion_inicial.ipynb
```

## Dataset

Fuente original:

https://www.kaggle.com/datasets/wordsforthewise/lending-club

Para la entrega previa se utiliza `lending_club_muestra.csv.gz`, una muestra representativa de 115,000 registros, 151 variables y cobertura temporal de 2007 a 2018.

El archivo de datos no se incluye en el repositorio debido a su tamaño.

## Ejecución en Google Colab

1. Abrir `notebooks/01_exploracion_inicial.ipynb` en Google Colab.
2. Seleccionar el ícono de carpeta en el panel izquierdo.
3. Subir `lending_club_muestra.csv.gz` al directorio `/content/`.
4. Ejecutar todas las celdas en orden.

El notebook busca automáticamente cualquier archivo cuyo nombre comience con `lending_club_muestra` y termine en `.csv.gz`.

## Dependencias

Google Colab ya incluye las principales dependencias. Si se ejecuta localmente, instalar:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Reproducibilidad

- Semilla aleatoria: 42.
- División estratificada: 70% entrenamiento, 15% validación y 15% prueba.
- Preprocesamiento y modelo dentro de un pipeline.
- Variables posteriores al préstamo excluidas.
- Regresión logística como baseline.
- PR-AUC como métrica principal.
- ROC-AUC como métrica secundaria.

