# Diagnóstico de datos y ajuste: segunda fase experimental

Esta fase contrasta hipótesis planteadas después de observar las pruebas del primer experimento. Los resultados anteriores permanecen intactos. No constituye el informe académico final.

## Pregunta y población

Se investiga si la disponibilidad de variables, la imputación, los extremos, la antigüedad de los datos, el plazo y los hiperparámetros mejoran la predicción de `Charged Off` frente a `Fully Paid`. La población principal son préstamos personales **aceptados a 36 meses**, con seguimiento casi completo para la cohorte de 2015. Los datos no representan a solicitantes rechazados.

El archivo registra pagos hasta marzo de 2019. En 2015, el 99,95% de los préstamos a 36 meses tiene estado terminal, frente al 67,08% de los de 60 meses. Las cifras corresponden al censo de la fuente, antes de excluir los IDs de la muestra inicial; por eso difieren ligeramente de las coberturas de la prueba original. Los préstamos activos no se etiquetan como pagados ni como incumplidos.

## Hipótesis y contrastes

| Hipótesis | Comprobación | Interpretación permitida |
|---|---|---|
| Faltantes e inválidos explican una parte del límite | Conteos por año; disponibilidad en entrenamiento; imputación, indicadores, filtros y recorte mediante CV | Existencia de faltantes no demuestra daño: se necesita mejora en la ablación |
| Seleccionar únicamente resultados terminales distorsiona cohortes recientes | Cobertura por plazo/año, diferencias observables entre terminales y no terminales, límites inferior/superior de la tasa final | Se puede medir selección y falta de identificación; no conocer el futuro de los activos |
| La población y la frecuencia cambian entre años | Tasas por plazo con cobertura y distribuciones de ingresos, FICO, deuda y cuentas | Un cambio entre 2013 y 2015 a 36 meses es más interpretable que una tasa de 2017 condicionada a terminar pronto |
| La muestra inicial es poco representativa | Distancia de variación total frente a terminales y frente a todos los aceptados | Un muestreo aleatorio de terminales puede representar bien a terminales y mal a todos los aceptados |
| El ajuste de hiperparámetros es insuficiente | Nuevas búsquedas temporales y comparación con parámetros anteriores sobre las mismas filas y variables | Atribución al ajuste en la validación, sin confundirlo con incorporar otra población o más información |
| SVM puede mejorar la discriminación | LinearSVC, SVC RBF exacta y aproximación RBF Nyström; comparación adicional con presupuesto igual | Diferenciar algoritmo, presupuesto de filas y aproximación de kernel |

Las recetas combinadas se exploran primero con LightGBM fijo; después se ajustan las familias sobre la receta seleccionada. Un contraste adicional cambia un factor a la vez. Las comparaciones primarias siempre validan sobre el mismo plazo y las mismas cohortes. Los intervalos y diferencias de AP no deben comparar poblaciones distintas.

En el cribado, los hiperparámetros anteriores fijan el máximo de árboles y se permite parada temprana por AP en cada corte; las iteraciones efectivas se conservan en `recipe_screen.json`. La ablación de un factor usa el número anterior de árboles completo, sin parada temprana, idéntico entre variantes. Por eso sus referencias pueden diferir ligeramente: las diferencias de limpieza se comparan dentro de cada bloque, sin restar directamente AP de bloques con reglas de ajuste distintas.

## Protocolo y reutilización de pruebas

`protocol_v2.json` fija validaciones 2013 H2 y 2014 H1, selección 2014 Q3, calibración octubre-noviembre y política de umbral diciembre. El ajuste definitivo llega a septiembre de 2014. La receta y el modelo se seleccionan antes de generar nuevas predicciones sobre los benchmarks 2015/2016.

Sin embargo, esos benchmarks ya fueron observados en la primera fase. Se describen como **evaluaciones retrospectivas reutilizadas**, no como una nueva prueba final independiente. Un intervalo bootstrap pareado mide variación condicional de muestreo, pero no elimina la reutilización adaptativa. Para una nueva confirmación final se requiere otra fuente o actualización con resultados maduros no utilizados.

2017/2018 no resuelven automáticamente el problema: solo 40,12% y 11,97% de sus préstamos a 36 meses tienen desenlace terminal. Excluir el resto cambia la población hacia préstamos que terminan pronto. Tampoco se reconstruyen fechas fiables de disponibilidad histórica de las etiquetas de entrenamiento.

La revisión mensual de 2016 utiliza un criterio de cobertura de 99,5%, fijado antes de calcular predicciones nuevas para ese subconjunto. Ningún mes lo cumple: enero alcanza 99,37% y febrero 98,44%. Se conserva el resultado negativo del criterio; no se baja el umbral para fabricar una cohorte madura. La tabla mensual permite estudiar los límites de tasas sin asignar desenlaces a los préstamos activos.

El presupuesto de búsqueda distingue ensayos ejecutados de configuraciones en espera. Se registró una interrupción administrativa para corregir ese contador, conservando los ensayos completos. La transición posterior entre búsqueda y selección se realiza por etapas para incluir los controles de parámetros anteriores antes de generar nuevas predicciones del benchmark.

Para las variantes RBF se incluyeron candidatos explícitos de gamma bajo y C alto dentro de los rangos y de los 12 intentos de cada estrategia. Es una búsqueda guiada por los resultados de desarrollo, no por nuevas predicciones del benchmark. Los candidatos y el motivo quedan en `outputs/research_v2/svm_manual_probes.json`; la cola de espera no aumenta el número de ensayos ejecutados previsto.

Algunas SVC RBF alcanzaron el límite numérico de 20.000 iteraciones. Antes de seleccionar se repiten esas mismas configuraciones con límite 100.000, manteniendo C, gamma, ponderaciones, filas y tolerancia. Estas repeticiones del solver se cuentan aparte, en `svm_solver_retries.json`; las soluciones no convergidas no son elegibles. Aumentar el límite no modifica las soluciones que ya convergieron antes de él.

Las interrupciones/reanudaciones y la inserción guiada de candidatos pueden cambiar la trayectoria de TPE aunque se conserve la semilla. Los CSV registran la secuencia efectivamente ejecutada y los JSON conservan los parámetros elegidos. La reconstrucción con configuraciones fijas permite verificar el resultado predictivo sin depender de repetir exactamente esa trayectoria de búsqueda. Una búsqueda nueva puede escoger otra configuración; no constituye una confirmación final adicional sobre una prueba ya observada.

Antes de generar predicciones nuevas del benchmark se añadió un control de disponibilidad desde enero de 2013: `tot_cur_bal`, `total_rev_hi_lim` y `num_rev_accts` tienen 51,98% de ausencia en 2012 y 0% en 2013. Se conserva el cribado original desde 2012 como referencia y se comparan cinco variantes adicionales con entrenamiento desde 2013 y parámetros fijos, sobre el mismo Q3. La elección final puede conservar 2012 o elegir 2013. No se aumenta la búsqueda de hiperparámetros para estas variantes. La revisión se registra antes de congelar; no responde a resultados nuevos de 2015/2016.

## Fuentes primarias

Antes del nuevo benchmark también se fijó un límite de predicción para las SVM RBF exactas no seleccionadas: todos los modelos se comparan sobre la misma muestra estratificada de 20.000 préstamos por año (semilla 42, huellas de IDs conservadas). La decisión responde al costo observado de 146–163 segundos para predecir 40.595 préstamos de validación, sin usar métricas nuevas del benchmark. El ganador seleccionado y todos los modelos no RBF se evalúan adicionalmente en las cohortes completas. Las dos tablas tienen alcances distintos y no se compara directamente su AP.

- [Documentación oficial de SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html): la SVM de kernel exacto escala mal con el número de filas. Se fija un límite de 12.000 en búsqueda y ajuste para esa variante; los otros métodos también se contrastan con las mismas 12.000 filas. Las puntuaciones SVM se calibran en un bloque temporal separado, sin utilizar validación aleatoria interna para producir probabilidades.
- [Aproximación de kernels en scikit-learn](https://scikit-learn.org/stable/modules/kernel_approximation.html): Nyström aproxima un mapa de características del kernel. Aquí se combina con LinearSVC; no se presenta como SVC RBF exacta.
- [Ausentes en CatBoost](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing): el tratamiento nativo de numéricas ausentes puede conservar información que la imputación elimina. Su utilidad concreta se contrasta empíricamente; ninguna estrategia se supone superior de antemano.
- [Fuente original de Lending Club en Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club): se conserva la procedencia y el SHA-256 del archivo original. Una fecha reciente de una página o de una copia no acredita actualización de los desenlaces del préstamo.

## Sensibilidad al alcance de información

Se realiza un diagnóstico separado incorporando `grade`, `sub_grade` e `int_rate`, asignados por Lending Club. Es información del modelo/precio de la plataforma disponible después de su evaluación. Si mejora el resultado, demuestra el valor de esa información adicional; no una mejora del mismo modelo independiente anterior a la clasificación. No puede ganar la selección principal ni presentarse sin advertir ese cambio de alcance.

Los resultados tabulares y las conclusiones cuantitativas se generan en `outputs/research_v2/`; el notebook 06 los presenta junto con sus límites. No se retocan los datos ni las métricas del experimento congelado original.
