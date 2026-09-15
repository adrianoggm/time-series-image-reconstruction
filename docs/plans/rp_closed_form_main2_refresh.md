# Plan de regeneración RP y actualización de `main2.tex`

Fecha: 2026-07-29

## Objetivo

Regenerar y validar todos los resultados de Recurrence Plot (RP) después de
sustituir la rejilla angular de 720 rotaciones por la solución cerrada de la
orientación, y actualizar posteriormente `doc/main2.tex` y
`doc/supplementary.tex` sin mezclar resultados antiguos y nuevos.

Los artefactos presentes en el árbol de trabajo indican que el cambio es
prometedor: elimina el error de discretización de la orientación en flotante,
mejora el NMAE del PNG limpio y reduce el tiempo de reconstrucción. No obstante,
algunos resultados mezclan cambios adicionales o se generan desde fuentes
obsoletas. Por ello, ninguna cifra se considera definitiva hasta superar los
controles de procedencia descritos en este documento.

## Principio de trabajo

El orden debe ser:

```text
Fijar el contrato del decoder
        ↓
Regenerar los seis folds RP
        ↓
Aplicar controles de integridad
        ↓
Recalcular fidelidad y robustez
        ↓
Reevaluar clasificación y estadística
        ↓
Actualizar main2.tex y supplementary.tex
        ↓
Comprobar coherencia y compilar
```

Las cifras que ya se han introducido en `main2.tex` se consideran provisionales
hasta verificar su trazabilidad. Ni la prueba inicial de 10 ventanas/30 series
ni la ablación posterior de 150 ventanas sustituyen al protocolo completo para
las cifras globales de 8.172 ventanas.

## Auditoría de consistencia del plan

Revisión realizada el 2026-07-29 contra el código, los scripts y los artefactos
que existen actualmente en el árbol de trabajo.

| Sección | Estado observado | Inconsistencia o riesgo | Corrección necesaria |
|---|---|---|---|
| Objetivo y principio | Correctos en intención | Hay resultados nuevos, pero no todos tienen la misma procedencia | Mantener el orden del plan y fijar una única fuente para cada cifra |
| Fase 0 | Parcialmente implementada | En `Reconstruct_RP`, el contrato metadata-free es `rp_range=None`, no la cadena `"diagonal"` | Documentar y probar el contrato real; reservar el rango para el oracle |
| Fase 1 | Existe una copia histórica | El llamado baseline `grid720` también usaba rango del operador y otro criterio de signo | Etiquetarlo como baseline de producción histórico, no como ablación causal de la rejilla |
| Fase 2 | Hay scripts para los seis folds | La ejecución se realiza en otro checkout WSL (`/home/adrianogarcia/tfg`) | Sincronizar código, verificar hashes de datos/splits y registrar el commit antes de ejecutar |
| Fase 3 | Existe un comprobador de regeneración | Suma LOTO y LOSO como si fueran 16.344 ventanas distintas y solo muestrea algunos controles | No publicar ese agregado duplicado; usar 8.172 ventanas únicas por protocolo y declarar el muestreo |
| Fase 4 | Existe la ablación aislada de 150 ventanas | El closed form mejora NMAE y tiempo, pero no mejora uniformemente el residuo de ciclo/imagen | Informar ambos resultados y no llamar “fallo” a toda selección no oracle sin mostrar el regret |
| Fase 5 | Ya existen generadores de fidelidad y robustez | Los scripts y resultados auxiliares están ignorados por Git y conviven varias definiciones de métrica | Elegir un generador canónico, versionarlo y congelar datos estructurados con definiciones explícitas |
| Fase 6 | Se han preparado reevaluación y estadística | `_make_accuracy_tables.py` y `_make_loso_tables.py` leen `.results_snapshot`, no los resultados nuevos | Prohibir esos emisores para esta actualización y usar `_emit_main2_numbers.py` con resultados vivos |
| Fase 7 | La explicación ya se ha empezado a actualizar | El autovector ordinario minimiza un residuo LS no normalizado, no necesariamente el cociente normalizado que devuelve `align_states` | Redactar la afirmación con el objetivo realmente optimizado o cambiar el código a un problema generalizado |
| Fase 8 | `main2.tex` ya contiene varias cifras nuevas | La fila grid/closed sobre 8.172 ventanas puede estar confundida por metadatos y selector; algunas afirmaciones sobre residuo son demasiado fuertes | Trazar cada número y separar la comparación histórica de la ablación de orientación pura |
| Fase 9 | El suplemento ya incluye la ablación | Presenta el closed form como mínimo exacto del residuo y destaca solo las métricas favorables | Precisar el objetivo LS y añadir la pequeña degradación observada en ciclo/residuo con PNG/ruido |
| Fase 10 | README ya apunta a `main2.tex` | Un barrido global encuentra planes históricos y gráficos que usan “grid” con otro significado | Buscar solo fuentes pertinentes y clasificar cada coincidencia antes de editar |
| Fase 11 | Hay scripts de compilación y referencias | El compilador puede continuar tras un fallo y el verificador de referencias no cubre todos los comandos | Convertir la compilación en una puerta estricta y revisar también citas, `eqref`, duplicados y referencias externas |
| Entregables | La mayoría están iniciados | Muchos scripts `_*.py`, `_*.sh` y `results/` no quedan versionados | Mover o forzar el seguimiento de los generadores y guardar hashes/procedencia de los arrays grandes |
| Finalización | El criterio conceptual es correcto | “Hay un PDF” no garantiza que proceda de los datos y fuentes definitivos | Exigir manifiesto cifra → artefacto → script → commit y compilación limpia |

### Bloqueos que deben resolverse antes de dar cifras por definitivas

1. Separar el baseline histórico —rejilla, rango del operador y selector
   antiguo— de la ablación pura donde solo cambia `orientation_method`.
2. Elegir un único checkout de ejecución o registrar una sincronización
   reproducible Windows → WSL con hashes.
3. Impedir que las tablas de clasificación se regeneren desde
   `.results_snapshot`.
4. Corregir la afirmación matemática sobre qué residuo minimiza exactamente el
   autovector de la matriz (2\times2).
5. Versionar los scripts de reproducción y un manifiesto de resultados, aunque
   los arrays grandes permanezcan fuera de Git.

## Fase 0. Cerrar el contrato del decoder antes de regenerar

### 0.1. Unificar la ruta metadata-free

`main2.tex` declara que el decoder RP recupera el nivel cero mediante la
diagonal del PNG y no necesita el rango real de la matriz RP. En el árbol de
trabajo actual, `src/reconstruct_tseries.py` ya ha dejado de construir ese rango
para producción y `src/recurrence_plots.py` interpreta `rp_range=None` como la
ruta metadata-free.

Antes de ejecutar el pipeline:

- [x] Hacer que la reconstrucción RP de producción pase `rp_range=None` a
  `Reconstruct_RP`; `decode_window` sí acepta internamente el modo
  `"diagonal"`.
- [ ] Reservar el rango real `(R_min, R_max)` únicamente para la variante
  *operator-range oracle* de la ablación.
- [ ] Confirmar que `orientation_method="closed_form"` es el valor
  predeterminado.
- [ ] Mantener `orientation_method="grid"` como baseline histórico y ruta de
  regresión.
- [x] Eliminar del pipeline de producción el cálculo costoso de `rp_range`
  cuando ya no sea necesario.
- [ ] Añadir una prueba que impida confundir `None` con la cadena
  `"diagonal"` en la frontera de `Reconstruct_RP`.

### 0.2. Revisar o retirar `scripts/regen_rp_recon.py`

El script ya selecciona `method="direct2ds"`, por lo que la advertencia anterior
ha quedado obsoleta. Sin embargo, su interfaz, docstring y nombre de copia
`X_all_rec_rho075.npy` siguen presentando `rho` como si controlara esta ruta. En
el decoder `direct2ds`, `rho` es inerte. Antes de usar el script debe renombrarse
o simplificarse para no generar una procedencia falsa.

Para la primera regeneración se recomienda usar `scripts.run_pipeline`.

### 0.3. Fijar el entorno de ejecución

Los scripts actuales ejecutan el pipeline en el checkout WSL
`/home/adrianogarcia/tfg`, mientras que el manuscrito se edita en el checkout de
Windows. Antes de cualquier regeneración:

- [ ] Sincronizar únicamente los archivos revisados mediante
  `_sync_rp_phase0.sh` o un procedimiento equivalente.
- [ ] Registrar hash/commit de ambos checkouts.
- [ ] Verificar checksums de ventanas, etiquetas, sujetos y particiones.
- [ ] Registrar versiones de Python, NumPy, SciPy y bibliotecas BLAS.
- [ ] No copiar resultados de vuelta hasta que la ejecución termine sin fallos.

### 0.4. Pruebas previas obligatorias

Ejecutar:

```powershell
python -m scripts._check_rp_decoders
python -m scripts._check_rp_production
python -m src.rp_decoders
```

Criterios de aceptación:

- [ ] Todas las comprobaciones terminan con código 0.
- [ ] La solución cerrada alcanza error de flotante próximo a precisión de
  máquina en los controles sintéticos.
- [ ] La opción `grid` sigue siendo ejecutable.
- [ ] Las cuatro posibilidades exactas —dos reflexiones y dos signos— llegan a
  la selección mediante RP firmada.
- [ ] No se introducen `NaN`, infinitos ni cambios de forma.

## Fase 1. Preservar la línea base anterior sin confundirla con la ablación

Antes de sobrescribir resultados:

- [ ] Guardar los `X_all_rec.npy` actuales con un nombre que identifique
  claramente el decoder de producción histórico.
- [ ] Guardar los JSON de clasificación RP anteriores.
- [ ] Registrar el commit, fecha, entorno y parámetros del decoder anterior.
- [ ] Mantener una tabla de correspondencia entre artefactos históricos y
  `closed_form`, incluyendo rango, orientación y selector de signo.

La copia actual llamada `X_all_rec_grid720.npy` no cambia únicamente la rejilla:
también empleaba el rango real del operador y un criterio de error firmado
dependiente de escala. Sirve como control de regresión de la producción
anterior, pero no demuestra causalmente el efecto de la orientación. La
comparación pura debe proceder de la Fase 4, manteniendo idénticos el PNG, la
inversión metadata-free y el selector.

## Fase 2. Regenerar las reconstrucciones RP

El encoder y los PNG no han cambiado. Por tanto, deben reutilizarse las imágenes
existentes y regenerarse únicamente las series reconstruidas.

Se necesitan seis ejecuciones:

- LOTO: folds 0, 1 y 2.
- LOSO: folds 0, 1 y 2.

Comando lógico recomendado, ejecutado en el checkout donde residen los datos:

```powershell
foreach ($sampling in @("loto", "loso")) {
    foreach ($fold in 0..2) {
        python -m scripts.run_pipeline `
            --stype RP `
            --sampling $sampling `
            --fold $fold `
            --skip-encode `
            --skip-validate
    }
}
```

No utilizar `--create-numpies`: las ventanas y particiones deben permanecer
idénticas a las usadas en el artículo. El helper `_regen_rp_6folds.sh` ya recorre
los seis casos con los hilos BLAS limitados, pero solo debe ejecutarse después
de la sincronización y del registro de hashes de la Fase 0.3. El hecho de que el
script termine no basta: el código usado en WSL debe coincidir con el revisado
en Windows.

## Fase 3. Controles de integridad de los seis folds

Comprobar en cada fold:

- [ ] Número esperado de ventanas de test.
- [ ] Forma final compatible con `(N, 128, 3)` después de normalizar la
  convención de ejes.
- [ ] Correspondencia exacta de índices, sujetos y etiquetas; si el control es
  muestral, indicar su tamaño y no describirlo como exhaustivo.
- [ ] Ausencia de valores no finitos.
- [ ] Rango físico reconstruido compatible con la calibración compartida.
- [ ] Pearson próximo a uno en el nivel agregado.
- [ ] NMAE limpio igual o inferior al baseline de rejilla.
- [ ] Error de solapamiento y residuo de ciclo documentados, aunque no se
  reduzcan de manera uniforme.
- [ ] Residuo de reproducción de la RP firmada coherente.

Si cualquier fold falla, detener el proceso antes de recalcular tablas o
clasificadores.

La condición frente al baseline histórico es solo una puerta de regresión: dado
que ese baseline incorpora más cambios que la rejilla, superarlo no demuestra
por sí solo que la orientación cerrada sea la causa de la mejora.

`_check_rp_regen.py` combina actualmente LOTO y LOSO en un campo `pooled` de
16.344 observaciones. Ese número cuenta dos veces las mismas 8.172 ventanas bajo
dos particiones y no se debe publicar. Para las tablas globales se utilizará un
solo protocolo como conjunto de ventanas únicas; LOTO y LOSO se mantendrán
separados para clasificación. Además, el Pearson plano no sustituye a la media
por ventana/eje que usa el manuscrito. La comprobación de orden actual toma 40
ventanas y la de ciclo 60 por fold: o se amplían a todo el fold o se declara
explícitamente ese muestreo.

## Fase 4. Ablación completa rejilla frente a solución cerrada

Repetir como mínimo sobre las 150 ventanas oficiales del protocolo de
robustez —30 por actividad y repartidas entre los 51 sujetos—, equivalentes a
450 series unidimensionales.

Comparar:

1. Rejilla de 720 ángulos con corrección de traslación.
2. Solución cerrada con corrección de traslación.
3. Opcionalmente, rejilla sin corrección de traslación como control histórico.

Condiciones de entrada:

- RP nativa en flotante.
- PNG limpio de 8 bits.
- PNG perturbado con `sigma ∈ {2, 4, 8, 16}`.
- Dos réplicas independientes por nivel de ruido, como en el protocolo actual.

Métricas mínimas:

- NMAE: media, desviación, mediana, percentil 95 y máximo.
- Pearson.
- Error de solapamiento.
- Residuo de ciclo/re-encoding.
- Tiempo por eje y por ventana.
- Frecuencia de selección no oracle y, sobre todo, magnitud del regret de NMAE.

La ablación debe generar un JSON o CSV reproducible además de la tabla
Markdown/LaTeX. No deben dependerse las cifras finales de un cálculo interactivo
no guardado.

La ablación existente de 150 ventanas es válida para aislar la orientación
porque mantiene la inversión diagonal en ambos brazos. Sus resultados exigen
una interpretación equilibrada: el closed form reduce NMAE y tiempo, pero el
residuo de RP firmada y el residuo de ciclo son ligeramente peores en algunas
condiciones PNG/ruido. Además, los conteos de “fallos” de la rejilla usan un
umbral prácticamente nulo y candidatos angulares casi equivalentes; deben
acompañarse del regret, no presentarse como errores categóricos graves.

El nivel denominado `float` representa el operador sin cuantización después de
la normalización/centrado definida por el decoder. Debe describirse así y no
como si fuera necesariamente la matriz original sin ninguna transformación.

## Fase 5. Recalcular fidelidad sobre las 8.172 ventanas

Recalcular para RP, agrupando los tres folds:

- MAE, MSE y Pearson en unidades físicas.
- NMAE y NRMSE.
- Error de cambios locales, `Delta-NMAE`.
- Distancia de forma PSD.
- Error de frecuencia dominante.
- Error de energía y masa por bandas.
- Resultados globales, por fold, por eje y por sujeto.

Los generadores `_make_fidelity_tables.py`, `_make_robustness_tables.py` y
`_make_error_tables.py` ya existen en el árbol de trabajo. Antes de aceptarlos
hay que auditar que MAE/NMAE/Pearson se agregan con la misma unidad estadística
que en el texto, escoger un único emisor por tabla y versionarlo. Los resultados
deben guardarse primero como datos estructurados y después convertirse en filas
LaTeX. Para el caso RP flotante se debe imprimir, por ejemplo,
`<10^{-12}` en lugar de redondearlo a `0.0000`.

Pregunta central que debe responder esta fase:

> ¿Confirma el conjunto completo que RP pasa de tener un pequeño término
> algorítmico en flotante a ser exacta hasta precisión numérica?

Solo si la respuesta es afirmativa podrá cambiarse esa afirmación en el
manuscrito.

La fidelidad de los PNG sobre 8.172 ventanas no responde por sí sola a esa
pregunta: para afirmar exactitud del operador flotante debe ejecutarse también
una ruta explícita sin cuantización. El resultado de 150 ventanas puede apoyar
la afirmación como ablación controlada, siempre que el texto declare ese tamaño
de muestra; no debe presentarse como una comprobación exhaustiva de las 8.172.

## Fase 6. Reevaluar los seis clasificadores

Los modelos originales fueron entrenados con señales originales, que no han
cambiado. En principio no es necesario reentrenarlos; sí es obligatorio volver
a evaluarlos sobre las nuevas reconstrucciones RP.

```powershell
foreach ($sampling in @("loto", "loso")) {
    foreach ($fold in 0..2) {
        python -m scripts.run_pipeline `
            --stype RP `
            --sampling $sampling `
            --fold $fold `
            --skip-encode `
            --skip-reconstruct `
            --force
    }
}
```

No usar `--train-new` salvo que falte algún modelo o se descubra que la caché no
corresponde a las particiones actuales.

Recalcular:

- [ ] Accuracy y macro-F1 de CNN, GRU, LSTM, KNN, RF y SVM.
- [ ] Resultados LOTO y LOSO por fold.
- [ ] Diferencias originales menos reconstrucción por sujeto.
- [ ] Intervalos bootstrap simultáneos sobre los 51 sujetos.
- [ ] TOST de equivalencia para márgenes de 0.01 y 0.02.
- [ ] Pruebas de Wilcoxon/McNemar que dependan de RP.
- [ ] Correcciones de Holm de las familias afectadas.

Aunque las predicciones cambien muy poco, no deben conservarse cifras
inferenciales calculadas con la reconstrucción anterior.

La reevaluación de los seis folds y los informes estadísticos deben leer los
JSON de `results/` regenerados. No utilizar `_make_accuracy_tables.py` ni
`_make_loso_tables.py` para esta actualización: actualmente leen
`.results_snapshot` y pueden reintroducir cifras antiguas o columnas mal
ordenadas. El emisor canónico será `_emit_main2_numbers.py`, después de comprobar
que recibe `stat_report_loto.json` y `stat_report_loso.json` nuevos. Se deben
ejecutar las familias inferenciales para ambos protocolos y guardar la
procedencia de las cachés de modelos.

## Fase 7. Actualizar la explicación del decoder en `main2.tex`

Sustituir la descripción de:

- 720 rotaciones.
- Dos reflexiones recorridas mediante rejilla.
- Lista corta de ocho candidatos.
- `n_theta=720` como ajuste fijo.

Por la nueva formulación:

1. Para cada reflexión se expresa el residuo de solapamiento de mínimos
   cuadrados, tras eliminar la traslación relativa, como una forma cuadrática
   en `[cos(theta), sin(theta)]`.
2. Se construye una matriz simétrica de 2x2.
3. Su autovector asociado al menor autovalor proporciona directamente el ángulo
   que minimiza el residuo LS no normalizado implementado.
4. Se conservan ambas reflexiones y ambos signos, dando cuatro candidatos.
5. La RP firmada selecciona el candidato final.
6. La traslación relativa se corrige explícitamente mediante
   `delta = -mean(b_i - a_{i+1})` y se reparte simétricamente entre coordenadas.

Actualizar también la complejidad:

- Rejilla: `O(n_theta N)` para la orientación.
- Solución cerrada: `O(N)` para la orientación.
- El coste global del decoder sigue dominado por la eigendescomposición MDS,
  aproximadamente `O(N^3)`.

La rejilla debe describirse únicamente como baseline histórico de ablación.

No afirmar que este autovector minimiza exactamente el cociente normalizado que
devuelve `align_states`: el código usa una eigendescomposición ordinaria de
(H=C^TC), que resuelve el numerador LS. Para sostener la afirmación sobre el
cociente habría que formular y probar un problema de autovalores generalizado.
También debe distinguirse la traslación relativa entre las dos coordenadas de
la traslación global indeterminada por las distancias.

La mejora de complejidad se refiere solo a la orientación. Aunque MDS siga
siendo (O(N^3)), la implementación completa también regenera una RP firmada
con trabajo (O(N^2)); no debe describirse el decoder entero como lineal.

## Fase 8. Actualizar cifras y afirmaciones de `main2.tex`

Revisar sistemáticamente:

| Parte de `main2.tex` | Contenido RP que puede cambiar |
|---|---|
| Abstract | MAE, exactitud flotante y resumen de contribución |
| Contributions | Descripción de la orientación cerrada |
| Workflow | Nombre de la etapa de orientación |
| Decoder contract | Metadatos, diagonal y complejidad |
| RP reconstruction method | Sustitución completa de la rejilla |
| Experimental settings | Eliminar `n_theta=720` como hiperparámetro |
| Pointwise fidelity | Fila RP de MAE, MSE y Pearson |
| Spectral fidelity | Todas las métricas RP |
| Robustness | Curva RP y descomposición de fuentes |
| Classification | Accuracy/F1 RP LOTO y LOSO |
| Equivalence | Intervalos, TOST y decisiones RP |
| Decoder ablations | Añadir grid720 frente a closed-form |
| Discussion | Posición relativa RP/GASF/GADF |
| Limitations | Coste MDS y alcance del test de ruido |
| Conclusion | MAE final y exactitud de la inversión flotante |

No sustituir números mediante búsqueda y reemplazo global. Cada cifra debe
trazarse hasta un artefacto regenerado.

El borrador actual ya incluye MAE RP `0.0142`, exactitud flotante y la nueva
explicación. Antes de conservarlo deben resolverse dos puntos concretos:

- La comparación `0.0148` frente a `0.0142` sobre 8.172 ventanas parece usar el
  baseline histórico, que no aísla únicamente la rejilla. Debe etiquetarse como
  “producción anterior” o recalcularse con ambos brazos idénticos salvo
  `orientation_method`.
- Las frases que vinculan la solución cerrada con un residuo de solapamiento
  exactamente mínimo deben limitarse al objetivo LS no normalizado. La pequeña
  degradación observada en ciclo/residuo PNG no invalida la mejora de NMAE, pero
  sí impide una afirmación universal de mejora en todas las métricas.

## Fase 9. Actualizar `supplementary.tex`

El suplemento contiene menciones explícitas a `n_theta=720` y cifras RP
dependientes de la reconstrucción anterior. Actualizar:

- Ablación geométrica RP.
- Comparación metadata-free frente al oracle de rango.
- Curvas e intervalos de robustez.
- Residuos de ciclo.
- Métricas espectrales adicionales.
- Resultados por eje.
- Resultados por fold.
- Clasificación completa.
- Intervalos de equivalencia individuales.
- Captions y texto interpretativo.

Añadir una tabla específica:

```text
Grid-720 baseline vs closed-form orientation
```

con flotante, PNG limpio, ruido, tiempo y error de solapamiento.

La tabla ya está presente en el suplemento actual. Debe conservar la separación
entre el estudio de 150 ventanas, que aísla la orientación, y cualquier
comparación histórica de los 8.172 casos. El texto debe añadir que el residuo de
ciclo/imagen no mejora en todas las condiciones y sustituir “fallos de
selección” por “selecciones no oracle” junto con el regret. La ecuación debe
nombrar de forma explícita el residuo LS que realmente optimiza el código.

## Fase 10. Barrido de consistencia textual

Buscar en las fuentes `main2.tex`, `supplementary.tex`, README, documentación y
comentarios del decoder, no en `.aux`, `.fls`, PDF ni planes históricos:

```text
720
n_theta
grid
eight lowest-overlap
eight-candidate
0.0171
0.00035
small algorithmic term
RP additionally carries
operator-range
top_k=8
rho
```

Cada coincidencia debe clasificarse como:

- descripción histórica de la ablación, que puede conservarse;
- descripción del método actual, que debe actualizarse;
- cifra antigua, que debe regenerarse;
- referencia legítima no relacionada, que no debe tocarse.

El README ya apunta a `doc/main2.tex`; solo hay que comprobar que no reaparece
ningún enlace a `doc/main.tex`. Las apariciones de `rho` en decoders geodésicos
pueden ser legítimas, pero son inconsistentes si describen la ruta `direct2ds`.

## Fase 11. Compilación y cierre

Orden de compilación:

1. `main2.tex`.
2. `supplementary.tex`, porque utiliza `\externaldocument{main2}`.

El script `_compile_papers.sh` debe abortar si falla `main2.tex`, antes de
intentar el suplemento. `_check_refs.py` es un apoyo, no una prueba completa:
actualmente no cubre todas las variantes de referencia/cita y puede marcar como
no usadas etiquetas que el suplemento consume mediante `externaldocument`.
Ejecutar `latexmk` hasta estabilizar referencias y revisar también etiquetas
duplicadas, `eqref`/`autoref`, citas y avisos de desbordamiento.

Comprobar:

- [ ] Ausencia de referencias o citas rotas.
- [ ] Tablas sin desbordamientos.
- [ ] Numeración coherente entre artículo y suplemento.
- [ ] Ninguna cifra RP contradictoria entre ambos documentos.
- [ ] PDF principal y suplemento generados correctamente.
- [ ] Los `.aux`, `.fls`, `.fdb_latexmk` y PDF corresponden a las fuentes
  definitivas.

## Entregables finales

- Decoder RP metadata-free con orientación cerrada como método predeterminado.
- Rejilla de 720 ángulos conservada como baseline de ablación.
- Seis `X_all_rec.npy` RP regenerados.
- Resultados estructurados de fidelidad, robustez y clasificación.
- Estadística inferencial RP recalculada.
- Ablación completa grid720/closed-form.
- `doc/main2.tex` actualizado.
- `doc/supplementary.tex` actualizado.
- README corregido para apuntar al manuscrito canónico.
- PDF principal y suplemento recompilados.
- Manifiesto reproducible con commit/hash de código, datos, splits, arrays y
  resultados utilizados por cada tabla.
- Generadores de tablas y estadística incluidos en Git. Los scripts auxiliares
  llamados `_*.py`/`_*.sh` y el directorio `results/` están ignorados en la
  configuración actual: habrá que renombrar/mover los que sean canónicos o
  añadirlos explícitamente.

## Condición de finalización

El trabajo se considerará terminado cuando toda cifra y afirmación RP del
manuscrito pueda trazarse a los nuevos artefactos, no queden descripciones de la
rejilla como método de producción y las conclusiones de fidelidad, robustez y
equivalencia se hayan recalculado con el decoder cerrado. La trazabilidad mínima
será:

```text
cifra del .tex → JSON/CSV fuente → script generador → arrays de entrada
               → hashes de datos/splits → commit y entorno de ejecución
```

No se dará por cerrada una fase solo porque su tabla o PDF exista: debe quedar
demostrado que no procede del snapshot antiguo, de un agregado LOTO+LOSO
duplicado ni del baseline histórico confundido.
