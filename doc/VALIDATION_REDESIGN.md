# Plan de mejora de la validación

> Estado: **rediseño aplicado**; los experimentos que requieren reentrenamiento o
> imágenes genuinamente generadas permanecen marcados como pendientes.
> Objetivo: reorientar el esfuerzo de validación desde comprobaciones redundantes
> hacia la pregunta que importa para el objetivo futuro (augmentación generativa):
> **¿qué ocurre cuando la imagen no es la salida perfecta del encoder?**

## v2 del paper

`doc/main2.tex` es el borrador v2 (derivado de `main.tex`) con la redesign
aplicada: evaluación reorganizada en Fidelidad→Robustez→Utilidad; NMAE/NRMSE/Δ-NMAE
(MRE eliminado); lente espectral (PSD, freq. dominante) sustituyendo al polinomio;
IC bootstrap por sujeto (±0.01) como resultado central de equivalencia; 3
clasificadores en el cuerpo (CNN/GRU/RF); regímenes de metadata. Cabecera del
fichero documenta el changelog y lo marcado como PENDING. Verificado: 0 labels
duplicadas, 0 refs rotas (no compilado con LaTeX; lo valida el autor).

## Implementación (código ya escrito)

| Componente | Fichero | Estado |
|---|---|---|
| Métricas de fidelidad + espectrales (NMAE, NRMSE, Pearson, Δ-NMAE, forma PSD, freq. dominante, energía, bandas, coherencia) | `src/fidelity_metrics.py` (+ auto-check `demo()`) | ✅ hecho y verificado |
| Tabla compacta fidelidad+espectral sobre 8172 ventanas | `scripts/_make_fidelity_tables.py` → `tab:fidelity_spectral` | ✅ genera datos |
| Wilcoxon clustered por sujeto + TOST (orig-vs-recon y pairwise) | `scripts/_stat_tests.py` §5a–5d | ✅ hecho |
| IC bootstrap simultáneo por clúster de sujeto (δ=0.01 y 0.02) | `scripts/_stat_tests.py` §5e | ✅ familia primaria de 18 comparaciones (RP/GASF/GADF) y familia MTF de control (6), por protocolo |
| Curvas de perturbación + cycle-consistency (bloque Robustez) | `scripts/_make_robustness_tables.py` → `tab:robustness` | ✅ 150 ventanas balanceadas, dos realizaciones por nivel no nulo, clipping/redondeo 8-bit y gate por ventana |
| 3 clasificadores en el cuerpo (CNN/GRU/RF) | `main2.tex` `tab:window_vs_subject_out` | ✅ hecho |
| Matriz de transferencia, InceptionTime, robustez a imágenes generadas, subject-out único + inner validation | — | ☐ requiere reentrenar / imágenes generadas |

## Diagnóstico

La validación actual tiene cuatro lentes (inspección visual, error punto a punto,
ajuste polinómico cúbico, seis clasificadores) + dos protocolos de partición +
muchas pruebas estadísticas. Es exhaustiva pero dispersa: dedica mucho espacio a
comprobaciones redundantes y poco a robustez fuera del round-trip perfecto.

## Cambio principal

Reemplazar las cuatro lentes por **tres bloques jerárquicos**:

```
Fidelidad  →  Robustez  →  Utilidad
```

- **Fidelidad**: qué recupera *matemáticamente* cada decoder cuando la imagen es válida.
- **Robustez**: qué ocurre cuando la imagen se aparta del round-trip perfecto.
- **Utilidad**: si la reconstrucción sirve para la tarea aguas abajo.

---

## Bloque 1 — Fidelidad de la inversión

Separar explícitamente **tres fuentes de error** a lo largo de la cadena

```
x  --E-->  M_float  --PNG-->  M_png  --D-->  x̂
```

| Entrada al decoder     | Qué mide                                   |
|------------------------|--------------------------------------------|
| Matriz float original  | Error propio del algoritmo inverso         |
| PNG de 8 bits          | Error introducido por la cuantización      |
| PNG perturbado         | Robustez fuera del round-trip perfecto (→ Bloque 2) |

Debe ser **una tabla común para los cuatro métodos** (RP, GASF, GADF, MTF), no solo
para GADF. Permite afirmaciones del tipo:
*"GADF is exact in floating point, and its observed error is almost entirely
attributable to 8-bit quantisation."*
✅ La tabla común se genera con `scripts/_make_robustness_tables.py` y se mantiene
en el suplemento: usa el operador float nativo, el round trip PNG y `σ=8` para los
cuatro métodos. Así se evita atribuir al inversor GADF el error de un mapeo de almacenamiento.

### Métricas a mantener
- **NMAE** (MAE normalizado por rango).
- **NRMSE** (RMSE normalizado por rango).
- **Pearson**.
- **Error normalizado en la primera diferencia** (conserva cambios locales, no solo amplitud global):

  `Δ-NMAE = (1/(r(N-1))) · Σ_{t=1}^{N-1} |(x_{t+1}-x_t) - (x̂_{t+1}-x̂_t)|`

  Si `r ≤ 10^-8 m/s²`, el eje se omite de las métricas relativas; no se divide
  por un epsilon arbitrario. Ninguna ventana evaluada quedó excluida por esta regla.

### Métrica a eliminar
- **MRE**: dividir por `|x_i| + 1e-8` deja que los cruces por cero del acelerómetro
  dominen el resultado. Sustituir por NMAE/NRMSE por rango.

---

## Bloque 2 — Robustez (sustituye al análisis polinómico) + perturbación

### 2a. Sustituir el ajuste polinómico cúbico por una lente dinámica/espectral

El ajuste `x(t) ≈ a3·t³ + a2·t² + a1·t + a0` es arbitrario para ventanas de
actividad humana: señales con oscilaciones muy distintas pueden dar coeficientes
cúbicos parecidos, y una señal periódica (marcha/carrera) no queda bien descrita
por un polinomio global.

Usar en su lugar métricas en frecuencia (más adecuadas para acelerometría, donde
caminar/correr/subir escaleras se distinguen por ritmo, energía y periodicidad):

- **Distancia de forma PSD** (Welch), con cada PSD normalizada a masa uno:

  `d_shape = ‖P_x / ΣP_x − P_x̂ / ΣP_x̂‖_1`

- **Error de frecuencia dominante**.
- **Error de energía total**.
- **Diferencia de masa espectral por bandas**.
- **Coherencia espectral media** (opcional).

**Elimina** todas las distancias Euclidean / Manhattan / cosine / Mahalanobis entre
coeficientes cúbicos.

- **Soft-DTW**: métrica *secundaria* para tolerar pequeños desplazamientos
  temporales — **no** principal, porque podría ocultar errores de alineación que el
  decoder debería evitar.

### 2b. Robustez fuera del round-trip perfecto
- **Curvas de perturbación**: decodificar PNGs perturbados y medir estabilidad local;
  no se interpreta el ruido gaussiano como sustituto de imágenes generadas.
- **Cycle consistency**: re-encodear la reconstrucción y medir el residuo
  `‖E(x̂) − M‖` en el dominio imagen.

### Tabla principal compacta (Fidelidad + espectral)

| Method | NMAE | Pearson | Δ-NMAE | PSD-shape distance | Dominant-freq error |
|--------|------|---------|-------|--------------|---------------------|
| RP     |      |         |       |              |                     |
| GASF   |      |         |       |              |                     |
| GADF   |      |         |       |              |                     |
| MTF    |      |         |       |              |                     |

---

## Bloque 3 — Utilidad (tarea)

### 3a. Equivalencia a nivel de sujeto (sustituye a McNemar agrupado)

El objetivo no es demostrar que hay diferencia, sino que la reconstrucción es
*suficientemente parecida* al original. Usar **no-inferioridad / equivalencia**:

- Hipótesis: `H0: |Δaccuracy| ≥ δ` con margen principal `δ = 0.01`
  (1 punto porcentual); `δ = 0.02` es una sensibilidad menos exigente.
- Procedimiento:
  1. Accuracy por sujeto de cada método.
  2. `Δ_s = accuracy_{s,recon} − accuracy_{s,original}`.
  3. **Bootstrap pareado por clúster**, remuestreando *sujetos completos*
     (preserva la dependencia intra-sujeto).
  4. Construir intervalos simultáneos sobre la familia confirmatoria de 18
     comparaciones (6 clasificadores × RP/GASF/GADF) mediante bootstrap conjunto
     y máximo desvío. Las 6 comparaciones de MTF forman una familia separada de
     control lossy. Declarar equivalencia solo si todo el intervalo correspondiente
     cae dentro de `[−δ, +δ]`.

Permite afirmar que cada intervalo simultáneo por clasificador queda dentro de la
región de equivalencia; el rango total que los abarca es solo un resumen descriptivo.
(La ausencia de significación en un test de diferencia **no** demuestra equivalencia.)

✅ Hecho: bootstrap simultáneo por sujeto como resultado central y TOST pareado
confirmatorio con margen principal ±0.01.

### 3b. Reducir de seis clasificadores a tres representativos
- **Convolucional temporal fuerte**: InceptionTime o 1D-CNN.
- **Recurrente**: GRU.
- **Clásico**: Random Forest o SVM.

Los seis pueden quedar en el **suplemento**; en el cuerpo, solo tres.

### 3c. Matriz de transferencia (enlaza con el objetivo futuro)

| Entrenamiento         | Evaluación    | Pregunta                                            |
|-----------------------|---------------|-----------------------------------------------------|
| Original              | Original      | Baseline                                            |
| Original              | Reconstrucción| ¿La reconstrucción entra en el mismo dominio? (lo actual) |
| Reconstrucción        | Original      | ¿Permite aprender patrones transferibles?           |
| Original + reconstrucción | Original  | ¿Aporta como augmentación controlada?               |

La fila 3 es una prueba más exigente; la fila 4 enlaza con la augmentación.

---

## Transversal — Coste de metadata

Los métodos no consumen la misma información; compararlos en **tres regímenes**:

| Régimen                  | Información disponible                     |
|--------------------------|-------------------------------------------|
| Image-only               | Solo imagen + configuración global        |
| Common calibration       | Imagen + `(x_min, x_max)`                 |
| Method-specific metadata | Toda la metadata propia del encoder       |

Recordatorio del contrato actual: RP necesita `(R_min, R_max)`; todos necesitan
`(x_min, x_max)` para unidades físicas; MTF mejoraría con medianas de bins + sign bit;
GASF/GADF reconstruyen la forma normalizada desde la imagen.

---

## Cambios de protocolo experimental
- **Grouped subject-out CV** como protocolo *principal* (no el window-level).
- **Inner grouped validation** para seleccionar Q, pivotes y parámetros
  — **nunca** hacer el sweep sobre el test.

## Workflow final

```
Subject-grouped outer split
        │
        ├── Inner grouped validation
        │      └── Selección de Q, pivotes y parámetros
        │
        └── Test subjects
               │
               ├── Exact float encoding
               ├── 8-bit PNG encoding
               └── Perturbed PNG encodings
                        │
                        ▼
                    Decoder
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
 Signal fidelity   Cycle consistency   Task utility
 NMAE / NRMSE      image re-encoding   frozen classifiers
 Pearson           residual            transfer matrix
 Δ-NMAE                                equivalence CI
 PSD-shape distance                    by subject
```

## Tabla de sustituciones concretas

Estado referido a `main2.tex` (v2):

| Elemento actual                         | Sustitución                                          | Estado |
|-----------------------------------------|------------------------------------------------------|--------|
| Visual como "lens"                      | Mantener solo 2–3 ejemplos ilustrativos              | ✅ (figura workflow ya no lo lista como lens; queda `fig:qualitative_main`) |
| MRE                                     | NMAE o NRMSE por rango                               | ✅ |
| Cuatro métricas de polinomio cúbico     | Δ-NMAE + forma PSD + error de frecuencia dominante | ✅ (`tab:fidelity_spectral`) |
| Window-level CV principal               | Grouped subject-out CV                               | ✅ (subject-out ahora primario; equivalencia por sujeto reportada bajo LOSO — RP/GASF/GADF dentro de ±0.01 en los 6 clf; window-level = referencia optimista) |
| Sweep de parámetros en test             | Inner grouped validation o degradar a análisis descriptivo | ⚠ Q de MTF se declara exploratorio; no hay claim confirmatorio de tuning |
| McNemar sobre 8.172 ventanas            | Bootstrap / equivalencia por sujeto                 | ✅ (IC-bootstrap δ=0.01 central + TOST, `tab:equivalence_ci`) |
| Seis clasificadores en el cuerpo        | Tres clasificadores diversos                        | ✅ (CNN/GRU/RF; los 6 al suplemento) |
| Solo imágenes perfectas                 | Curvas de perturbación + cycle consistency          | ✅ (`tab:robustness`, los 4 métodos) |
| Tabla de error solo con PNG             | Tabla común operador float / 8-bit / perturbado (4 métodos) | ✅ movida al suplemento; GADF float nativo exacto y RP conserva error algorítmico no nulo |
