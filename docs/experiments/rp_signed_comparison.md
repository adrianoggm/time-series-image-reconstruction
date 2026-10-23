# Comparativa RP: qué aporta el signo y qué aporta el decodificador

Plan experimental acordado el 2026-10-23 para la parte comparativa de RP del
benchmark multi-dataset (roadmap M4 y M7, issues #64, #65, #67 y #85). Se
ejecuta sobre los siete datasets (WISDM, PAMAP2, STEAD, Swarm MAGx, PTB-XL,
NinaPro DB2, ESC-50), con exactamente las mismas ventanas para todos los métodos.

## Punto de partida

No comparamos dos algoritmos sobre la misma representación. Trabajamos con un
**Signed RP**, mientras que los métodos clásicos se formularon para el RP
convencional, sin signo. Hay que separar tres cosas: reproducir el método
clásico, adaptarlo a nuestro Signed RP y demostrar qué aporta la nueva
reconstrucción.

Con estados adyacentes $s_i=(x_i,x_{i+1})$:

- RP convencional (distancias): $D_{ij}=\lVert s_i-s_j\rVert_2$.
- Signed RP (`main3.tex`, ec. `eq:FormulaRP`; `src/recurrence_plots.py::varRP2`):
  $R_{ij}=\sigma_{ij}\lVert s_i-s_j\rVert_2$, con $\sigma_{ij}=+1$ si
  $\lVert s_i-s_j\rVert_2=0$ o $\hat u_{ij}\cdot v\ge\cos(\pi/4)$, y $-1$ en otro caso,
  donde $\hat u_{ij}=(s_i-s_j)/\lVert s_i-s_j\rVert_2$ y $v=(1,1)^\top/\sqrt2$.

En precisión exacta $|R_{ij}|=D_{ij}$: el Signed RP contiene las mismas
distancias continuas que el RP convencional y añade información direccional.
Esa información puede resolver la ambigüedad de orientación (reflexión) que
deja el MDS, pero no todas: la invariancia a traslaciones sigue ahí.

> **Corrección respecto a la nota original.** El umbral es $\pi/4$ (coseno
> $\ge\sqrt2/2$), no $3\pi/4$; el código y el manuscrito coinciden. Consecuencia
> para el punto 4: un desplazamiento $d$ y su opuesto $-d$ forman ángulos
> $\theta$ y $\pi-\theta$ con $v$. Si $\theta\le\pi/4$ o $\theta\ge3\pi/4$ reciben
> signos distintos (par informativo). Si $\pi/4<\theta<3\pi/4$ —desplazamientos
> cercanos a la antidiagonal $(1,-1)$— **ambos reciben $-1$** y el par no
> distingue la señal de su inversa.

## Experimento A — ¿qué aporta el signo?

Todo igual salvo el acceso al signo.

| Variante | Información | Reconstrucción |
|---|---|---|
| RP unsigned | $\lvert R^{signed}\rvert$ | MDS 2-D + solapamiento, sin signo |
| RP signed | $R^{signed}$ | el mismo MDS 2-D + solapamiento, orientación por el signo |

Métricas: MAE, NRMSE, correlación y **porcentaje de ventanas en que el signo
elige bien la orientación**. Como cota, un *oracle* que elige entre las dos
orientaciones conociendo la señal original, solo en evaluación y etiquetado
como oracle; no es un resultado de reconstrucción autónoma.

Si el signed mejora, la mejora se atribuye a la información direccional, no a
haber sustituido Jaccard por MDS 2-D.

## Experimento B — ¿qué aporta nuestro decodificador?

Misma entrada (Signed RP) para ambos.

| Decoder | Geometría | Orientación |
|---|---|---|
| Legacy adaptado | Jaccard + Dijkstra + MDS 1-D | comprobación con el Signed RP |
| RP nuevo | distancias continuas + MDS 2-D + solapamiento | comprobación con el Signed RP |

Mide cuánto mejora la reconstrucción gracias a la recuperación geométrica
bidimensional y al solapamiento entre estados, con la representación controlada.

## Experimento C — RP clásico de la literatura

Reproducción independiente del procedimiento de Hirata et al. (2008) sobre el
RP convencional correspondiente. No se presenta como si Hirata lo hubiera
diseñado para Signed RP. Si el método publicado no especifica cómo recuperar la
señal completa con su orientación y longitud, se separa el algoritmo original
del posprocesamiento que añadimos para poder calcular métricas.

## Matriz de experimentos

| Método | Entrada | Qué demuestra |
|---|---|---|
| Hirata clásico | RP convencional | referencia histórica |
| Legacy adaptado | Signed RP | reconstrucción anterior |
| Direct-2D unsigned | RP continuo sin signo | ablación del signo |
| Direct-2D signed | Signed RP | propuesta completa |
| GEO-2D signed | Signed RP | ablación geométrica adicional |

Todo en los siete datasets, con las mismas ventanas, y **por separado** en
matrices de coma flotante y en PNG de 8 bits: la normalización y la
cuantización pueden afectar de forma distinta al RP firmado y al no firmado.

## Comprobación adicional: cuánta información de signo hay

Con la regla angular no todas las parejas de estados distinguen una orientación
de su inversa (ver la corrección arriba). Medir, por dataset:

1. proporción de parejas con información direccional que diferencia ambas orientaciones;
2. en cuántas ventanas la orientación correcta resulta inequívoca;
3. en qué condiciones el signo no permite resolverla;
4. qué pasa al aumentar la cuantización o añadir perturbaciones pequeñas.

Es especialmente relevante para sismología, ECG, magnetometría y el resto de
dominios, donde la geometría de los estados adyacentes puede ser muy distinta
de la del acelerómetro.

## Afirmación que buscamos sostener

No que el Signed RP sea universalmente invertible, sino:

> The signed recurrence representation provides additional directional
> constraints that can reduce the orientation ambiguity inherent to
> distance-based reconstruction. Combined with two-dimensional
> multidimensional scaling and adjacent-state overlap consistency, these
> constraints enable an improved operational reconstruction of temporal
> trajectories.

Dos aportaciones experimentales separables: el valor real de la información
direccional (A) y que nuestro procedimiento geométrico la aprovecha mejor que la
reconstrucción anterior (B), con Hirata (C) como referencia histórica.

## Qué existe ya en el código

| Pieza | Estado |
|---|---|
| Direct-2D signed | producción: `rp_decoders.decode_window(source="direct", signed=True)` |
| Direct-2D unsigned | parcial: la ablación "Without signed selection" de `scripts/_ablation_rp_2d.py` (`signed=False`) quita la selección por signo; falta la variante que solo recibe $\lvert R\rvert$ |
| GEO-2D signed | `source="geo"` (issue #67) |
| Legacy adaptado | `source="jaccard", dims=1`, pero con las desviaciones de `docs/audit/m0_baseline.md` (Jaccard sobre todos los pares, MDS clásico, ρ = 0,60); restaurar la versión fiel en #65 |
| Hirata clásico | no implementado (#64) |
| Oracle de orientación | el *regret* frente al candidato NMAE-óptimo de `scripts/_ablation_rp_orientation.py` es un oracle sobre los cuatro candidatos O(2) |
| Proporción de parejas informativas | no existe |
