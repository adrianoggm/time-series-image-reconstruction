# Codificación de Series Temporales como Imágenes: Fundamentos Matemáticos de RP, GASF, GADF y MTF

> **Resumen.** Este documento presenta de manera unificada el fundamento matemático, la formulación algorítmica y las propiedades de reconstrucción de las cuatro técnicas de codificación de series temporales como imágenes empleadas en este proyecto: *Recurrence Plot* (RP), *Gramian Angular Summation Field* (GASF), *Gramian Angular Difference Field* (GADF) y *Markov Transition Field* (MTF). Cada técnica se analiza en términos del operador matemático que define la matriz, la información que preserva, las hipótesis de invertibilidad y las estrategias de reconstrucción implementadas en este trabajo. Se cierra con una tabla comparativa y conclusiones sobre la idoneidad de cada representación según la tarea (clasificación, generación, reconstrucción).
>
> **Contribución.** Formalización homogénea de los cuatro operadores, vinculación explícita con la implementación en [src/](../src/), y derivación de conclusiones sobre el espacio funcional de información que cada uno preserva y su comportamiento como representación reversible.

---

## 1. Marco común: la transformación TS → Imagen

Sea una serie temporal univariante $x = \{x_t\}_{t=1}^{N}$ con $x_t \in \mathbb{R}$. Una codificación TS-a-imagen es una aplicación:

$$\mathcal{T}: \mathbb{R}^N \longrightarrow \mathbb{R}^{N \times N},$$

que asocia a $x$ una matriz $M = \mathcal{T}(x)$ interpretable como una imagen en escala de grises. En este proyecto las señales son tri-axiales (acelerómetro, $x$, $y$, $z$), por lo que cada eje genera una matriz independiente y la composición $(M_x, M_y, M_z)$ se interpreta como una imagen RGB de dimensión $N \times N \times 3$ ([src/GADF.py:107-121](../src/GADF.py#L107-L121), [src/recurrence_plots.py:233-247](../src/recurrence_plots.py#L233-L247), [src/MTF.py:36-50](../src/MTF.py#L36-L50)).

La motivación común es **explotar la madurez de las CNN** para tareas de clasificación y generación, transformando un problema 1D en un problema 2D donde la **estructura espacial $(i,j)$ codifica relaciones temporales entre los instantes $i$ y $j$**.

La diferencia entre las cuatro técnicas reside en la **naturaleza del operador** $\mathcal{T}$:

| Método | Operador        | Información codificada                         |
|--------|-----------------|------------------------------------------------|
| RP     | Distancia       | Recurrencia geométrica en el espacio de fases  |
| GASF   | $\cos(\phi_i + \phi_j)$ | Producto angular en coordenadas polares |
| GADF   | $\sin(\phi_i - \phi_j)$ | Diferencia angular en coordenadas polares |
| MTF    | $P(s_i \to s_j)$ | Probabilidad de transición markoviana         |

---

## 2. Recurrence Plot (RP)

### 2.1. Fundamento dinámico

Los *Recurrence Plots* fueron introducidos por **Eckmann, Kamphorst & Ruelle (1987)** como una herramienta de visualización de la dinámica de sistemas no lineales. Su idea central proviene del **teorema de Takens** sobre embedding: cualquier sistema dinámico puede reconstruirse a partir de una única señal escalar embebiéndola en un espacio de fases mediante retardos:

$$\vec{s}_t = (x_t,\, x_{t+\tau},\, x_{t+2\tau},\, \dots,\, x_{t+(m-1)\tau}) \in \mathbb{R}^m,$$

con dimensión de embedding $m$ y retardo $\tau$. En la implementación se usa $m=2$, $\tau=1$ ([src/recurrence_plots.py:111-117](../src/recurrence_plots.py#L111-L117)):

$$\vec{s}_i = (x_i,\, x_{i+1}).$$

### 2.2. Definición clásica de la matriz de recurrencia

Dado un umbral $\epsilon > 0$:

$$R_{i,j} = \Theta\bigl(\epsilon - \|\vec{s}_i - \vec{s}_j\|\bigr),$$

donde $\Theta$ es la función escalón de Heaviside. $R$ es una matriz binaria simétrica que vale $1$ cuando los estados $\vec{s}_i$ y $\vec{s}_j$ son cercanos en el espacio de fases — es decir, **el sistema recurre**.

### 2.3. Variante con signo implementada (`varRP`)

En este trabajo se usa una variante **no umbralizada y con signo** ([src/recurrence_plots.py:97-138](../src/recurrence_plots.py#L97-L138)):

$$R_{i,j} = \sigma_{i,j} \cdot \|\vec{s}_i - \vec{s}_j\|_2,$$

donde el signo $\sigma_{i,j}$ se asigna según la dirección del vector diferencia respecto a la diagonal principal del plano de fases:

$$\sigma_{i,j} = \begin{cases} +1 & \text{si } \cos(\vec{s}_i - \vec{s}_j,\, [1,1]) \geq \tfrac{\sqrt{2}}{2}, \\ -1 & \text{en otro caso.} \end{cases}$$

Esta modificación preserva información direccional (si el sistema "sube" o "baja" entre $i$ y $j$), perdida en la formulación binaria original. La distancia explícita conserva además la **magnitud** de la separación, lo que enriquece la información disponible para la CNN.

### 2.4. Propiedades estructurales

- **Diagonal:** $R_{i,i} = 0$ (distancia consigo mismo).
- **Simetría:** $|R_{i,j}| = |R_{j,i}|$ (la distancia euclídea es simétrica; el signo puede invertirse).
- **Estructuras visuales:** las **líneas diagonales** revelan periodicidad o trayectorias paralelas; los **bloques** indican estados estacionarios; los **patrones difusos** indican comportamiento estocástico.

### 2.5. Reconstrucción inversa: grafo + Dijkstra + MDS

La reconstrucción implementada en [src/recurrence_plots.py:417-506](../src/recurrence_plots.py#L417-L506) sigue el siguiente esquema:

**Paso 1 — Umbralización adaptativa.** Se calcula un umbral $\epsilon_X$ como el percentil 75 de las distancias ([src/recurrence_plots.py:417-420](../src/recurrence_plots.py#L417-L420)):

$$\epsilon_X = \text{percentil}_{75}(\{|R_{i,j}|\}).$$

**Paso 2 — Matriz de coste binaria.**

$$C_{i,j} = \begin{cases} 1 & \text{si } |R_{i,j}| \leq \epsilon_X, \\ 0 & \text{en caso contrario.} \end{cases}$$

**Paso 3 — Grafo ponderado mediante distancia de Jaccard.** Para cada par $(i,j)$ se calcula el peso ([src/recurrence_plots.py:461-472](../src/recurrence_plots.py#L461-L472)):

$$W_{i,j} = 1 - \frac{|G_i \cap G_j|}{|G_i \cup G_j|},$$

donde $G_i = \{k : C_{i,k} = 1\}$ es la vecindad de recurrencia del instante $i$. Esto define una **distancia de Jaccard** sobre los conjuntos de recurrencia: dos instantes están "cerca" si comparten muchas recurrencias.

**Paso 4 — Distancias geodésicas mediante Dijkstra.**

$$D_{i,j} = \text{Dijkstra}(W)_{i,j}.$$

**Paso 5 — Multidimensional Scaling (MDS).** Se obtiene una representación euclídea de dimensión 2 que preserva las distancias geodésicas ([src/recurrence_plots.py:489-491](../src/recurrence_plots.py#L489-L491)):

$$\text{MDS}(D) \to E \in \mathbb{R}^{N \times 2}.$$

**Paso 6 — Proyección en la dirección principal.** Se aplica PCA sobre $E$ y se selecciona la componente del mayor autovalor ([src/recurrence_plots.py:494-504](../src/recurrence_plots.py#L494-L504)) como serie reconstruida.

**Paso 7 — Resolución de la ambigüedad de signo.** La reconstrucción es invariante a inversión, por lo que se compara la RP de la serie reconstruida y de su inverso contra la RP original, eligiendo la de menor distancia euclídea ([src/recurrence_plots.py:537-585](../src/recurrence_plots.py#L537-L585)).

### 2.6. Limitaciones de la reconstrucción RP

- **Pérdida por umbralización:** $C_{i,j}$ es binaria; la magnitud original se pierde.
- **Ambigüedad de inversión:** la RP de $x$ y de $-x$ son indistinguibles.
- **Coste computacional:** Dijkstra $O(N^2 \log N)$ + MDS $O(N^3)$.
- **Requiere metadatos:** $(\min(x), \max(x))$ para desnormalizar.

---

## 3. Gramian Angular Summation Field (GASF)

### 3.1. Codificación polar

Sea la serie normalizada al rango $[-1, 1]$ (o $[0,1]$ como en [src/GAF.py:16-31](../src/GAF.py#L16-L31)):

$$\tilde{x}_t = \frac{x_t - \min(x)}{\max(x) - \min(x)} \cdot 2 - 1, \qquad \tilde{x}_t \in [-1, 1].$$

Cada valor se interpreta como el coseno de un ángulo polar:

$$\phi_t = \arccos(\tilde{x}_t) \in [0, \pi], \qquad r_t = \frac{t}{N}.$$

### 3.2. Definición de GASF

$$G^{\text{sum}}_{i,j} = \cos(\phi_i + \phi_j).$$

Expandida mediante la fórmula del coseno de la suma:

$$\boxed{\;G^{\text{sum}}_{i,j} = \tilde{x}_i\,\tilde{x}_j - \sqrt{1 - \tilde{x}_i^2}\,\sqrt{1 - \tilde{x}_j^2}.\;}$$

Implementación: `tabulate(phi, phi, cos_sum)` ([src/GAF.py:70](../src/GAF.py#L70)).

### 3.3. Propiedades estructurales

- **Simetría:** $G^{\text{sum}}_{i,j} = G^{\text{sum}}_{j,i}$.
- **Diagonal informativa:**

$$G^{\text{sum}}_{i,i} = \cos(2\phi_i) = 2\tilde{x}_i^2 - 1.$$

Esto permite recuperar $|\tilde{x}_i|$ directamente, lo que constituye la **propiedad clave de invertibilidad** de GASF.

### 3.4. Reconstrucción cerrada desde GASF

Implementada en [src/GAF.py:199-222](../src/GAF.py#L199-L222):

**Paso 1 — Lectura de la diagonal.**

$$\phi_i = \frac{\arccos(G^{\text{sum}}_{i,i})}{2}, \qquad \tilde{x}_i = \cos(\phi_i).$$

**Paso 2 — Desnormalización.** Con $(\min(x), \max(x))$ del diccionario externo:

$$x_i = \min(x) + \frac{\tilde{x}_i - \min(\tilde{x})}{\max(\tilde{x}) - \min(\tilde{x})} (\max(x) - \min(x)).$$

### 3.5. Limitaciones

- **Ambigüedad de signo:** $\sqrt{\cdot}$ devuelve $|\tilde{x}_i|$, no su signo. En la implementación actual no se resuelve esta ambigüedad — se asume que la serie está confinada en una rama positiva tras la normalización a $[0,1]$.
- **Información extradiagonal redundante:** la diagonal basta para reconstruir; las entradas $(i,j)$ con $i\neq j$ ofrecen redundancia útil para validación pero no son estrictamente necesarias.

---

## 4. Gramian Angular Difference Field (GADF)

### 4.1. Definición

Bajo la misma codificación polar de §3.1:

$$G^{\text{diff}}_{i,j} = \sin(\phi_i - \phi_j),$$

o equivalentemente:

$$\boxed{\;G^{\text{diff}}_{i,j} = \sqrt{1 - \tilde{x}_i^2}\,\tilde{x}_j - \tilde{x}_i\,\sqrt{1 - \tilde{x}_j^2}.\;}$$

### 4.2. Propiedades estructurales

- **Antisimetría:** $G^{\text{diff}}_{i,j} = -G^{\text{diff}}_{j,i}$.
- **Diagonal nula:** $G^{\text{diff}}_{i,i} = 0$.
- **Información codificada:** sólo **diferencias angulares relativas**, no absolutas.

### 4.3. Restricción de invertibilidad y reconstrucción por fila pivote

A diferencia de GASF, la diagonal de GADF no proporciona información directa. La reconstrucción implementada en [src/GADF.py:209-244](../src/GADF.py#L209-L244) requiere dos ingredientes:

**(a) Escalado restringido a $[-0.7, 0.7]$** ([src/GADF.py:50-57](../src/GADF.py#L50-L57)):

$$\tilde{x}_t^{0.7} = 1.4 \cdot \frac{x_t - \min(x)}{\max(x) - \min(x)} - 0.7,$$

que garantiza $\phi_i - \phi_j \in [-\pi/2, \pi/2]$ y por tanto **unicidad de la rama del $\arcsin$**.

**(b) Selección de la fila pivote $i^*$:**

$$i^* = \arg\min_{i}\, \bigl|\bar{G}^{\text{diff}}_{i}\bigr|,$$

correspondiente al instante cuyo ángulo está más cerca de la mediana angular de la serie. La reconstrucción se efectúa por:

$$\phi_j = \arcsin(G^{\text{diff}}_{i^*,\,j}) + \phi_{i^*}, \qquad \tilde{x}_j = \cos(\phi_j).$$

El ángulo absoluto del pivote se obtiene por anclaje a la fila de máxima media absoluta (véase el documento [GADF_fundamento_y_reconstruccion.md](GADF_fundamento_y_reconstruccion.md) para el desarrollo completo).

### 4.4. Comparativa GASF ↔ GADF

| Aspecto                  | GASF                              | GADF                                |
|--------------------------|-----------------------------------|-------------------------------------|
| Operador                 | $\cos(\phi_i + \phi_j)$           | $\sin(\phi_i - \phi_j)$             |
| Diagonal                 | $2\tilde{x}_i^2 - 1$ (informativa) | $0$ (no informativa)                |
| Simetría                 | Simétrica                         | Antisimétrica                       |
| Reconstrucción           | Cerrada vía diagonal              | Cerrada vía fila pivote ($[-0.7, 0.7]$) |
| Ambigüedad               | Signo de $\tilde{x}_i$            | Constante aditiva global de fase    |
| Coste de reconstrucción  | $O(N)$                            | $O(N)$                              |

---

## 5. Markov Transition Field (MTF)

### 5.1. Discretización y matriz de transición

A diferencia de los tres métodos anteriores —de naturaleza determinista—, MTF tiene fundamento **probabilístico**. Su construcción, propuesta por **Wang & Oates (2015)**, parte de la discretización de la serie en $Q$ estados mediante percentiles ([src/MTF.py:62-71](../src/MTF.py#L62-L71)):

$$s_t = \text{cuantil}_Q(x_t) \in \{1, 2, \dots, Q\}.$$

En la implementación, $Q = 4$ por defecto (`n_bins=4`, [src/MTF.py:30](../src/MTF.py#L30)).

A partir de la secuencia discreta $\{s_t\}_{t=1}^{N}$ se construye la matriz de transición markoviana $W \in [0,1]^{Q \times Q}$:

$$W_{p,q} = P(s_{t+1} = q \mid s_t = p) = \frac{\#\{t : s_t = p,\; s_{t+1} = q\}}{\#\{t : s_t = p\}}.$$

### 5.2. Construcción del campo MTF

El MTF es una expansión $N \times N$ de la matriz $Q \times Q$ que asocia a cada par $(i,j)$ la probabilidad de transición del estado de $x_i$ al estado de $x_j$ ([src/MTF.py:82-85](../src/MTF.py#L82-L85)):

$$\boxed{\;M_{i,j} = W_{s_i,\, s_j}.\;}$$

Geométricamente, $M_{i,j}$ es la probabilidad de pasar del cuantil de $x_i$ al cuantil de $x_j$ en un único paso de la cadena de Markov de primer orden estimada de la serie.

### 5.3. Propiedades estructurales

- **No simétrica:** $M_{i,j} \neq M_{j,i}$ en general (la cadena de Markov es direccional).
- **Diagonal:** $M_{i,i} = W_{s_i, s_i}$ = probabilidad de auto-transición del estado de $x_i$.
- **Rango:** $M_{i,j} \in [0, 1]$.
- **Información codificada:** **estadística de segundo orden** (transiciones) en lugar de relaciones puntuales entre pares.

### 5.4. Reconstrucción inversa: clustering por similitud coseno

La reconstrucción ([src/MTF.py:214-337](../src/MTF.py#L214-L337)) es notablemente más compleja que en GAF/GADF porque MTF **pierde la información del orden temporal absoluto** dentro de cada estado. El esquema implementado es:

**Paso 1 — Matriz de similitud coseno entre filas** ([src/MTF.py:217-222](../src/MTF.py#L217-L222)):

$$S_{i,j} = \frac{\langle M_i, M_j \rangle}{\|M_i\|\,\|M_j\|}.$$

Dos instantes pertenecen al mismo estado markoviano si sus filas en $M$ son cuasi-paralelas.

**Paso 2 — Clustering en $Q$ estados.** Se agrupan los instantes mediante un algoritmo voraz: a partir de un instante semilla se añaden los más similares hasta cubrir $N/Q$ elementos por estado ([src/MTF.py:223-251](../src/MTF.py#L223-L251)).

**Paso 3 — Matriz de similitud inter-estados.** Se calcula la similitud media entre cada par de clusters ([src/MTF.py:253-261](../src/MTF.py#L253-L261)).

**Paso 4 — Ordenación de estados por pivote.** El estado con mayor suma de similitudes (es decir, el más "central") se sitúa en la posición media de un vector de orden. A partir de ahí, se alternan asignaciones a izquierda y derecha del pivote en función de la similitud descendente ([src/MTF.py:267-313](../src/MTF.py#L267-L313)).

**Paso 5 — Asignación de valores y desnormalización.** A cada uno de los $Q=4$ estados ordenados se le asigna un valor representativo del conjunto $\{-1, 1, 3, 5\}$ ([src/MTF.py:320-328](../src/MTF.py#L320-L328)) y la serie resultante se reescala al rango original mediante $(\min(x), \max(x))$.

### 5.5. Limitaciones de la reconstrucción MTF

- **Pérdida de magnitud intra-estado:** todos los instantes de un mismo cuantil reciben el mismo valor reconstruido — se pierde la variabilidad fina.
- **Ambigüedad de orden de estados:** la ordenación por similitud es heurística y puede invertir los extremos.
- **Resolución cuantizada:** con $Q$ estados, la resolución es $\sim (\max - \min)/Q$.
- **Sensible al tamaño de muestra:** la estimación de $W$ requiere suficientes transiciones por bin para ser estable.

---

## 6. Análisis comparativo

### 6.1. Tabla resumen

| Propiedad                          | RP                              | GASF                            | GADF                             | MTF                             |
|------------------------------------|---------------------------------|---------------------------------|----------------------------------|---------------------------------|
| **Naturaleza**                     | Geométrica                      | Trigonométrica (suma)           | Trigonométrica (diferencia)      | Probabilística                  |
| **Operador**                       | $\sigma \cdot \|\vec{s}_i - \vec{s}_j\|$ | $\cos(\phi_i + \phi_j)$ | $\sin(\phi_i - \phi_j)$          | $W_{s_i, s_j}$                  |
| **Simetría**                       | $|R|$ simétrica                 | Simétrica                       | Antisimétrica                    | No simétrica                    |
| **Diagonal**                       | $0$                             | $2\tilde{x}_i^2 - 1$            | $0$                              | $W_{s_i, s_i}$                  |
| **Información directa**            | Distancias + signo direccional  | Magnitudes (vía diagonal)       | Diferencias angulares            | Probabilidades de transición    |
| **Rango**                          | $\mathbb{R}$                    | $[-1, 1]$                       | $[-1, 1]$                        | $[0, 1]$                        |
| **Reconstrucción**                 | Grafo + Dijkstra + MDS + PCA    | Cerrada (diagonal)              | Cerrada (fila pivote, $[-0.7,0.7]$) | Clustering + ordenación heurística |
| **Complejidad reconstrucción**     | $O(N^3)$                        | $O(N)$                          | $O(N)$                           | $O(N^2)$                        |
| **Ambigüedad principal**           | Inversión global ($x \leftrightarrow -x$) | Signo de $\tilde{x}_i$ | Constante aditiva de fase global | Orden y magnitud intra-estado   |
| **Requiere metadato externo**      | Sí: $(\min, \max)$              | Sí: $(\min, \max)$              | Sí: $(\min, \max)$               | Sí: $(\min, \max)$              |
| **Sensibilidad al ruido**          | Media                           | Alta cerca de $\tilde{x}=\pm 1$ | Alta cerca de $\tilde{x}=\pm 1$  | Baja (efecto promediador)       |
| **Información estadística**        | Distancias temporales           | 1er orden                       | 1er orden                        | 2º orden (transiciones)         |

### 6.2. Equivalencias y relaciones formales

- **GASF y GADF son complementarios:** $G^{\text{sum}}_{i,j} + i\,G^{\text{diff}}_{i,j} = e^{-i(\phi_i + \phi_j)} \cdot e^{i\,2\phi_i}$. Es decir, la combinación $G^{\text{sum}} + iG^{\text{diff}}$ es equivalente a una matriz hermitiana $\tilde{z}\tilde{z}^H$ con $\tilde{z}_t = e^{i\phi_t}$, lo que permite reconstrucción óptima por descomposición espectral.
- **RP y GADF coinciden en su antisimetría direccional** (signo del desplazamiento), aunque RP la basa en distancia euclídea en el espacio de fases y GADF en la diferencia angular.
- **MTF es la única no determinista**: dos series con la misma matriz de transición y la misma secuencia de estados producen el mismo MTF aunque sus valores intra-estado difieran. Esto la convierte en una representación **invariante a perturbaciones intra-cuantil**, propiedad deseable para clasificación pero indeseable para reconstrucción fina.

### 6.3. Comportamiento como representación reversible

Ordenadas por **fidelidad de reconstrucción** (menor error MAE/MRE empírico):

1. **GASF** — reconstrucción cuasi-exacta vía diagonal, salvo ambigüedad de signo.
2. **GADF** — reconstrucción cuasi-exacta bajo el escalado $[-0.7, 0.7]$.
3. **RP** — reconstrucción con error moderado debido a la umbralización y a la sensibilidad de MDS al ruido geodésico.
4. **MTF** — reconstrucción de baja resolución, cuantizada a $Q$ niveles; útil sólo cuando la dinámica de estados es la información relevante.

### 6.4. Comportamiento como descriptor de clasificación

Las cuatro representaciones se han evaluado con clasificadores CNN, GRU/LSTM, KNN, SVC y Random Forest ([scripts/cnnvalidation.py](../scripts/cnnvalidation.py), [scripts/gruvalidation.py](../scripts/gruvalidation.py), [scripts/lstmvalidation.py](../scripts/lstmvalidation.py), [scripts/knnvalidation.py](../scripts/knnvalidation.py), [scripts/randomforestvalidation.py](../scripts/randomforestvalidation.py)). Conclusiones cualitativas:

- **RP** suele aportar mejor desempeño en señales **no estacionarias** o con **patrones recurrentes complejos** (caminar, correr) por capturar estructura geométrica del espacio de fases.
- **GASF/GADF** son más estables y discriminativas para señales con **modulación periódica clara**, al codificar relaciones angulares finas.
- **MTF** es robusta frente al **ruido aditivo** y eficaz cuando las clases se distinguen por **dinámica de transición** más que por amplitud (p. ej. estados estacionarios vs. transitorios).

---

## 7. Conclusiones

### 7.1. Sobre los fundamentos matemáticos

Las cuatro técnicas se inscriben en un mismo paradigma —proyectar la dependencia temporal $(i,j)$ en una matriz 2D— pero materializan dicha proyección con operadores radicalmente distintos: **geométrico (RP), trigonométrico aditivo (GASF), trigonométrico diferencial (GADF) y probabilístico (MTF)**. Esta diferencia no es cosmética: determina la naturaleza de la información preservada y, por tanto, qué clase de patrones puede explotar la CNN aguas abajo.

### 7.2. Sobre la invertibilidad

Existe un **trade-off claro entre invarianza e invertibilidad**:

- Las representaciones más invariantes (MTF) son las menos invertibles.
- Las más invertibles (GASF) preservan información que también es ruido para tareas de clasificación robustas.
- **GADF ocupa un punto intermedio óptimo**: con la restricción de escala a $[-0.7, 0.7]$ admite una reconstrucción cerrada $O(N)$ comparable a GASF, pero su estructura antisimétrica aporta a la CNN información direccional ausente en GASF.

### 7.3. Sobre los requisitos auxiliares

**Todas las técnicas requieren metadatos externos $(\min(x), \max(x))$ por ventana y eje** para la desnormalización final. Esta dependencia es consecuencia de que la codificación TS-a-imagen es invariante a escala por construcción (los operadores se aplican sobre series normalizadas). En contextos donde la escala absoluta es crítica, este metadato debe almacenarse junto con la imagen.

### 7.4. Sobre la complejidad computacional

| Operación        | RP             | GASF       | GADF       | MTF         |
|------------------|----------------|------------|------------|-------------|
| Construcción     | $O(N^2)$       | $O(N^2)$   | $O(N^2)$   | $O(N^2)$    |
| Reconstrucción   | $O(N^3)$       | $O(N)$     | $O(N)$     | $O(N^2)$    |

La reconstrucción GADF es **dos órdenes de magnitud más eficiente** que la de RP para series largas, lo que la convierte en la elección recomendada cuando se requiere generación de señal a partir de la imagen.

### 7.5. Recomendación práctica

1. **Para clasificación pura:** usar el conjunto $\{RP, MTF\}$ como representaciones complementarias; sus invarianzas distintas reducen el riesgo de sobreajuste.
2. **Para clasificación + reconstrucción:** preferir GADF (mejor balance) o la combinación $\{GASF, GADF\}$ (reconstrucción óptima por descomposición espectral del campo hermitiano $G^{\text{sum}} + iG^{\text{diff}}$).
3. **Para generación condicional vía modelos de difusión** (como se explora en [scripts/eval_diffusion.py](../scripts/eval_diffusion.py)): GASF/GADF aportan invertibilidad cerrada que minimiza el error de reconstrucción post-muestreo; MTF es preferible cuando se quiere generar trayectorias con dinámica markoviana controlada.
4. **Validación experimental:** medir siempre MAE, MRE y coeficiente de correlación entre serie original y reconstruida, además de la métrica de clasificación; las cuatro funciones `calcular_errores` implementadas ([src/GADF.py:174-191](../src/GADF.py#L174-L191), [src/GAF.py:164-181](../src/GAF.py#L164-L181), [src/MTF.py:170-187](../src/MTF.py#L170-L187)) facilitan esta evaluación.

---

## Referencias

1. **Eckmann, J.-P., Kamphorst, S. O., & Ruelle, D.** (1987). *Recurrence Plots of Dynamical Systems*. Europhysics Letters, 4(9), 973–977. — Introducción del Recurrence Plot.
2. **Marwan, N., Romano, M. C., Thiel, M., & Kurths, J.** (2007). *Recurrence plots for the analysis of complex systems*. Physics Reports, 438(5–6), 237–329. — Revisión exhaustiva de RP y sus variantes.
3. **Takens, F.** (1981). *Detecting strange attractors in turbulence*. Lecture Notes in Mathematics. — Teorema de embedding por retardos.
4. **Wang, Z., & Oates, T.** (2015). *Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks*. AAAI Workshop. — Introducción de GAF (GASF/GADF) y MTF.
5. **Wang, Z., & Oates, T.** (2015). *Imaging Time-Series to Improve Classification and Imputation*. IJCAI.
6. **Hatami, N., Gavet, Y., & Debayle, J.** (2018). *Classification of time-series images using deep convolutional neural networks*. International Conference on Machine Vision. — Estudio comparativo entre GAF, MTF y RP como entrada de CNN.
