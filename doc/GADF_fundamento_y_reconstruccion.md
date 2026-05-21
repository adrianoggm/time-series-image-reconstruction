# Fundamento Matemático y Reconstrucción del Gramian Angular Difference Field (GADF)

> **Resumen.** Este documento presenta el fundamento matemático y algorítmico del *Gramian Angular Difference Field* (GADF) y describe cómo se puede reconstruir la serie temporal original a partir de dicha representación. Se desarrolla la transformación polar/gramiana, la expresión cerrada de GADF, las propiedades de invertibilidad (y sus ambigüedades intrínsecas), y un esquema práctico de reconstrucción numérica tal y como se implementa en [src/GADF.py](../src/GADF.py).
>
> **Contribución.** Se formaliza la transformación directa GADF, se demuestra la invertibilidad bajo restricción de escala a $[-0.7, 0.7]$, y se documenta un esquema de reconstrucción por **fila pivote** que explota la estructura antisimétrica del campo gramiano.

---

## 1. Introducción

### 1.1. Contexto

La representación de series temporales como imágenes facilita el uso de modelos de procesamiento de imágenes, en particular redes neuronales convolucionales (CNNs). Dos transformaciones populares en este marco son el *Gramian Angular Summation Field* (GASF) y el *Gramian Angular Difference Field* (GADF), introducidas por **Wang & Oates (2015)**. Ambas mapean una serie temporal unidimensional a matrices —interpretables como imágenes— que capturan dependencias temporales mediante relaciones trigonométricas entre pares de instantes.

### 1.2. Problema

El objetivo de este apartado es doble:

1. **Formalizar** matemáticamente cómo se construye GADF y bajo qué condiciones la transformación admite una inversión analítica.
2. **Documentar** el esquema de reconstrucción implementado en [src/GADF.py:209-244](../src/GADF.py#L209-L244), que permite recuperar la serie original cuasi-exactamente a partir de la imagen GADF y de un metadato de calibración mínimo.

---

## 2. Notación y preprocesado

Sea la serie temporal original de longitud $N$:

$$x = \{x_t\}_{t=1}^{N}, \qquad x_t \in \mathbb{R}.$$

La normalización al rango $[-1, 1]$, necesaria para garantizar el dominio de definición de $\arccos$, se define como:

$$\tilde{x}_t = \frac{x_t - \min(x)}{\max(x) - \min(x)} \cdot 2 - 1, \qquad \tilde{x}_t \in [-1, 1].$$

Esta operación se corresponde con `normalize_series` en [src/GADF.py:42-49](../src/GADF.py#L42-L49). Para la **variante GADF de la implementación** se emplea, sin embargo, un escalado más restringido al intervalo $[-0.7, 0.7]$ mediante `normalize_series07` ([src/GADF.py:50-57](../src/GADF.py#L50-L57)):

$$\tilde{x}_t^{0.7} = 1.4 \cdot \frac{x_t - \min(x)}{\max(x) - \min(x)} - 0.7, \qquad \tilde{x}_t^{0.7} \in [-0.7, 0.7].$$

Esta restricción es **crucial para la invertibilidad** del método propuesto y se justifica en §6.2.

---

## 3. Codificación angular (polar)

A cada punto $(t, \tilde{x}_t)$ se le asocia un ángulo polar $\phi_t$ y un radio relativo $r_t$:

$$\phi_t = \arccos(\tilde{x}_t) \in [0, \pi], \qquad r_t = \frac{t}{N} \in (0, 1].$$

La información de la magnitud queda codificada en $\phi_t$ (puesto que $\cos\phi_t = \tilde{x}_t$) y la posición temporal puede incorporarse en $r_t$ si se desea representar gráficamente la serie en coordenadas polares.

> **Nota de implementación.** En [src/GADF.py:66](../src/GADF.py#L66) se calcula `phi = np.arccos(normalized_series)`, donde la serie ya ha sido escalada al rango $[0,1]$ (no $[-1,1]$) mediante `normalize_to_zero_one`. Ambas son convenciones equivalentes salvo una traslación constante del eje angular.

---

## 4. Definición de GASF y GADF

Usando las representaciones angulares, se definen las matrices gramianas que codifican combinaciones trigonométricas entre pares de ángulos.

### 4.1. Gramian Angular Summation Field (GASF)

$$G^{\text{sum}}_{i,j} = \cos(\phi_i + \phi_j).$$

Expandiendo mediante la fórmula del coseno de la suma:

$$G^{\text{sum}}_{i,j} = \tilde{x}_i \tilde{x}_j - \sqrt{1 - \tilde{x}_i^2}\,\sqrt{1 - \tilde{x}_j^2}.$$

### 4.2. Gramian Angular Difference Field (GADF)

$$G^{\text{diff}}_{i,j} = \sin(\phi_i - \phi_j).$$

Usando la identidad $\sin\phi = \sqrt{1 - \cos^2\phi} = \sqrt{1 - \tilde{x}^2}$ (con $\phi \in [0,\pi]$ y la raíz no negativa), se obtiene la forma cerrada:

$$\boxed{\;G^{\text{diff}}_{i,j} = \sqrt{1 - \tilde{x}_i^2}\,\tilde{x}_j - \tilde{x}_i\,\sqrt{1 - \tilde{x}_j^2}.\;}$$

La implementación de esta operación en [src/GADF.py:39-41](../src/GADF.py#L39-L41) se realiza vectorialmente mediante `tabulate(phi, phi, sin_diff)` ([src/GADF.py:83](../src/GADF.py#L83)).

### 4.3. Observaciones estructurales

| Propiedad        | GASF                                      | GADF                                  |
|------------------|-------------------------------------------|---------------------------------------|
| Simetría         | Simétrica: $G^{\text{sum}}_{i,j} = G^{\text{sum}}_{j,i}$ | Antisimétrica: $G^{\text{diff}}_{i,j} = -G^{\text{diff}}_{j,i}$ |
| Diagonal         | $G^{\text{sum}}_{i,i} = \cos(2\phi_i) = 2\tilde{x}_i^2 - 1$ | $G^{\text{diff}}_{i,i} = 0$           |
| Información directa | Magnitudes (vía diagonal)              | Diferencias relativas (no absolutas)  |

### 4.4. Composición RGB tri-axial

En el contexto experimental de este proyecto, las series acelerométricas son tri-axiales (ejes $x$, $y$, $z$). Por ello, se construye una imagen RGB superponiendo los tres campos GADF independientes en los canales rojo, verde y azul ([src/GADF.py:107-121](../src/GADF.py#L107-L121)), generando una representación de dimensión $N \times N \times 3$ que preserva la información completa de los tres ejes.

---

## 5. Invertibilidad y reconstrucción — análisis teórico

### 5.1. Reconstrucción desde GASF (caso simple)

Por la diagonal de GASF:

$$G^{\text{sum}}_{i,i} = \cos(2\phi_i) = 2\tilde{x}_i^2 - 1 \;\;\Longrightarrow\;\; \tilde{x}_i = \pm\sqrt{\frac{G^{\text{sum}}_{i,i} + 1}{2}}.$$

Esto recupera el **valor absoluto** de $\tilde{x}_i$ en cada paso. La ambigüedad de signo se resuelve localmente mediante continuidad temporal o la información de signo de las entradas extradiagonales. Por tanto, GASF contiene una ruta directa y numéricamente estable hacia la reconstrucción.

### 5.2. Reconstrucción desde GADF — dificultades

GADF codifica **únicamente diferencias angulares** mediante el seno:

$$G^{\text{diff}}_{i,j} = \sin(\phi_i - \phi_j).$$

Esto introduce dos limitaciones intrínsecas:

1. **Diagonal nula** ($G^{\text{diff}}_{i,i} = 0$): no hay lectura directa de $\phi_i$.
2. **Ambigüedad de rama**: $\sin(\theta) = \sin(\pi - \theta)$, por lo que sólo conocer $\sin(\Delta\phi)$ no determina $\Delta\phi$ de forma única si $\Delta\phi$ no está restringido.

Conocer **todas las diferencias exactas** $\phi_i - \phi_j$ (sin ambigüedad de rama) determina el vector $\phi$ salvo una **constante aditiva global**. Con sólo el seno, la recuperación exige hipótesis adicionales sobre el rango de las diferencias.

Formalmente, la reconstrucción requiere resolver el sistema no lineal y sobre-determinado:

$$\text{Encontrar } \phi \in [0,\pi]^N \text{ tal que } \sin(\phi_i - \phi_j) = G^{\text{diff}}_{i,j}, \;\forall i, j.$$

---

## 6. Reconstrucción cerrada por fila pivote (método implementado)

La implementación en [src/GADF.py:209-244](../src/GADF.py#L209-L244) propone un esquema de reconstrucción **cerrada** —no iterativa— que aprovecha la estructura antisimétrica de $G^{\text{diff}}$ y resuelve la ambigüedad de rama mediante la restricción de escala a $[-0.7, 0.7]$.

### 6.1. Identidad fundamental para la inversión por filas

Considérese la fila $i$-ésima de $G^{\text{diff}}$. Por la identidad trigonométrica del seno de una diferencia:

$$G^{\text{diff}}_{i,j} = \sin(\phi_i - \phi_j) \;\;\Longrightarrow\;\; \phi_i - \phi_j = \arcsin(G^{\text{diff}}_{i,j}),$$

bajo el supuesto $\phi_i - \phi_j \in [-\pi/2, \pi/2]$. Despejando:

$$\boxed{\;\phi_j = \phi_i - \arcsin(G^{\text{diff}}_{i,j}), \quad \forall j \in \{1,\dots,N\}.\;}$$

Es decir, **conocer una única fase de referencia $\phi_i$ determina unívocamente todas las demás fases** a partir de la fila $i$-ésima de la matriz GADF.

### 6.2. Justificación del escalado restringido a $[-0.7, 0.7]$

La validez de la identidad anterior exige que $\arcsin$ opere en su rama principal $[-\pi/2, \pi/2]$. Si $\tilde{x}_t \in [-1,1]$, entonces $\phi_t \in [0,\pi]$ y por tanto $\phi_i - \phi_j \in [-\pi, \pi]$, lo que **excede el dominio de invertibilidad** del arcoseno.

Al restringir $\tilde{x}_t^{0.7} \in [-0.7, 0.7]$:

$$\phi_t = \arccos(\tilde{x}_t^{0.7}) \in [\arccos(0.7),\, \arccos(-0.7)] \approx [0.795,\, 2.346],$$

con lo que $\phi_i - \phi_j \in [-1.55,\, 1.55] \subset [-\pi/2, \pi/2]$, **asegurando la unicidad de la rama del $\arcsin$** y por tanto la invertibilidad estricta del campo. Las constantes `MINVALUE = -0.7` y `MAXVALUE = 0.7` ([src/GADF.py:216-217](../src/GADF.py#L216-L217)) materializan esta cota teórica.

### 6.3. Selección de la fila pivote: criterio de mínima media absoluta

El algoritmo selecciona la fila $i^*$ cuya media es más próxima a cero ([src/GADF.py:218-221](../src/GADF.py#L218-L221)):

$$i^* = \arg\min_{i} \left| \frac{1}{N} \sum_{j=1}^{N} G^{\text{diff}}_{i,j} \right|.$$

**Justificación.** Dado que $G^{\text{diff}}_{i,j} = \sin(\phi_i - \phi_j)$, la media de la fila $i$ es proporcional a la diferencia angular promedio entre $\phi_i$ y el resto de fases de la señal. Por la antisimetría de $G^{\text{diff}}$, la fila cuya media tiende a cero corresponde a un instante $i^*$ cuyo ángulo $\phi_{i^*}$ se sitúa aproximadamente en la **mediana angular** de la serie, es decir, $\phi_{i^*} \approx \overline{\phi}$. Esta elección minimiza la magnitud máxima de las diferencias $|\phi_{i^*} - \phi_j|$, manteniéndolas dentro del dominio invertible del $\arcsin$ y reduciendo el error de aproximación.

Simétricamente, la fila de **máxima** media absoluta $i_{\max} = \arg\max_i |\bar{G}_i|$ corresponde al instante de fase más extrema, y se utiliza como anclaje para reconstruir el ángulo absoluto del pivote.

### 6.4. Recuperación del ángulo pivote absoluto

La identidad de §6.1 reconstruye fases *relativas*. Para fijar el ángulo absoluto del pivote se emplea ([src/GADF.py:222](../src/GADF.py#L222)):

$$\phi_{i^*} = \arcsin(G^{\text{diff}}_{i_{\max},\, i^*}) + \phi_{\text{ref}},$$

donde $\phi_{\text{ref}} = 0.7$ es la cota superior del escalado normalizado. Esta operación corresponde a un desplazamiento de la rama del $\arcsin$ que sitúa el pivote en el centro del intervalo angular válido.

### 6.5. Reconstrucción de la serie normalizada

Aplicando la identidad de §6.1 con $\phi_{i^*}$ como referencia ([src/GADF.py:224-227](../src/GADF.py#L224-L227)):

$$\phi_j = \arcsin(G^{\text{diff}}_{i^*,\, j}) + \phi_{i^*}, \qquad \tilde{x}_j = \cos(\phi_j).$$

### 6.6. Desnormalización y recuperación de la escala física

La transformación polar es invariante a la escala, por lo que la serie reconstruida $\tilde{x}_j$ se halla en un rango arbitrario. Para recuperar las unidades físicas originales se requiere un **diccionario externo de calibración** que almacene $(\min(x), \max(x))$ por cada eje y muestra ([src/GADF.py:242](../src/GADF.py#L242)):

$$x_j = \min(x) + \frac{\tilde{x}_j - \min(\tilde{x})}{\max(\tilde{x}) - \min(\tilde{x})} \cdot (\max(x) - \min(x)).$$

Esta dependencia de un metadato externo constituye la principal limitación del esquema: la imagen GADF preserva la *forma* de la señal pero no su *escala absoluta*, hecho consistente con la invariancia por escala del campo gramiano.

---

## 7. Métodos alternativos de reconstrucción

### 7.1. Optimización no lineal (caso general)

Cuando no se dispone del escalado restringido a $[-0.7, 0.7]$ o cuando se quiere mejorar la robustez frente al ruido, se puede formular el problema como una **estimación por mínimos cuadrados no lineales**:

- **Parámetros:** vector $\phi = (\phi_1, \dots, \phi_N)$ (desconocido). Se fija una referencia global, p. ej. $\phi_1 = 0$, ya que la solución es ambigua salvo una constante aditiva.
- **Función de coste:**

$$\min_{\phi} \;\sum_{i<j} w_{ij}\,\bigl(\sin(\phi_i - \phi_j) - G^{\text{diff}}_{i,j}\bigr)^2,$$

donde $w_{ij} \geq 0$ pondera la confianza (mayor peso en pares cercanos temporalmente).

- **Optimización:** descenso por **Gauss–Newton** o **Levenberg–Marquardt** con inicialización apropiada. Una inicialización razonable puede derivarse integrando diferencias aproximadas si $\arcsin(G^{\text{diff}}_{i,i+1})$ está en rango pequeño, o resolviendo una versión linealizada para pequeñas diferencias ($\sin(\Delta) \approx \Delta$).

### 7.2. Formulación compleja

Considérese $z_t = e^{i\phi_t}$. Entonces:

$$\sin(\phi_i - \phi_j) = \operatorname{Im}(z_i \bar{z}_j).$$

Denotando la matriz hermitiana $Z = z z^H$, su parte imaginaria contiene GADF. Recuperar $z$ a partir de $\operatorname{Im}(Z)$ es análogo a un problema de **factorización de matriz con restricción de módulo** ($|z_t| = 1$): una tarea no convexa pero abordable por métodos espectrales (SVD de bajo rango) seguidos de refinamiento por optimización y proyección en la variedad de fases unitarias.

### 7.3. Combinación GASF + GADF

Cuando ambos campos están disponibles, la estrategia óptima es:
1. Usar **GASF** para obtener $|\tilde{x}_i|$ por la diagonal.
2. Usar **GADF** para resolver la ambigüedad de signo y afinar la reconstrucción mediante consistencia cruzada.

---

## 8. Análisis del error de reconstrucción

El error total puede descomponerse en tres contribuciones independientes:

1. **Error de truncamiento del escalado** ($\epsilon_{\text{sat}}$): proviene del recorte en `np.where(... >= 0.7, 0.7, ...)`; no nulo únicamente en las muestras saturadas.
2. **Error de cuantización RGB** ($\epsilon_q$): al codificar la matriz como imagen de 8 bits ([src/GADF.py:152](../src/GADF.py#L152)), la precisión queda limitada a $2^{-8}$ por canal, lo que se traduce en un error angular máximo de $\Delta\phi \approx 1/255$ rad.
3. **Error de aliasing del $\arcsin$** ($\epsilon_a$): **nulo por construcción** si se respeta el escalado a $[-0.7, 0.7]$.

El error agregado en la serie reconstruida puede acotarse por:

$$|x_j - \hat{x}_j| \leq |\sin(\phi_j)| \cdot (\Delta\phi_q + \Delta\phi_{i^*}) \cdot (\max(x) - \min(x)),$$

donde $\Delta\phi_{i^*}$ es el error en la estimación del pivote. La función `calcular_errores` ([src/GADF.py:174-191](../src/GADF.py#L174-L191)) cuantifica empíricamente el MAE y el MRE promedio sobre la señal reconstruida.

---

## 9. Propiedades numéricas y consideraciones prácticas

- **Ruido y estabilidad:** GADF depende implícitamente de raíces cuadradas $\sqrt{1 - \tilde{x}^2}$; los valores $\tilde{x} \approx \pm 1$ generan inestabilidad numérica. La normalización a $[-0.7, 0.7]$ y el truncamiento (*clipping*) reducen estos errores.
- **Ambigüedades:** sin GASF o información adicional (signo inicial, muestra de referencia), la reconstrucción desde GADF sólo es posible salvo constante aditiva global y con posibles ambigüedades de rama. La continuidad temporal o una regularización adecuada elimina soluciones no físicas.
- **Complejidad computacional:** la construcción de GADF es $\mathcal{O}(N^2)$ en tiempo y memoria. La optimización iterativa para reconstrucción es $\mathcal{O}(N^2)$ por iteración si se usan todos los pares; para series largas se recomiendan pesos locales o submuestreo. **El método de fila pivote implementado, por contraste, es $\mathcal{O}(N)$ una vez identificada la fila pivote.**

---

## 10. Limitaciones teóricas

- **No es universalmente invertible:** GADF por sí sola no garantiza reconstrucción única sin hipótesis adicionales sobre $\phi$ (monotonía, rango restringido de diferencias, o información adicional).
- **Sensibilidad al preprocesado:** el escalado inicial y el manejo de extremos son críticos para conservar la invertibilidad numérica.
- **Pérdida de escala absoluta:** la invariancia por escala del campo gramiano impide recuperar la amplitud sin metadatos externos.

---

## 11. Recomendaciones prácticas

- Cuando esté disponible, **preferir GASF** para reconstrucción directa por la información explícita de su diagonal.
- Si sólo se dispone de GADF:
  - aplicar el **método de fila pivote** con escalado restringido a $[-0.7, 0.7]$ (esquema cerrado, óptimo en tiempo);
  - alternativamente, usar pesos que prioricen pares cercanos temporalmente en una formulación por optimización;
  - inicializar con integraciones locales de $\arcsin$ o con una estimación de fase por PCA/espectral;
  - añadir regularización de suavidad temporal: $\lambda \sum_t (\phi_{t+1} - \phi_t)^2$.
- **Validar** la reconstrucción midiendo RMSE entre la serie original y la reconstruida, y probar robustez frente a ruido sintético.

---

## 12. Conclusión

GADF es una representación rica que codifica las **diferencias angulares** entre instantes de una serie temporal y es especialmente valiosa para tareas de clasificación y análisis estructural mediante CNNs. Matemáticamente, GADF refleja la matriz de senos de diferencias angulares y por tanto contiene información **relativa**, no absoluta; su inversión general exige resolver un problema no lineal sujeto a ambigüedades de fase y rama.

El método de reconstrucción propuesto e implementado en este trabajo demuestra que, pese a esta limitación teórica general, el GADF admite una **reconstrucción cuasi-exacta y cerrada** cuando se cumplen dos condiciones:

1. El escalado de la serie original se restringe a $[-0.7, 0.7]$ para garantizar la unicidad de la rama del $\arcsin$.
2. Se dispone de los extremos $(\min(x), \max(x))$ como metadato auxiliar de desnormalización.

El criterio de selección del pivote por **mínima media absoluta de fila** proporciona un anclaje robusto, fundamentado en la estructura antisimétrica del campo gramiano, que minimiza la propagación del error angular en la inversión. En la práctica, la combinación de GASF (para magnitud) y GADF (para relaciones) ofrece la ruta más estable y completa a la reconstrucción.

---

## Referencias

1. **Wang, Z., & Oates, T.** (2015). *Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks*. AAAI Workshop on Statistical, Structural and Relational Pattern Recognition. — Introducción original de GAF/GADF.
2. **Wang, Z., & Oates, T.** (2015). *Imaging Time-Series to Improve Classification and Imputation*. International Joint Conference on Artificial Intelligence (IJCAI).
3. Literatura posterior sobre GAF/GADF y su uso en aprendizaje profundo (búsqueda recomendada: *"Gramian Angular Field reconstruction"*).
